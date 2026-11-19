# -*- coding: utf-8 -*-
"""
MisarRouter: Adaptive Context-Aware Arabic NLP Preprocessing Router.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from .pipelines import (
    SUPPORTED_EMOJI_MODES,
    DialectPreservePipeline,
    MSALightPipeline,
    NoisySocialPipeline,
)
from .profiler import LinguisticProfile, LinguisticProfiler
from .tokenizer import (
    HuggingFaceTokenizerAdapter,
    TokenizerProtocol,
    load_default_tokenizer,
)


@dataclass
class RoutingResult:
    original_text: str
    processed_text: str
    route_name: str
    route_display_ar: str
    decision_reason: str
    profile: LinguisticProfile
    latency_ms: float
    tokenizer_backend: str
    is_real_tokenizer: bool
    raw_tokens: List[str]
    processed_tokens: List[str]
    raw_token_count: int
    processed_token_count: int
    unk_count_raw: int
    unk_count_processed: int
    tfr_raw: float
    tfr_processed: float
    token_reduction_pct: float
    auto_healed_count: int = 0
    clitic_fixed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "processed_text": self.processed_text,
            "route_name": self.route_name,
            "route_display_ar": self.route_display_ar,
            "decision_reason": self.decision_reason,
            "profile": self.profile.to_dict(),
            "latency_ms": round(self.latency_ms, 4),
            "tokenizer_backend": self.tokenizer_backend,
            "is_real_tokenizer": self.is_real_tokenizer,
            "raw_tokens": list(self.raw_tokens),
            "processed_tokens": list(self.processed_tokens),
            "raw_token_count": self.raw_token_count,
            "processed_token_count": self.processed_token_count,
            "unk_count_raw": self.unk_count_raw,
            "unk_count_processed": self.unk_count_processed,
            "tfr_raw": round(self.tfr_raw, 3),
            "tfr_processed": round(self.tfr_processed, 3),
            "token_reduction_pct": round(self.token_reduction_pct, 2),
            "auto_healed_count": self.auto_healed_count,
            "clitic_fixed_count": self.clitic_fixed_count,
        }


class MisarRouter:
    """
    Context-aware preprocessing router for Arabic NLP Transformer models.
    Profiles input text in O(n) time and routes it dynamically to the optimal
    micro-pipeline while measuring real WordPiece Token Fragmentation Ratio (TFR).
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        model_key: str = "arabertv02",
        allow_vocab_download: bool = True,
        preserve_orthography: bool = True,
        arabert_placeholders: bool = False,
        noise_threshold: float = 0.18,
        dialect_threshold: float = 0.10,
        auto_heal: bool = True,
        emoji_mode: str = "strip",
        arabizi_mode: bool = False,
    ):
        if emoji_mode not in SUPPORTED_EMOJI_MODES:
            raise ValueError(f"Unsupported emoji_mode '{emoji_mode}'. Choose from {SUPPORTED_EMOJI_MODES}.")

        if tokenizer is None:
            self.tokenizer: TokenizerProtocol = load_default_tokenizer(
                model_key=model_key,
                allow_download=allow_vocab_download,
            )
        elif hasattr(tokenizer, "has_token") and hasattr(tokenizer, "is_real_vocab"):
            self.tokenizer = tokenizer
        else:
            self.tokenizer = HuggingFaceTokenizerAdapter(tokenizer)

        self.model_key = model_key
        self.noise_threshold = noise_threshold
        self.dialect_threshold = dialect_threshold
        self.auto_heal = auto_heal
        self.emoji_mode = emoji_mode
        self.arabizi_mode = arabizi_mode
        self.profiler = LinguisticProfiler()
        self.msa_pipeline = MSALightPipeline(preserve_orthography=preserve_orthography)
        self.dialect_pipeline = DialectPreservePipeline(
            preserve_orthography=preserve_orthography,
            tokenizer=self.tokenizer,
            auto_heal=auto_heal,
            arabizi_mode=arabizi_mode,
            emoji_mode=emoji_mode,
        )
        self.noisy_pipeline = NoisySocialPipeline(
            preserve_orthography=preserve_orthography,
            arabert_placeholders=arabert_placeholders,
            tokenizer=self.tokenizer,
            auto_heal=auto_heal,
            emoji_mode=emoji_mode,
            arabizi_mode=arabizi_mode,
        )
        # Fair static regex normalizer mirroring classical AraBERT/tnkeeh-style ة->ه / أ->ا cleanup
        self._legacy_baseline = NoisySocialPipeline(
            preserve_orthography=False,
            arabert_placeholders=False,
            tokenizer=None,
            auto_heal=False,
            emoji_mode="strip",
        )

    def _select_and_process(self, text: str, profile: LinguisticProfile):
        if profile.total_chars == 0:
            return (
                "FAST_MSA",
                "مسار الفصحى المباشر فائق السرعة",
                "نص فارغ أو يتكون من مسافات فقط تم تمريره تلقائياً",
                "",
                0,
                0,
            )

        if profile.elongation_noise_score >= self.noise_threshold or profile.repeated_char_runs >= 1:
            hybrid = len(profile.detected_markers) > 0
            route_name = "NOISY_SOCIAL"
            route_display = (
                "مسار تنقية التواصل الاجتماعي (مع صون اللهجة)"
                if hybrid
                else "مسار تنقية شبكات التواصل والضجيج اللفظي"
            )
            reason = (
                f"ارتفاع مؤشر الضجيج والمدود ({profile.elongation_noise_score:.2f} >= {self.noise_threshold:.2f})"
                + (f" مع رصد سمات لهجية ({', '.join(profile.detected_markers[:3])})" if hybrid else " وتكرار الحروف غير القياسي")
            )
            processed, healed, clitic_fixed = self.noisy_pipeline.process_with_stats(
                text, apply_dialect_cleanup=hybrid
            )
            return route_name, route_display, reason, processed, healed, clitic_fixed

        if profile.dialect_ratio >= self.dialect_threshold or len(profile.detected_markers) >= 1:
            route_name = "DIALECT_PRESERVE"
            route_display = "مسار التطبيع اللهجي الحافظ للأدوات الدلالية"
            markers_preview = "، ".join(dict.fromkeys(profile.detected_markers[:3]))
            neg_note = " — تم صون أدوات النفي اللهجية" if profile.has_protected_negation else ""
            reason = f"رصد سمات لهجية بارزة ({profile.dialect_ratio:.1%} من الكلمات: {markers_preview}){neg_note}"
            processed, healed, clitic_fixed = self.dialect_pipeline.process_with_stats(text)
            return route_name, route_display, reason, processed, healed, clitic_fixed

        if self.arabizi_mode and profile.latin_chars > 0:
            # Transliterate opt-in Arabizi inside MSA/mixed lines as well
            processed, healed, clitic_fixed = self.dialect_pipeline.process_with_stats(text)
            return (
                "DIALECT_PRESERVE",
                "مسار التطبيع اللهجي الحافظ للأدوات الدلالية",
                "تفعيل محول العربيزي والتطبيع السياقي على مدخلات مختلطة",
                processed,
                healed,
                clitic_fixed,
            )

        route_name = "FAST_MSA"
        route_display = "مسار الفصحى المباشر فائق السرعة"
        if profile.arabic_chars == 0 and profile.total_chars > 0:
            reason = "نص غير عربي أو أرقام ورموز تم تمريره عبر المسار المباشر دون تعديل"
        elif profile.diacritic_density >= 0.25:
            reason = f"نص فصيح أو تراثي مشكول (كثافة التشكيل {profile.diacritic_density:.2f}) تم تفريغ حركاته مع حفظ الرسم الإملائي"
        else:
            reason = "نص فصحى معياري نقي تم إعفاؤه من التحليل الصرفي الثقيل مع صون الهمزات والتاء المربوطة"
        processed, healed, clitic_fixed = self.msa_pipeline.process_with_stats(text)
        if self.emoji_mode != "strip":
            from .pipelines import MULTISPACE_REGEX, process_emojis
            processed = MULTISPACE_REGEX.sub(" ", process_emojis(processed, self.emoji_mode)).strip()
        return route_name, route_display, reason, processed, healed, clitic_fixed

    def preprocess(self, text: str) -> str:
        """Fast drop-in preprocessing returning only the cleaned routed string."""
        if not text or not text.strip():
            return ""
        profile = self.profiler.analyze(text)
        _, _, _, processed, _, _ = self._select_and_process(text, profile)
        return processed

    def route(self, text: str) -> RoutingResult:
        t0 = time.perf_counter()
        profile = self.profiler.analyze(text)
        route_name, route_display, reason, processed, healed, clitic_fixed = self._select_and_process(text, profile)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_tokens = self.tokenizer.tokenize(text)
        proc_tokens = self.tokenizer.tokenize(processed)
        raw_count = len(raw_tokens)
        proc_count = len(proc_tokens)
        if self.tokenizer.is_real_vocab:
            unk_raw = sum(1 for tok in raw_tokens if tok == "[UNK]" or not self.tokenizer.has_token(tok))
            unk_proc = sum(1 for tok in proc_tokens if tok == "[UNK]" or not self.tokenizer.has_token(tok))
        else:
            unk_raw = raw_tokens.count("[UNK]")
            unk_proc = proc_tokens.count("[UNK]")

        raw_words = max(1, profile.word_count) if profile.word_count > 0 else 1
        proc_word_count = len(processed.split())
        proc_words = max(1, proc_word_count) if proc_word_count > 0 else 1

        tfr_raw = (raw_count / raw_words) if raw_count > 0 else 0.0
        tfr_proc = (proc_count / proc_words) if proc_count > 0 else 0.0
        reduction_pct = ((raw_count - proc_count) / raw_count * 100.0) if raw_count > 0 else 0.0

        return RoutingResult(
            original_text=text,
            processed_text=processed,
            route_name=route_name,
            route_display_ar=route_display,
            decision_reason=reason,
            profile=profile,
            latency_ms=elapsed_ms,
            tokenizer_backend=self.tokenizer.backend_name,
            is_real_tokenizer=self.tokenizer.is_real_vocab,
            raw_tokens=raw_tokens,
            processed_tokens=proc_tokens,
            raw_token_count=raw_count,
            processed_token_count=proc_count,
            unk_count_raw=unk_raw,
            unk_count_processed=unk_proc,
            tfr_raw=tfr_raw,
            tfr_processed=tfr_proc,
            token_reduction_pct=reduction_pct,
            auto_healed_count=healed,
            clitic_fixed_count=clitic_fixed,
        )

    def route_batch(self, texts: Iterable[str]) -> List[RoutingResult]:
        return [self.route(t) for t in texts]

    def preprocess_batch(self, texts: Iterable[str]) -> List[str]:
        """High-throughput batch preprocessing without per-item double tokenization overhead."""
        return [self.preprocess(t) for t in texts]

    def preprocess_stream(self, texts: Iterable[str]) -> Iterable[str]:
        """Memory-constant generator wrapper for streaming large Arabic corpora."""
        for t in texts:
            yield self.preprocess(t)

    def hf_dataset_map(
        self,
        batch: Dict[str, Any],
        text_column: str = "text",
        output_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Drop-in batched callable for Hugging Face `datasets.Dataset.map(..., batched=True)`.
        Preserves unrelated columns cleanly.
        """
        target = output_column or text_column
        items = batch[text_column]
        cleaned = [self.preprocess(item) if isinstance(item, str) else item for item in items]
        updated = dict(batch)
        updated[target] = cleaned
        return updated

    def compare_baselines(self, text: str) -> Dict[str, Any]:
        """
        Three-way scientific comparison: Raw Input vs Static Destructive Baseline (ة->ه, أ->ا)
        vs Misar Adaptive v2 Router.
        """
        misar_res = self.route(text)
        legacy_text = self._legacy_baseline.process(text, apply_dialect_cleanup=False)
        legacy_tokens = self.tokenizer.tokenize(legacy_text)
        legacy_count = len(legacy_tokens)
        if self.tokenizer.is_real_vocab:
            legacy_unk = sum(1 for tok in legacy_tokens if tok == "[UNK]" or not self.tokenizer.has_token(tok))
        else:
            legacy_unk = legacy_tokens.count("[UNK]")
        legacy_words = max(1, len(legacy_text.split())) if legacy_text else 1
        legacy_tfr = (legacy_count / legacy_words) if legacy_count > 0 else 0.0
        vs_legacy_pct = (
            ((legacy_count - misar_res.processed_token_count) / legacy_count * 100.0)
            if legacy_count > 0
            else 0.0
        )

        return {
            "misar": misar_res.to_dict(),
            "legacy_text": legacy_text,
            "legacy_tokens": legacy_tokens,
            "legacy_token_count": legacy_count,
            "legacy_unk_count": legacy_unk,
            "legacy_tfr": round(legacy_tfr, 3),
            "vs_legacy_reduction_pct": round(vs_legacy_pct, 2),
        }

    def wrap_hf_pipeline(self, hf_pipeline: Callable[..., Any]) -> Callable[..., Any]:
        """
        Wrap any Hugging Face Transformers pipeline so Arabic inputs are automatically
        routed and cleaned by Misar before model inference.
        """
        def _wrapped(inputs: Any, *args: Any, **kwargs: Any) -> Any:
            if isinstance(inputs, str):
                cleaned = self.preprocess(inputs)
            elif isinstance(inputs, list):
                cleaned = [self.preprocess(item) if isinstance(item, str) else item for item in inputs]
            else:
                cleaned = inputs
            return hf_pipeline(cleaned, *args, **kwargs)

        return _wrapped


_DEFAULT_ROUTER: Optional[MisarRouter] = None


def _get_default_router() -> MisarRouter:
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = MisarRouter()
    return _DEFAULT_ROUTER


def route(text: str) -> RoutingResult:
    return _get_default_router().route(text)


def preprocess(text: str) -> str:
    return _get_default_router().preprocess(text)


def preprocess_batch(texts: Iterable[str]) -> List[str]:
    return _get_default_router().preprocess_batch(texts)
