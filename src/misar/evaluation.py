# -*- coding: utf-8 -*-
"""
Scientific Downstream & Cross-Model Evaluation Suite for Misar (مِسار).
Quantifies minimal-pair semantic disambiguation, colloquial negation protection,
sentiment emoji retention, and quadratic self-attention FLOPs savings across models.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .benchmark import BENCHMARK_CORPUS
from .router import MisarRouter
from .tokenizer import PRETRAINED_MODEL_CONFIGS, load_default_tokenizer

# Minimal orthographic pairs where legacy static normalization (ة->ه, أ/إ->ا) causes semantic collision.
SEMANTIC_MINIMAL_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("عالمة", "عالمه"),
    ("مديرة", "مديره"),
    ("سنة", "سنه"),
    ("جنة", "جنه"),
    ("قصة", "قصه"),
    ("كتابة", "كتابه"),
    ("حملة", "حمله"),
    ("مسلمة", "مسلمه"),
    ("مهمة", "مهمه"),
    ("أسر", "اسر"),
)

NEGATION_SAFETY_CASES: Tuple[Tuple[str, str], ...] = (
    ("المكتبة مو معقدة أبداً", "مو"),
    ("السيرفر مب بطيء في الإصدار الجديد", "مب"),
    ("النظام ماكو أسرع منه في التشغيل", "ماكو"),
    ("الخدمة مافيه أي تأخير الحين", "مافيه"),
    ("النسخة الجديدة مافيها أخطاء", "مافيها"),
    ("التطبيق مش بطيء في المعالجة", "مش"),
)

SENTIMENT_EMOJI_CASES: Tuple[Tuple[str, str], ...] = (
    ("استجابة الخادم بطيييئة مرررة 😡😡👎", "سلبي"),
    ("الإصدارررر الجديييييد ممتاز ومستقر 😍🔥🚀", "إيجابي"),
    ("البرمجه صارت سلسة في المنصه ✅👍", "إيجابي"),
    ("في هذه المنظومة تحديات تقنية معقدة 💔😤", "سلبي"),
)


def evaluate_system(router: Optional[MisarRouter] = None) -> Dict[str, Any]:
    """Evaluate semantic safety, disambiguation, polarity, and token/attention efficiency."""
    active = router or MisarRouter()
    sentiment_router = MisarRouter(
        tokenizer=active.tokenizer,
        model_key=active.model_key,
        emoji_mode="sentiment",
        auto_heal=active.auto_heal,
    )

    misar_distinct = 0
    legacy_distinct = 0
    for word_a, word_b in SEMANTIC_MINIMAL_PAIRS:
        if active.preprocess(word_a) != active.preprocess(word_b):
            misar_distinct += 1
        if active._legacy_baseline.process(word_a) != active._legacy_baseline.process(word_b):
            legacy_distinct += 1

    neg_kept = sum(
        1 for text, marker in NEGATION_SAFETY_CASES if marker in active.preprocess(text)
    )
    sent_kept = sum(
        1
        for text, tag in SENTIMENT_EMOJI_CASES
        if tag in sentiment_router.preprocess(text)
    )

    tot_raw = 0
    tot_legacy = 0
    tot_misar = 0
    tot_unk_raw = 0
    tot_unk_misar = 0
    for item in BENCHMARK_CORPUS:
        comp = active.compare_baselines(item["text"])
        tot_raw += comp["misar"]["raw_token_count"]
        tot_legacy += comp["legacy_token_count"]
        tot_misar += comp["misar"]["processed_token_count"]
        tot_unk_raw += comp["misar"]["unk_count_raw"]
        tot_unk_misar += comp["misar"]["unk_count_processed"]

    token_red_raw = ((tot_raw - tot_misar) / tot_raw * 100.0) if tot_raw else 0.0
    token_red_leg = ((tot_legacy - tot_misar) / tot_legacy * 100.0) if tot_legacy else 0.0
    attn_saved_raw = ((1.0 - (tot_misar / tot_raw) ** 2) * 100.0) if tot_raw else 0.0

    return {
        "model_key": active.model_key,
        "tokenizer_backend": active.tokenizer.backend_name,
        "is_real_tokenizer": active.tokenizer.is_real_vocab,
        "minimal_pairs_tested": len(SEMANTIC_MINIMAL_PAIRS),
        "misar_disambiguation_pct": round(misar_distinct / len(SEMANTIC_MINIMAL_PAIRS) * 100.0, 2),
        "legacy_disambiguation_pct": round(legacy_distinct / len(SEMANTIC_MINIMAL_PAIRS) * 100.0, 2),
        "negation_preservation_pct": round(neg_kept / len(NEGATION_SAFETY_CASES) * 100.0, 2),
        "sentiment_polarity_pct": round(sent_kept / len(SENTIMENT_EMOJI_CASES) * 100.0, 2),
        "total_raw_tokens": tot_raw,
        "total_legacy_tokens": tot_legacy,
        "total_misar_tokens": tot_misar,
        "total_unk_raw": tot_unk_raw,
        "total_unk_misar": tot_unk_misar,
        "token_reduction_vs_raw_pct": round(token_red_raw, 2),
        "token_reduction_vs_legacy_pct": round(token_red_leg, 2),
        "attention_flops_savings_pct": round(attn_saved_raw, 2),
    }


def run_evaluation(
    model_keys: Optional[Sequence[str]] = None,
    offline: bool = False,
    json_output: bool = False,
) -> List[Dict[str, Any]]:
    """Run cross-model scientific evaluation and print a clean summary table."""
    keys = list(model_keys) if model_keys else ["arabertv02"]
    reports: List[Dict[str, Any]] = []

    for key in keys:
        tok = load_default_tokenizer(model_key=key, allow_download=not offline)
        router = MisarRouter(tokenizer=tok, model_key=key)
        reports.append(evaluate_system(router))

    if json_output:
        import json
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return reports

    print("=" * 110)
    print("Misar Scientific Cross-Model Evaluation | Disambiguation · Polarity · Tokens · O(L^2) Attention Savings")
    print("=" * 110)
    print(f"{'Model Key':<14} | {'Backend':<36} | {'Disamb M/L':<11} | {'Neg%':>5} | {'Sent%':>5} | {'Tok -Raw/-Leg':<16} | {'O(L^2)':>7}")
    print("-" * 110)
    for rep in reports:
        dis = f"{rep['misar_disambiguation_pct']:.0f}% / {rep['legacy_disambiguation_pct']:.0f}%"
        tok_str = f"-{rep['token_reduction_vs_raw_pct']:.1f}% / -{rep['token_reduction_vs_legacy_pct']:.1f}%"
        attn_str = f"-{rep['attention_flops_savings_pct']:.1f}%"
        backend = str(rep["tokenizer_backend"])[:36]
        print(
            f"{rep['model_key']:<14} | {backend:<36} | {dis:<11} | "
            f"{rep['negation_preservation_pct']:>4.0f}% | {rep['sentiment_polarity_pct']:>4.0f}% | "
            f"{tok_str:<16} | {attn_str:>7}"
        )
    print("=" * 110)
    return reports
