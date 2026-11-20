# -*- coding: utf-8 -*-
"""
Specialized Transformer-aware Arabic preprocessing micro-pipelines for Misar.
Key architectural finding: Modern Arabic Transformer vocabularies (e.g., AraBERTv2)
preserve Hamzas (أ/إ/آ) and Taa Marbuta (ة). Destructive legacy normalization (ة -> ه)
actively increases subword fragmentation; Misar defaults to orthography-preserving
cleaning while offering optional legacy normalization for classical search engines.
"""

import re
from typing import Optional, Tuple

from .lexicons import (
    ARABIC_INDIC_DIGITS,
    ARABIZI_CHAR_MAP,
    ARABIZI_LEXICON,
    ARABIZI_TECH_PROTECT_REGEX,
    CLITIC_LATIN_REGEX,
    DIALECT_SPELLING_MAP,
    EMAIL_REGEX,
    EMOJI_REGEX,
    EXPRESSIVE_MARRA_SPELL_REGEX,
    GEMINATED_FALLBACK_PATTERNS,
    HASHTAG_REGEX,
    MENTION_REGEX,
    NEGATIVE_EMOJIS,
    POSITIVE_EMOJIS,
    POSSESSIVE_AMBIGUOUS_H_WORDS,
    PUNCT_REPEAT_REGEX,
    REPEATED_CHARS_REGEX,
    TASHKEEL_REGEX,
    TATWEEL_REGEX,
    URL_REGEX,
    ZERO_WIDTH_REGEX,
    normalize_unicode_safety,
)
from .tokenizer import TokenizerProtocol

ALEF_WASLA_REGEX = re.compile(r"\u0671")
LEGACY_ALEF_REGEX = re.compile(r"[إأآٱ]")
LEGACY_TAA_REGEX = re.compile(r"ة\b")
LEGACY_YAA_REGEX = re.compile(r"ى\b")
MULTISPACE_REGEX = re.compile(r"\s+")
MARKDOWN_HEADER_REGEX = re.compile(r"(^|\n)\s*#{1,6}\s+")
MARKDOWN_FORMAT_REGEX = re.compile(r"[*~`]")
ARABIC_FINAL_HAA_WORD_REGEX = re.compile(r"\b([\u0621-\u064A]{4,})\b")
EMOJI_RUN_REGEX = re.compile(r"[\U00010000-\U0010FFFF\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF\uFE0F\u200D]+", flags=re.UNICODE)
ARABIZI_CANDIDATE_REGEX = re.compile(r"\b([A-Za-z][A-Za-z0-9]*|[2356789][A-Za-z]{2,}[0-9]*)\b")
SUPPORTED_EMOJI_MODES = ("strip", "keep", "sentiment")


def clean_code_switching_clitics(text: str) -> Tuple[str, int]:
    """Attach Arabic clitics separated by kashida before Latin tokens (e.g., الـ Backend -> الBackend)."""
    return CLITIC_LATIN_REGEX.subn(r"\1\2\3", text)


def heal_orthography_with_vocab(text: str, tokenizer: Optional[TokenizerProtocol]) -> Tuple[str, int]:
    """
    O(1) per-word vocabulary-verified orthography repair: heals dropped Taa Marbuta (ه -> ة)
    only when the raw word is not a single vocab token, not an ambiguous possessive pronoun,
    and the Taa Marbuta form exists cleanly in the loaded Transformer vocabulary.
    """
    if tokenizer is None or not getattr(tokenizer, "is_real_vocab", False):
        return text, 0

    repaired = 0

    def _heal_match(match: re.Match) -> str:
        nonlocal repaired
        word = match.group(1)
        if not word.endswith("ه"):
            return word
        stem = word[1:] if len(word) > 4 and word[0] in ("و", "ف", "ب", "ل", "ك") else word
        if word in POSSESSIVE_AMBIGUOUS_H_WORDS or stem in POSSESSIVE_AMBIGUOUS_H_WORDS:
            return word
        if tokenizer.has_token(word):
            return word
        candidate = word[:-1] + "ة"
        if tokenizer.has_token(candidate):
            repaired += 1
            return candidate
        return word

    healed = ARABIC_FINAL_HAA_WORD_REGEX.sub(_heal_match, text)
    return healed, repaired


def process_emojis(text: str, emoji_mode: str = "strip") -> str:
    """Handle emoji runs according to strip, keep (deduplicated), or sentiment-tag policy."""
    if emoji_mode == "strip":
        return EMOJI_REGEX.sub(" ", text)

    if emoji_mode == "keep":
        def _keep_dedup(match: re.Match) -> str:
            chars = [ch for ch in match.group(0) if ch not in ("\uFE0F", "\u200D")]
            deduped = []
            for ch in chars:
                if not deduped or deduped[-1] != ch:
                    deduped.append(ch)
            return f" {''.join(deduped)} " if deduped else " "

        return EMOJI_RUN_REGEX.sub(_keep_dedup, text)

    if emoji_mode == "sentiment":
        def _tag_sentiment(match: re.Match) -> str:
            run = match.group(0)
            pos = sum(1 for ch in run if ch in POSITIVE_EMOJIS)
            neg = sum(1 for ch in run if ch in NEGATIVE_EMOJIS)
            if neg > pos:
                return " سلبي "
            if pos > neg:
                return " إيجابي "
            return " "

        tagged = EMOJI_RUN_REGEX.sub(_tag_sentiment, text)
        return re.sub(r"\b(سلبي|إيجابي)(\s+\1)+\b", r"\1", tagged)

    raise ValueError(f"Unsupported emoji_mode '{emoji_mode}'. Choose from {SUPPORTED_EMOJI_MODES}.")


def normalize_arabizi(text: str) -> str:
    """Opt-in Franco-Arabic transliteration guarded against software/protocol tokens."""
    try:
        from .arabizi import decode_arabizi
        return decode_arabizi(text)
    except Exception:
        return text


def strip_markup_noise(text: str) -> str:
    """Strip Markdown syntax (headers, asterisks, formatting noise) to protect Transformers."""
    t = MARKDOWN_HEADER_REGEX.sub(r"\1", text)
    return MARKDOWN_FORMAT_REGEX.sub("", t)


def collapse_elongations(text: str, tokenizer: Optional[TokenizerProtocol] = None) -> str:
    """
    Reduce exaggerated character stretches (3+ consecutive identical chars) while
    preserving valid Arabic geminated letters (e.g., الللله -> الله, متكرررر -> متكرر)
    using loaded Transformer vocabulary lookup when available.
    """
    t = text
    for pattern, replacement in GEMINATED_FALLBACK_PATTERNS:
        t = pattern.sub(replacement, t)

    if not REPEATED_CHARS_REGEX.search(t):
        return t

    has_vocab = tokenizer is not None and getattr(tokenizer, "is_real_vocab", False)

    def _fix_word(match: re.Match) -> str:
        word = match.group(0)
        if not REPEATED_CHARS_REGEX.search(word):
            return word
        punct_match = re.search(r"([.,!؟؛،]+)$", word)
        trailing_punct = punct_match.group(1) if punct_match else ""
        clean_word = word[:-len(trailing_punct)] if trailing_punct else word
        cand_single = REPEATED_CHARS_REGEX.sub(r"\1", clean_word)
        if not has_vocab:
            return cand_single + trailing_punct
        if tokenizer.has_token(cand_single):
            return cand_single + trailing_punct
        cand_double = REPEATED_CHARS_REGEX.sub(r"\1\1", clean_word)
        if tokenizer.has_token(cand_double):
            return cand_double + trailing_punct
        return cand_single + trailing_punct

    return re.sub(r"\S+", _fix_word, t)


class MSALightPipeline:
    """
    Ultra-fast Modern Standard Arabic & Classical diacritized pass.
    Strips diacritics/tatweel/zero-width marks and normalizes Alef Wasla while
    preserving Transformer-native orthography (أ/إ/آ, ة, ى) by default.
    """

    def __init__(
        self,
        preserve_orthography: bool = True,
        normalize_digits: bool = False,
        emoji_mode: str = "strip",
    ):
        self.preserve_orthography = preserve_orthography
        self.normalize_digits = normalize_digits
        self.emoji_mode = emoji_mode

    def process_with_stats(self, text: str) -> Tuple[str, int, int]:
        t = normalize_unicode_safety(text)
        t = strip_markup_noise(t)
        t = ZERO_WIDTH_REGEX.sub("", t)
        t, clitic_fixed = clean_code_switching_clitics(t)
        t = process_emojis(t, self.emoji_mode)
        t = TASHKEEL_REGEX.sub("", t)
        t = TATWEEL_REGEX.sub("", t)
        if self.preserve_orthography:
            t = ALEF_WASLA_REGEX.sub("ا", t)
        else:
            t = LEGACY_ALEF_REGEX.sub("ا", t)
            t = LEGACY_TAA_REGEX.sub("ه", t)
            t = LEGACY_YAA_REGEX.sub("ي", t)
        if self.normalize_digits:
            t = t.translate(ARABIC_INDIC_DIGITS)
        if REPEATED_CHARS_REGEX.search(t):
            t = collapse_elongations(t)
        return MULTISPACE_REGEX.sub(" ", t).strip(), 0, clitic_fixed

    def process(self, text: str) -> str:
        cleaned, _, _ = self.process_with_stats(text)
        return cleaned


class DialectPreservePipeline:
    """
    Dialect-safe preprocessing pass.
    Protects colloquial negation and sentiment particles (مو، مب، مافيه، ماكو)
    while applying vocabulary-aligned gentle spelling normalization and optional Auto-Heal.
    """

    def __init__(
        self,
        preserve_orthography: bool = True,
        tokenizer: Optional[TokenizerProtocol] = None,
        auto_heal: bool = True,
        arabizi_mode: bool = False,
        emoji_mode: str = "strip",
    ):
        self.preserve_orthography = preserve_orthography
        self.tokenizer = tokenizer
        self.auto_heal = auto_heal
        self.arabizi_mode = arabizi_mode
        self.emoji_mode = emoji_mode

    def process_with_stats(self, text: str) -> Tuple[str, int, int]:
        t = normalize_unicode_safety(text)
        t = strip_markup_noise(t)
        t = ZERO_WIDTH_REGEX.sub("", t)
        t, clitic_fixed = clean_code_switching_clitics(t)
        t = process_emojis(t, self.emoji_mode)
        if self.arabizi_mode:
            t = normalize_arabizi(t)
        t = TASHKEEL_REGEX.sub("", t)
        t = TATWEEL_REGEX.sub("", t)
        t = ALEF_WASLA_REGEX.sub("ا", t)
        t = EXPRESSIVE_MARRA_SPELL_REGEX.sub(r"\1مرة", t)
        t = collapse_elongations(t, self.tokenizer)

        for pattern, replacement in DIALECT_SPELLING_MAP:
            t = pattern.sub(replacement, t)

        healed = 0
        if self.preserve_orthography and self.auto_heal:
            t, healed = heal_orthography_with_vocab(t, self.tokenizer)

        if not self.preserve_orthography:
            t = LEGACY_ALEF_REGEX.sub("ا", t)
            t = LEGACY_TAA_REGEX.sub("ه", t)
            t = LEGACY_YAA_REGEX.sub("ي", t)

        return MULTISPACE_REGEX.sub(" ", t).strip(), healed, clitic_fixed

    def process(self, text: str) -> str:
        cleaned, _, _ = self.process_with_stats(text)
        return cleaned


class NoisySocialPipeline:
    """
    Social-media denoising pass with optional hybrid dialect spelling normalization,
    vocab-verified orthography healing, clitic hygiene, Arabizi support, and emoji policy.
    """

    def __init__(
        self,
        preserve_orthography: bool = True,
        arabert_placeholders: bool = False,
        tokenizer: Optional[TokenizerProtocol] = None,
        auto_heal: bool = True,
        emoji_mode: str = "strip",
        arabizi_mode: bool = False,
    ):
        self.preserve_orthography = preserve_orthography
        self.arabert_placeholders = arabert_placeholders
        self.tokenizer = tokenizer
        self.auto_heal = auto_heal
        self.emoji_mode = emoji_mode
        self.arabizi_mode = arabizi_mode

    def process_with_stats(self, text: str, apply_dialect_cleanup: bool = False) -> Tuple[str, int, int]:
        t = normalize_unicode_safety(text)
        t = strip_markup_noise(t)
        t = ZERO_WIDTH_REGEX.sub("", t)
        t, clitic_fixed = clean_code_switching_clitics(t)
        t = TATWEEL_REGEX.sub("", t)

        if self.arabert_placeholders:
            t = EMAIL_REGEX.sub(" [بريد] ", t)
            t = URL_REGEX.sub(" [رابط] ", t)
            t = MENTION_REGEX.sub(" [مستخدم] ", t)
        else:
            t = EMAIL_REGEX.sub(" ", t)
            t = URL_REGEX.sub(" ", t)
            t = MENTION_REGEX.sub(" ", t)

        t = HASHTAG_REGEX.sub(lambda m: m.group(1).replace("_", " "), t)
        t = process_emojis(t, self.emoji_mode)
        if self.arabizi_mode:
            t = normalize_arabizi(t)
        t = PUNCT_REPEAT_REGEX.sub(r"\1", t)
        t = TASHKEEL_REGEX.sub("", t)
        t = ALEF_WASLA_REGEX.sub("ا", t)
        if apply_dialect_cleanup:
            t = EXPRESSIVE_MARRA_SPELL_REGEX.sub(r"\1مرة", t)
        t = collapse_elongations(t, self.tokenizer)

        if apply_dialect_cleanup:
            for pattern, replacement in DIALECT_SPELLING_MAP:
                t = pattern.sub(replacement, t)

        healed = 0
        if self.preserve_orthography and self.auto_heal:
            t, healed = heal_orthography_with_vocab(t, self.tokenizer)

        if not self.preserve_orthography:
            t = LEGACY_ALEF_REGEX.sub("ا", t)
            t = LEGACY_TAA_REGEX.sub("ه", t)
            t = LEGACY_YAA_REGEX.sub("ي", t)

        return MULTISPACE_REGEX.sub(" ", t).strip(), healed, clitic_fixed

    def process(self, text: str, apply_dialect_cleanup: bool = False) -> str:
        cleaned, _, _ = self.process_with_stats(text, apply_dialect_cleanup=apply_dialect_cleanup)
        return cleaned
