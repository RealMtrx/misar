# -*- coding: utf-8 -*-
"""
Specialized Transformer-aware Arabic preprocessing micro-pipelines for Misar.
Key architectural finding: Modern Arabic Transformer vocabularies (e.g., AraBERTv2)
preserve Hamzas (أ/إ/آ) and Taa Marbuta (ة). Destructive legacy normalization (ة -> ه)
actively increases subword fragmentation; Misar defaults to orthography-preserving
cleaning while offering optional legacy normalization for classical search engines.
"""

import re
from typing import Optional

from .lexicons import (
    ARABIC_INDIC_DIGITS,
    DIALECT_SPELLING_MAP,
    EMAIL_REGEX,
    EMOJI_REGEX,
    EXPRESSIVE_MARRA_SPELL_REGEX,
    GEMINATED_FALLBACK_PATTERNS,
    HASHTAG_REGEX,
    MENTION_REGEX,
    PUNCT_REPEAT_REGEX,
    REPEATED_CHARS_REGEX,
    TASHKEEL_REGEX,
    TATWEEL_REGEX,
    URL_REGEX,
    ZERO_WIDTH_REGEX,
)
from .tokenizer import TokenizerProtocol

ALEF_WASLA_REGEX = re.compile(r"\u0671")
LEGACY_ALEF_REGEX = re.compile(r"[إأآٱ]")
LEGACY_TAA_REGEX = re.compile(r"ة\b")
LEGACY_YAA_REGEX = re.compile(r"ى\b")
MULTISPACE_REGEX = re.compile(r"\s+")
MARKDOWN_HEADER_REGEX = re.compile(r"(^|\n)\s*#{1,6}\s+")
MARKDOWN_FORMAT_REGEX = re.compile(r"[*~`]")


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
        cand_single = REPEATED_CHARS_REGEX.sub(r"\1", word)
        if not has_vocab:
            return cand_single
        if tokenizer.has_token(cand_single):
            return cand_single
        cand_double = REPEATED_CHARS_REGEX.sub(r"\1\1", word)
        if tokenizer.has_token(cand_double):
            return cand_double
        return cand_single

    return re.sub(r"\S+", _fix_word, t)


class MSALightPipeline:
    """
    Ultra-fast Modern Standard Arabic & Classical diacritized pass.
    Strips diacritics/tatweel/zero-width marks and normalizes Alef Wasla while
    preserving Transformer-native orthography (أ/إ/آ, ة, ى) by default.
    """

    def __init__(self, preserve_orthography: bool = True, normalize_digits: bool = False):
        self.preserve_orthography = preserve_orthography
        self.normalize_digits = normalize_digits

    def process(self, text: str) -> str:
        t = strip_markup_noise(text)
        t = ZERO_WIDTH_REGEX.sub("", t)
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
        return MULTISPACE_REGEX.sub(" ", t).strip()


class DialectPreservePipeline:
    """
    Dialect-safe preprocessing pass.
    Protects colloquial negation and sentiment particles (مو، مش، مفيش، مب، ابي)
    while applying vocabulary-aligned gentle spelling normalization.
    """

    def __init__(
        self,
        preserve_orthography: bool = True,
        tokenizer: Optional[TokenizerProtocol] = None,
    ):
        self.preserve_orthography = preserve_orthography
        self.tokenizer = tokenizer

    def process(self, text: str) -> str:
        t = strip_markup_noise(text)
        t = ZERO_WIDTH_REGEX.sub("", t)
        t = TASHKEEL_REGEX.sub("", t)
        t = TATWEEL_REGEX.sub("", t)
        t = ALEF_WASLA_REGEX.sub("ا", t)
        t = EXPRESSIVE_MARRA_SPELL_REGEX.sub(r"\1مرة", t)
        t = collapse_elongations(t, self.tokenizer)

        for pattern, replacement in DIALECT_SPELLING_MAP:
            t = pattern.sub(replacement, t)

        if not self.preserve_orthography:
            t = LEGACY_ALEF_REGEX.sub("ا", t)
            t = LEGACY_TAA_REGEX.sub("ه", t)
            t = LEGACY_YAA_REGEX.sub("ي", t)

        return MULTISPACE_REGEX.sub(" ", t).strip()


class NoisySocialPipeline:
    """
    Social-media denoising pass with optional hybrid dialect spelling normalization.
    Unpacks hashtags, removes emojis/URLs/mentions/tatweel, collapses stretches,
    and optionally applies AraBERTv2 special placeholders ([رابط], [مستخدم], [بريد]).
    """

    def __init__(
        self,
        preserve_orthography: bool = True,
        arabert_placeholders: bool = False,
        tokenizer: Optional[TokenizerProtocol] = None,
    ):
        self.preserve_orthography = preserve_orthography
        self.arabert_placeholders = arabert_placeholders
        self.tokenizer = tokenizer

    def process(self, text: str, apply_dialect_cleanup: bool = False) -> str:
        t = strip_markup_noise(text)
        t = ZERO_WIDTH_REGEX.sub("", t)
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
        t = EMOJI_REGEX.sub(" ", t)
        t = PUNCT_REPEAT_REGEX.sub(r"\1", t)
        t = TASHKEEL_REGEX.sub("", t)
        t = ALEF_WASLA_REGEX.sub("ا", t)
        if apply_dialect_cleanup:
            t = EXPRESSIVE_MARRA_SPELL_REGEX.sub(r"\1مرة", t)
        t = collapse_elongations(t, self.tokenizer)

        if apply_dialect_cleanup:
            for pattern, replacement in DIALECT_SPELLING_MAP:
                t = pattern.sub(replacement, t)

        if not self.preserve_orthography:
            t = LEGACY_ALEF_REGEX.sub("ا", t)
            t = LEGACY_TAA_REGEX.sub("ه", t)
            t = LEGACY_YAA_REGEX.sub("ي", t)

        return MULTISPACE_REGEX.sub(" ", t).strip()
