# -*- coding: utf-8 -*-
"""
Linear-time O(n) Linguistic Profiler for Arabic text characterization.
Computes dialect ratio (P_dial), diacritic density (D_diac), elongation/social
noise score (E_noise), Shannon character entropy (H_char), and code-switching ratio.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Set

from .lexicons import (
    ARABIC_CHAR_REGEX,
    ARABIC_WORD_REGEX,
    EMAIL_REGEX,
    EMOJI_REGEX,
    EXPRESSIVE_MARRA_REGEX,
    HASHTAG_REGEX,
    LATIN_CHAR_REGEX,
    MENTION_REGEX,
    NORMALIZED_DIALECT_MAP,
    PROTECTED_DIALECT_NEGATIONS,
    PUNCT_REPEAT_REGEX,
    REPEATED_CHARS_REGEX,
    TASHKEEL_REGEX,
    TATWEEL_REGEX,
    URL_REGEX,
    normalize_token_for_lookup,
    normalize_unicode_safety,
)


@dataclass
class LinguisticProfile:
    total_chars: int
    arabic_chars: int
    latin_chars: int
    word_count: int
    dialect_ratio: float
    diacritic_density: float
    elongation_noise_score: float
    shannon_entropy: float
    code_switching_ratio: float
    detected_markers: List[str] = field(default_factory=list)
    detected_regions: List[str] = field(default_factory=list)
    has_protected_negation: bool = False
    repeated_char_runs: int = 0
    social_artifact_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_chars": self.total_chars,
            "arabic_chars": self.arabic_chars,
            "latin_chars": self.latin_chars,
            "word_count": self.word_count,
            "dialect_ratio": round(self.dialect_ratio, 4),
            "diacritic_density": round(self.diacritic_density, 4),
            "elongation_noise_score": round(self.elongation_noise_score, 4),
            "shannon_entropy": round(self.shannon_entropy, 4),
            "code_switching_ratio": round(self.code_switching_ratio, 4),
            "detected_markers": list(self.detected_markers),
            "detected_regions": list(self.detected_regions),
            "has_protected_negation": self.has_protected_negation,
            "repeated_char_runs": self.repeated_char_runs,
            "social_artifact_count": self.social_artifact_count,
        }


def compute_shannon_entropy(text: str) -> float:
    """Compute character-level Shannon entropy H = -sum(p_i * log2(p_i)) excluding spaces."""
    compact = [ch for ch in text if not ch.isspace()]
    n = len(compact)
    if n == 0:
        return 0.0
    counts = Counter(compact)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)
    return entropy


class LinguisticProfiler:
    """Extracts structural and sociolinguistic features of Arabic text in O(n) time with O(1) LRU caching."""

    def __init__(self, cache_size: int = 1024):
        self.cache_size = cache_size
        self._cache: Dict[str, LinguisticProfile] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def analyze(self, text: str) -> LinguisticProfile:
        if text in self._cache:
            return self._cache[text]

        trimmed = normalize_unicode_safety(text.strip()) if text else ""
        total_chars = len(trimmed)
        if total_chars == 0:
            empty_profile = LinguisticProfile(
                total_chars=0,
                arabic_chars=0,
                latin_chars=0,
                word_count=0,
                dialect_ratio=0.0,
                diacritic_density=0.0,
                elongation_noise_score=0.0,
                shannon_entropy=0.0,
                code_switching_ratio=0.0,
            )
            if self.cache_size > 0 and len(self._cache) < self.cache_size:
                self._cache[text] = empty_profile
            return empty_profile

        arabic_chars = len(ARABIC_CHAR_REGEX.findall(trimmed))
        latin_chars = len(LATIN_CHAR_REGEX.findall(trimmed))
        alpha_total = arabic_chars + latin_chars
        code_switching_ratio = (latin_chars / alpha_total) if (arabic_chars > 0 and latin_chars > 0) else 0.0

        diacritics_count = len(TASHKEEL_REGEX.findall(trimmed))
        diacritic_density = (diacritics_count / max(1, arabic_chars)) if arabic_chars > 0 else 0.0

        tatweel_count = len(TATWEEL_REGEX.findall(trimmed))
        repeated_matches = list(REPEATED_CHARS_REGEX.finditer(trimmed))
        repeated_char_runs = len(repeated_matches)
        excess_repeated_chars = sum(len(m.group(0)) - 1 for m in repeated_matches)
        emojis_count = len(EMOJI_REGEX.findall(trimmed))
        punct_repeats = len(PUNCT_REPEAT_REGEX.findall(trimmed))
        hashtags = len(HASHTAG_REGEX.findall(trimmed))
        urls_mentions = len(URL_REGEX.findall(trimmed)) + len(MENTION_REGEX.findall(trimmed)) + len(EMAIL_REGEX.findall(trimmed))

        noise_penalty = (
            tatweel_count * 1.5
            + repeated_char_runs * 2.5
            + excess_repeated_chars * 1.0
            + emojis_count * 1.2
            + punct_repeats * 1.5
            + hashtags * 2.0
            + urls_mentions * 2.5
        )
        elongation_noise_score = min(1.0, noise_penalty / max(15.0, total_chars * 0.4))

        all_tokens = trimmed.split()
        arabic_words = ARABIC_WORD_REGEX.findall(trimmed)
        detected_markers: List[str] = []
        region_set: Set[str] = set()
        has_protected_negation = False

        if arabic_words:
            for word in arabic_words:
                clean_surface = TATWEEL_REGEX.sub("", TASHKEEL_REGEX.sub("", word))
                if EXPRESSIVE_MARRA_REGEX.match(clean_surface):
                    matched = ("مرره", "gulf_saudi")
                    norm_w = clean_surface
                else:
                    norm_w = normalize_token_for_lookup(word)
                    matched = NORMALIZED_DIALECT_MAP.get(norm_w)
                    if matched is None and len(norm_w) > 3 and norm_w[0] in ("و", "ف", "ب", "ل"):
                        matched = NORMALIZED_DIALECT_MAP.get(norm_w[1:])
                if matched is not None:
                    canonical, region = matched
                    detected_markers.append(canonical)
                    region_set.add(region)
                    if norm_w in PROTECTED_DIALECT_NEGATIONS or (len(norm_w) > 3 and norm_w[1:] in PROTECTED_DIALECT_NEGATIONS):
                        has_protected_negation = True
            dialect_ratio = len(detected_markers) / len(arabic_words)
        else:
            dialect_ratio = 0.0

        shannon_entropy = compute_shannon_entropy(trimmed)

        profile = LinguisticProfile(
            total_chars=total_chars,
            arabic_chars=arabic_chars,
            latin_chars=latin_chars,
            word_count=len(all_tokens),
            dialect_ratio=dialect_ratio,
            diacritic_density=diacritic_density,
            elongation_noise_score=elongation_noise_score,
            shannon_entropy=shannon_entropy,
            code_switching_ratio=code_switching_ratio,
            detected_markers=detected_markers,
            detected_regions=sorted(region_set),
            has_protected_negation=has_protected_negation,
            repeated_char_runs=repeated_char_runs,
            social_artifact_count=hashtags + urls_mentions,
        )

        if self.cache_size > 0:
            if len(self._cache) >= self.cache_size:
                del self._cache[next(iter(self._cache))]
            self._cache[text] = profile

        return profile
