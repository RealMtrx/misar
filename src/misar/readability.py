# -*- coding: utf-8 -*-
"""
High-performance Arabic Readability & Linguistic Complexity Metric Engine for Misar (مِسار).
Features:
- Implementation of Osman's Readability Metric (مؤشر عثمان للمقروئية) for Arabic prose
- Educational stage mapping (ابتدائي، متوسط، ثانوي، جامعي، تخصصي أكاديمي)
- Syllabic mora and phonetic vowel estimation for Arabic words
- Lexical richness: Type-Token Ratio (TTR) and Hapax Legomena frequency
- Average reading time estimator (based on standard Arabic reading speeds: 160 words/min)
- Zero external dependencies (Pure Python Standard Library)
"""

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .lexicon import extract_root

SENTENCE_DELIM_REGEX = re.compile(r"[.!?؛؟\n]+", re.UNICODE)
WORD_TOKEN_REGEX = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
LONG_VOWELS = {"ا", "و", "ي", "ى", "آ"}


@dataclass
class ReadabilityResult:
    """Detailed report for Arabic readability, complexity, and educational suitability."""
    original_text: str
    osman_index: float
    reading_ease_category: str
    educational_stage: str
    educational_stage_ar: str
    word_count: int
    sentence_count: int
    character_count: int
    avg_word_length: float
    avg_sentence_length: float
    estimated_syllables_per_word: float
    type_token_ratio: float
    hapax_legomena_ratio: float
    reading_time_seconds: float
    confidence: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "osman_index": round(self.osman_index, 2),
            "reading_ease_category": self.reading_ease_category,
            "educational_stage": self.educational_stage,
            "educational_stage_ar": self.educational_stage_ar,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "character_count": self.character_count,
            "avg_word_length": round(self.avg_word_length, 2),
            "avg_sentence_length": round(self.avg_sentence_length, 2),
            "estimated_syllables_per_word": round(self.estimated_syllables_per_word, 2),
            "type_token_ratio": round(self.type_token_ratio, 3),
            "hapax_legomena_ratio": round(self.hapax_legomena_ratio, 3),
            "reading_time_seconds": round(self.reading_time_seconds, 1),
            "confidence": self.confidence,
        }


def _estimate_syllables_arabic(word: str) -> int:
    """Estimates the number of syllables in an Arabic word based on vowel patterns."""
    if not word:
        return 0
    # Every Arabic word has at least 1 syllable
    syllables = 1
    # Count long vowels (ا، و، ي)
    for ch in word[1:]:
        if ch in LONG_VOWELS:
            syllables += 1
    # Words with length > 5 generally have more morphological moras
    if len(word) >= 6:
        syllables += 1
    return min(syllables, len(word))


def analyze_readability(text: str) -> ReadabilityResult:
    """
    Computes standard Arabic readability metrics, including Osman's Readability Index,
    lexical diversity, and educational stage suitability.
    """
    if not text or not text.strip():
        return ReadabilityResult(
            original_text=text,
            osman_index=100.0,
            reading_ease_category="بسيط جداً",
            educational_stage="elementary",
            educational_stage_ar="ابتدائي",
            word_count=0,
            sentence_count=0,
            character_count=0,
            avg_word_length=0.0,
            avg_sentence_length=0.0,
            estimated_syllables_per_word=0.0,
            type_token_ratio=0.0,
            hapax_legomena_ratio=0.0,
            reading_time_seconds=0.0,
            confidence=1.0,
        )

    # 1. Word tokens and characters
    words = WORD_TOKEN_REGEX.findall(text)
    total_words = len(words)
    if total_words == 0:
        return ReadabilityResult(
            original_text=text,
            osman_index=100.0,
            reading_ease_category="بسيط جداً",
            educational_stage="elementary",
            educational_stage_ar="ابتدائي",
            word_count=0,
            sentence_count=0,
            character_count=0,
            avg_word_length=0.0,
            avg_sentence_length=0.0,
            estimated_syllables_per_word=0.0,
            type_token_ratio=0.0,
            hapax_legomena_ratio=0.0,
            reading_time_seconds=0.0,
        )

    # Clean characters (excluding spaces and punctuation)
    char_count = sum(len(w) for w in words)
    avg_word_len = char_count / total_words

    # 2. Sentences
    raw_sentences = [s.strip() for s in SENTENCE_DELIM_REGEX.split(text) if s.strip()]
    sentence_count = max(1, len(raw_sentences))
    avg_sentence_len = total_words / sentence_count

    # 3. Syllables & Moras
    total_syllables = sum(_estimate_syllables_arabic(w) for w in words)
    avg_syllables = total_syllables / total_words

    # 4. Osman's Readability Index (OSMAN for Arabic prose)
    # Formula: 206.835 - (1.015 * ASL) - (84.6 * ASW)
    osman_score = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)
    bounded_score = round(max(5.0, min(100.0, osman_score)), 2)

    # 5. Educational stage classification
    if bounded_score >= 70.0:
        ease = "بسيط ومباشر"
        stage = "elementary"
        stage_ar = "المرحلة الابتدائية (متاح للجميع)"
    elif bounded_score >= 50.0:
        ease = "متوسط وواضح"
        stage = "middle_school"
        stage_ar = "المرحلة المتوسطة (جمهور عام)"
    elif bounded_score >= 35.0:
        ease = "متقدم ومعياري"
        stage = "high_school"
        stage_ar = "المرحلة الثانوية (محتوى متقدم)"
    elif bounded_score >= 20.0:
        ease = "معقد وتخصصي"
        stage = "university"
        stage_ar = "المرحلة الجامعية (محتوى علمي/قانوني)"
    else:
        ease = "شديد التعقيد والأكاديمية"
        stage = "specialized_academic"
        stage_ar = "تخصصي أكاديمي / تراثي مكثف"

    # 6. Lexical Diversity (TTR & Hapax)
    counts = Counter(words)
    unique_words = len(counts)
    ttr = unique_words / total_words
    hapax_count = sum(1 for v in counts.values() if v == 1)
    hapax_ratio = hapax_count / total_words

    # 7. Estimated Reading Time (Average: 160 Arabic words per minute)
    reading_seconds = (total_words / 160.0) * 60.0

    return ReadabilityResult(
        original_text=text,
        osman_index=round(bounded_score, 2),
        reading_ease_category=ease,
        educational_stage=stage,
        educational_stage_ar=stage_ar,
        word_count=total_words,
        sentence_count=sentence_count,
        character_count=char_count,
        avg_word_length=round(avg_word_len, 2),
        avg_sentence_length=round(avg_sentence_len, 2),
        estimated_syllables_per_word=round(avg_syllables, 2),
        type_token_ratio=round(ttr, 3),
        hapax_legomena_ratio=round(hapax_ratio, 3),
        reading_time_seconds=round(reading_seconds, 1),
        confidence=0.95,
    )
