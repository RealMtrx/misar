# -*- coding: utf-8 -*-
"""
High-precision Arabic Grapheme-to-Phoneme (G2P) & Phonetic Transcription Engine for Misar (مِسار).
Features:
- Standard International Phonetic Alphabet (IPA) conversion for Arabic text
- Phonological assimilation rules: Solar letters (اللام الشمسية) and Lunar letters (اللام القمرية)
- Pharyngealized emphatic consonant realizations (ص، ض، ط، ظ)
- Short and long vowel duration mappings (a, u, i, aː, uː, iː)
- Automatic short-vowel restoration integration via misar.diacritize
- Zero external dependencies (Pure Python Standard Library)
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Diacritic Unicode Constants
FATHA = "\u064E"      # َ
DAMMA = "\u064F"      # ُ
KASRA = "\u0650"      # ِ
SUKUN = "\u0652"      # ْ
SHADDA = "\u0651"     # ّ
FATHATAN = "\u064B"   # ً
DAMMATAN = "\u064C"   # ٌ
KASRATAN = "\u064D"   # ٍ

ALL_DIACRITICS = {FATHA, DAMMA, KASRA, SUKUN, SHADDA, FATHATAN, DAMMATAN, KASRATAN}

# Solar (Sun) Letters that assimilate the Lam of the definite article
SUN_LETTERS = {"ت", "ث", "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ل", "ن"}

# Arabic Consonants to IPA Base Mapping
ARABIC_IPA_CONSONANTS: Dict[str, str] = {
    "ء": "ʔ",
    "أ": "ʔ",
    "إ": "ʔ",
    "ؤ": "ʔ",
    "ئ": "ʔ",
    "آ": "ʔaː",
    "ب": "b",
    "ت": "t",
    "ث": "θ",
    "ج": "dʒ",
    "ح": "ħ",
    "خ": "x",
    "د": "d",
    "ذ": "ð",
    "ر": "r",
    "ز": "z",
    "س": "s",
    "ش": "ʃ",
    "ص": "sˤ",
    "ض": "dˤ",
    "ط": "tˤ",
    "ظ": "ðˤ",
    "ع": "ʕ",
    "غ": "ɣ",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ه": "h",
    "ة": "h",
    "و": "w",
    "ي": "j",
    "ى": "aː",
}


@dataclass
class PhonemizationResult:
    """Detailed report for Arabic phonetic transcription and speech synthesis metrics."""
    original_text: str
    diacritized_text: str
    ipa_transcription: str
    phoneme_tokens: List[str]
    syllable_count: int
    estimated_audio_duration_sec: float
    sun_letter_assimilations_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "diacritized_text": self.diacritized_text,
            "ipa_transcription": self.ipa_transcription,
            "phoneme_tokens": list(self.phoneme_tokens),
            "syllable_count": self.syllable_count,
            "estimated_audio_duration_sec": round(self.estimated_audio_duration_sec, 3),
            "sun_letter_assimilations_count": self.sun_letter_assimilations_count,
        }


def _phonemize_word(word: str) -> Tuple[str, List[str], int, int]:
    """
    Phonemizes a single diacritized Arabic word into IPA and token list.
    Returns: (ipa_string, tokens_list, syllable_count, sun_assimilations)
    """
    if not word:
        return "", [], 0, 0

    sun_count = 0
    w = word
    prefix_ipa = ""
    tokens: List[str] = []

    # Handle Definite Article الـ and Sun Letter Assimilation
    if w.startswith("ال") and len(w) >= 3:
        next_char = w[2]
        # Skip diacritics if next_char is a diacritic
        char_idx = 2
        while char_idx < len(w) and w[char_idx] in ALL_DIACRITICS:
            char_idx += 1

        if char_idx < len(w):
            first_noun_char = w[char_idx]
            if first_noun_char in SUN_LETTERS:
                sun_count += 1
                sun_ipa = ARABIC_IPA_CONSONANTS.get(first_noun_char, first_noun_char)
                prefix_ipa = f"ʔa{sun_ipa}"
                tokens.extend(["ʔ", "a", sun_ipa])
                w = w[char_idx:]
            else:
                prefix_ipa = "ʔal"
                tokens.extend(["ʔ", "a", "l"])
                w = w[2:]
        else:
            prefix_ipa = "ʔal"
            tokens.extend(["ʔ", "a", "l"])
            w = w[2:]

    # Parse remaining characters and diacritics
    n = len(w)
    i = 0
    word_tokens: List[str] = []
    syllables = 0

    while i < n:
        ch = w[i]

        # Diacritics
        if ch == FATHA:
            # Check for long vowel ا
            if i + 1 < n and w[i + 1] in ("ا", "ى"):
                word_tokens.append("aː")
                syllables += 1
                i += 2
                continue
            word_tokens.append("a")
            syllables += 1
            i += 1
            continue
        elif ch == DAMMA:
            # Check for long vowel و
            if i + 1 < n and w[i + 1] == "و":
                word_tokens.append("uː")
                syllables += 1
                i += 2
                continue
            word_tokens.append("u")
            syllables += 1
            i += 1
            continue
        elif ch == KASRA:
            # Check for long vowel ي
            if i + 1 < n and w[i + 1] == "ي":
                word_tokens.append("iː")
                syllables += 1
                i += 2
                continue
            word_tokens.append("i")
            syllables += 1
            i += 1
            continue
        elif ch == SUKUN:
            i += 1
            continue
        elif ch == SHADDA:
            if word_tokens:
                last_tok = word_tokens[-1]
                # Geminate consonant
                word_tokens.append(last_tok)
            i += 1
            continue
        elif ch == FATHATAN:
            word_tokens.append("an")
            syllables += 1
            i += 1
            # Skip trailing silent alif
            if i < n and w[i] == "ا":
                i += 1
            continue
        elif ch == DAMMATAN:
            word_tokens.append("un")
            syllables += 1
            i += 1
            continue
        elif ch == KASRATAN:
            word_tokens.append("in")
            syllables += 1
            i += 1
            continue

        # Regular Consonants / Glides
        if ch in ARABIC_IPA_CONSONANTS:
            ipa_c = ARABIC_IPA_CONSONANTS[ch]
            word_tokens.append(ipa_c)
        elif ch.isalpha():
            word_tokens.append(ch.lower())
        else:
            word_tokens.append(ch)

        i += 1

    tokens.extend(word_tokens)
    full_ipa = prefix_ipa + "".join(word_tokens)
    if syllables == 0:
        syllables = max(1, len(full_ipa) // 3)

    return full_ipa, tokens, syllables, sun_count


def phonemize_arabic(text: str, auto_diacritize: bool = True) -> PhonemizationResult:
    """
    Converts Arabic text into International Phonetic Alphabet (IPA) representation
    and acoustic phoneme tokens for Text-to-Speech (TTS) models.
    """
    t = "" if text is None else str(text).strip()
    if not t:
        return PhonemizationResult(
            original_text=t,
            diacritized_text="",
            ipa_transcription="",
            phoneme_tokens=[],
            syllable_count=0,
            estimated_audio_duration_sec=0.0,
            sun_letter_assimilations_count=0,
        )

    # 1. Automatic Short Vowel Restoration if text lacks diacritics
    has_diacritics = any(c in ALL_DIACRITICS for c in t)
    if auto_diacritize and not has_diacritics:
        from .diacritize import diacritize_text
        diac_res = diacritize_text(t)
        diacritized = diac_res.diacritized_text
    else:
        diacritized = t

    # 2. Tokenize and phonemize words
    words = diacritized.split()
    ipa_words: List[str] = []
    all_tokens: List[str] = []
    total_syllables = 0
    total_sun = 0

    for w in words:
        ipa_w, tok_w, syl_w, sun_w = _phonemize_word(w)
        ipa_words.append(ipa_w)
        all_tokens.extend(tok_w)
        total_syllables += syl_w
        total_sun += sun_w

    ipa_result = " ".join(ipa_words)
    # Standard Arabic speech velocity is ~4.5 syllables per second
    estimated_duration = max(0.1, total_syllables / 4.5)

    return PhonemizationResult(
        original_text=t,
        diacritized_text=diacritized,
        ipa_transcription=ipa_result,
        phoneme_tokens=all_tokens,
        syllable_count=total_syllables,
        estimated_audio_duration_sec=estimated_duration,
        sun_letter_assimilations_count=total_sun,
    )
