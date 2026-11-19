# -*- coding: utf-8 -*-
"""
Curated Arabic dialect lexicons, regex patterns, and orthographic spelling maps for Misar.
"""

import re
from typing import Dict, Set, Tuple

TASHKEEL_REGEX = re.compile(r"[\u064B-\u065F\u0670]")
TATWEEL_REGEX = re.compile(r"\u0640")
DECORATION_MARKS_REGEX = re.compile(r"[\u0640\u064B-\u065F\u0670]")
REPEATED_CHARS_REGEX = re.compile(r"([^\s\d\W_])\1{2,}")
EMOJI_REGEX = re.compile(
    r"[\U00010000-\U0010FFFF\u2600-\u27BF\uFE0F]",
    flags=re.UNICODE,
)
PUNCT_REPEAT_REGEX = re.compile(r"([!?.،؛؟]){2,}")
HASHTAG_REGEX = re.compile(r"#([\w]+)")
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_REGEX = re.compile(r"@[A-Za-z0-9_]+")
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ARABIC_WORD_REGEX = re.compile(r"[\u0621-\u064A\u0671-\u06D3]+")
ARABIC_CHAR_REGEX = re.compile(r"[\u0621-\u064A\u0671-\u06D3]")
LATIN_CHAR_REGEX = re.compile(r"[A-Za-z]")
ZERO_WIDTH_REGEX = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")

ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# High-precision dialect markers audited to eliminate false positives on Standard Arabic (MSA).
EXPRESSIVE_MARRA_REGEX = re.compile(r"^[وفب]?مر{2,}[ةه]$")
EXPRESSIVE_MARRA_SPELL_REGEX = re.compile(r"\b([وف]?)مر{2,}[ةه]\b")

DIALECT_MARKERS: Dict[str, Set[str]] = {
    "gulf_saudi": {
        "يجنن", "يهبل", "حقت", "حقتي", "حقتك",
        "هاذا", "هاذي", "هذول", "هذولي", "الحين", "دحين", "تكفى", "تكفون",
        "ابغى", "أبغى", "ابغا", "تبغى", "يبغى", "نبغى", "تبي", "يبي",
        "شويتين", "كذا", "كذه", "شلون", "شلونك", "شلونكم",
        "ايش", "إيش", "وشو", "وش", "عشان", "شسالفة",
        "يبيلها", "يبيله", "وشلون", "مافيه", "مافي", "ماكو", "مو", "مب",
        "بالحيل", "وايد", "سالفة", "سالفه", "طقطقة", "طقطقه",
        "ياخي", "ياخوي", "ماهو", "ماهي",
        "خرابيط", "تضبيط", "كفو", "واللهي", "زبط", "انبسطت", "بغيت",
    },
    "egyptian": {
        "مش", "علشان", "كده", "كدا", "اوي", "أوي",
        "ده", "عاوز", "عايز", "عايزة", "عايزه", "عاوزين", "عايزين",
        "بتعمل", "بيعمل", "بنعمل", "مفيش", "مافيش", "شوية", "شويه", "تظبيط",
        "ازيك", "إزيك", "ازيكم", "ازاي", "إزاي", "فين", "ليه",
        "كويس", "كويسة", "كويسه", "جدع", "برضه", "برضو", "دلوقتي", "ياباشا",
        "ياعم", "مابلاش", "بلاش", "افتكس",
        "النهاردة", "النهارده", "امبارح", "بص", "بتاع", "بتاعة",
    },
    "levantine": {
        "هيك", "هاد", "هادا", "هيدي", "هيدا", "شو", "ليش", "منشان", "كرمال", "كتير",
        "هلأ", "هلق", "بدي", "بدك", "بدنا", "مافي", "مو", "شلونك",
        "منيح", "منيحة", "منيحه", "بركي", "يقبرني", "هون", "هونا", "يلا",
        "زلمة", "مزبوط", "قديش", "وين",
    },
    "maghrebi": {
        "ديال", "ديالي", "ديالك", "برشا", "بزاف", "واش", "كاين", "مزيان", "دابا",
        "شكون", "علاش", "زوين", "زوينة", "واخا", "لاباس",
        "شنو", "باش", "غادي", "بصح", "خويا", "كيفاش",
    },
}

PROTECTED_DIALECT_NEGATIONS: Set[str] = {
    "مو", "مش", "مب", "مافي", "مافيه", "مفيش", "مافيش",
    "ماكو", "ماهو", "ماهي", "بلاش", "مابلاش",
}

# Vocabulary-aligned gentle dialect normalizations (preserve AraBERTv2/CAMeLBERT tokens).
DIALECT_SPELLING_MAP: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern), replacement)
    for pattern, replacement in (
        (r"\b([وف]?)هاذا\b", r"\1هذا"),
        (r"\b([وف]?)هاذي\b", r"\1هذه"),
        (r"\b([وف]?)هذي\b", r"\1هذه"),
        (r"\b([وف]?)هذول\b", r"\1هؤلاء"),
        (r"\b([وف]?)هذولي\b", r"\1هؤلاء"),
        (r"\b([وف]?)اللى\b", r"\1اللي"),
        (r"\b([وف]?)دا\b", r"\1ده"),
        (r"\b([وف]?)كدا\b", r"\1كده"),
        (r"\b([وف]?)كتير\b", r"\1كثير"),
        (r"\b([وف]?)شويتين\b", r"\1شوية"),
        (r"\b([وف]?)مر{2,}[ةه]\b", r"\1مرة"),
        (r"\b([وف]?)علشان\b", r"\1عشان"),
        (r"\b([وف]?)برضو\b", r"\1برضه"),
        (r"\b([وف]?)مابلاش\b", r"\1بلاش"),
        (r"\bياباشا\b", "يا باشا"),
        (r"\bشسالفة\b", "ايش السالفة"),
        (r"\b([وف]?)ابغا\b", r"\1أبغى"),
    )
)

# Common Arabic words with legitimate consecutive double letters (e.g., ال + ل).
GEMINATED_FALLBACK_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(p), r)
    for p, r in (
        (r"\bالل{2,}ه\b", "الله"),
        (r"\bوالل{2,}ه\b", "والله"),
        (r"\bبالل{2,}ه\b", "بالله"),
        (r"\bالل{2,}يل\b", "الليل"),
        (r"\bالل{2,}غة\b", "اللغة"),
        (r"\bالل{2,}غات\b", "اللغات"),
    )
)


def normalize_token_for_lookup(token: str) -> str:
    t = TASHKEEL_REGEX.sub("", token)
    t = TATWEEL_REGEX.sub("", t)
    t = REPEATED_CHARS_REGEX.sub(r"\1", t)
    t = re.sub(r"[إأآٱ]", "ا", t)
    t = re.sub(r"ة\b", "ه", t)
    t = re.sub(r"ى\b", "ي", t)
    return t


NORMALIZED_DIALECT_MAP: Dict[str, Tuple[str, str]] = {}
for _region, _words in DIALECT_MARKERS.items():
    for _word in _words:
        _clean = normalize_token_for_lookup(_word)
        if _clean not in NORMALIZED_DIALECT_MAP:
            NORMALIZED_DIALECT_MAP[_clean] = (_word, _region)
