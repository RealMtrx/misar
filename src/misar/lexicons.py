# -*- coding: utf-8 -*-
"""
Low-Level Regex Patterns, Normalization Tables & Fast Orthographic Maps for Misar.

Architecture Note:
    - `misar.lexicons` (plural): Low-level module providing precompiled regexes
      (tashkeel, tatweel, emoji, urls, hashtags), dialect negation token sets,
      and static orthographic mapping dictionaries used inside the pipeline passes.
    - `misar.lexicon` (singular): High-level semantic lexicon providing word lookups,
      morphological tri-consonantal root extraction, definitions, and sentence-level
      dialect-to-MSA explanation.
"""

import re
import unicodedata
from typing import Dict, Set, Tuple

TASHKEEL_REGEX = re.compile(r"[\u064B-\u065F\u0670]")
TATWEEL_REGEX = re.compile(r"\u0640")
DECORATION_MARKS_REGEX = re.compile(r"[\u0640\u064B-\u065F\u0670]")
REPEATED_CHARS_REGEX = re.compile(r"([^\s\d\W_])\1{2,}")
EMOJI_REGEX = re.compile(
    r"[\U00010000-\U0010FFFF\u2600-\u27BF\u2300-\u23FF\u2B00-\u2BFF\uFE0F]",
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
ZERO_WIDTH_REGEX = re.compile(r"[\u00AD\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")
CLITIC_LATIN_REGEX = re.compile(
    r"(^|[\s\"'(\[{])(ال|وال|بال|كال|فال|لل|[وبفلك])\u0640+\s*([A-Za-z][A-Za-z0-9_.-]*)"
)

ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

UNICODE_VARIANT_TRANS = str.maketrans({
    "\u06A9": "ك",
    "\u06AA": "ك",
    "\u06CC": "ي",
    "\u06D2": "ي",
    "\u06C0": "ة",
    "\u06C1": "ه",
    "\u06BE": "ه",
    "\u00A0": " ",
    "\u201C": '"',
    "\u201D": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u2013": "-",
    "\u2014": "-",
})


def normalize_unicode_safety(text: str) -> str:
    """Fold PDF Arabic Presentation Forms (NFKC) and Farsi/Urdu keyboard lookalikes without breaking أإآةى."""
    if not text:
        return text
    return unicodedata.normalize("NFKC", text).translate(UNICODE_VARIANT_TRANS)

# Nouns referring to male roles/kinship ending in possessive -ه where +ة is a valid feminine noun;
# protected from Vocab Auto-Heal so "مديره" (his manager) is never turned into "مديرة" (female manager).
POSSESSIVE_AMBIGUOUS_H_WORDS: Set[str] = {
    "مديره", "وزيره", "سفيره", "رئيسه", "عميده", "زميله", "صديقه", "شقيقه",
    "حبيبه", "رفيقه", "طبيبه", "عالمه", "معلمه", "طالبه", "كاتبه", "تاجره",
    "شاعره", "والده", "جده", "خاله", "عمه", "أميره", "اميره",
}

POSITIVE_EMOJIS: Set[str] = {
    "😍", "🥰", "😘", "😊", "😁", "😄", "😃", "🥳", "🤩", "🙌", "👏",
    "👍", "👌", "🔥", "🚀", "✨", "🌟", "⭐", "💯", "✅", "♥", "❤", "💖", "💙", "💚",
}

NEGATIVE_EMOJIS: Set[str] = {
    "😡", "🤬", "😠", "👎", "💔", "😢", "😭", "😞", "😔", "😤", "🤮",
    "🤢", "💩", "⚠", "❌", "😒", "🙁", "😣", "😫", "😩",
}

ARABIZI_TECH_PROTECT_REGEX = re.compile(
    r"^(base64|utf8|utf16|utf32|http2|http3|sha256|sha512|md5|web3|p2p|b2b|b2c|mp3|mp4|h264|x86|arm64|v[0-9]+|gpt[0-9a-z]*|py3[0-9]*|llama[0-9]*|bert[a-z0-9-]*)$",
    flags=re.IGNORECASE,
)

ARABIZI_LEXICON: Dict[str, str] = {
    "salam": "سلام",
    "3alekom": "عليكم",
    "3alaykom": "عليكم",
    "mar7aba": "مرحبا",
    "ahlan": "أهلا",
    "kefak": "كيفك",
    "shlonak": "شلونك",
    "7elw": "حلو",
    "7elwa": "حلوة",
    "3ajabni": "عجبني",
    "masha2allah": "ماشاء الله",
    "insha2allah": "ان شاء الله",
    "5alas": "خلاص",
    "3arab": "عرب",
    "3arabi": "عربي",
    "tamam": "تمام",
    "akeed": "أكيد",
    "shukran": "شكرا",
    "mashakil": "مشاكل",
    "habibi": "حبيبي",
    "yalla": "يلا",
}

ARABIZI_CHAR_MAP: Dict[str, str] = {
    "2": "أ", "3": "ع", "5": "خ", "6": "ط", "7": "ح", "8": "غ", "9": "ق",
    "a": "ا", "b": "ب", "t": "ت", "j": "ج", "d": "د", "r": "ر", "z": "ز",
    "s": "س", "f": "ف", "k": "ك", "l": "ل", "m": "م", "n": "ن", "h": "ه",
    "w": "و", "o": "و", "u": "و", "y": "ي", "i": "ي", "e": "ي",
}

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
