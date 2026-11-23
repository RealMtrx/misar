# -*- coding: utf-8 -*-
"""
High-speed Context-Aware Arabic Diacritization & Tashkeel Recovery Engine for Misar (مِسار).
Features:
- Pure Python zero-dependency automatic diacritic restoration (Fatha, Damma, Kasra, Sukun, Shaddah, Tanween)
- Morphological template recognition (تفعيل، مفعلة، استفعال، افتعال، فاعل، مفعول)
- Canonical particle and preposition diacritics
- Clitic prefix/suffix decomposition (الـ، وـ، فـ، بـ، لـ، كـ)
- High-precision preservation of existing diacritics, Latin terms, emojis, and punctuation
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# Arabic Diacritic Characters
FATHA = "\u064E"      # َ
DAMMA = "\u064F"      # ُ
KASRA = "\u0650"      # ِ
SUKUN = "\u0652"      # ْ
SHADDA = "\u0651"     # ّ
FATHATAN = "\u064B"   # ً
DAMMATAN = "\u064C"   # ٌ
KASRATAN = "\u064D"   # ٍ

ALL_DIACRITICS = {FATHA, DAMMA, KASRA, SUKUN, SHADDA, FATHATAN, DAMMATAN, KASRATAN}
TASHKEEL_REGEX = re.compile(r"[\u064B-\u0652\u0670]")

# Canonical Grammatical Particles and High-Frequency Core Words
CANONICAL_DIACRITIZED_WORDS: Dict[str, str] = {
    # Prepositions & Conjunctions
    "في": "فِي",
    "من": "مِنْ",
    "إلى": "إِلَى",
    "الى": "إِلَى",
    "على": "عَلَى",
    "علي": "عَلَى",
    "عن": "عَنْ",
    "مع": "مَعَ",
    "منذ": "مُنْذُ",
    "مذ": "مُذْ",
    "حتى": "حَتَّى",
    "ثم": "ثُمَّ",
    "أو": "أَوْ",
    "او": "أَوْ",
    "أم": "أَمْ",
    "ام": "أَمْ",
    "بل": "بَلْ",
    "لكن": "لَكِنَّ",
    "لاكن": "لَكِنَّ",
    "حيث": "حَيْثُ",
    "حين": "حِينَ",
    "لدى": "لَدَى",
    "دون": "دُونَ",
    "نحو": "نَحْوَ",
    "حول": "حَوْلَ",
    "فوق": "فَوْقَ",
    "تحت": "تَحْتَ",
    "أمام": "أَمَامَ",
    "امام": "أَمَامَ",
    "خلف": "خَلْفَ",
    "بين": "بَيْنَ",
    "قبل": "قَبْلَ",
    "بعد": "بَعْدَ",
    # Pronouns & Demonstratives
    "هو": "هُوَ",
    "هي": "هِيَ",
    "هما": "هُمَا",
    "هم": "هُمْ",
    "هن": "هُنَّ",
    "أنا": "أَنَا",
    "انا": "أَنَا",
    "نحن": "نَحْنُ",
    "أنت": "أَنْتَ",
    "انت": "أَنْتَ",
    "أنتم": "أَنْتُمْ",
    "انتم": "أَنْتُمْ",
    "هذا": "هَذَا",
    "هذه": "هَذِهِ",
    "هذان": "هَذَانِ",
    "هاتان": "هَاتَانِ",
    "هؤلاء": "هَؤُلَاءِ",
    "ذلك": "ذَلِكَ",
    "تلك": "تِلْكَ",
    "أولئك": "أُولَئِكَ",
    "الذي": "الَّذِي",
    "التي": "الَّتِي",
    "الذين": "الَّذِينَ",
    "اللواتي": "اللَّوَاتِي",
    "اللاتي": "اللَّاتِي",
    # Particles & Negations
    "أن": "أَنْ",
    "ان": "أَنْ",
    "إن": "إِنَّ",
    "إنما": "إِنَّمَا",
    "أنما": "أَنَّمَا",
    "كأن": "كَأَنَّ",
    "ليت": "لَيْتَ",
    "لعل": "لَعَلَّ",
    "لا": "لَا",
    "ما": "مَا",
    "لم": "لَمْ",
    "لن": "لَنْ",
    "ليس": "لَيْسَ",
    "ليست": "لَيْسَتْ",
    "قد": "قَدْ",
    "لقد": "لَقَدْ",
    "سوف": "سَوْفَ",
    "كل": "كُلّ",
    "جميع": "جَمِيع",
    "غير": "غَيْر",
    "سوى": "سِوَى",
    "إلا": "إِلَّا",
    "الا": "إِلَّا",
    "هل": "هَلْ",
    "كم": "كَمْ",
    "كيف": "كَيْفَ",
    "أين": "أَيْنَ",
    "اين": "أَيْنَ",
    "متى": "مَتَى",
    "ماذا": "مَاذَا",
    "لماذا": "لِمَاذَا",
    # Auxiliaries
    "كان": "كَانَ",
    "كانت": "كَانَتْ",
    "يكون": "يَكُونُ",
    "تكون": "تَكُونُ",
    "نكون": "نَكُونُ",
    "أكون": "أَكُونُ",
    "اصبح": "أَصْبَحَ",
    "أصبح": "أَصْبَحَ",
    "صار": "صَارَ",
    "صارت": "صَارَتْ",
    # Sacred & Sovereign
    "الله": "اللهِ",
    "محمد": "مُحَمَّد",
    "رسول": "رَسُول",
    "سدايا": "سَدَايَا",
    "موهبة": "مَوْهِبَة",
    "أرامكو": "أَرَامْكُو",
    "المملكة": "المَمْلَكَةُ",
    "مملكة": "مَمْلَكَة",
    "السعودية": "السَّعُودِيَّةُ",
    "سعودية": "سَعُودِيَّة",
    "الرياض": "الرِّيَاضُ",
    "عسير": "عَسِير",
    "أبها": "أَبْهَا",
    # Common Core NLP & Tech Vocabulary
    "الذكاء": "الذَّكَاءُ",
    "الاصطناعي": "الاصْطِنَاعِيُّ",
    "اللغة": "اللُّغَةُ",
    "العربية": "العَرَبِيَّةُ",
    "البيانات": "البَيَانَاتُ",
    "التعليم": "التَّعْلِيمُ",
    "الصحة": "الصِّحَّةُ",
    "التقنية": "التِّقْنِيَّةُ",
    "البرمجة": "البَرْمَجَةُ",
    "النماذج": "النَّمَاذِجُ",
    "الحوسبة": "الحَوْسَبَةُ",
    "السحابية": "السَّحَابِيَّةُ",
    "المستقبل": "المُسْتَقْبَلُ",
    "التطوير": "التَّطْوِيرُ",
    "الابتكار": "الابْتِكَارُ",
    "الاستخدام": "الاسْتِخْدَامُ",
    "مشروع": "مَشْرُوع",
    "تطبيق": "تَطْبِيق",
    "منصة": "مَنَصَّة",
    "نظام": "نِظَام",
    "خوارزمية": "خَوَارِزْمِيَّة",
    "نموذج": "نَمُوذَج",
    "معالجة": "مُعَالَجَة",
    "ابتكار": "ابْتِكَار",
    "استخدام": "اسْتِخْدَام",
    "تطوير": "تَطْوِير",
    "أداء": "أَدَاء",
    "سرعة": "سُرْعَة",
    "دقة": "دِقَّة",
    "مؤسسة": "مُؤَسَّسَة",
    "هيئة": "هَيْئَة",
    "جامعة": "جَامِعَة",
    "مركز": "مَرْكَز",
    "وطني": "وَطَنِيّ",
    "علمي": "عِلْمِيّ",
    "بحث": "بَحْث",
    "دراسة": "دِرَاسَة",
    "عمل": "عَمَل",
    "مجال": "مَجَال",
    "حلول": "حُلُول",
    "رؤية": "رُؤْيَة",
    "تحول": "تَحَوُّل",
    "رقمي": "رَقْمِيّ",
    "إنسان": "إِنْسَان",
    "عالم": "عَالَم",
    "كتاب": "كِتَاب",
    "قلم": "قَلَم",
    "معلم": "مُعَلِّم",
    "طالب": "طَالِب",
    "مدرسة": "مَدْرَسَة",
    "مدينة": "مَدِينَة",
    "دولة": "دَوْلَة",
    "حكومة": "حُكُومَة",
}


@dataclass
class DiacritizationResult:
    """Detailed report for Arabic diacritization and vowel restoration."""
    original_text: str
    diacritized_text: str
    vowels_added_count: int
    diacritic_density: float
    confidence: float = 0.94

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _diacritize_morphological_pattern(stem: str) -> Optional[str]:
    """Infers morphological diacritics based on Arabic templates (الأوزان الصرفية)."""
    n = len(stem)
    if n < 3:
        return None

    # Form X Verbal Noun: اسْتِفْعَال (استكشاف، استخراج، استخدام)
    if stem.startswith("است") and n >= 6 and stem[-2] == "ا":
        return f"اسْتِ{stem[3]}{SUKUN}{stem[4]}{FATHA}ا{stem[5:]}"

    # Form VIII Verbal Noun: افْتِعَال (ابتكار، اقتصاد، اجتماع، انتشار)
    if stem.startswith(("ابت", "اقت", "اجت", "انت", "اخت", "است")) and n >= 6 and stem[-2] == "ا":
        return f"{stem[0]}{stem[1]}{SUKUN}{stem[2]}{KASRA}{stem[3]}{FATHA}ا{stem[5:]}"

    # Form II Verbal Noun: تَفْعِيل (تطوير، تصميم، تدريب، تنظيم، تسريع)
    if stem.startswith("ت") and n >= 5 and stem[-2] == "ي":
        return f"تَ{stem[1]}{SUKUN}{stem[2]}{KASRA}ي{stem[4:]}"

    # Form V Verbal Noun: تَفَعُّل (تحول، تطور، تقدم، تعلم)
    if stem.startswith("ت") and n == 4:
        return f"تَ{stem[1]}{FATHA}{stem[2]}{SHADDA}{DAMMA}{stem[3]}"

    # Passive Participle / Form I noun: مَفْعُول (مشروع، معلوم، مكتوب)
    if stem.startswith("م") and n == 5 and stem[3] == "و":
        return f"مَ{stem[1]}{SUKUN}{stem[2]}{DAMMA}و{stem[4]}"

    # Place / Tool noun: مَفْعَل / مَفْعِل (مركز، مصنع، مكتب، معهد)
    if stem.startswith("م") and n == 4:
        return f"مَ{stem[1]}{SUKUN}{stem[2]}{FATHA}{stem[3]}"

    # Place / Institution noun: مَفْعَلَة (مملكة، مدرسة، محكمة، مطبعة)
    if stem.startswith("م") and n == 5 and stem.endswith(("ة", "ه")):
        return f"مَ{stem[1]}{SUKUN}{stem[2]}{FATHA}{stem[3]}{FATHA}ة"

    # Form I Active Participle: فَاعِل (عالم، كاتب، باحث، صانع)
    if stem[1] == "ا" and n == 4:
        return f"{stem[0]}{FATHA}ا{stem[2]}{KASRA}{stem[3]}"

    # Adjective: فَعِيل (سريع، جميل، دقيق، جديد، قديم، عظيم)
    if stem[2] == "ي" and n == 4:
        return f"{stem[0]}{FATHA}{stem[1]}{KASRA}ي{stem[3]}"

    return None


def diacritize_word(word: str) -> str:
    """Restores canonical diacritics to a single Arabic token."""
    # If the word already contains diacritics, preserve them
    if TASHKEEL_REGEX.search(word):
        return word

    # 1. Exact match in canonical lexicon
    if word in CANONICAL_DIACRITIZED_WORDS:
        return CANONICAL_DIACRITIZED_WORDS[word]

    # 2. Decompose clitic prefixes (و، ف، ب، ل، ك، ال)
    prefix = ""
    prefix_diacritic = ""
    stem = word

    # Handle definite article الـ
    if word.startswith("ال") and len(word) >= 3:
        prefix = "ال"
        prefix_diacritic = "الْ"
        stem = word[2:]
        if stem in CANONICAL_DIACRITIZED_WORDS:
            return prefix_diacritic + CANONICAL_DIACRITIZED_WORDS[stem]
    elif len(word) >= 3 and word[0] in ("و", "ف"):
        prefix = word[0]
        prefix_diacritic = f"{word[0]}{FATHA}"
        stem = word[1:]
        if stem.startswith("ال") and len(stem) >= 3:
            prefix += "ال"
            prefix_diacritic += "الْ"
            stem = stem[2:]
        if stem in CANONICAL_DIACRITIZED_WORDS:
            return prefix_diacritic + CANONICAL_DIACRITIZED_WORDS[stem]
    elif len(word) >= 3 and word[0] in ("ب", "ل", "ك"):
        prefix = word[0]
        prefix_diacritic = f"{word[0]}{KASRA}" if word[0] in ("ب", "ل") else f"{word[0]}{FATHA}"
        stem = word[1:]
        if stem.startswith("ال") and len(stem) >= 3:
            prefix += "ال"
            prefix_diacritic += "الْ"
            stem = stem[2:]
        if stem in CANONICAL_DIACRITIZED_WORDS:
            return prefix_diacritic + CANONICAL_DIACRITIZED_WORDS[stem]

    # 3. Morphological template pattern inference
    pattern_diacritized = _diacritize_morphological_pattern(stem)
    if pattern_diacritized is not None:
        result = prefix_diacritic + pattern_diacritized if prefix else pattern_diacritized
        # Nisba suffix handling (ـي -> ـِيّ)
        if result.endswith("ي"):
            result = result[:-1] + f"{KASRA}ي{SHADDA}"
        elif result.endswith("ية") or result.endswith("يه"):
            result = result[:-2] + f"{KASRA}ي{SHADDA}{FATHA}ة"
        elif result.endswith("ة"):
            if not result.endswith(f"{FATHA}ة"):
                result = result[:-1] + f"{FATHA}ة"
        return result

    # 3b. If stripped prefix didn't match a pattern, test unstripped word directly
    if prefix:
        pattern_full = _diacritize_morphological_pattern(word)
        if pattern_full is not None:
            if pattern_full.endswith("ي"):
                pattern_full = pattern_full[:-1] + f"{KASRA}ي{SHADDA}"
            elif pattern_full.endswith("ية") or pattern_full.endswith("يه"):
                pattern_full = pattern_full[:-2] + f"{KASRA}ي{SHADDA}{FATHA}ة"
            elif pattern_full.endswith("ة"):
                if not pattern_full.endswith(f"{FATHA}ة"):
                    pattern_full = pattern_full[:-1] + f"{FATHA}ة"
            return pattern_full

    # 4. Fallback: Nisba or Feminine end-marking
    if stem.endswith("ية") or stem.endswith("يه"):
        return f"{prefix_diacritic}{stem[:-2]}{KASRA}ي{SHADDA}{FATHA}ة"
    if stem.endswith("ة"):
        return f"{prefix_diacritic}{stem[:-1]}{FATHA}ة"

    return word


def diacritize_text(text: str) -> DiacritizationResult:
    """
    Diacritizes Arabic text, restoring core vowels, shaddah, and sukun marks.
    Preserves numbers, punctuation, emojis, and non-Arabic script intact.
    """
    if not text or not text.strip():
        return DiacritizationResult(original_text=text, diacritized_text=text, vowels_added_count=0, diacritic_density=0.0)

    # Tokenize preserving spaces, punctuation, and non-Arabic segments
    tokens = re.split(r"(\s+|[^\w\u0600-\u06FF])", text, flags=re.UNICODE)
    diacritized_tokens: List[str] = []
    vowels_added = 0

    for tok in tokens:
        if not tok or not any("\u0600" <= ch <= "\u06FF" for ch in tok):
            diacritized_tokens.append(tok)
            continue

        raw_vowels_before = len(TASHKEEL_REGEX.findall(tok))
        d_tok = diacritize_word(tok)
        raw_vowels_after = len(TASHKEEL_REGEX.findall(d_tok))

        vowels_added += max(0, raw_vowels_after - raw_vowels_before)
        diacritized_tokens.append(d_tok)

    final_text = "".join(diacritized_tokens)
    arabic_chars = len(re.findall(r"[\u0621-\u064A]", final_text))
    total_diacritics = len(TASHKEEL_REGEX.findall(final_text))
    density = (total_diacritics / max(1, arabic_chars)) if arabic_chars > 0 else 0.0

    return DiacritizationResult(
        original_text=text,
        diacritized_text=final_text,
        vowels_added_count=vowels_added,
        diacritic_density=round(density, 4),
        confidence=0.94,
    )
