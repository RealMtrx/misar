# -*- coding: utf-8 -*-
"""
Microsecond Arabic Dialect & Register Classifier for Misar.

Provides lightweight, deterministic, zero-dependency sub-dialect classification across:
- Saudi & Gulf (`saudi_gulf`): Najdi, Hijazi, Emirati, Kuwaiti markers
- Levantine (`levantine`): Syrian, Lebanese, Jordanian, Palestinian
- Egyptian (`egyptian`): Cairene, Sa'idi
- Maghrebi (`maghrebi`): Moroccan, Algerian, Tunisian
- Modern Standard Arabic (`msa`): Standard formal Arabic

Also provides discourse register identification (`social_slang`, `formal_official`,
`tech_dev`, `classical_heritage`) and semantic canonicalization for LLM prompt caching.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Regional Dialect Lexicon Signatures
# ---------------------------------------------------------------------------

SAUDI_GULF_MARKERS: Set[str] = {
    "وش", "وشو", "ايش", "إيش", "أبغى", "ابغى", "تبغى", "يبغى", "نبغى", "أبي", "ابي",
    "تبي", "يبي", "نبي", "ودنا", "ودي", "ودك", "حقت", "حقته", "حقها", "حقهم", "حقتي",
    "حقتك", "الحين", "دحين", "ياخي", "ياخوك", "تكفى", "تكفون", "شسالفة", "شلونك", "شلونكم",
    "بالحيل", "مره", "مررة", "مرهـ", "زين", "شوي", "شويات", "شويتين", "مافيه", "مافي",
    "ماكو", "مب", "مو", "عشان", "علشانه", "هذا", "هذي", "هذول", "هذولي", "سالفة", "خلاص",
    "عسى", "تراه", "تراها", "تراهم", "ترا", "تراي", "عطني", "سوا", "سوى", "قاعد", "قاعدين",
    "ارحب", "أرحب", "هلا_والله", "ماشاء_الله",
}

LEVANTINE_MARKERS: Set[str] = {
    "شو", "بدي", "بدك", "بده", "بدها", "بدنا", "بدهم", "هيك", "هيكا", "هلق", "هلا",
    "هلقيت", "إسا", "إسه", "عم", "منيح", "منيحة", "مناح", "شلونك", "كيفك", "كيفكن",
    "يا_زلمة", "زلمة", "شب", "صبايا", "ما_بدي", "مو_هيك", "هاد", "هادا", "هادي", "هادول",
    "كتير", "كتيرة", "شوي", "شوية", "عشان", "مشان", "رمسة", "هلأ", "وينك", "وينكن",
    "بلش", "عم_يحكي", "عم_نحكي", "يلا",
}

EGYPTIAN_MARKERS: Set[str] = {
    "إيه", "ايه", "عايز", "عايزة", "عايزين", "كده", "كدا", "دلوقتي", "دلوءتي", "علشان",
    "عشان", "أوي", "اوي", "ازيك", "ازيكم", "فين", "ليه_كده", "مش_عايز", "بتاع", "بتاعتي",
    "بتاعته", "بتاعتنا", "بتوع", "خالص", "برضه", "برضو", "يلا_بينا", "أهو", "اهو",
    "شغال", "كويس", "جدع", "طب", "طب_ليه", "يعني_إيه", "مفيش", "معلش", "حاجة", "حاجه",
    "كويس_جداً", "إزاي", "ازاي", "يا_عم", "يا_باشا", "يا_فندم",
}

MAGHREBI_MARKERS: Set[str] = {
    "ديال", "ديالي", "ديالنا", "ديالكم", "ديالهم", "بزاف", "دابا", "واش", "برشا", "شنو",
    "كاين", "ماكاينش", "حنا", "زوين", "زوينة", "مزيان", "مزيانة", "علاش", "هكا", "هكاك",
    "غادي", "نتا", "نتوما", "كاينة", "باهي", "برشا_برشا", "شكون", "وقتاش", "كفاش",
    "بلاتي", "صافي", "دير", "درتي", "خدمة",
}

MSA_FORMAL_MARKERS: Set[str] = {
    "الذي", "التي", "الذين", "اللواتي", "اللاتي", "اللذان", "اللتان",
    "حيثما", "بيد_أن", "بيد", "غير_أن", "علاوة_على", "فضلاً_عن", "بناءً_على",
    "نظراً_لأن", "الأمر_الذي", "سوف", "كذلك", "إذ", "منذ", "حينئذ", "عندئذ",
    "إنما", "ريثما", "طالما", "سيما", "لاسيما", "لذا", "لذلك", "هنالك",
    "شريطة_أن", "وفقاً_لـ", "بموجب", "استناداً_إلى",
}

# ---------------------------------------------------------------------------
# Canonical Normalization Mapping (For LLM Prompt Cache Optimization)
# ---------------------------------------------------------------------------

CANONICAL_INTERROGATIVES: Dict[str, str] = {
    "وش": "ما",
    "وشو": "ما",
    "ايش": "ما",
    "إيش": "ما",
    "شو": "ما",
    "إيه": "ما",
    "ايه": "ما",
    "شنو": "ما",
    "واش": "ما",
    "عايز": "أريد",
    "عايزة": "أريد",
    "عايزين": "نريد",
    "بدي": "أريد",
    "بدنا": "نريد",
    "ابغى": "أريد",
    "أبغى": "أريد",
    "ابي": "أريد",
    "أبي": "أريد",
    "نبي": "نريد",
    "نبغى": "نريد",
    "ازاي": "كيف",
    "إزاي": "كيف",
    "شلون": "كيف",
    "كفاش": "كيف",
    "فين": "أين",
    "وين": "أين",
    "علاش": "لماذا",
    "ليه": "لماذا",
    "دلوقتي": "الآن",
    "الحين": "الآن",
    "دحين": "الآن",
    "هلق": "الآن",
    "دابا": "الآن",
    "أوي": "جداً",
    "اوي": "جداً",
    "بزاف": "جداً",
    "برشا": "جداً",
    "كتير": "كثيراً",
    "مش": "ليس",
    "مو": "ليس",
    "ماكو": "لا يوجد",
    "مافيه": "لا يوجد",
    "مفيش": "لا يوجد",
}

# ---------------------------------------------------------------------------
# Register Keywords
# ---------------------------------------------------------------------------

TECH_DEV_MARKERS: Set[str] = {
    "api", "backend", "frontend", "database", "server", "docker", "cloud",
    "token", "json", "rest", "graphql", "python", "javascript", "code",
    "خادم", "قواعد_البيانات", "واجهة_البرمجة", "برمجة", "تطبيق", "نظام",
    "استضافة", "سحابة", "معمارية", "توكن", "كود", "مكتبة", "إطار_عمل",
}

LEGAL_FORMAL_MARKERS: Set[str] = {
    "المادة", "الفقرة", "المرسوم", "اللائحة", "القرار", "الوزاري",
    "المجلس", "الاتفاقية", "العقد", "البند", "الطرفان", "النظام",
    "الرسمي", "الجريدة_الرسمية", "الوزارة", "الهيئة", "المحكمة",
}

CLASSICAL_HERITAGE_MARKERS: Set[str] = {
    "روى", "حدثنا", "أخبرنا", "عن", "قال", "تعالى", "رسول_الله", "صلى_الله_عليه_وسلم",
    "رضي_الله_عنه", "رحمه_الله", "القرآن", "السنة", "الإسناد", "الفقه",
}


@dataclass
class DialectClassification:
    """Telemetry report for Arabic dialect and register classification."""
    primary_dialect: str
    confidence: float
    dialect_scores: Dict[str, float]
    register: str
    register_confidence: float
    detected_markers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_dialect": self.primary_dialect,
            "confidence": round(self.confidence, 3),
            "dialect_scores": {k: round(v, 3) for k, v in self.dialect_scores.items()},
            "register": self.register,
            "register_confidence": round(self.register_confidence, 3),
            "detected_markers": list(self.detected_markers),
        }


class DialectClassifier:
    """
    High-speed deterministic Arabic dialect and register classifier (< 0.1 ms on CPU).
    """

    def __init__(self) -> None:
        self.word_split_regex = re.compile(r"[\u0621-\u064A\u0671-\u06D3A-Za-z0-9_]+")
        self.diacritic_regex = re.compile(r"[\u064B-\u065F\u0670]")

    def classify(self, text: str) -> DialectClassification:
        """Classify incoming Arabic text into primary dialect, scores, and register."""
        if not text or not text.strip():
            return DialectClassification(
                primary_dialect="msa",
                confidence=1.0,
                dialect_scores={"msa": 1.0, "saudi_gulf": 0.0, "levantine": 0.0, "egyptian": 0.0, "maghrebi": 0.0},
                register="formal_official",
                register_confidence=1.0,
                detected_markers=[],
            )

        words = [w.lower() for w in self.word_split_regex.findall(text)]
        if not words:
            return DialectClassification(
                primary_dialect="msa",
                confidence=1.0,
                dialect_scores={"msa": 1.0, "saudi_gulf": 0.0, "levantine": 0.0, "egyptian": 0.0, "maghrebi": 0.0},
                register="formal_official",
                register_confidence=1.0,
                detected_markers=[],
            )

        # Count dialect markers
        counts: Dict[str, int] = {
            "saudi_gulf": 0,
            "levantine": 0,
            "egyptian": 0,
            "maghrebi": 0,
            "msa": 0,
        }
        detected_markers: List[str] = []

        for w in words:
            matched = False
            if w in SAUDI_GULF_MARKERS:
                counts["saudi_gulf"] += 1
                detected_markers.append(f"saudi_gulf:{w}")
                matched = True
            if w in LEVANTINE_MARKERS:
                counts["levantine"] += 1
                detected_markers.append(f"levantine:{w}")
                matched = True
            if w in EGYPTIAN_MARKERS:
                counts["egyptian"] += 1
                detected_markers.append(f"egyptian:{w}")
                matched = True
            if w in MAGHREBI_MARKERS:
                counts["maghrebi"] += 1
                detected_markers.append(f"maghrebi:{w}")
                matched = True
            if w in MSA_FORMAL_MARKERS:
                counts["msa"] += 1
                detected_markers.append(f"msa:{w}")
                matched = True

        total_dialect_hits = (
            counts["saudi_gulf"] + counts["levantine"] + counts["egyptian"] + counts["maghrebi"]
        )

        # Softmax-like normalized distribution
        scores: Dict[str, float] = {}
        if total_dialect_hits == 0:
            scores = {
                "msa": 0.85 if counts["msa"] > 0 else 0.70,
                "saudi_gulf": 0.08,
                "levantine": 0.08,
                "egyptian": 0.07,
                "maghrebi": 0.07,
            }
            # Normalize to sum to 1.0
            sum_s = sum(scores.values())
            scores = {k: v / sum_s for k, v in scores.items()}
            primary = "msa"
            confidence = scores["msa"]
        else:
            base_msa = 0.10 if counts["msa"] == 0 else 0.10 + 0.15 * counts["msa"]
            raw_scores = {
                "saudi_gulf": counts["saudi_gulf"] * 1.5,
                "levantine": counts["levantine"] * 1.5,
                "egyptian": counts["egyptian"] * 1.5,
                "maghrebi": counts["maghrebi"] * 1.6,
                "msa": base_msa,
            }
            total_raw = sum(raw_scores.values())
            scores = {k: v / total_raw for k, v in raw_scores.items()}
            primary = max(scores.keys(), key=lambda k: scores[k])
            confidence = scores[primary]

        # Register classification
        register, reg_conf = self._classify_register(text, words, primary)

        return DialectClassification(
            primary_dialect=primary,
            confidence=confidence,
            dialect_scores=scores,
            register=register,
            register_confidence=reg_conf,
            detected_markers=detected_markers[:15],
        )

    def _classify_register(self, text: str, words: List[str], primary_dialect: str) -> Tuple[str, float]:
        """Classify text discourse register: formal_official, tech_dev, classical_heritage, or social_slang."""
        # 1. Classical Heritage: high diacritic density or religious markers
        diac_count = len(self.diacritic_regex.findall(text))
        arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        diac_density = (diac_count / arabic_chars) if arabic_chars > 0 else 0.0
        heritage_hits = sum(1 for w in words if w in CLASSICAL_HERITAGE_MARKERS)

        if diac_density >= 0.25 or heritage_hits >= 2:
            return "classical_heritage", 0.90 if diac_density >= 0.30 else 0.80

        # 2. Tech Dev: latin code-switching or tech keywords
        tech_hits = sum(1 for w in words if w in TECH_DEV_MARKERS)
        latin_chars = sum(1 for c in text if ("a" <= c <= "z" or "A" <= c <= "Z"))
        code_switch_ratio = (latin_chars / len(text)) if text else 0.0

        if tech_hits >= 2 or code_switch_ratio >= 0.15:
            return "tech_dev", 0.88 if tech_hits >= 3 else 0.75

        # 3. Legal / Formal Official
        legal_hits = sum(1 for w in words if w in LEGAL_FORMAL_MARKERS)
        if legal_hits >= 2 or (primary_dialect == "msa" and ("السيد" in text or "الوزارة" in text or "القرار" in text)):
            return "formal_official", 0.85

        # 4. Social Slang
        if primary_dialect != "msa" or any(c in text for c in ["😂", "❤️", "🔥", "✨", "#"]):
            return "social_slang", 0.82

        return "formal_official", 0.70

    def canonicalize(self, text: str) -> str:
        """
        Normalize variant dialectal questions to a canonicalized semantic representation.
        Significantly boosts LLM Prompt Cache hit rates (e.g. on OpenAI, Anthropic, ALLaM).
        """
        if not text or not text.strip():
            return ""

        def _sub_word(m: re.Match) -> str:
            w = m.group(0)
            clean = w.strip()
            return CANONICAL_INTERROGATIVES.get(clean, CANONICAL_INTERROGATIVES.get(clean.lower(), w))

        word_pat = re.compile(r"[\u0621-\u064A\u0671-\u06D3]+")
        canonicalized = word_pat.sub(_sub_word, text)
        return " ".join(canonicalized.split())


_DEFAULT_CLASSIFIER: Optional[DialectClassifier] = None


def _get_classifier() -> DialectClassifier:
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = DialectClassifier()
    return _DEFAULT_CLASSIFIER


def classify_dialect(text: str) -> DialectClassification:
    """Convenience helper to classify Arabic dialect and register."""
    return _get_classifier().classify(text)


def canonicalize(text: str) -> str:
    """Convenience helper to canonicalize dialectal questions for LLM prompt caching."""
    return _get_classifier().canonicalize(text)
