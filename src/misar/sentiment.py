# -*- coding: utf-8 -*-
"""
High-performance Arabic Sentiment & Emotion Analyzer for Misar (مِسار).
Features:
- Dialect-aware Arabic sentiment scoring (Saudi/Gulf, Levantine, Egyptian, Maghrebi, MSA)
- Nuanced dialectal negation inverter ("مو حلو", "مش بطال", "ما قصرت", "ولا غلطة")
- Valence intensifiers ("مرررة", "جداً", "وايد", "بزاف", "حيل")
- Emotional resonance and polarity classification with zero external dependencies
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Positive Lexicon: Classical MSA + Modern Dialects (Saudi, Gulf, Egyptian, Levantine, Maghrebi)
POSITIVE_WORDS: Dict[str, float] = {
    # MSA / Formal
    "ممتاز": 1.0, "رائع": 1.0, "جميل": 0.8, "متقن": 0.9, "مبدع": 0.9,
    "ناجح": 0.8, "متألق": 0.9, "مفيد": 0.7, "سريع": 0.6, "دقيق": 0.8,
    "احترافي": 0.9, "قوي": 0.7, "ثابت": 0.6, "مستقر": 0.7, "أفضل": 0.9,
    "أحسن": 0.8, "عظيم": 0.9, "مبهر": 1.0, "باهر": 0.9, "تطور": 0.7,
    "شكر": 0.7, "شكرا": 0.7, "ممتن": 0.8, "سعادة": 0.8, "فخور": 0.9,
    "أمان": 0.7, "آمن": 0.8, "ثقة": 0.8, "موثوق": 0.9, "سهل": 0.7,
    # Saudi & Gulf Dialectal
    "بطل": 0.9, "كفو": 1.0, "رهيب": 1.0, "فنان": 0.9, "فخم": 1.0,
    "خيال": 0.9, "جبار": 1.0, "عسل": 0.8, "زين": 0.7, "حلو": 0.8,
    "يجنن": 1.0, "يطير العقل": 1.0, "ما قصرت": 1.0, "بيض الله وجهك": 1.0,
    "يسلمو": 0.8, "يعطيك العافية": 0.8, "كويس": 0.7, "تمام": 0.7,
    # Levantine & Egyptian
    "منيح": 0.7, "كتير منيح": 0.9, "حلو كتير": 0.9, "جامد": 0.9, "تحفة": 1.0,
    "فل": 0.8, "مظبوط": 0.7, "هايل": 0.9, "شغال": 0.6,
    # Maghrebi
    "مزيان": 0.8, "واعر": 0.9, "مليح": 0.7, "باهي": 0.7, "غايه": 0.8,
}

# Negative Lexicon: Classical MSA + Modern Dialects
NEGATIVE_WORDS: Dict[str, float] = {
    # MSA / Formal
    "سيء": -0.9, "رديء": -0.9, "فاشل": -1.0, "ضعيف": -0.7, "بطيء": -0.7,
    "معقد": -0.7, "خسارة": -0.8, "كارثة": -1.0, "خطأ": -0.7, "أخطاء": -0.7,
    "عطل": -0.8, "انقطاع": -0.8, "خلل": -0.8, "مشكلة": -0.7, "صعب": -0.6,
    "مكلف": -0.6, "غير آمن": -0.9, "خطر": -0.8, "تالف": -0.9, "فوضى": -0.8,
    "تراجع": -0.7, "رديء جدا": -1.0, "مؤسف": -0.8, "حزين": -0.7, "قلق": -0.6,
    # Saudi & Gulf Dialectal
    "خايس": -1.0, "زبالة": -1.0, "تعبان": -0.8, "خربان": -0.9, "يعلق": -0.8,
    "يهنق": -0.8, "مقرف": -0.9, "مقلب": -0.9, "نكبة": -1.0, "ضايع": -0.8,
    "شين": -0.8, "ردي": -0.9, "مهزلة": -1.0, "مسخرة": -1.0,
    # Levantine & Egyptian
    "عاطل": -0.8, "مش منيح": -0.8, "وحش": -0.8, "زفت": -1.0, "مقرف": -0.9,
    "بشع": -0.8, "مهبب": -0.9, "منيل": -0.9,
    # Maghrebi
    "خايب": -0.8, "ماشي مزيان": -0.8, "عيان": -0.8, "مكفس": -0.9,
}

# Negation particles across dialects
NEGATION_PARTICLES: Set[str] = {
    "مو", "مش", "ما", "ليس", "مهوب", "مب", "ماني", "ماكو", "مافيه", "مافيها",
    "ماشي", "مافيش", "بدون", "غير", "لا", "لن", "لم",
}

# Idiomatic phrases where negation inverts to positive praise
POSITIVE_IDIOMATIC_NEGATIONS: Dict[str, float] = {
    "ما قصرت": 1.0,
    "ما قصرتوا": 1.0,
    "ما قصر": 1.0,
    "ولا غلطة": 1.0,
    "ولا غلطه": 1.0,
    "ما فيه مثله": 1.0,
    "ما في مثله": 1.0,
    "ما في زيه": 1.0,
    "مش بطال": 0.5,
    "مو بطال": 0.5,
    "مو شين": 0.6,
    "مش شين": 0.6,
}

# Valence Intensifiers (multiply score)
INTENSIFIERS: Dict[str, float] = {
    "جدا": 1.4, "جداً": 1.4, "مرررة": 1.5, "مرة": 1.3, "مرره": 1.3,
    "كتير": 1.4, "كثير": 1.3, "وايد": 1.4, "بزاف": 1.4, "حيل": 1.4,
    "خالص": 1.3, "للغاية": 1.5, "بشدة": 1.4, "فوق الوصف": 1.6,
}

# Valence Diminishers (multiply score)
DIMINISHERS: Dict[str, float] = {
    "شوي": 0.6, "شوية": 0.6, "قليلا": 0.6, "قليلاً": 0.6, "نوعا ما": 0.7, "حبة": 0.7,
}

# Sentiment Emojis
POSITIVE_EMOJIS: Dict[str, float] = {
    "😍": 1.0, "🔥": 0.8, "🚀": 0.8, "❤️": 0.9, "👍": 0.8, "✅": 0.7,
    "👏": 0.8, "🌟": 0.8, "✨": 0.7, "🎉": 0.8, "💯": 0.9, "🥰": 0.9, "👌": 0.7,
}

NEGATIVE_EMOJIS: Dict[str, float] = {
    "😡": -1.0, "👎": -0.9, "💔": -0.8, "😤": -0.8, "🤮": -1.0,
    "🤬": -1.0, "❌": -0.7, "⚠️": -0.6, "🤦": -0.7, "😢": -0.7,
}


@dataclass
class SentimentResult:
    """Detailed sentiment analysis report for an Arabic text."""
    label: str                  # "إيجابي" (Positive), "سلبي" (Negative), "محايد" (Neutral), "مختلط" (Mixed)
    score: float                # Bounded in [-1.0, 1.0]
    confidence: float           # Estimated confidence in [0.0, 1.0]
    emotion: str                # Emotion category e.g. "حماس ورضا", "غضب واستياء", "محايد وموضوعي"
    positive_tokens: List[str] = field(default_factory=list)
    negative_tokens: List[str] = field(default_factory=list)
    negations_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _lookup_sentiment(word: str) -> Optional[Tuple[str, float]]:
    """Helper to lookup sentiment considering definite article 'ال' and feminine 'ة/ه'."""
    if word in POSITIVE_WORDS:
        return ("pos", POSITIVE_WORDS[word])
    if word in NEGATIVE_WORDS:
        return ("neg", NEGATIVE_WORDS[word])
    w_clean = word[2:] if word.startswith("ال") and len(word) > 3 else word
    if w_clean in POSITIVE_WORDS:
        return ("pos", POSITIVE_WORDS[w_clean])
    if w_clean in NEGATIVE_WORDS:
        return ("neg", NEGATIVE_WORDS[w_clean])
    if w_clean.endswith(("ة", "ه")) and len(w_clean) > 3:
        w_base = w_clean[:-1]
        if w_base in POSITIVE_WORDS:
            return ("pos", POSITIVE_WORDS[w_base])
        if w_base in NEGATIVE_WORDS:
            return ("neg", NEGATIVE_WORDS[w_base])
    return None


def analyze_sentiment(text: str) -> SentimentResult:
    """
    Performs fast, accurate rule-and-lexicon sentiment analysis on Arabic text.
    Handles dialectal slang, emojis, valence shifters, and contextual negations.
    """
    if not text or not text.strip():
        return SentimentResult(
            label="محايد",
            score=0.0,
            confidence=0.5,
            emotion="محايد وموضوعي",
        )

    # 1. Check for whole-phrase idiomatic expressions first
    for idiom, score in POSITIVE_IDIOMATIC_NEGATIONS.items():
        if idiom in text:
            return SentimentResult(
                label="إيجابي",
                score=score,
                confidence=0.95,
                emotion="ثناء ورضا",
                positive_tokens=[idiom],
                negative_tokens=[],
                negations_detected=[idiom],
            )

    # 2. Tokenize words while keeping punctuation and emojis
    words = re.findall(r"[\w\u0600-\u06FF]+|[^\s\w]", text, re.UNICODE)
    
    total_score = 0.0
    matched_pos: List[str] = []
    matched_neg: List[str] = []
    matched_negations: List[str] = []
    
    i = 0
    n = len(words)
    
    while i < n:
        w = words[i].strip()
        
        # Emoji check
        if w in POSITIVE_EMOJIS:
            total_score += POSITIVE_EMOJIS[w]
            matched_pos.append(w)
            i += 1
            continue
        if w in NEGATIVE_EMOJIS:
            total_score += NEGATIVE_EMOJIS[w]
            matched_neg.append(w)
            i += 1
            continue

        # Lookahead for Negation: e.g. "مو حلو", "مش بطيء", "ماكو مشاكل"
        if w in NEGATION_PARTICLES and i + 1 < n:
            next_w = words[i + 1].strip()
            lookup_next = _lookup_sentiment(next_w)
            if lookup_next is not None:
                pol, score_val = lookup_next
                if pol == "pos":
                    val = -0.85 * score_val
                    total_score += val
                    matched_neg.append(f"{w} {next_w}")
                    matched_negations.append(w)
                    i += 2
                    continue
                elif pol == "neg":
                    val = 0.75 * abs(score_val)
                    total_score += val
                    matched_pos.append(f"{w} {next_w}")
                    matched_negations.append(w)
                    i += 2
                    continue

        # Normal single word scoring
        current_multiplier = 1.0
        # Check if preceding word was an intensifier
        if i > 0 and words[i - 1].strip() in INTENSIFIERS:
            current_multiplier = INTENSIFIERS[words[i - 1].strip()]
        elif i > 0 and words[i - 1].strip() in DIMINISHERS:
            current_multiplier = DIMINISHERS[words[i - 1].strip()]

        lookup_curr = _lookup_sentiment(w)
        if lookup_curr is not None:
            pol, score_val = lookup_curr
            val = score_val * current_multiplier
            total_score += val
            if pol == "pos":
                matched_pos.append(w)
            else:
                matched_neg.append(w)

        i += 1

    # Normalize score between -1.0 and 1.0
    count_matches = len(matched_pos) + len(matched_neg)
    if count_matches == 0:
        return SentimentResult(
            label="محايد",
            score=0.0,
            confidence=0.6,
            emotion="محايد وموضوعي",
        )

    normalized_score = max(-1.0, min(1.0, total_score / max(1.0, count_matches * 0.9)))

    # Classification logic
    if len(matched_pos) > 0 and len(matched_neg) > 0 and abs(normalized_score) < 0.25:
        label = "مختلط"
        emotion = "آراء متباينة / مختلطة"
    elif normalized_score >= 0.20:
        label = "إيجابي"
        emotion = "حماس ورضا وثناء" if normalized_score >= 0.6 else "انطباع إيجابي"
    elif normalized_score <= -0.20:
        label = "سلبي"
        emotion = "غضب واستياء حاد" if normalized_score <= -0.6 else "انطباع سلبي ونقد"
    else:
        label = "محايد"
        emotion = "محايد وموضوعي"

    confidence = min(0.98, round(0.55 + (0.15 * count_matches), 2))

    return SentimentResult(
        label=label,
        score=round(normalized_score, 3),
        confidence=confidence,
        emotion=emotion,
        positive_tokens=matched_pos,
        negative_tokens=matched_neg,
        negations_detected=matched_negations,
    )
