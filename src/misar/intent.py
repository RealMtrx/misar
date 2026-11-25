# -*- coding: utf-8 -*-
"""
High-performance Arabic Intent Classification & Aspect-Based Sentiment Analysis (ABSA) for Misar (مِسار).
Features:
- Multi-dialect intent detection: COMPLAINT, INQUIRY, SUPPORT_REQUEST, PURCHASE_INTENT, CANCELLATION, PRAISE
- Granular Aspect-Based Sentiment Analysis (ABSA) for: Speed, Price, Support, Quality, Usability, Delivery
- Colloquial dialect support across Saudi/Gulf, Egyptian, Levantine, Maghrebi, and MSA
- Pure Python standard library (Zero mandatory external dependencies)
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

WORD_TOKEN_REGEX = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)

# Intent Keyword Dictionaries
INTENT_PATTERNS: Dict[str, Dict[str, Any]] = {
    "COMPLAINT": {
        "display_ar": "شكوى وبلاغ استياء",
        "keywords": {
            "اشتكي", "اشتكيت", "شكوى", "معطل", "خربان", "مشكلة", "مشاكل", "سيء", "سيئة",
            "زفت", "تالف", "ما يشتغل", "مو شغال", "مش شغال", "مافيه فايدة", "نصابين",
            "سرقة", "تأخير", "انقطاع", "مهزلة", "فاشل", "فاشلة", "ضيعتوني", "تعبت",
            "حسبي الله", "سوء معاملة", "مشتكي", "تلاعب", "خصم بدون وجه حق", "حرامية",
        },
        "weight": 1.25,
    },
    "INQUIRY": {
        "display_ar": "استفسار وسؤال عام",
        "keywords": {
            "كم", "كيف", "متى", "وين", "فين", "هل", "ممكن اعرف", "وش طريقة", "ايش طريقة",
            "شنو", "شو هو", "طريقة الاستخدام", "شروط", "أوقات العمل", "ساعات الدوام",
            "الرابط", "رابط التسجيل", "استفسار", "استفسر", "استفسارات", "معلومات", "تفاصيل",
        },
        "weight": 1.10,
    },
    "SUPPORT_REQUEST": {
        "display_ar": "طلب دعم فني ومساعدة",
        "keywords": {
            "مساعدة", "ساعدوني", "تكفون ساعدوني", "تكفون", "الدعم الفني", "خدمة العملاء",
            "ابي حل", "ابغى حل", "بدي حل", "عايز حل", "ارسلوا كود", "رمز التحقق", "كلمة المرور",
            "نسيت الرمز", "تحديث بيانات", "حل مشكلتي", "تفعيل الحساب", "توثيق الحساب", "تواصلوا معي",
        },
        "weight": 1.20,
    },
    "PURCHASE_INTENT": {
        "display_ar": "طلب شراء واشتراك جديد",
        "keywords": {
            "اشترك", "اشتراك", "اشترى", "شراء", "ابي اشتري", "ابغى اطلب", "كيف اطلب",
            "عايز اشتري", "بدي اطلب", "كود خصم", "عروض", "باقة", "باقات", "اسعار الباقات",
            "تجديد اشتراك", "طريقة الدفع", "فيزا", "مدى", "تقسيط", "تمارا", "تابي",
        },
        "weight": 1.15,
    },
    "CANCELLATION": {
        "display_ar": "طلب إلغاء واسترجاع أموال",
        "keywords": {
            "الغي", "الغاء", "إلغاء", "ابي الغي", "ابغى اكنسل", "كنسل", "استرجاع", "استرداد",
            "ردوا فلوسي", "رجعوا المبلغ", "استرجاع المبلغ", "الغاء الطلب", "فسخ العقد",
            "ما ابيه", "ما بدي اياه", "مش عايزه", "ايقاف الاشتراك", "حذف حسابي",
        },
        "weight": 1.30,
    },
    "PRAISE": {
        "display_ar": "مدح وشكر وثناء",
        "keywords": {
            "شكرا", "شكراً", "مشكورين", "ما قصرتوا", "ماقصرتوا", "افضل تطبيق", "احسن خدمة",
            "ممتاز", "ممتازة", "رائع", "رائعة", "مبدعين", "الله يسعدكم", "يعطيكم العافية",
            "كفو", "ولا غلطة", "بيض الله وجيهكم", "جزاكم الله خير", "خدمة سريعة", "تجاوب سريع",
        },
        "weight": 1.15,
    },
}

# Aspect-Based Sentiment Lexicon
ASPECT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "speed": {
        "display_ar": "السرعة والأداء",
        "triggers": {"سريع", "سريعة", "بطيء", "بطيئة", "يعلق", "تعليق", "فوري", "فورية", "تأخير", "متأخر", "سرعة", "بطء"},
        "positive": {"سريع", "سريعة", "فوري", "فورية", "ممتاز", "خفيف", "ثواني"},
        "negative": {"بطيء", "بطيئة", "يعلق", "تعليق", "تأخير", "متأخر", "ثقيل", "يطول"},
    },
    "price": {
        "display_ar": "السعر والتكلفة",
        "triggers": {"سعر", "اسعار", "أسعار", "غالي", "غالية", "رخيص", "رخيصة", "تكلفة", "مبالغ فيه", "خصم", "رسوم"},
        "positive": {"رخيص", "رخيصة", "مناسب", "معقول", "خصم ممتاز", "ببلاش", "مجاني", "عادل"},
        "negative": {"غالي", "غالية", "مبالغ فيه", "نار", "مرتفع", "رسوم إضافية", "استغلال"},
    },
    "support": {
        "display_ar": "الدعم وخدمة العملاء",
        "triggers": {"الدعم", "دعم", "خدمة العملاء", "الموظف", "الموظفة", "ردوا", "يردون", "تجاوب", "تواصل"},
        "positive": {"متجاوبين", "خدومين", "سريعين", "ذوق", "محترمين", "حلوا المشكلة", "ممتازين"},
        "negative": {"ما يردون", "تطنيش", "سيئين", "أسلوب زفت", "تصريف", "لا حياة لمن تنادي"},
    },
    "quality": {
        "display_ar": "الجودة والاعتمادية",
        "triggers": {"جودة", "خدمة", "الخدمة", "منتج", "المنتج", "شغال", "خربان", "دقة", "موثوق", "نظام"},
        "positive": {"عالية", "ممتازة", "أصلي", "موثوق", "جبار", "نظيف", "ولا غلطة", "توب"},
        "negative": {"رديئة", "خربان", "مغشوش", "مقلد", "تعبان", "صفر", "أفشل"},
    },
    "usability": {
        "display_ar": "سهولة التطبيق والواجهة",
        "triggers": {"تطبيق", "التطبيق", "واجهة", "الواجهة", "برنامج", "موقع", "الموقع", "استخدام"},
        "positive": {"سهل", "سلس", "واضح", "بسيط", "مرتب", "مريح"},
        "negative": {"معقد", "يلخبط", "صعب", "معيوب", "حوسة"},
    },
    "delivery": {
        "display_ar": "التوصيل والشحن",
        "triggers": {"توصيل", "التوصيل", "شحن", "الشحن", "مندوب", "المندوب", "طلب", "الطلب", "شحنة"},
        "positive": {"وصل بدري", "سريع", "بنفس اليوم", "محترم", "سليم"},
        "negative": {"متأخر", "ضاعت", "ما وصل", "تأخير", "مكسور"},
    },
}


@dataclass
class AspectSentiment:
    """Represents detected sentiment towards a specific aspect of a product or service."""
    aspect_key: str
    aspect_display_ar: str
    polarity: str           # "positive", "negative", "neutral"
    score: float            # -1.0 to +1.0
    evidence_snippet: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IntentResult:
    """Detailed report for Arabic intent classification and aspect sentiment analysis."""
    original_text: str
    primary_intent: str
    primary_intent_ar: str
    confidence: float
    intent_scores: Dict[str, float]
    aspects: List[AspectSentiment]
    overall_sentiment_polarity: str
    matched_intent_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "primary_intent": self.primary_intent,
            "primary_intent_ar": self.primary_intent_ar,
            "confidence": round(self.confidence, 2),
            "intent_scores": {k: round(v, 3) for k, v in self.intent_scores.items()},
            "aspects": [a.to_dict() for a in self.aspects],
            "overall_sentiment_polarity": self.overall_sentiment_polarity,
            "matched_intent_keywords": self.matched_intent_keywords,
        }


def classify_intent(text: str) -> IntentResult:
    """
    Classifies the user intent and extracts granular aspect-based sentiments from Arabic prose and social feedback.
    """
    t = "" if text is None else str(text).strip()
    if not t:
        return IntentResult(
            original_text=t,
            primary_intent="GENERAL_FEEDBACK",
            primary_intent_ar="رأي عام / غير محدد",
            confidence=0.50,
            intent_scores={"GENERAL_FEEDBACK": 1.0},
            aspects=[],
            overall_sentiment_polarity="neutral",
            matched_intent_keywords=[],
        )

    words = WORD_TOKEN_REGEX.findall(t)
    clean_text = " " + " ".join(words) + " "

    # 1. Score Intents
    intent_scores: Dict[str, float] = {}
    matched_keywords_all: List[str] = []

    for intent, cfg in INTENT_PATTERNS.items():
        score = 0.0
        kw_set = cfg["keywords"]
        weight = cfg["weight"]
        for kw in kw_set:
            if kw in clean_text or kw in t:
                score += (1.0 * weight)
                matched_keywords_all.append(kw)
        intent_scores[intent] = score

    # Check for question marks (Inquiry boost)
    if any(q in t for q in ("?", "؟")):
        intent_scores["INQUIRY"] = intent_scores.get("INQUIRY", 0.0) + 1.2

    # Find dominant intent
    best_intent = "GENERAL_FEEDBACK"
    best_score = 0.0
    for intent, sc in intent_scores.items():
        if sc > best_score:
            best_score = sc
            best_intent = intent

    if best_score == 0.0:
        best_intent = "GENERAL_FEEDBACK"
        confidence = 0.60
        display_ar = "رأي عام / تعليق عادي"
    else:
        confidence = min(0.98, max(0.65, 0.60 + (best_score * 0.12)))
        display_ar = INTENT_PATTERNS[best_intent]["display_ar"]

    # 2. Extract Aspect Sentiments (ABSA)
    detected_aspects: List[AspectSentiment] = []

    for aspect_key, cfg in ASPECT_TAXONOMY.items():
        triggers = cfg["triggers"]
        # Check if aspect is mentioned
        has_trigger = any(trig in t for trig in triggers)
        if not has_trigger:
            continue

        # Evaluate aspect polarity in the text
        pos_hits = sum(1 for p in cfg["positive"] if p in t)
        neg_hits = sum(1 for n in cfg["negative"] if n in t)

        # Inversion checks (e.g. مو سريع, مش بطيء)
        if "مو سريع" in t or "مش سريع" in t or "غير سريع" in t:
            neg_hits += 1
            pos_hits = max(0, pos_hits - 1)
        if "مو بطيء" in t or "مش بطيء" in t:
            pos_hits += 1
            neg_hits = max(0, neg_hits - 1)

        if pos_hits > neg_hits:
            polarity = "positive"
            score = 0.75
        elif neg_hits > pos_hits:
            polarity = "negative"
            score = -0.75
        else:
            polarity = "neutral"
            score = 0.0

        # Find evidence snippet around trigger
        matched_trigger = next((trig for trig in triggers if trig in t), "")
        snippet = t
        if matched_trigger:
            idx = t.find(matched_trigger)
            start = max(0, idx - 20)
            end = min(len(t), idx + len(matched_trigger) + 25)
            snippet = t[start:end].strip()

        detected_aspects.append(AspectSentiment(
            aspect_key=aspect_key,
            aspect_display_ar=cfg["display_ar"],
            polarity=polarity,
            score=score,
            evidence_snippet=snippet,
        ))

    # 3. Overall Polarity
    if best_intent in ("COMPLAINT", "CANCELLATION"):
        overall_pol = "negative"
    elif best_intent == "PRAISE":
        overall_pol = "positive"
    else:
        pos_aspects = sum(1 for a in detected_aspects if a.polarity == "positive")
        neg_aspects = sum(1 for a in detected_aspects if a.polarity == "negative")
        if pos_aspects > neg_aspects:
            overall_pol = "positive"
        elif neg_aspects > pos_aspects:
            overall_pol = "negative"
        else:
            overall_pol = "neutral"

    return IntentResult(
        original_text=t,
        primary_intent=best_intent,
        primary_intent_ar=display_ar,
        confidence=confidence,
        intent_scores=intent_scores,
        aspects=detected_aspects,
        overall_sentiment_polarity=overall_pol,
        matched_intent_keywords=matched_keywords_all,
    )
