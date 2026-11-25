# -*- coding: utf-8 -*-
"""
Unit tests for Misar Arabic Intent Classification & Aspect-Based Sentiment Analysis (ABSA).
"""

import unittest

import misar
from misar import MisarRouter
from misar.intent import AspectSentiment, IntentResult, classify_intent


class TestArabicCustomerIntentAndABSA(unittest.TestCase):
    """Exhaustive tests for customer intent detection and granular aspect-based sentiment analysis."""

    def test_complaint_intent_detection(self) -> None:
        text = "التطبيق معطل وخربان وما يشتغل عندي من الصبح وسيئ جدا!"
        res = classify_intent(text)
        self.assertIsInstance(res, IntentResult)
        self.assertEqual(res.primary_intent, "COMPLAINT")
        self.assertGreater(res.confidence, 0.7)
        self.assertGreater(len(res.matched_intent_keywords), 0)

    def test_inquiry_intent_detection(self) -> None:
        text = "كم سعر الاشتراك ومتى تبدأ أوقات العمل في فرعكم؟"
        res = classify_intent(text)
        self.assertEqual(res.primary_intent, "INQUIRY")
        self.assertGreater(res.confidence, 0.6)

    def test_support_request_detection(self) -> None:
        text = "تكفون ساعدوني نسيت كلمة المرور والدعم الفني ما ردوا علي"
        res = classify_intent(text)
        self.assertIn(res.primary_intent, {"SUPPORT_REQUEST", "COMPLAINT"})
        self.assertTrue(any(kw in res.matched_intent_keywords for kw in ["ساعدوني", "الدعم الفني", "كلمة المرور", "تكفون"]))

    def test_purchase_intent_detection(self) -> None:
        text = "ابي اشترك في باقة المحترفين هل فيه كود خصم متاح للدفع بتمارا؟"
        res = classify_intent(text)
        self.assertEqual(res.primary_intent, "PURCHASE_INTENT")

    def test_cancellation_intent_detection(self) -> None:
        text = "ابغى الغي اشتراكي وارجعوا المبلغ والفلوس لحسابي فورا"
        res = classify_intent(text)
        self.assertEqual(res.primary_intent, "CANCELLATION")

    def test_praise_intent_detection(self) -> None:
        text = "ما قصرتوا وبيض الله وجيهكم افضل خدمة وتجاوب سريع ورائع شكرا لكم"
        res = classify_intent(text)
        self.assertEqual(res.primary_intent, "PRAISE")

    def test_aspect_based_sentiment_multiaspect(self) -> None:
        text = "التطبيق سريع جدا وممتاز بس السعر غالي والدعم الفني بطيء في الرد"
        res = classify_intent(text)
        self.assertGreaterEqual(len(res.aspects), 2)
        aspect_map = {a.aspect_key: a for a in res.aspects}
        if "speed" in aspect_map:
            self.assertEqual(aspect_map["speed"].polarity, "positive")
        if "price" in aspect_map:
            self.assertEqual(aspect_map["price"].polarity, "negative")

    def test_router_integration(self) -> None:
        router = MisarRouter()
        res = router.classify_intent("خدمة رائعة وممتازة شكرا لكم")
        self.assertIsInstance(res, IntentResult)
        self.assertEqual(res.primary_intent, "PRAISE")

    def test_top_level_exports(self) -> None:
        res = misar.classify_intent("اريد الغاء الحساب")
        self.assertIsInstance(res, IntentResult)
        d = res.to_dict()
        self.assertIn("primary_intent", d)
        self.assertIn("aspects", d)


if __name__ == "__main__":
    unittest.main()
