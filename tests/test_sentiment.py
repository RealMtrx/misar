# -*- coding: utf-8 -*-
"""
Unit tests for Misar (مِسار) Arabic Sentiment & Emotion Analyzer.
"""

import unittest
from misar.sentiment import SentimentResult, analyze_sentiment


class TestArabicSentiment(unittest.TestCase):
    def test_positive_sentiment(self):
        res = analyze_sentiment("الخدمة ممتازة وسريعة جداً وبيض الله وجهكم 😍🔥")
        self.assertEqual(res.label, "إيجابي")
        self.assertGreater(res.score, 0.4)
        self.assertGreater(len(res.positive_tokens), 0)
        self.assertIn("حماس", res.emotion)

    def test_negative_sentiment(self):
        res = analyze_sentiment("التطبيق فاشل وسيئ وخايس مرررة 😡👎")
        self.assertEqual(res.label, "سلبي")
        self.assertLess(res.score, -0.4)
        self.assertGreater(len(res.negative_tokens), 0)

    def test_dialect_negation_inversion(self):
        # "مو حلو" should be detected as negative (inverting "حلو")
        res = analyze_sentiment("أداء الموقع مو حلو أبداً ويعلق")
        self.assertEqual(res.label, "سلبي")
        self.assertLess(res.score, 0.0)
        self.assertTrue(any("مو" in n for n in res.negations_detected))

    def test_positive_idiomatic_negation(self):
        # "ما قصرت" / "ما قصرتوا" is strong praise, not negative
        res = analyze_sentiment("ما قصرتوا بيض الله وجهكم على هذا العمل الرائع")
        self.assertEqual(res.label, "إيجابي")
        self.assertGreaterEqual(res.score, 0.8)

    def test_mild_positive_negation(self):
        # "مش بطال" is mild positive/relief
        res = analyze_sentiment("التطبيق مش بطال وشغال كويس")
        self.assertEqual(res.label, "إيجابي")
        self.assertGreater(res.score, 0.0)

    def test_neutral_and_empty_text(self):
        res_empty = analyze_sentiment("")
        self.assertEqual(res_empty.label, "محايد")
        self.assertEqual(res_empty.score, 0.0)

        res_neutral = analyze_sentiment("تم انعقاد الاجتماع الدوري في مقر المؤسسة اليوم")
        self.assertEqual(res_neutral.label, "محايد")

    def test_to_dict_structure(self):
        res = analyze_sentiment("شكراً جزيلاً لكم")
        d = res.to_dict()
        self.assertIn("label", d)
        self.assertIn("score", d)
        self.assertIn("confidence", d)
        self.assertIn("emotion", d)
        self.assertIn("positive_tokens", d)
        self.assertIn("negative_tokens", d)
        self.assertIn("negations_detected", d)


if __name__ == "__main__":
    unittest.main()
