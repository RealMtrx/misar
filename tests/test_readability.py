# -*- coding: utf-8 -*-
"""
Unit tests for Misar (مِسار) Arabic Readability & Osman Metric Engine.
"""

import unittest
from misar.readability import ReadabilityResult, analyze_readability
from misar.router import MisarRouter


SIMPLE_STORY = (
    "ذهب أحمد إلى المدرسة في الصباح الباكر وهو يشعر بالسعادة والنشاط. "
    "التقى أحمد بمعلمه وزملائه في ساحة المدرسة الجميلة ولعب معهم بالكرة."
)

ACADEMIC_PROSE = (
    "تقتضي معالجة اللغات الطبيعية استيعاب المحددات الصرفية والبنى النحوية التوليدية "
    "لضمان سلامة النمذجة الرياضية للمتجهات الدلالية في الفضاءات الهندسية متعددة الأبعاد "
    "وتحقيق الاستمثال المنطقي للشبكات العصبية العميقة عبر خوارزميات الانحدار التدريجي."
)


class TestArabicReadability(unittest.TestCase):
    def test_readability_metrics_sanity(self):
        res = analyze_readability(SIMPLE_STORY)
        self.assertIsInstance(res, ReadabilityResult)
        self.assertGreater(res.word_count, 10)
        self.assertEqual(res.sentence_count, 2)
        self.assertGreater(res.avg_word_length, 2.0)
        self.assertGreater(res.estimated_syllables_per_word, 1.0)
        self.assertTrue(0.0 <= res.type_token_ratio <= 1.0)
        self.assertTrue(0.0 <= res.hapax_legomena_ratio <= 1.0)
        self.assertGreater(res.reading_time_seconds, 0.0)

    def test_comparative_complexity(self):
        simple_res = analyze_readability(SIMPLE_STORY)
        academic_res = analyze_readability(ACADEMIC_PROSE)

        # Longer words and longer sentences in academic text lead to lower Osman index (harder)
        self.assertGreater(academic_res.avg_word_length, simple_res.avg_word_length)
        self.assertLess(academic_res.osman_index, simple_res.osman_index)
        self.assertIn(academic_res.educational_stage, {"specialized_academic", "university", "secondary"})

    def test_empty_and_whitespace(self):
        res = analyze_readability("")
        self.assertEqual(res.word_count, 0)
        self.assertEqual(res.sentence_count, 0)
        self.assertEqual(res.osman_index, 100.0)
        self.assertEqual(res.reading_time_seconds, 0.0)

        res_spaces = analyze_readability("    ")
        self.assertEqual(res_spaces.word_count, 0)

    def test_router_integration(self):
        router = MisarRouter()
        res = router.readability(SIMPLE_STORY)
        self.assertIsInstance(res, ReadabilityResult)
        self.assertGreater(res.word_count, 0)
        self.assertIn("osman_index", res.to_dict())


if __name__ == "__main__":
    unittest.main()
