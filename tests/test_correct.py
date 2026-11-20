# -*- coding: utf-8 -*-
"""
Unit tests for Misar (مِسار) Arabic Grammar & Spell Correction Engine.
"""

import unittest
from misar.correct import CorrectionResult, correct_arabic_text


class TestArabicCorrection(unittest.TestCase):
    def test_hamza_corrections(self):
        # مسئول -> مسؤول, شئ -> شيء, دفئ -> دفء
        res = correct_arabic_text("كل مسئول عن شئ معين في هذا العمل")
        self.assertIn("مسؤول", res.corrected_text)
        self.assertIn("شيء", res.corrected_text)
        self.assertGreater(res.changes_count, 0)

    def test_alef_wasla_corrections(self):
        # أبتكار -> ابتكار, إستخدام -> استخدام
        res = correct_arabic_text("قامت المنظمة في أبتكار أساليب جديدة عبر إستخدام النماذج")
        self.assertIn("ابتكار", res.corrected_text)
        self.assertIn("استخدام", res.corrected_text)

    def test_composite_phrases(self):
        res = correct_arabic_text("سنبدأ العمل إنشاء الله قريباً")
        self.assertIn("إن شاء الله", res.corrected_text)

    def test_taa_marbuta_disambiguation(self):
        # منظمه -> منظمة, but preserving pronoun 'عمله' / 'مديره'
        res = correct_arabic_text("قامت منظمه جديده بدعم مديره في عمله")
        self.assertIn("منظمة", res.corrected_text)
        self.assertIn("جديدة", res.corrected_text)
        self.assertIn("مديره", res.corrected_text)  # protected
        self.assertIn("عمله", res.corrected_text)   # protected

    def test_clean_input(self):
        clean_text = "ابتكار النماذج اللغوية في اللغة العربية"
        res = correct_arabic_text(clean_text)
        self.assertEqual(res.corrected_text, clean_text)
        self.assertEqual(res.changes_count, 0)
        self.assertEqual(res.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
