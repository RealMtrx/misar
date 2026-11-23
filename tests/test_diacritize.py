# -*- coding: utf-8 -*-
"""
Unit tests for Misar (مِسار) Arabic Diacritization & Tashkeel Recovery Engine.
"""

import unittest
from misar.diacritize import (
    ALL_DIACRITICS,
    DiacritizationResult,
    diacritize_text,
    diacritize_word,
)
from misar.router import MisarRouter, diacritize_text as router_diacritize_text


class TestArabicDiacritization(unittest.TestCase):
    def test_canonical_particles(self):
        text = "في ومن وإلى وعلى وعن ومع وحتى ثم ولكن"
        res = diacritize_text(text)
        self.assertIn("فِي", res.diacritized_text)
        self.assertIn("مِنْ", res.diacritized_text)
        self.assertIn("إِلَى", res.diacritized_text)
        self.assertIn("عَلَى", res.diacritized_text)
        self.assertIn("عَنْ", res.diacritized_text)
        self.assertIn("مَعَ", res.diacritized_text)
        self.assertIn("حَتَّى", res.diacritized_text)
        self.assertIn("ثُمَّ", res.diacritized_text)
        self.assertIn("لَكِنَّ", res.diacritized_text)
        self.assertGreater(res.vowels_added_count, 10)

    def test_morphological_patterns(self):
        # تفعيل
        res_tatweer = diacritize_word("تطوير")
        self.assertEqual(res_tatweer, "تَطْوِير")
        res_tahdeeth = diacritize_word("تحديث")
        self.assertEqual(res_tahdeeth, "تَحْدِيث")

        # استفعال
        res_istikhdam = diacritize_word("استخدام")
        self.assertEqual(res_istikhdam, "اسْتِخْدَام")

        # فاعل & مفعول
        res_katib = diacritize_word("كاتب")
        self.assertEqual(res_katib, "كَاتِب")
        res_maktoob = diacritize_word("مكتوب")
        self.assertEqual(res_maktoob, "مَكْتُوب")

    def test_clitics_handling(self):
        res = diacritize_text("وفي المملكة")
        self.assertTrue(any(c in ALL_DIACRITICS for c in res.diacritized_text))
        self.assertIn("وَفِي", res.diacritized_text)

    def test_preserves_existing_diacritics(self):
        pre_diacritized = "وَضَعَ الْخَوَارِزْمِيُّ أُسُسَ الْجَبْرِ"
        res = diacritize_text(pre_diacritized)
        self.assertIn("الْخَوَارِزْمِيُّ", res.diacritized_text)
        self.assertIn("أُسُسَ", res.diacritized_text)

    def test_preserves_latin_and_emojis(self):
        mixed = "تطوير واجهات API الحديثة مع Python 3.12 🚀"
        res = diacritize_text(mixed)
        self.assertIn("API", res.diacritized_text)
        self.assertIn("Python 3.12", res.diacritized_text)
        self.assertIn("🚀", res.diacritized_text)
        self.assertIn("تَطْوِير", res.diacritized_text)

    def test_router_integration(self):
        router = MisarRouter()
        res = router.diacritize("تطوير المنظومة في المملكة")
        self.assertIsInstance(res, DiacritizationResult)
        self.assertIn("تَطْوِير", res.diacritized_text)

        # Unified process with diacritize=True
        proc = router.process("تطوير المنظومه في المملكه", correct=True, diacritize=True)
        self.assertIn("تَطْوِير", proc)
        self.assertTrue(any(c in ALL_DIACRITICS for c in proc))

    def test_empty_and_whitespace(self):
        res = diacritize_text("")
        self.assertEqual(res.diacritized_text, "")
        self.assertEqual(res.vowels_added_count, 0)
        res_spaces = diacritize_text("   ")
        self.assertEqual(res_spaces.diacritized_text, "   ")


if __name__ == "__main__":
    unittest.main()
