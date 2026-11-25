# -*- coding: utf-8 -*-
"""
Tests for Misar Regional Sub-Dialect & Register Classifier and Prompt Caching.
"""

import unittest

import misar
from misar import MisarRouter
from misar.classifier import canonicalize, classify_dialect


class TestArabicDialectAndRegisterClassifier(unittest.TestCase):
    """Exhaustive tests for Arabic regional sub-dialect and register classifier."""

    def test_saudi_gulf_subdialect_detection(self) -> None:
        text = "وش تبغى تسوي الحين يا خوي؟ تبي نروح سوا؟"
        res = classify_dialect(text)
        self.assertEqual(res.primary_dialect, "saudi_gulf")
        self.assertGreater(res.confidence, 0.8)
        self.assertIn("saudi_gulf:وش", res.detected_markers)

    def test_levantine_subdialect_detection(self) -> None:
        text = "شو بدك تعمل هلق يا زلمة؟ هيك كتير منيح!"
        res = classify_dialect(text)
        self.assertEqual(res.primary_dialect, "levantine")
        self.assertGreater(res.confidence, 0.8)
        self.assertIn("levantine:شو", res.detected_markers)

    def test_egyptian_subdialect_detection(self) -> None:
        text = "عايز ايه دلوقتي يا باشا؟ كده كويس اوي والله"
        res = classify_dialect(text)
        self.assertEqual(res.primary_dialect, "egyptian")
        self.assertGreater(res.confidence, 0.8)
        self.assertIn("egyptian:دلوقتي", res.detected_markers)

    def test_maghrebi_subdialect_detection(self) -> None:
        text = "هاد الدار ديالنا وزوينة بزاف دابا نمشيو"
        res = classify_dialect(text)
        self.assertEqual(res.primary_dialect, "maghrebi")
        self.assertGreater(res.confidence, 0.7)
        self.assertIn("maghrebi:بزاف", res.detected_markers)

    def test_msa_formal_classification(self) -> None:
        text = "عقد مجلس الوزراء جلسته الأسبوعية برئاسة خادم الحرمين الشريفين لمناقشة الميزانية العامة."
        res = classify_dialect(text)
        self.assertEqual(res.primary_dialect, "msa")
        self.assertEqual(res.register, "formal_official")

    def test_register_discourse_categories(self) -> None:
        # 1. Classical Heritage
        heritage = "رَوَى الإِمَامُ البُخَارِيُّ فِي صَحِيحِهِ عَنْ أَمِيرِ المُؤْمِنِينَ"
        res_h = classify_dialect(heritage)
        self.assertEqual(res_h.register, "classical_heritage")

        # 2. Tech Dev
        tech = "قمنا بإعداد خادم REST API وقواعد البيانات Docker Cloud لربط التطبيق"
        res_t = classify_dialect(tech)
        self.assertEqual(res_t.register, "tech_dev")

        # 3. Social Slang
        slang = "وش مسوي اليوم يا وحش 🔥😂 ارحب مليون"
        res_s = classify_dialect(slang)
        self.assertEqual(res_s.register, "social_slang")

    def test_canonicalize_interrogatives_prompt_cache(self) -> None:
        # Gulf
        self.assertEqual(canonicalize("وش تبغى تسوي الحين؟"), "ما تبغى تسوي الآن؟")
        # Egyptian
        self.assertEqual(canonicalize("عايز ايه دلوقتي؟"), "أريد ما الآن؟")
        # Levantine
        self.assertEqual(canonicalize("شو بدك تسوي هلق؟"), "ما بدك تسوي الآن؟")
        # Maghrebi
        self.assertEqual(canonicalize("واش دابا نمشيو بزاف؟"), "ما الآن نمشيو جداً؟")

    def test_top_level_package_and_router_integration(self) -> None:
        router = MisarRouter()
        clf = router.classify("وش تبغى الحين؟")
        self.assertEqual(clf.primary_dialect, "saudi_gulf")

        canon = router.canonicalize("عايز ايه دلوقتي؟")
        self.assertEqual(canon, "أريد ما الآن؟")

        self.assertEqual(misar.__version__, "3.1.0")
        self.assertTrue(callable(misar.classify))
        self.assertTrue(callable(misar.canonicalize))
        self.assertTrue(callable(misar.route_parallel))
        self.assertTrue(callable(misar.process_dataset_file))


if __name__ == "__main__":
    unittest.main(verbosity=2)
