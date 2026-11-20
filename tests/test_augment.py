# -*- coding: utf-8 -*-
"""
Tests for Misar Synthetic Data Augmentation & Stress Testing Engine.
"""

import unittest

from misar import (
    evaluate_augmentation_recovery,
    generate_stress_suite,
    inject_arabizi_noise,
    inject_diacritics_noise,
    inject_orthographic_noise,
    inject_tatweel_noise,
)


class TestDataAugmentation(unittest.TestCase):
    """Exhaustive tests for synthetic perturbations and model stress-testing."""

    def test_orthographic_noise_injection(self) -> None:
        """Verify realistic human typos (hamza, taa marbuta/haa, etc.)."""
        text = "المملكة العربية السعودية أرسلت خطاباً رسمياً"
        noisy = inject_orthographic_noise(text, typo_rate=0.5, seed=123)
        self.assertNotEqual(text, noisy)
        self.assertTrue(len(noisy) > 0)

    def test_tatweel_noise_injection(self) -> None:
        """Verify kashida elongation insertion on connectable letters."""
        text = "السلام عليكم ورحمة الله"
        stretched = inject_tatweel_noise(text, rate=0.8, max_stretch=4, seed=456)
        self.assertIn("ـ", stretched)
        self.assertGreater(len(stretched), len(text))

    def test_diacritics_noise_injection(self) -> None:
        """Verify random tashkeel (harakat) injection."""
        text = "كتب الطالب الدرس في الفصل"
        diacritized = inject_diacritics_noise(text, rate=0.8, seed=789)
        has_haraka = any("\u064B" <= ch <= "\u0652" or ch == "\u0651" for ch in diacritized)
        self.assertTrue(has_haraka)

    def test_arabizi_noise_injection(self) -> None:
        """Verify youth code-switching substitution."""
        text = "مرحبا يا حبيبي كيفك اليوم وخلاص تمام"
        noisy = inject_arabizi_noise(text, replacement_rate=0.9, seed=101)
        self.assertTrue(any(w in noisy for w in ("mar7aba", "7abibi", "keefak", "5alas", "tamam")))

    def test_generate_stress_suite(self) -> None:
        """Verify generation of all 6 standard perturbation variants."""
        suite = generate_stress_suite("مرحبا بك في اللقاء العلمي")
        expected_keys = {
            "original_clean",
            "orthographic_typos",
            "tatweel_elongation",
            "heavy_diacritics",
            "arabizi_code_switching",
            "adversarial_composite",
        }
        self.assertEqual(set(suite.keys()), expected_keys)
        for val in suite.values():
            self.assertTrue(len(val) > 0)

    def test_evaluate_augmentation_recovery(self) -> None:
        """Verify Misar's token reduction on synthetic noisy variants."""
        res = evaluate_augmentation_recovery("الــــســـــلام علـــيكم يا حبيبي كيفك 7abibi")
        self.assertIn("tatweel_elongation", res)
        self.assertIn("adversarial_composite", res)
        for variant, stats in res.items():
            self.assertIn("raw_token_count", stats)
            self.assertIn("cleaned_token_count", stats)
            self.assertIn("tokens_saved", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
