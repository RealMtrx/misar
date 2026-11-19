# -*- coding: utf-8 -*-
"""
Tests for Misar WordPiece Tokenizer & Authentic AraBERTv2 / CAMeLBERT Vocabularies.
"""

import unittest

from misar import MisarRouter, WordPieceTokenizer
from misar.tokenizer import (
    PRETRAINED_MODEL_CONFIGS,
    PRETRAINED_VOCAB_URLS,
    load_default_tokenizer,
)


class TestRealAraBERTv2WordPiece(unittest.TestCase):
    """Empirical proof against the authentic 64,000-token AraBERTv2 vocabulary."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.tok = WordPieceTokenizer.from_pretrained("arabertv02")
            cls.router = MisarRouter(tokenizer=cls.tok)
            cls.legacy_router = MisarRouter(tokenizer=cls.tok, preserve_orthography=False)
            cls.available = True
        except Exception:
            cls.available = False

    def test_real_arabert_vocab_orthography_beats_legacy(self):
        if not self.available:
            self.skipTest("AraBERTv2 vocab unavailable offline")
        msa = "تعتمد الحوسبة السحابية الحديثة على هندسة النماذج اللغوية الضخمة لتحسين كفاءة المعالجة وتقليل زمن الاستدلال"
        res_misar = self.router.route(msa)
        res_legacy = self.legacy_router.route(msa)
        self.assertTrue(res_misar.is_real_tokenizer)
        # Misar keeps ة intact -> strictly fewer subword tokens than destructive legacy ة->ه
        self.assertLess(res_misar.processed_token_count, res_legacy.processed_token_count)
        self.assertAlmostEqual(res_misar.tfr_processed, 1.0, delta=0.15)

    def test_real_arabert_vocab_reduces_noisy_and_diacritized_tokens(self):
        if not self.available:
            self.skipTest("AraBERTv2 vocab unavailable offline")
        noisy = "السيرفررررر بطييييء مررررة ومافيه استجابة في API أبدددداً 😡😡👎 #هندسة_البرمجيات"
        res_noisy = self.router.route(noisy)
        self.assertLess(res_noisy.processed_token_count, res_noisy.raw_token_count)
        self.assertGreater(res_noisy.token_reduction_pct, 40.0)

        classical = "وَضَعَ الْخَوَارِزْمِيُّ أُسُسَ الْجَبْرِ وَالْحِسَابِ لِيَبْنِيَ مِنْهَا الْعُلَمَاءُ مَنَاهِجَ الْبَرْمَجَةِ الْحَدِيثَةِ"
        res_class = self.router.route(classical)
        # Without Misar, AraBERTv2 collapses diacritized words into [UNK] (OOV collapse).
        # Misar recovers 100% of words to valid AraBERTv2 vocabulary tokens (0 [UNK]).
        self.assertGreaterEqual(res_class.unk_count_raw, 8)
        self.assertEqual(res_class.unk_count_processed, 0)
        self.assertLess(res_class.processed_token_count, res_class.raw_token_count)

    def test_real_arabert_splits_tatweel_overhead(self):
        if not self.available:
            self.skipTest("AraBERTv2 vocab unavailable offline")
        stretched = "الســــلام علــــيكم"
        res = self.router.route(stretched)
        self.assertEqual(res.route_name, "NOISY_SOCIAL")
        self.assertEqual(res.processed_text, "السلام عليكم")
        self.assertIn("السلام", res.raw_tokens)
        self.assertIn("عليكم", res.raw_tokens)
        self.assertGreater(res.token_reduction_pct, 50.0)
        self.assertEqual(res.unk_count_processed, 0)

    def test_multi_model_registry_and_do_lower_case_parity(self):
        for key in ("arabertv02", "camelbert-mix", "marbertv2", "arbertv2", "qarib"):
            self.assertIn(key, PRETRAINED_VOCAB_URLS)
            self.assertIn(key, PRETRAINED_MODEL_CONFIGS)
        self.assertTrue(PRETRAINED_MODEL_CONFIGS["marbertv2"]["do_lower_case"])
        self.assertTrue(PRETRAINED_MODEL_CONFIGS["arbertv2"]["do_lower_case"])
        self.assertFalse(PRETRAINED_MODEL_CONFIGS["arabertv02"]["do_lower_case"])

        mini_vocab = {"[UNK]": 0, "ap": 1, "##i": 2, "البرمجة": 3}
        tok_lower = WordPieceTokenizer(vocab=mini_vocab, do_lower_case=True)
        self.assertEqual(tok_lower.tokenize("API البرمجة"), ["ap", "##i", "البرمجة"])

        tok_cased = WordPieceTokenizer(vocab=mini_vocab, do_lower_case=False)
        self.assertEqual(tok_cased.tokenize("API"), ["[UNK]"])

        tok_marbert = load_default_tokenizer("marbertv2", allow_download=False)
        if tok_marbert.is_real_vocab:
            self.assertNotIn("[UNK]", tok_marbert.tokenize("API Backend"))

    def test_real_vocab_auto_heal_repairs_broken_haa_without_corrupting_pronouns(self):
        if not self.available:
            self.skipTest("AraBERTv2 vocab unavailable offline")
        text = "ياخي البرمجه في المنصه هاذي مهمه بس بيته وملفه ومديره تمام"
        res = self.router.route(text)
        self.assertEqual(res.route_name, "DIALECT_PRESERVE")
        self.assertIn("البرمجة", res.processed_text)
        self.assertIn("المنصة", res.processed_text)
        self.assertIn("مهمة", res.processed_text)
        self.assertIn("هذه", res.processed_text)
        self.assertIn("بيته", res.processed_text)
        self.assertIn("ملفه", res.processed_text)
        self.assertIn("مديره", res.processed_text)
        self.assertGreaterEqual(res.auto_healed_count, 3)
        self.assertLess(res.processed_token_count, res.raw_token_count)
        self.assertEqual(res.unk_count_processed, 0)

    def test_real_arabert_recovers_presentation_forms_and_farsi_zero_unk(self):
        if not self.available:
            self.skipTest("AraBERTv2 vocab unavailable offline")
        raw = "ﺍﻟﺒﺮﻣﺠﺔ ﻻ تقبل کود سرييييیع جــــديدة"
        res = self.router.route(raw)
        self.assertGreaterEqual(res.unk_count_raw, 4)
        self.assertEqual(res.unk_count_processed, 0)
        self.assertEqual(res.processed_text, "البرمجة لا تقبل كود سريع جديدة")
        self.assertGreater(res.attention_savings_pct, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
