# -*- coding: utf-8 -*-
"""
Comprehensive tests for Misar (مِسار) Unified Core Engine & LRU Cache Stabilization.
Verifies:
- O(1) in-memory LRU cache hit/miss semantics and microsecond speeds
- Unified process() pipeline orchestrating correction, NER, PII masking, and dialect translation
- Extreme edge-case safety (None, empty, numbers, foreign scripts, BiDi overrides, giant texts)
- Zero-crash robustness guarantees
"""

import unittest
from misar import (
    MisarRouter,
    UnifiedProcessResult,
    clear_cache,
    get_cache_stats,
    process,
    route,
)


class TestUnifiedCoreEngine(unittest.TestCase):
    def setUp(self):
        clear_cache()
        self.router = MisarRouter(cache_size=64)

    def tearDown(self):
        clear_cache()

    def test_lru_cache_hit_and_speed(self):
        text = "هذا نص فصيح لاختبار التخزين المؤقت فائق السرعة"
        # First call: cache miss
        res1 = self.router.route(text)
        info1 = self.router.cache_info()
        self.assertEqual(info1["misses"], 1)
        self.assertEqual(info1["hits"], 0)
        self.assertEqual(info1["size"], 1)

        # Second call: cache hit (< 0.01 ms)
        res2 = self.router.route(text)
        info2 = self.router.cache_info()
        self.assertEqual(info2["hits"], 1)
        self.assertEqual(res2.processed_text, res1.processed_text)
        self.assertLessEqual(res2.latency_ms, 0.05)

        # Clear cache verification
        self.router.clear_cache()
        info3 = self.router.cache_info()
        self.assertEqual(info3["size"], 0)
        self.assertEqual(info3["hits"], 0)

    def test_unified_process_basic(self):
        # Basic mode returns a clean string by default
        text = "  توثيق البرمجه   في المنصة دقيق وموثوق   "
        out = self.router.process(text, correct=True)
        self.assertIsInstance(out, str)
        self.assertTrue(len(out) > 0)
        self.assertIn("البرمجة", out)

    def test_unified_process_telemetry(self):
        text = "توثيق البرمجه في المنصة دقيق وموثوق"
        res = self.router.process(text, return_telemetry=True)
        self.assertIsInstance(res, UnifiedProcessResult)
        self.assertEqual(res.route_name, "FAST_MSA")
        self.assertGreater(res.tokens_raw, 0)
        self.assertGreater(res.tokens_processed, 0)
        self.assertIsInstance(res.to_dict(), dict)

    def test_unified_process_composite_flags(self):
        # Text with spelling mistakes, dialect, NER entity, and Saudi ID PII
        raw = "كل مسئول في سدايا وش يبغى يسوي بهويتي 1000000008؟"
        res = self.router.process(
            raw,
            correct=True,
            translate_msa=True,
            mask_ner=True,
            mask_pii=True,
            return_telemetry=True,
        )
        self.assertIsInstance(res, UnifiedProcessResult)
        # 1. Spelling corrected (مسئول -> مسؤول)
        self.assertIn("مسؤول", res.processed_text)
        # 2. Dialect translated (وش -> ماذا, يبغى -> يريد)
        self.assertTrue("ماذا" in res.processed_text or "يريد" in res.processed_text)
        # 3. NER masked (سدايا -> [ORG_محجوب])
        self.assertIn("[ORG_محجوب]", res.processed_text)
        # 4. PII masked (1000000008 -> [رقم_هوية_محجوب])
        self.assertIn("[رقم_هوية_محجوب]", res.processed_text)
        self.assertNotIn("1000000008", res.processed_text)

    def test_zero_crash_edge_cases(self):
        # None input
        res_none = self.router.process(None, return_telemetry=True)
        self.assertEqual(res_none.processed_text, "")
        self.assertEqual(res_none.tokens_raw, 0)

        # Empty and whitespace
        self.assertEqual(self.router.process(""), "")
        self.assertEqual(self.router.process("   \n\t  "), "")

        # Pure English / Latin
        en_out = self.router.process("Transformer Attention FLOPs Optimization", return_telemetry=True)
        self.assertEqual(en_out.route_name, "FAST_MSA")

        # Pure numbers and punctuation
        num_out = self.router.process("1234567890 . , ! ? : ;", return_telemetry=True)
        self.assertEqual(num_out.tokens_raw, num_out.tokens_processed)

        # Malicious BiDi and Zero-width characters
        bidi_text = "\u202E\u200B\uFEFFأهلاً بالعالم\u202C"
        cleaned_bidi = self.router.process(bidi_text)
        self.assertNotIn("\u202E", cleaned_bidi)
        self.assertNotIn("\u200B", cleaned_bidi)
        self.assertIn("أهلا", cleaned_bidi)

        # Large stress text (5,000 words repetition)
        big_text = "الذكاء الاصطناعي والحوسبة السحابية المتقدمة في المملكة العربية السعودية. " * 300
        big_res = self.router.process(big_text, return_telemetry=True)
        self.assertGreater(big_res.tokens_raw, 1000)
        self.assertGreater(big_res.tokens_processed, 1000)

    def test_convenience_submodule_delegation(self):
        # Verify router direct convenience methods
        corr = self.router.correct("مسئول")
        self.assertIn("مسؤول", corr.corrected_text)

        ner = self.router.extract_ner("سدايا وموهبة في الرياض", mask=True)
        self.assertIn("[ORG_محجوب]", ner.masked_text)

        trans = self.router.translate("وش تبغى تسوي الحين")
        self.assertIn("ماذا تريد", trans.msa_text)

        sent = self.router.sentiment("الخدمة ممتازة ومبتكرة جداً")
        self.assertGreater(sent.score, 0.5)

        docs = ["معالجة اللغة العربية", "حوسبة سحابية ونظم تشغيل"]
        hits = self.router.search("اللغة العربية", docs)
        self.assertGreater(len(hits), 0)


if __name__ == "__main__":
    unittest.main()
