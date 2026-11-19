# -*- coding: utf-8 -*-
"""
Comprehensive automated test suite for Misar (مِسار) v1.0.0.
Run with: python -m unittest discover -s tests -v  (or: pytest)
"""

import json
import threading
import unittest
import urllib.error
import urllib.request

from misar import (
    HeuristicSubwordTokenizer,
    LinguisticProfiler,
    MSALightPipeline,
    MisarRouter,
    WordPieceTokenizer,
    compute_shannon_entropy,
)
from misar.web.server import bind_server


class TestLinguisticProfiler(unittest.TestCase):
    def setUp(self):
        self.profiler = LinguisticProfiler()

    def test_empty_and_whitespace(self):
        for sample in ("", "   \t\n  "):
            p = self.profiler.analyze(sample)
            self.assertEqual(p.total_chars, 0)
            self.assertEqual(p.word_count, 0)
            self.assertEqual(p.shannon_entropy, 0.0)

    def test_shannon_entropy_math(self):
        # Single repeated symbol has 0 bits of entropy
        self.assertAlmostEqual(compute_shannon_entropy("ااااا"), 0.0, places=5)
        # 2 equiprobable symbols -> 1.0 bit
        self.assertAlmostEqual(compute_shannon_entropy("اباب"), 1.0, places=5)
        # Rich Arabic text > 2.5 bits
        self.assertGreater(compute_shannon_entropy("أعلنت الهيئة السعودية للبيانات والذكاء الاصطناعي"), 3.0)

    def test_code_switching_ratio(self):
        p = self.profiler.analyze("أطلقنا مشروع Misar لتسريع AraBERT")
        self.assertGreater(p.code_switching_ratio, 0.2)
        self.assertLess(p.code_switching_ratio, 0.7)

    def test_protected_negation_flag(self):
        p = self.profiler.analyze("المكان مو حلو أبداً والخدمة مش كويسة")
        self.assertTrue(p.has_protected_negation)
        self.assertIn("مو", p.detected_markers)
        self.assertIn("مش", p.detected_markers)


class TestMisarRoutingAndPipelines(unittest.TestCase):
    def setUp(self):
        self.router_offline = MisarRouter(tokenizer=HeuristicSubwordTokenizer())

    def test_empty_and_non_arabic_passthrough(self):
        res_empty = self.router_offline.route("")
        self.assertEqual(res_empty.route_name, "FAST_MSA")
        self.assertEqual(res_empty.processed_text, "")

        res_en = self.router_offline.route("Hello Misar 2027")
        self.assertEqual(res_en.route_name, "FAST_MSA")
        self.assertEqual(res_en.processed_text, "Hello Misar 2027")

    def test_msa_preserves_transformer_orthography(self):
        text = "أعلنت الهيئة السعودية للبيانات والذكاء الاصطناعي عن إطلاق مبادرة وطنية"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "FAST_MSA")
        # Must preserve أ, إ, ة for AraBERTv2/CAMeLBERT vocabulary integrity
        self.assertIn("أعلنت", res.processed_text)
        self.assertIn("الهيئة", res.processed_text)
        self.assertIn("السعودية", res.processed_text)
        self.assertIn("إطلاق", res.processed_text)

    def test_legacy_msa_normalization_opt_in(self):
        legacy = MSALightPipeline(preserve_orthography=False)
        out = legacy.process("الهيئة السعودية")
        self.assertEqual(out, "الهيئه السعوديه")

    def test_classical_diacritized_text(self):
        text = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "FAST_MSA")
        self.assertGreater(res.profile.diacritic_density, 0.5)
        self.assertEqual(res.processed_text, "بسم الله الرحمن الرحيم")

    def test_saudi_dialect_negation_preserved(self):
        text = "المكان مو حلو أبداً وما أبي أرجع الحين"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "DIALECT_PRESERVE")
        self.assertIn("مو", res.processed_text)
        self.assertIn("أبي", res.processed_text)
        self.assertTrue(res.profile.has_protected_negation)

    def test_egyptian_dialect_normalization(self):
        text = "الفكرة مش بطالة خالص وعلشان كدا عاوزين تظبيط"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "DIALECT_PRESERVE")
        self.assertIn("مش", res.processed_text)
        self.assertIn("عشان", res.processed_text)
        self.assertIn("كده", res.processed_text)

    def test_hybrid_noisy_social_plus_dialect(self):
        text = "المطعممممم سيييييء جداااااا ومفيش أسلوب علشان كدا 😡😡 #تجربة_فاشلة https://example.com"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "NOISY_SOCIAL")
        self.assertIn("المطعم", res.processed_text)
        self.assertIn("سيء", res.processed_text)
        self.assertIn("مفيش", res.processed_text)
        self.assertIn("عشان", res.processed_text)
        self.assertIn("تجربة فاشلة", res.processed_text)
        self.assertNotIn("#", res.processed_text)
        self.assertNotIn("https://", res.processed_text)

    def test_gemination_preservation(self):
        text = "والللله المكان مو حلو ابدا"
        res = self.router_offline.route(text)
        self.assertIn("والله", res.processed_text)

    def test_batch_and_hf_pipeline_wrapper(self):
        texts = ["أعلنت الهيئة السعودية", "المطعممممم سيييييء #فاشل"]
        results = self.router_offline.route_batch(texts)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].route_name, "FAST_MSA")
        self.assertEqual(results[1].route_name, "NOISY_SOCIAL")

        captured = []
        wrapped = self.router_offline.wrap_hf_pipeline(lambda x: captured.append(x) or x)
        wrapped("المطعممممم سيييييء #فاشل")
        self.assertEqual(captured[0], "المطعم سيء فاشل")

    def test_single_elongated_word_collapsed(self):
        text = "السلام" + "م" * 30 + " عليكم"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "NOISY_SOCIAL")
        self.assertEqual(res.processed_text, "السلام عليكم")
        self.assertGreaterEqual(res.profile.repeated_char_runs, 1)
        self.assertGreaterEqual(res.profile.elongation_noise_score, 0.5)

    def test_msa_homographs_do_not_trigger_dialect_false_positives(self):
        msa_sentences = [
            "أصدرت عدة دول عربية بياناً مشتركاً لأول مرة عن صافي الأرباح",
            "جاء أبي في حاجة ماسة إلى كتاب فخم وزينة الحياة الدنيا",
            "حيت اللجنة العلمية الجهود البحثية ولم تجد طريقاً سوى الابتكار",
            "في آية كريمة توصل الطرفان إلى حل ودي حول منتدى نسوي لخدمة البدو",
            "مع خالص الشكر والتقدير لحفظ حقك في هذه المبادرة",
        ]
        for sentence in msa_sentences:
            res = self.router_offline.route(sentence)
            self.assertEqual(res.route_name, "FAST_MSA", f"False dialect on: {sentence} -> {res.profile.detected_markers}")
            self.assertEqual(res.profile.detected_markers, [])

    def test_expressive_marra_triggers_dialect_preserve(self):
        res = self.router_offline.route("الفندق مرره جميل")
        self.assertEqual(res.route_name, "DIALECT_PRESERVE")
        self.assertIn("مرة", res.processed_text)
        res_noisy = self.router_offline.route("الفندق مررره جميل")
        self.assertEqual(res_noisy.route_name, "NOISY_SOCIAL")
        self.assertIn("مرة", res_noisy.processed_text)

    def test_offline_tokenizer_no_double_tokens_or_false_unks(self):
        toks_hash = self.router_offline.tokenizer.tokenize("#تجربة_فاشلة")
        self.assertEqual(toks_hash, ["#", "##تجربة", "##فاشلة"])
        toks_rep = self.router_offline.tokenizer.tokenize("السلاممممم")
        self.assertEqual(toks_rep, ["السلا", "##ممممم"])
        res = self.router_offline.route("أعلنت الهيئة السعودية للبيانات")
        self.assertEqual(res.unk_count_raw, 0)
        self.assertEqual(res.unk_count_processed, 0)

    def test_markdown_formatting_stripped_cleanly(self):
        text = "# أهمية العلم\n**المقدمة:** إن العلم نور"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "FAST_MSA")
        self.assertEqual(res.processed_text, "أهمية العلم المقدمة: إن العلم نور")


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
        msa = "أعلنت الهيئة السعودية للبيانات والذكاء الاصطناعي عن إطلاق المبادرات الوطنية"
        res_misar = self.router.route(msa)
        res_legacy = self.legacy_router.route(msa)
        self.assertTrue(res_misar.is_real_tokenizer)
        # Misar keeps ة intact -> strictly fewer subword tokens than destructive legacy ة->ه
        self.assertLess(res_misar.processed_token_count, res_legacy.processed_token_count)
        self.assertAlmostEqual(res_misar.tfr_processed, 1.0, delta=0.15)

    def test_real_arabert_vocab_reduces_noisy_and_diacritized_tokens(self):
        if not self.available:
            self.skipTest("AraBERTv2 vocab unavailable offline")
        noisy = "المطعممممم سيييييء جداااااا ومفيش أسلوب خالصصصصص 😡😡😡 #تجربة_فاشلة"
        res_noisy = self.router.route(noisy)
        self.assertLess(res_noisy.processed_token_count, res_noisy.raw_token_count)
        self.assertGreater(res_noisy.token_reduction_pct, 40.0)

        classical = "إِنَّ الَّذِي مَلَأَ اللُّغَاتِ مَحَاسِناً، جَعَلَ الجَمَالَ وَسِرَّهُ فِي الضَّادِ"
        res_class = self.router.route(classical)
        # Without Misar, AraBERTv2 collapses diacritized words into [UNK] (10/10 OOV collapse).
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


class TestMisarHTTPServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = MisarRouter(tokenizer=HeuristicSubwordTokenizer())
        cls.server = bind_server(cls.router, host="127.0.0.1", preferred_port=8190)
        cls.port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_get_routes_and_security_headers(self):
        with urllib.request.urlopen(f"{self.base}/?v=2", timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
            self.assertEqual(resp.headers.get("X-Frame-Options"), "SAMEORIGIN")
            body = resp.read().decode("utf-8")
            self.assertIn("مِسار", body)

        with urllib.request.urlopen(f"{self.base}/api/health", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["status"], "ok")

    def test_post_validation_and_error_codes(self):
        for bad_body in (b"{broken", b"[1, 2]", b'{"text": 123}'):
            req = urllib.request.Request(
                f"{self.base}/api/route",
                data=bad_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(ctx.exception.code, 400)
            ctx.exception.close()

        req_ok = urllib.request.Request(
            f"{self.base}/api/batch",
            data=json.dumps({"texts": ["مرحبا", "المطعمممم سييييء"]}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req_ok, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(len(payload["results"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
