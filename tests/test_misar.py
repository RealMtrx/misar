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
from misar.cli import build_parser
from misar.tokenizer import PRETRAINED_MODEL_CONFIGS, PRETRAINED_VOCAB_URLS, load_default_tokenizer
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
        self.assertGreater(compute_shannon_entropy("تعتمد الحوسبة السحابية الحديثة على هندسة النماذج اللغوية الضخمة"), 3.0)

    def test_code_switching_ratio(self):
        p = self.profiler.analyze("أطلقنا مشروع Misar لتسريع AraBERT")
        self.assertGreater(p.code_switching_ratio, 0.2)
        self.assertLess(p.code_switching_ratio, 0.7)

    def test_protected_negation_flag(self):
        p = self.profiler.analyze("الكود مو معقد أبداً والسيرفر مب بطيء وماكو أخطاء")
        self.assertTrue(p.has_protected_negation)
        self.assertIn("مو", p.detected_markers)
        self.assertIn("مب", p.detected_markers)
        self.assertIn("ماكو", p.detected_markers)


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
        text = "أعلنت اللجنة الهندسية عن إطلاق الحوسبة السحابية ومعالجة البيانات وإدارة الخوادم"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "FAST_MSA")
        # Must preserve أ, إ, ة for AraBERTv2/CAMeLBERT vocabulary integrity
        self.assertIn("أعلنت", res.processed_text)
        self.assertIn("الحوسبة", res.processed_text)
        self.assertIn("السحابية", res.processed_text)
        self.assertIn("وإدارة", res.processed_text)

    def test_legacy_msa_normalization_opt_in(self):
        legacy = MSALightPipeline(preserve_orthography=False)
        out = legacy.process("الحوسبة السحابية")
        self.assertEqual(out, "الحوسبه السحابيه")

    def test_classical_diacritized_text(self):
        text = "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "FAST_MSA")
        self.assertGreater(res.profile.diacritic_density, 0.5)
        self.assertEqual(res.processed_text, "بسم الله الرحمن الرحيم")

    def test_saudi_dialect_negation_preserved(self):
        text = "الكود مو معقد أبداً وما أبي أبدل السيرفر الحين"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "DIALECT_PRESERVE")
        self.assertIn("مو", res.processed_text)
        self.assertIn("أبي", res.processed_text)
        self.assertTrue(res.profile.has_protected_negation)

    def test_gulf_and_levantine_dialect_normalization(self):
        text = "المكتبة هاذي كتير مرتبة وشويتين زحمة بس ماكو تعقيد في API"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "DIALECT_PRESERVE")
        self.assertIn("هذه", res.processed_text)
        self.assertIn("كثير", res.processed_text)
        self.assertIn("شوية", res.processed_text)
        self.assertIn("ماكو", res.processed_text)

    def test_hybrid_noisy_social_plus_dialect(self):
        text = "السيرفررررر بطييييء مررررة ومافيه استجابة شويتين 😡😡 #هندسة_البرمجيات https://example.com"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "NOISY_SOCIAL")
        self.assertIn("السيرفر", res.processed_text)
        self.assertIn("بطيء", res.processed_text)
        self.assertIn("مرة", res.processed_text)
        self.assertIn("مافيه", res.processed_text)
        self.assertIn("شوية", res.processed_text)
        self.assertIn("هندسة البرمجيات", res.processed_text)
        self.assertNotIn("#", res.processed_text)
        self.assertNotIn("https://", res.processed_text)

    def test_gemination_preservation(self):
        text = "والللله الكود مو معقد ابدا"
        res = self.router_offline.route(text)
        self.assertIn("والله", res.processed_text)

    def test_batch_and_hf_pipeline_wrapper(self):
        texts = ["تعتمد الحوسبة السحابية", "السيرفررررر بطييييء #مشكلة"]
        results = self.router_offline.route_batch(texts)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].route_name, "FAST_MSA")
        self.assertEqual(results[1].route_name, "NOISY_SOCIAL")

        captured = []
        wrapped = self.router_offline.wrap_hf_pipeline(lambda x: captured.append(x) or x)
        wrapped("السيرفررررر بطييييء #مشكلة")
        self.assertEqual(captured[0], "السيرفر بطيء مشكلة")

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
        res = self.router_offline.route("السيرفر مرره سريع")
        self.assertEqual(res.route_name, "DIALECT_PRESERVE")
        self.assertIn("مرة", res.processed_text)
        res_noisy = self.router_offline.route("السيرفر مررره سريع")
        self.assertEqual(res_noisy.route_name, "NOISY_SOCIAL")
        self.assertIn("مرة", res_noisy.processed_text)

    def test_offline_tokenizer_no_double_tokens_or_false_unks(self):
        toks_hash = self.router_offline.tokenizer.tokenize("#ذكاء_مفيد")
        self.assertEqual(toks_hash, ["#", "##ذكاء", "##مفيد"])
        toks_rep = self.router_offline.tokenizer.tokenize("السلاممممم")
        self.assertEqual(toks_rep, ["السلا", "##ممممم"])
        res = self.router_offline.route("تعتمد الحوسبة السحابية الحديثة")
        self.assertEqual(res.unk_count_raw, 0)
        self.assertEqual(res.unk_count_processed, 0)

    def test_markdown_formatting_stripped_cleanly(self):
        text = "# أهمية العلم\n**المقدمة:** إن العلم نور"
        res = self.router_offline.route(text)
        self.assertEqual(res.route_name, "FAST_MSA")
        self.assertEqual(res.processed_text, "أهمية العلم المقدمة: إن العلم نور")

    def test_clitic_latin_code_switching_cleanup(self):
        text = "تطوير الـ Backend وربط بـ Docker عبر لـ API"
        res = self.router_offline.route(text)
        self.assertEqual(res.processed_text, "تطوير الBackend وربط بDocker عبر لAPI")
        self.assertGreaterEqual(res.clitic_fixed_count, 3)

    def test_emoji_polarity_modes(self):
        neg = "السيرفررررر بطييييء 😡😡👎"
        pos = "الاصدارررر سرييييع 😍🔥🚀"

        r_strip = MisarRouter(tokenizer=HeuristicSubwordTokenizer(), emoji_mode="strip")
        self.assertEqual(r_strip.route(neg).processed_text, "السيرفر بطيء")

        r_keep = MisarRouter(tokenizer=HeuristicSubwordTokenizer(), emoji_mode="keep")
        out_keep = r_keep.route(neg).processed_text
        self.assertIn("😡", out_keep)
        self.assertNotIn("😡😡", out_keep)

        r_sent = MisarRouter(tokenizer=HeuristicSubwordTokenizer(), emoji_mode="sentiment")
        self.assertEqual(r_sent.route(neg).processed_text, "السيرفر بطيء سلبي")
        self.assertEqual(r_sent.route(pos).processed_text, "الاصدار سريع إيجابي")

    def test_arabizi_transliteration_protects_tech_tokens(self):
        router_az = MisarRouter(tokenizer=HeuristicSubwordTokenizer(), arabizi_mode=True)
        out = router_az.preprocess("salam 3alekom al-nizam 7elw in base64 utf8 http2")
        self.assertIn("سلام", out)
        self.assertIn("عليكم", out)
        self.assertIn("حلو", out)
        self.assertIn("base64", out)
        self.assertIn("utf8", out)
        self.assertIn("http2", out)

    def test_fast_batch_stream_and_hf_dataset_map(self):
        texts = ["تعتمد الحوسبة السحابية", "السيرفررررر بطييييء 😡 #مشكلة"]
        batch_out = self.router_offline.preprocess_batch(texts)
        self.assertEqual(batch_out, ["تعتمد الحوسبة السحابية", "السيرفر بطيء مشكلة"])

        stream_out = list(self.router_offline.preprocess_stream(texts))
        self.assertEqual(stream_out, batch_out)

        hf_batch = {"text": texts, "label": [1, 0]}
        mapped = self.router_offline.hf_dataset_map(hf_batch, text_column="text", output_column="clean_text")
        self.assertEqual(mapped["label"], [1, 0])
        self.assertEqual(mapped["clean_text"], batch_out)

    def test_compare_baselines_three_way_structure(self):
        comp = self.router_offline.compare_baselines("تعتمد الحوسبة السحابية الحديثة")
        self.assertIn("legacy_text", comp)
        self.assertIn("legacy_tokens", comp)
        self.assertIn("legacy_token_count", comp)
        self.assertIn("vs_legacy_reduction_pct", comp)
        self.assertIn("ة", comp["misar"]["processed_text"])
        self.assertNotIn("ة", comp["legacy_text"])

    def test_cli_parser_v2_options(self):
        parser = build_parser()
        args = parser.parse_args(["route", "نص", "--model", "marbertv2", "--emoji-mode", "sentiment", "--arabizi", "--compare", "--offline"])
        self.assertEqual(args.model, "marbertv2")
        self.assertEqual(args.emoji_mode, "sentiment")
        self.assertTrue(args.arabizi)
        self.assertTrue(args.compare)
        self.assertTrue(args.offline)


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
            data=json.dumps({"texts": ["مرحبا", "السيرفررررر بطييييء"]}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req_ok, timeout=5) as resp:
            self.assertEqual(resp.status, 200)
            payload = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(len(payload["results"]), 2)

    def test_v2_three_way_comparison_and_options(self):
        with urllib.request.urlopen(f"{self.base}/api/health", timeout=5) as resp:
            health = json.loads(resp.read().decode("utf-8"))
            self.assertIn("available_models", health)
            self.assertGreaterEqual(len(health["available_models"]), 5)

        req = urllib.request.Request(
            f"{self.base}/api/route",
            data=json.dumps(
                {
                    "text": "السيرفررررر بطييييء 😡👎 والـ API ممتاز 😍",
                    "emoji_mode": "sentiment",
                    "auto_heal": True,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("legacy_text", data)
            self.assertIn("legacy_token_count", data)
            self.assertIn("vs_legacy_reduction_pct", data)
            self.assertIn("سلبي", data["processed_text"])
            self.assertIn("إيجابي", data["processed_text"])
            self.assertIn("الAPI", data["processed_text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
