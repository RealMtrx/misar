# -*- coding: utf-8 -*-
"""
Unit tests for Misar (مِسار) Arabic Text Summarizer & Keyphrase Extractor.
"""

import unittest
from misar.summarize import (
    KeyphraseSpan,
    SummarizationResult,
    extract_keyphrases,
    summarize_text,
)
from misar.router import MisarRouter


SAMPLE_ARTICLE = (
    "أعلنت المملكة العربية السعودية عن استثمارات كبرى في البنية التحتية للحوسبة السحابية ومراكز البيانات المتقدمة. "
    "تهدف هذه المبادرات إلى تمكين الشركات التقنية والناشئة من تسريع الابتكار في تطبيقات الذكاء الاصطناعي وتوطين التقنيات السيادية. "
    "وتعتبر هذه الخطوة ركيزة أساسية ضمن مستهدفات رؤية 2030 لتحويل المملكة إلى مركز إقليمي وعالمي للبيانات. "
    "ومن المتوقع أن تسهم هذه الاستثمارات في خلق آلاف الوظائف التقنية النوعية وتعزيز الاقتصاد الرقمي الوطني."
)


class TestArabicSummarization(unittest.TestCase):
    def test_extractive_summarization(self):
        res = summarize_text(SAMPLE_ARTICLE, num_sentences=2)
        self.assertIsInstance(res, SummarizationResult)
        self.assertEqual(res.summary_sentence_count, 2)
        self.assertEqual(res.original_sentence_count, 4)
        self.assertLess(res.compression_ratio, 1.0)
        self.assertTrue(len(res.summary_text) < len(res.original_text))

        # Extracted sentences must belong to original text
        for sentence in res.extracted_sentences:
            self.assertIn(sentence, SAMPLE_ARTICLE)

    def test_keyphrase_extraction(self):
        phrases = extract_keyphrases(SAMPLE_ARTICLE, top_k=5)
        self.assertGreaterEqual(len(phrases), 1)
        for kp in phrases:
            self.assertIsInstance(kp, KeyphraseSpan)
            self.assertGreater(kp.score, 0.0)
            self.assertGreaterEqual(kp.frequency, 1)

        # High-salience phrases should contain key domains
        phrase_texts = " ".join(k.phrase for k in phrases)
        self.assertTrue(
            "السعودية" in phrase_texts
            or "الذكاء" in phrase_texts
            or "المملكة" in phrase_texts
            or "البيانات" in phrase_texts
        )

    def test_single_sentence(self):
        single = "تعتبر الرياض عاصمة المملكة العربية السعودية ومركزها المالي الرئيسي."
        res = summarize_text(single, num_sentences=2)
        self.assertEqual(res.original_sentence_count, 1)
        self.assertEqual(res.summary_sentence_count, 1)
        self.assertEqual(res.summary_text.strip(), single.strip())

    def test_empty_and_whitespace(self):
        res = summarize_text("")
        self.assertEqual(res.summary_text, "")
        self.assertEqual(res.summary_sentence_count, 0)
        self.assertEqual(len(res.keyphrases), 0)

        kp_empty = extract_keyphrases("   ")
        self.assertEqual(len(kp_empty), 0)

    def test_router_integration(self):
        router = MisarRouter()
        res = router.summarize(SAMPLE_ARTICLE, num_sentences=2)
        self.assertIsInstance(res, SummarizationResult)
        self.assertEqual(res.summary_sentence_count, 2)

        kps = router.extract_keyphrases(SAMPLE_ARTICLE, top_k=3)
        self.assertGreaterEqual(len(kps), 1)


if __name__ == "__main__":
    unittest.main()
