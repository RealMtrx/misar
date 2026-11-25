# -*- coding: utf-8 -*-
"""
Unit tests for Misar Arabic Semantic Similarity & Plagiarism Detection Engine.
"""

import unittest

import misar
from misar import MisarRouter
from misar.similarity import MatchedSpan, SimilarityResult, check_plagiarism, compute_similarity


class TestArabicSemanticSimilarity(unittest.TestCase):
    """Exhaustive tests for Arabic text similarity, paraphrase detection, and plagiarism analysis."""

    def test_identical_texts_similarity(self) -> None:
        text = "المملكة العربية السعودية دولة رائدة في الذكاء الاصطناعي والابتكار."
        res = compute_similarity(text, text)
        self.assertIsInstance(res, SimilarityResult)
        self.assertAlmostEqual(res.overall_similarity, 1.0, places=2)
        self.assertAlmostEqual(res.lexical_similarity, 1.0, places=2)
        self.assertAlmostEqual(res.ngram_similarity, 1.0, places=2)
        self.assertAlmostEqual(res.lcs_ratio, 1.0, places=2)
        self.assertFalse(res.is_paraphrase)
        self.assertTrue(res.is_plagiarism)

    def test_completely_unrelated_texts(self) -> None:
        text1 = "الطقس حار جدا في فصل الصيف في صحراء الربع الخالي."
        text2 = "الرياضيات التطبيقية فرع هام لحساب المتجهات والمصفوفات الجبرية."
        res = compute_similarity(text1, text2)
        self.assertLess(res.overall_similarity, 0.25)
        self.assertFalse(res.is_paraphrase)
        self.assertFalse(res.is_plagiarism)
        self.assertEqual(len(res.matched_spans), 0)

    def test_paraphrase_detection(self) -> None:
        text1 = "يقوم الطلاب بكتابة مقالات دراسية في مكتبة الجامعة."
        text2 = "كتب الطالب مقالة دراسية داخل مكاتب الجامعات."
        res = compute_similarity(text1, text2)
        self.assertGreater(res.root_similarity, 0.25)
        self.assertTrue(res.is_paraphrase)

    def test_exact_copied_span_detection(self) -> None:
        text1 = "تعتبر معالجة اللغة الطبيعية فرعا أساسيا من فروع علوم الحاسب الآلي والذكاء الاصطناعي الحديث."
        text2 = "في هذا البحث نرى أن معالجة اللغة الطبيعية فرعا أساسيا من فروع علوم الحاسب ومجالات التقنية."
        res = compute_similarity(text1, text2)
        self.assertGreater(res.verbatim_match_count, 0)
        self.assertTrue(any("معالجة اللغة الطبيعية" in s.text for s in res.matched_spans))

    def test_corpus_plagiarism_scanning(self) -> None:
        corpus = [
            "تطوير البرمجيات يتطلب فهما عميقا للهياكل البيانية وخوارزميات البحث السريع.",
            "تاريخ الأندلس مليء بالحضارة المعمارية والشعر والأدب العربي الفريد.",
            "المملكة العربية السعودية تشهد نهضة تقنية كبرى في جميع القطاعات الحكومية والخاصة.",
        ]
        candidate = "نلاحظ أن المملكة العربية السعودية تشهد نهضة تقنية كبرى ومشاريع واعدة."
        results = check_plagiarism(candidate, corpus, threshold=0.3)
        self.assertGreaterEqual(len(results), 1)
        top_match = results[0]
        self.assertIsInstance(top_match, SimilarityResult)
        self.assertEqual(top_match.text2, corpus[2])
        self.assertGreater(top_match.overall_similarity, 0.4)

    def test_router_integration(self) -> None:
        router = MisarRouter()
        res = router.similarity("الذكاء الاصطناعي", "الذكاء الاصطناعي الحديث")
        self.assertIsInstance(res, SimilarityResult)
        self.assertGreater(res.overall_similarity, 0.5)

        hits = router.check_plagiarism("نص اختباري", ["نص اختباري متطابق"])
        self.assertGreater(len(hits), 0)

    def test_top_level_exports(self) -> None:
        res = misar.compute_similarity("كلمة أولى", "كلمة ثانية")
        self.assertIsInstance(res, SimilarityResult)
        d = res.to_dict()
        self.assertIn("overall_similarity", d)
        self.assertIn("matched_spans", d)


if __name__ == "__main__":
    unittest.main()
