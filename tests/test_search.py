# -*- coding: utf-8 -*-
"""
Unit tests for Misar (مِسار) in-memory Arabic BM25 & Semantic Search Engine.
"""

import unittest
from misar.search import BM25SearchEngine, SearchResult, search_arabic


class TestBM25Search(unittest.TestCase):
    def setUp(self):
        self.engine = BM25SearchEngine(use_roots=True, remove_stopwords=True)
        self.docs = [
            "تعتمد الحوسبة السحابية على مراكز البيانات والخوادم الضخمة لتشغيل النماذج",
            "ابتكار محرك مِسار لمعالجة وتوجيه نصوص اللغة العربية والذكاء الاصطناعي",
            "تأمين الواجهات البرمجية وتشفير البيانات الحساسة وحجب الهجمات السيبرانية",
            "استرجاع المستندات في قواعد البيانات وأنظمة البحث الدلالي الذكي RAG",
        ]
        self.engine.add_documents(self.docs)

    def test_indexing_counts(self):
        self.assertEqual(self.engine.total_docs, 4)
        self.assertGreater(len(self.engine.index), 10)
        self.assertGreater(len(self.engine.root_index), 10)

    def test_exact_surface_search(self):
        results = self.engine.search("الذكاء الاصطناعي", top_k=2)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top.doc_id, "doc_2")
        self.assertIn("الذكاء", top.text)
        self.assertGreater(top.score, 0.0)

    def test_morphological_root_matching(self):
        # Searching for 'معالجات' or 'يعالج' should match doc_2 ('معالجة') via root 'ع-ل-ج'
        results = self.engine.search("المعالجات اللغوية", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].doc_id, "doc_2")

    def test_empty_and_stopword_query(self):
        self.assertEqual(self.engine.search(""), [])
        self.assertEqual(self.engine.search("   "), [])
        self.assertEqual(self.engine.search("في من على إلى"), [])

    def test_convenience_helper(self):
        res = search_arabic("الحوسبة السحابية", self.docs, top_k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].doc_id, "doc_1")
        self.assertIsInstance(res[0], SearchResult)
        d = res[0].to_dict()
        self.assertIn("score", d)
        self.assertIn("snippet", d)
        self.assertIn("matched_terms", d)


if __name__ == "__main__":
    unittest.main()
