# -*- coding: utf-8 -*-
"""
Tests for Misar Semantic Discourse RAG Chunker & LangChain Splitter.
"""

import unittest

from misar.rag import ArabicDiscourseChunker, MisarArabicSplitter, chunk_document


class TestArabicDiscourseRAGChunker(unittest.TestCase):
    """Exhaustive tests for Arabic RAG semantic chunking and negation binding."""

    def test_semantic_chunking_discourse_boundaries(self) -> None:
        doc = (
            "قرر مجلس الإدارة اعتماد خطة التحول الرقمي السنوية. "
            "ومن الجدير بالذكر أن الاستثمارات تجاوزت الأهداف المرسومة. "
            "علاوة على ذلك، وافق الشركاء على الهيكلة ولم يعترض أي عضو على الميزانية. "
            "بناءً على ذلك، سينطلق العمل مطلع الربع القادم."
        )
        chunks = chunk_document(doc, max_tokens=22, overlap_tokens=4)
        self.assertGreaterEqual(len(chunks), 2)
        # Verify all chunks have non-empty text and valid token counts
        for c in chunks:
            self.assertTrue(len(c.text) > 0)
            self.assertGreater(c.token_count, 0)
            self.assertLessEqual(c.token_count, 28)

    def test_negation_particle_never_severed(self) -> None:
        # Text engineered with negation particle 'ولم' right at sentence break
        doc = (
            "انتهت الجلسة الأولى من المداولات البرلمانية ولم "
            "يوافق النواب على المقترح المعدل لضريبة الشركات."
        )
        chunker = ArabicDiscourseChunker(max_tokens=15, overlap_tokens=0)
        chunks = chunker.chunk(doc)
        # Invariant: Neither chunk should end with bare negation 'ولم'
        for c in chunks:
            words = c.raw_text.strip().split()
            if words:
                self.assertNotEqual(words[-1], "ولم", f"Stranded negation found at end of chunk: {c.raw_text}")

    def test_char_offsets_and_provenance(self) -> None:
        doc = "بدأ الاجتماع في موعده المحدد. ومن ناحية أخرى تم تسليم التقارير الفنية في الموعد."
        chunks = chunk_document(doc, max_tokens=15, overlap_tokens=0)
        for c in chunks:
            slice_text = doc[c.start_char:c.end_char].strip()
            self.assertEqual(slice_text, c.raw_text)

    def test_langchain_splitter_interface(self) -> None:
        splitter = MisarArabicSplitter(max_tokens=25, overlap_tokens=4)
        text = "الذكاء الاصطناعي يغير العالم العربي. بالإضافة إلى ذلك يسهم في تطوير الرعاية الصحية."
        splits = splitter.split_text(text)
        self.assertIsInstance(splits, list)
        self.assertGreater(len(splits), 0)

        docs = splitter.create_documents([text], metadatas=[{"source": "medical_report"}])
        self.assertEqual(len(docs), len(splits))
        self.assertEqual(docs[0]["metadata"]["source"], "medical_report")
        self.assertIn("token_count", docs[0]["metadata"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
