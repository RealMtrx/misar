# -*- coding: utf-8 -*-
"""
Tests for Misar Multi-Dialect Semantic Lexicon & Morphological Root Extractor.
"""

import unittest

import misar
from misar import MisarRouter
from misar.lexicon import explain_text, extract_root, lookup_word


class TestArabicLexiconAndMorphology(unittest.TestCase):
    """Exhaustive tests for Misar Arabic Lexicon and Morphological Dictionary."""

    def test_lexicon_lookup_multi_dialect(self) -> None:
        # Saudi / Gulf
        entry_sa = lookup_word("أبخص")
        self.assertIsNotNone(entry_sa)
        self.assertEqual(entry_sa.dialect, "saudi_gulf")
        self.assertEqual(entry_sa.root, "ب-خ-ص")
        self.assertEqual(entry_sa.lemma_msa, "أعلم وأدرى")

        # Egyptian
        entry_eg = lookup_word("دلوقتي")
        self.assertIsNotNone(entry_eg)
        self.assertEqual(entry_eg.dialect, "egyptian")
        self.assertEqual(entry_eg.lemma_msa, "الآن")

        # Levantine
        entry_lev = lookup_word("هيك")
        self.assertIsNotNone(entry_lev)
        self.assertEqual(entry_lev.dialect, "levantine")

        # Maghrebi
        entry_mag = lookup_word("بزاف")
        self.assertIsNotNone(entry_mag)
        self.assertEqual(entry_mag.dialect, "maghrebi")

        # Iraqi
        entry_irq = lookup_word("يمعود")
        self.assertIsNotNone(entry_irq)
        self.assertEqual(entry_irq.dialect, "iraqi")

        # Yemeni
        entry_yem = lookup_word("ناهي")
        self.assertIsNotNone(entry_yem)
        self.assertEqual(entry_yem.dialect, "yemeni")

    def test_extract_root_morphological_patterns(self) -> None:
        self.assertEqual(extract_root("استكشاف"), "ك-ش-ف")
        self.assertEqual(extract_root("تطبيق"), "ط-ب-ق")
        self.assertEqual(extract_root("مكتوب"), "ك-ت-ب")
        self.assertEqual(extract_root("افتتاح"), "ف-ت-ح")
        self.assertEqual(extract_root("كاتب"), "ك-ت-ب")
        self.assertEqual(extract_root("عالم"), "ع-ل-م")
        self.assertEqual(extract_root("مشاريع"), "ش-ر-ع")

    def test_sentence_explanation_and_canonical_msa(self) -> None:
        text = "وش تبغى تسوي الحين يا يمعود؟"
        res = explain_text(text)
        self.assertEqual(res.original_text, text)
        self.assertGreaterEqual(res.dialect_tokens_count, 3)
        self.assertGreater(res.dialect_density_pct, 40.0)
        self.assertEqual(res.dominant_dialect, "saudi_gulf")
        self.assertIn("الآن", res.canonical_msa_text)

    def test_expanded_dialect_vocabulary_v250(self) -> None:
        lex = misar.get_lexicon()
        self.assertGreaterEqual(len(lex.entries), 150)

        # Saudi Najdi / Hijazi additions
        e_azhaleh = lookup_word("أزهله")
        self.assertIsNotNone(e_azhaleh)
        self.assertEqual(e_azhaleh.dialect, "saudi_gulf")

        e_dandoon = lookup_word("دندون")
        self.assertIsNotNone(e_dandoon)
        self.assertEqual(e_dandoon.dialect, "saudi_gulf")

        e_dahein = lookup_word("دحين")
        self.assertIsNotNone(e_dahein)
        self.assertEqual(e_dahein.dialect, "saudi_gulf")

        # Levantine addition
        e_kteer = lookup_word("كتير")
        self.assertIsNotNone(e_kteer)
        self.assertEqual(e_kteer.dialect, "levantine")

    def test_router_integration_with_lexicon(self) -> None:
        router = MisarRouter()
        entry = router.lookup("أبخص")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.root, "ب-خ-ص")

        expl = router.explain_text("دلوقتي كل حاجة تمام")
        self.assertGreaterEqual(expl.dialect_tokens_count, 1)

        self.assertEqual(misar.__version__, "3.1.0")
        self.assertTrue(callable(misar.lookup_word))
        self.assertTrue(callable(misar.explain_text))
        self.assertTrue(callable(misar.extract_root))


if __name__ == "__main__":
    unittest.main(verbosity=2)
