# -*- coding: utf-8 -*-
"""
Tests for Misar Domain-Specific Lexicons (FinTech, GovTech, HealthTech).
"""

import unittest

import misar
from misar.lexicon import explain_text, extract_root, lookup_word


class TestDomainLexicons(unittest.TestCase):
    """Exhaustive unit tests for FinTech, GovTech, and HealthTech domain vocabularies."""

    def test_fintech_lexicon_entries(self) -> None:
        """Verify Saudi FinTech & Banking terminology coverage."""
        terms = ["آيبان", "سمة", "سداد", "مدى", "مرابحة", "تورق", "استقطاع"]
        for term in terms:
            entry = lookup_word(term)
            self.assertIsNotNone(entry, f"FinTech term '{term}' should be found in lexicon")
            self.assertEqual(entry.dialect, "fintech")
            self.assertTrue(len(entry.definition) > 0)
            self.assertTrue(len(entry.lemma_msa) > 0)

        # Detailed verification of IBAN
        iban_entry = lookup_word("آيبان")
        self.assertIn("رقم الحساب المصرفي الدولي", iban_entry.definition)

        # Detailed verification of SIMAH
        simah_entry = lookup_word("سمة")
        self.assertIn("التقرير الائتماني", simah_entry.definition)

    def test_govtech_lexicon_entries(self) -> None:
        """Verify Saudi GovTech & Citizen digital platforms coverage."""
        terms = ["أبشر", "ناجز", "توكلنا", "قوى", "بلدي", "صحتي", "مدد", "اعتماد"]
        for term in terms:
            entry = lookup_word(term)
            self.assertIsNotNone(entry, f"GovTech term '{term}' should be found in lexicon")
            self.assertEqual(entry.dialect, "govtech")
            self.assertTrue(len(entry.definition) > 0)

        # Detailed verification of Absher
        absher_entry = lookup_word("أبشر")
        self.assertIn("المنصة الوطنية الموحدة", absher_entry.definition)

        # Detailed verification of Najiz
        najiz_entry = lookup_word("ناجز")
        self.assertIn("وزارة العدل", najiz_entry.definition)

    def test_healthtech_lexicon_entries(self) -> None:
        """Verify HealthTech clinical symptoms and medical facilities coverage."""
        terms = ["سخونة", "كحة", "دوخة", "مغص", "رشح", "زكام", "صداع", "مستوصف"]
        for term in terms:
            entry = lookup_word(term)
            self.assertIsNotNone(entry, f"HealthTech term '{term}' should be found in lexicon")
            self.assertEqual(entry.dialect, "healthtech")
            self.assertTrue(len(entry.definition) > 0)

        # Detailed verification of clinic
        clinic_entry = lookup_word("مستوصف")
        self.assertIn("منشأة صحية", clinic_entry.definition)

        # Detailed verification of fever
        fever_entry = lookup_word("سخونة")
        self.assertIn("حمى", fever_entry.lemma_msa)

    def test_total_lexicon_expansion_v260(self) -> None:
        """Ensure lexicon vocabulary size satisfies v2.6.0 milestone (>= 200 curated entries)."""
        lex = misar.get_lexicon()
        self.assertGreaterEqual(len(lex.entries), 200, "Lexicon must contain at least 200 distinct curated entries")
        self.assertGreaterEqual(lex.total_entries, 200)

    def test_domain_sentence_explanation(self) -> None:
        """Verify explain_text correctly recognizes domain keywords and translates them to canonical MSA."""
        sentence = "طلب العميل تحويل آيبان عبر سداد بسبب سخونة ودوخة وراجع مستوصف عبر توكلنا"
        explanation = explain_text(sentence)

        self.assertGreaterEqual(explanation.dialect_tokens_count, 3)
        self.assertTrue(len(explanation.canonical_msa_text) > 0)
        found_dialects = {t.entry.dialect for t in explanation.tokens if t.entry}
        self.assertIn("fintech", found_dialects)
        self.assertIn("healthtech", found_dialects)
        self.assertIn("govtech", found_dialects)

    def test_domain_morphological_root_extraction(self) -> None:
        """Verify root extraction on specialized domain terms."""
        self.assertEqual(extract_root("مرابحة"), "ر-ب-ح")
        self.assertEqual(extract_root("استقطاع"), "ق-ط-ع")
        self.assertEqual(extract_root("مستوصف"), "و-ص-ف")


if __name__ == "__main__":
    unittest.main(verbosity=2)
