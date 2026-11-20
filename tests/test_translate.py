# -*- coding: utf-8 -*-
"""
Unit tests for Misar (مِسار) Dialect-to-MSA Parallel Translation Engine.
"""

import unittest
from misar.translate import TranslationResult, translate_to_msa


class TestDialectTranslation(unittest.TestCase):
    def test_saudi_gulf_translation(self):
        text = "وش تبغى تسوي الحين يا خوي؟ ترا السيرفر يبيله تحديث"
        res = translate_to_msa(text)
        self.assertEqual(res.detected_dialect, "saudi_gulf")
        self.assertIn("ماذا تريد أن تفعل", res.msa_text)
        self.assertIn("الآن", res.msa_text)
        self.assertIn("يحتاج إلى", res.msa_text)
        self.assertIn("يا أخي", res.msa_text)
        self.assertGreater(res.rules_applied_count, 0)

    def test_egyptian_translation(self):
        text = "عايز ايه دلوقتي يا باشا؟ التطبيق مش بطيء خالص"
        res = translate_to_msa(text)
        self.assertEqual(res.detected_dialect, "egyptian")
        self.assertIn("ماذا تريد", res.msa_text)
        self.assertIn("الآن", res.msa_text)
        self.assertIn("يا سيدي", res.msa_text)
        self.assertIn("ليس بطيئاً", res.msa_text)

    def test_levantine_translation(self):
        text = "شو بدك هلأ؟ هاد النظام كتير منيح"
        res = translate_to_msa(text)
        self.assertEqual(res.detected_dialect, "levantine")
        self.assertIn("ماذا تريد", res.msa_text)
        self.assertIn("الآن", res.msa_text)
        self.assertIn("هذا", res.msa_text)
        self.assertIn("جيد جداً", res.msa_text)

    def test_maghrebi_translation(self):
        text = "واش كاين دابا؟ الخدمة ديالكم مزيانة بزاف"
        res = translate_to_msa(text)
        self.assertEqual(res.detected_dialect, "maghrebi")
        self.assertIn("ماذا يوجد", res.msa_text)
        self.assertIn("الآن", res.msa_text)
        self.assertIn("الخاص بـ", res.msa_text)
        self.assertIn("جيد جداً", res.msa_text)

    def test_msa_pass_through(self):
        msa = "تعتمد الحوسبة السحابية على مراكز البيانات الضخمة"
        res = translate_to_msa(msa)
        self.assertEqual(res.msa_text, msa)
        self.assertEqual(res.rules_applied_count, 0)


if __name__ == "__main__":
    unittest.main()
