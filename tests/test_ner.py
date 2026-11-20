# -*- coding: utf-8 -*-
"""
Unit tests for Misar (مِسار) Arabic Named Entity Recognition (NER) & Masking.
"""

import unittest
from misar.ner import EntitySpan, NERResult, extract_entities


class TestArabicNER(unittest.TestCase):
    def test_organization_and_location_extraction(self):
        text = "زار مسؤولو موهبة وسدايا مدينة أبها في منطقة عسير"
        res = extract_entities(text)
        cats = {e.category: e.text for e in res.entities}
        self.assertIn("ORG", cats)
        self.assertIn("LOC", cats)
        orgs = [e.text for e in res.entities if e.category == "ORG"]
        locs = [e.text for e in res.entities if e.category == "LOC"]
        self.assertTrue(any("موهبة" in o for o in orgs))
        self.assertTrue(any("سدايا" in o for o in orgs))
        self.assertTrue(any("أبها" in l or "عسير" in l for l in locs))

    def test_person_entity_extraction(self):
        text = "التقى المهندس منير الذيب مع الدكتور خالد في الرياض"
        res = extract_entities(text)
        pers = [e.text for e in res.entities if e.category == "PER"]
        self.assertGreater(len(pers), 0)
        self.assertTrue(any("منير" in p for p in pers))

    def test_money_and_date_extraction(self):
        text = "تم رصد ميزانية بقيمة 1500000 ريال لدعم المؤتمر عام 2027 م"
        res = extract_entities(text)
        cats = [e.category for e in res.entities]
        self.assertIn("MONEY", cats)
        self.assertIn("DATE", cats)

    def test_entity_masking(self):
        text = "زار منير الذيب مقر سدايا في الرياض"
        res = extract_entities(text, mask_entities=True)
        self.assertIn("[ORG_محجوب]", res.masked_text)
        self.assertIn("[LOC_محجوب]", res.masked_text)
        self.assertNotIn("سدايا", res.masked_text)
        self.assertNotIn("الرياض", res.masked_text)


if __name__ == "__main__":
    unittest.main()
