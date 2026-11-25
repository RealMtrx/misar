# -*- coding: utf-8 -*-
"""
Unit tests for Misar Arabic Grapheme-to-Phoneme (G2P) & Phonetic Transcription Engine.
"""

import unittest

import misar
from misar import MisarRouter
from misar.phonemize import PhonemizationResult, phonemize_arabic


class TestArabicPhonemizationAndG2P(unittest.TestCase):
    """Exhaustive tests for Arabic IPA phonemization, solar/lunar assimilation, and audio estimation."""

    def test_solar_letter_assimilation(self) -> None:
        # Solar letters: الشمس -> ash-shams (assimilated Lam)
        res = phonemize_arabic("الشمس")
        self.assertIsInstance(res, PhonemizationResult)
        self.assertGreaterEqual(res.sun_letter_assimilations_count, 1)
        self.assertTrue("ʃ" in res.ipa_transcription)

    def test_lunar_letter_preservation(self) -> None:
        # Lunar letters: القمر -> al-qamar (preserved Lam /l/)
        res = phonemize_arabic("القمر")
        self.assertIsInstance(res, PhonemizationResult)
        self.assertEqual(res.sun_letter_assimilations_count, 0)
        self.assertIn("ʔal", res.ipa_transcription)
        self.assertIn("q", res.ipa_transcription)

    def test_emphatic_consonant_transcriptions(self) -> None:
        # Check pharyngealized consonants (ص، ض، ط، ظ)
        res_sad = phonemize_arabic("صبر")
        self.assertIn("sˤ", res_sad.ipa_transcription)

        res_ta = phonemize_arabic("طريق")
        self.assertIn("tˤ", res_ta.ipa_transcription)

    def test_syllables_and_duration_estimation(self) -> None:
        text = "المملكة العربية السعودية وطن العطاء والنماء والخير"
        res = phonemize_arabic(text)
        self.assertGreater(res.syllable_count, 5)
        self.assertGreater(res.estimated_audio_duration_sec, 0.5)
        self.assertIsInstance(res.phoneme_tokens, list)
        self.assertGreater(len(res.phoneme_tokens), 0)

    def test_router_integration(self) -> None:
        router = MisarRouter()
        res = router.phonemize("صباح الخير")
        self.assertIsInstance(res, PhonemizationResult)
        self.assertIn("sˤ", res.ipa_transcription)

    def test_top_level_exports(self) -> None:
        res = misar.phonemize_arabic("السلام عليكم ورحمة الله وبركاته")
        self.assertIsInstance(res, PhonemizationResult)
        d = res.to_dict()
        self.assertIn("ipa_transcription", d)
        self.assertIn("estimated_audio_duration_sec", d)


if __name__ == "__main__":
    unittest.main()
