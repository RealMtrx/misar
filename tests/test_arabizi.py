# -*- coding: utf-8 -*-
"""
Tests for Misar Arabizi (Franco-Arabic) Decoupler & Transliteration Engine.
"""

import unittest

from misar import MisarRouter, decode_arabizi, is_arabizi_text, is_arabizi_token


class TestArabiziDecoder(unittest.TestCase):
    """Exhaustive tests for Arabizi / Franco-Arabic transliteration and tokenizer protection."""

    def test_is_arabizi_detection(self) -> None:
        """Verify reliable detection of Arabizi tokens and sentences."""
        self.assertTrue(is_arabizi_token("mar7aba"))
        self.assertTrue(is_arabizi_token("7abibi"))
        self.assertTrue(is_arabizi_token("5alas"))
        self.assertTrue(is_arabizi_token("3afwan"))
        self.assertTrue(is_arabizi_token("shlonak"))

        # English tech terms should NOT be marked as Arabizi
        self.assertFalse(is_arabizi_token("API"))
        self.assertFalse(is_arabizi_token("Python"))
        self.assertFalse(is_arabizi_token("Docker"))
        self.assertFalse(is_arabizi_token("vLLM"))

        # Sentence-level detection
        self.assertTrue(is_arabizi_text("mar7aba ya 7abibi keefak?"))
        self.assertTrue(is_arabizi_text("shlonak al7amdulillah ana b5air"))
        self.assertFalse(is_arabizi_text("مرحبا بك في المملكة العربية السعودية"))
        self.assertFalse(is_arabizi_text("Welcome to the sovereign inference server"))

    def test_decode_arabizi_vocabulary(self) -> None:
        """Verify transliteration of common Arabizi expressions into Arabic."""
        text = "mar7aba ya 7abibi keefak shlonak al7amdulillah"
        decoded = decode_arabizi(text)

        self.assertIn("مرحبا", decoded)
        self.assertIn("حبيبي", decoded)
        self.assertIn("كيفك", decoded)
        self.assertIn("شلونك", decoded)
        self.assertIn("الحمد لله", decoded)

    def test_preserves_technical_english_code_switching(self) -> None:
        """Verify that English words, technical terms, and URLs remain untouched."""
        mixed = "mar7aba send the API request using Python with Docker"
        decoded = decode_arabizi(mixed)

        self.assertIn("مرحبا", decoded)
        self.assertIn("API", decoded)
        self.assertIn("Python", decoded)
        self.assertIn("Docker", decoded)

    def test_router_integration_with_arabizi_mode(self) -> None:
        """Verify MisarRouter preprocesses and converts Arabizi into clean Arabic."""
        router = MisarRouter(arabizi_mode=True)
        raw = "mar7aba ya 7abibi keefak shlonak"
        res = router.preprocess(raw)

        self.assertIn("مرحبا", res)
        self.assertIn("حبيبي", res)
        self.assertIn("كيفك", res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
