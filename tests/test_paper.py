# -*- coding: utf-8 -*-
"""
Tests for Misar Automated Academic Paper & LaTeX Generator.
"""

import unittest

from misar import generate_academic_paper, generate_markdown_summary


class TestPaperGenerator(unittest.TestCase):
    """Exhaustive tests for academic paper synthesis and publication formatting."""

    def test_latex_paper_synthesis(self) -> None:
        """Verify LaTeX document structure, IEEE format, tables, and BibTeX citations."""
        latex = generate_academic_paper()
        self.assertIn(r"\documentclass[conference]{IEEEtran}", latex)
        self.assertIn("Muneer Jarab Matrak Al-Theeb", latex)
        self.assertIn("National Olympiad for Scientific Creativity", latex)
        self.assertIn(r"\begin{abstract}", latex)
        self.assertIn(r"\begin{table}", latex)
        self.assertIn("AraBERTv2", latex)
        self.assertIn("MARBERTv2", latex)
        self.assertIn("SDAIA ALLaM-7B", latex)
        self.assertIn(r"\begin{thebibliography}", latex)
        self.assertIn(r"\end{document}", latex)

    def test_markdown_summary_synthesis(self) -> None:
        """Verify markdown summary format and verified metrics."""
        md = generate_markdown_summary()
        self.assertIn("# Misar (مِسار)", md)
        self.assertIn("45.88%", md)
        self.assertIn("70.7%", md)
        self.assertIn("0.30 ms", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)
