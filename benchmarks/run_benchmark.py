# -*- coding: utf-8 -*-
"""
Reproducible Real-Vocabulary Benchmark entry script for Misar (مِسار) v1.0.0.
Compares Raw Input vs. Legacy Destructive Normalization vs. Misar Adaptive Router
against the authentic 64,000-token AraBERTv2 WordPiece vocabulary.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from misar.benchmark import run_benchmark


def main() -> None:
    report_path = Path(__file__).parent / "latest_benchmark_report.json"
    run_benchmark(model_key="arabertv02", offline=False, save_report_path=report_path)


if __name__ == "__main__":
    main()
