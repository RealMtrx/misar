# -*- coding: utf-8 -*-
"""
Tests for Misar High-Throughput Multi-Core Parallel Processing Engine.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from misar.parallel import process_dataset_file, route_parallel


class TestParallelBatchEngine(unittest.TestCase):
    """Tests for multi-core parallel processing of large datasets."""

    def test_route_parallel_small_and_large(self) -> None:
        samples = [
            "وش تبغى يا غالي الحين؟",
            "السلام عليكم ورحمة الله وبركاته",
            "عايز ايه دلوقتي يا باشا؟؟؟",
            "نظام مِسار لمعالجة اللغة العربية الطبيعية الفائقة",
        ] * 4  # 16 items

        results = route_parallel(samples, num_workers=2, chunk_size=4)
        self.assertEqual(len(results), 16)
        for r in results:
            self.assertTrue(len(r.processed_text) > 0)
            self.assertIn(r.route_name, ("FAST_MSA", "DIALECT_PRESERVE", "NOISY_SOCIAL"))
            self.assertIsNotNone(r.profile)

    def test_process_dataset_file_jsonl_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # 1. JSONL Test
            in_jsonl = tmp / "data.jsonl"
            out_jsonl = tmp / "data_out.jsonl"
            lines = [
                {"id": 1, "text": "وششش تبغىىى يا خوووي؟؟؟"},
                {"id": 2, "text": "بسم الله الرحمن الرحيم"},
                {"id": 3, "text": "دلوقتي كل حااااجة تمامممم"},
            ]
            with open(in_jsonl, "w", encoding="utf-8") as f:
                for item in lines:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            summary = process_dataset_file(
                input_file=in_jsonl,
                output_file=out_jsonl,
                text_key="text",
                num_workers=2,
            )
            self.assertEqual(summary["total_records"], 3)
            self.assertTrue(out_jsonl.exists())

            # Verify contents
            with open(out_jsonl, "r", encoding="utf-8") as f:
                out_lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(out_lines), 3)
            self.assertNotIn("وششش", out_lines[0]["text"])

            # 2. CSV Test
            in_csv = tmp / "data.csv"
            out_csv = tmp / "data_out.csv"
            with open(in_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "text"])
                writer.writeheader()
                for item in lines:
                    writer.writerow(item)

            summary_csv = process_dataset_file(
                input_file=in_csv,
                output_file=out_csv,
                text_key="text",
                num_workers=2,
            )
            self.assertEqual(summary_csv["total_records"], 3)
            self.assertTrue(out_csv.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
