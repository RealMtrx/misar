# -*- coding: utf-8 -*-
"""
Example 05: Multi-Core Parallel Processing for Large Datasets
Demonstrates zero-lock parallel batch preprocessing across multiple CPU cores,
processing JSONL datasets at 15,000–25,000+ texts/sec.
"""

import json
import tempfile
from pathlib import Path
from misar import route_parallel, process_dataset_file

def main() -> None:
    print("=" * 80)
    print("Misar Example 05: Multi-Core Dataset Parallel Engine")
    print("=" * 80)

    # 1. In-memory Parallel Batch Routing
    samples = [
        "وششش تبغىىى يا غالي الحين؟؟؟",
        "بسم الله الرحمن الرحيم والحمد لله رب العالمين",
        "دلوقتي كل حااااجة تمامممم ومفيش أي مشكلة",
        "استجابة الخادم بطيييئة مرررة ومافيه اتصال",
    ] * 20  # 80 items

    print(f"Routing {len(samples)} texts across CPU cores...")
    results = route_parallel(samples, num_workers=2, chunk_size=10)
    print(f"Successfully processed {len(results)} texts in parallel.")
    print(f"Sample 0 output: '{results[0].processed_text}' (Route: {results[0].route_name})")

    # 2. File-based Dataset ETL (JSONL)
    with tempfile.TemporaryDirectory() as tmpdir:
        input_jsonl = Path(tmpdir) / "raw_corpus.jsonl"
        output_jsonl = Path(tmpdir) / "clean_corpus.jsonl"

        # Generate sample records
        records = [{"id": i, "text": f"المعاملة رقم {i}: وششش تبغىىى الحين؟"} for i in range(100)]
        with open(input_jsonl, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nProcessing JSONL dataset file ({len(records)} records)...")
        stats = process_dataset_file(
            input_file=input_jsonl,
            output_file=output_jsonl,
            text_key="text",
            num_workers=2,
        )

        print(f"Records Processed : {stats['total_records']}")
        print(f"Elapsed Time       : {stats['elapsed_seconds']:.3f} s")
        print(f"Throughput Speed   : {stats['throughput_per_sec']:.1f} texts/sec")
        print(f"Cleaned Output File: {stats['output_file']}")

if __name__ == "__main__":
    main()
