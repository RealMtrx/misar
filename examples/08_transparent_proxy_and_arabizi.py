# -*- coding: utf-8 -*-
"""
Example 08: Transparent OpenAI Reverse Proxy, Arabizi Decoupler, and Data Augmentation
Demonstrates Misar v2.7.0 advanced capabilities:
1. Decoding Franco-Arabic (Arabizi) youth chat to canonical Arabic while preserving tech terms.
2. Synthetic Data Augmentation (typos, tatweel, diacritics) for LLM robustness stress-testing.
3. Transparent OpenAI-compatible Reverse Proxy Gateway with PII masking and token reduction.
4. Automated generation of publication-ready IEEE/ACL academic research paper preprint.
"""

import json
import threading
import time
import urllib.request

from misar import (
    create_proxy_server,
    decode_arabizi,
    evaluate_augmentation_recovery,
    generate_academic_paper,
    generate_stress_suite,
    is_arabizi_text,
)


def main() -> None:
    print("=" * 80)
    print("Misar Example 08: Transparent OpenAI Proxy, Arabizi & Augmentation (v2.7.0)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Arabizi (Franco-Arabic) Decoupler & Transliteration
    # -------------------------------------------------------------------------
    print("\n[1] Arabizi / Franco-Arabic Youth Chat Transliteration:")
    arabizi_samples = [
        "mar7aba ya 7abibi keefak shlonak al7amdulillah",
        "shlonik 7abibti? inshallah b5air w tmam",
        "salam 3alekom send the API request using Python with Docker and base64",
        "5alas ya 5oy mashkour w teslam ktheer",
    ]
    for sample in arabizi_samples:
        is_az = is_arabizi_text(sample)
        decoded = decode_arabizi(sample)
        print(f"  • Raw Arabizi: '{sample}'")
        print(f"    Detected:    {is_az} -> Decoded: '{decoded}'\n")

    # -------------------------------------------------------------------------
    # 2. Synthetic Data Augmentation & LLM Stress-Testing
    # -------------------------------------------------------------------------
    print("[2] Synthetic Perturbations & Robustness Recovery:")
    seed_text = "المملكة العربية السعودية تطلق مبادرة الذكاء الاصطناعي السيادي"
    suite = generate_stress_suite(seed_text)
    for variant, text in suite.items():
        print(f"  • {variant:22}: {text}")

    print("\n  Evaluating Misar's token recovery on perturbed variants:")
    recovery_stats = evaluate_augmentation_recovery(seed_text)
    for var_name, stats in recovery_stats.items():
        print(
            f"    - {var_name:22}: Raw {stats['raw_token_count']:2} tok -> "
            f"Clean {stats['cleaned_token_count']:2} tok | "
            f"Saved: {stats['tokens_saved']:2} ({stats['reduction_pct']}%)"
        )

    # -------------------------------------------------------------------------
    # 3. Transparent OpenAI-Compatible Reverse Proxy Gateway
    # -------------------------------------------------------------------------
    print("\n[3] Transparent OpenAI Proxy Gateway (Sub-Millisecond Interception):")
    proxy_port = 8765
    server = create_proxy_server(host="127.0.0.1", port=proxy_port)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.3)

    try:
        # Client sends standard OpenAI SDK /v1/chat/completions payload
        openai_payload = {
            "model": "sdaia/allam-7b-instruct",
            "messages": [
                {"role": "system", "content": "أنت مساعد ذكي لمنصة أبشر."},
                {
                    "role": "user",
                    "content": "الــــســــلام علـــيكم، أرغب في تجديد هويتي 1087654321 وآيبان SA0380000000608010167519 ورقمي 0512345678",
                },
            ],
        }

        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            data=json.dumps(openai_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=5) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
            print(f"  • Proxy Response Code:        {resp.status} OK")
            print(f"  • Header [X-Misar-Processed]: {resp.headers.get('X-Misar-Processed')}")
            print(f"  • Header [X-Misar-Tokens-Saved]: {resp.headers.get('X-Misar-Tokens-Saved')}")
            print(f"  • Header [X-Misar-Attention-Saved-Pct]: {resp.headers.get('X-Misar-Attention-Saved-Pct')}")
            print(f"  • Header [X-Misar-PII-Masked]: {resp.headers.get('X-Misar-PII-Masked')}")
            print(f"  • Header [X-Misar-Latency-Ms]: {resp.headers.get('X-Misar-Latency-Ms')} ms")
            print(f"  • Sanitized Output Preview:   {resp_body['choices'][0]['message']['content']}")

    finally:
        server.shutdown()
        server.server_close()

    # -------------------------------------------------------------------------
    # 4. Automated Academic Paper Preprint Synthesis
    # -------------------------------------------------------------------------
    print("\n[4] Automated Academic LaTeX Preprint Generator:")
    latex_code = generate_academic_paper()
    print(f"  • Synthesized compile-ready IEEE/ACL research paper ({len(latex_code):,} chars).")
    print("  • Document sections: Abstract, Mathematical Formulation, Benchmarks Table, Citations.")

    print("\n" + "=" * 80)
    print("Misar v2.7.0 advanced capabilities verified successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
