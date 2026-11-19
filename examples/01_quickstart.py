# -*- coding: utf-8 -*-
"""
Example 01: Quickstart Routing & Token Optimization
Demonstrates how Misar dynamically routes Arabic text across FAST_MSA, DIALECT_PRESERVE,
and NOISY_SOCIAL micro-pipelines, measuring token reduction against authentic AraBERTv2 vocab.
"""

from misar import MisarRouter, route

def main() -> None:
    print("=" * 80)
    print("Misar Example 01: Adaptive Routing & Token Telemetry")
    print("=" * 80)

    # 1. Simple module-level helper
    sample = "استجابة الخادم بطيييئة مرررة ومافيه أي اتصال في المنظومة أبدددداً 😡😡👎 #انقطاع_الخدمة"
    res = route(sample)

    print(f"\n[Input Text]:\n  {res.original_text}")
    print(f"\n[Route Selected]: {res.route_name} ({res.route_display_ar})")
    print(f"[Decision Reason]: {res.decision_reason}")
    print(f"[Processed Text]:\n  {res.processed_text}")
    print(f"\n[WordPiece Tokens]: {res.raw_token_count} -> {res.processed_token_count} (Saved: {res.token_reduction_pct:.1f}%)")
    print(f"[Latency]: {res.latency_ms:.3f} ms on CPU")
    print(f"[Attention Flops Saved]: {res.attention_savings_pct:.1f}%")

    # 2. Advanced 3-Way Baseline Comparison (Raw vs Legacy Static vs Misar)
    router = MisarRouter(model_key="arabertv02")
    comp = router.compare_baselines("توثيق البرمجه في المنصه هاذي دقيق وموثوق ومستعدين للإطلاق إن شاء الله")

    print("\n" + "-" * 80)
    print("3-Way Comparison on Orthography & Feminine Noun Preservation:")
    print("-" * 80)
    print(f"1. Raw Input    : {comp['misar']['original_text']}")
    print(f"2. Legacy Static: {comp['legacy_text']} (Tokens: {comp['legacy_token_count']}, UNK: {comp['legacy_unk_count']})")
    print(f"3. Misar v2+    : {comp['misar']['processed_text']} (Tokens: {comp['misar']['processed_token_count']}, Auto-Heal: {comp['misar']['auto_healed_count']})")
    print(f"-> Misar vs Legacy Gain: {comp['vs_legacy_reduction_pct']:+.1f}% fewer fragmented subwords")

if __name__ == "__main__":
    main()
