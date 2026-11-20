# -*- coding: utf-8 -*-
"""
Example 09: Arabic In-Memory BM25 Semantic Search & Dialect Sentiment Analyzer
Demonstrates Misar v2.8.0 zero-dependency features for RAG retrieval and sentiment mining.
"""

from misar import (
    BM25SearchEngine,
    analyze_sentiment,
    benchmark_text,
    search_arabic,
)


def main():
    print("=" * 80)
    print("Misar v2.8.0 Example 09: In-Memory BM25 Search & Dialect Sentiment Analyzer")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Arabic In-Memory BM25 Search with Morphological Roots
    # ---------------------------------------------------------
    print("\n--- [1] Arabic Okapi BM25 Search with Morphological Root Matching ---")
    documents = [
        "تعتمد الحوسبة السحابية الحديثة على مراكز البيانات الضخمة لتسريع استدلال النماذج اللغوية.",
        "ابتكار محرك مِسار لمعالجة وتوجيه نصوص اللغة العربية ورفع كفاءة نماذج الذكاء الاصطناعي.",
        "تأمين الواجهات البرمجية وتشفير البيانات الحساسة وحجب هجمات حقن الأوامر والـ Jailbreak.",
        "استرجاع المستندات في قواعد البيانات وأنظمة البحث الدلالي الذكي RAG عبر الفهرسة المعجمية.",
    ]

    engine = BM25SearchEngine(use_roots=True, remove_stopwords=True)
    engine.add_documents(documents)
    print(f"Indexed {engine.total_docs} documents successfully.")

    query = "المعالجات اللغوية والذكاء الاصطناعي"
    print(f"Searching for: '{query}'")
    results = engine.search(query, top_k=2)

    for r in results:
        print(f"\n#Rank {r.rank} | Doc: {r.doc_id} | BM25 Score: {r.score:.4f}")
        print(f"  Matched Terms: {', '.join(r.matched_terms)}")
        print(f"  Snippet: {r.snippet}")

    # ---------------------------------------------------------
    # 2. Dialect-Aware Sentiment Analysis & Negation Inversion
    # ---------------------------------------------------------
    print("\n--- [2] Dialect Sentiment Analysis & Contextual Negation Handling ---")
    samples = [
        "الخدمة ممتازة وسريعة جداً وبيض الله وجهكم على هذا الإنجاز 😍🔥",
        "التطبيق مو حلو ويعلق وما نبي نجدد الاشتراك أبدددداً 😡👎",
        "ما قصرتوا ولا غلطة في الشغل يعطيكم العافية يا أبطال ✅",
        "النظام مش بطال وشغال تمام وبنعتمد التحديث الجديد إن شاء الله",
    ]

    for s in samples:
        res = analyze_sentiment(s)
        print(f"\nنص: {s}")
        print(f"  النتيجة: {res.label} (درجة القطبية: {res.score:+.2f}, الثقة: {res.confidence * 100:.0f}%)")
        print(f"  الانطباع: {res.emotion}")
        if res.positive_tokens:
            print(f"  كلمات إيجابية: {', '.join(res.positive_tokens)}")
        if res.negative_tokens:
            print(f"  كلمات سلبية/منفية: {', '.join(res.negative_tokens)}")
        if res.negations_detected:
            print(f"  أدوات نفي تم رصدها: {', '.join(res.negations_detected)}")

    # ---------------------------------------------------------
    # 3. Cross-Model 7-Tokenizer Benchmark
    # ---------------------------------------------------------
    print("\n--- [3] Real Cross-Model Token Optimization Benchmark ---")
    bench_text = "أداء الخادم الحين صار ممتاز واستجابة واجهة الـ API سريعة"
    report = benchmark_text(bench_text, offline=True)
    print(f"Benchmark for: '{bench_text}'")
    print(f"Average Token Reduction : {report['avg_token_reduction_pct']}%")
    print(f"Attention FLOPs Saved   : {report['avg_attention_savings_pct']}%")
    for r in report["results"]:
        if "reduction_pct" in r:
            print(f"  • {r['model_key']:<15}: {r['raw_tokens']} -> {r['processed_tokens']} tokens (-{r['reduction_pct']}%)")

    print("\n" + "=" * 80)
    print("Example 09 completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
