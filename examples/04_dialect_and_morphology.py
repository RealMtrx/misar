# -*- coding: utf-8 -*-
"""
Example 04: Dialect Classification, Semantic Lexicon & Root Extractor
Demonstrates sub-dialect classification, multi-dialect dictionary lookups,
trilateral root extraction, and full sentence deconstruction.
"""

from misar import classify_dialect, canonicalize, lookup_word, explain_text, extract_root

def main() -> None:
    print("=" * 80)
    print("Misar Example 04: Dialect Intelligence, Lexicon & Morphology")
    print("=" * 80)

    # 1. Microsecond Sub-Dialect & Register Classification
    dialect_text = "وش تبغى تسوي الحين يا خوي؟ تبي نروح سوا؟"
    clf = classify_dialect(dialect_text)
    print(f"Text: '{dialect_text}'")
    print(f"-> Primary Dialect : {clf.primary_dialect} (Confidence: {clf.confidence:.1%})")
    print(f"-> Discourse Reg   : {clf.register} (Confidence: {clf.register_confidence:.1%})")
    print(f"-> Detected Markers: {clf.detected_markers}")

    # 2. Semantic Dictionary Lookup
    print("\n" + "-" * 80)
    print("Dictionary Lookup (معجم مِسار الدلالي):")
    print("-" * 80)
    for word in ("أبخص", "بزاف", "دلوقتي", "يمعود", "ناهي"):
        entry = lookup_word(word)
        if entry:
            print(f"• [{entry.word}] ({entry.dialect}) -> الفصيح: '{entry.lemma_msa}' | الجذر: {entry.root} | {entry.definition}")

    # 3. Morphological Trilateral Root Extraction
    print("\n" + "-" * 80)
    print("Morphological Root Extraction (استخراج الجذور الصرفية):")
    print("-" * 80)
    words = ["استكشاف", "تطبيق", "مكتوب", "افتتاح", "كاتب", "عالم", "مشاريع"]
    for w in words:
        print(f"• {w:<12} -> الجذر: {extract_root(w)}")

    # 4. Sentence Deconstruction & Canonical MSA Reconstruction
    print("\n" + "-" * 80)
    print("Sentence Deconstruction & MSA Reconstruction:")
    print("-" * 80)
    sentence = "وش تبغى تسوي الحين يا يمعود؟ بزاف عليك الشغل!"
    explanation = explain_text(sentence)
    print(f"Original Text : {explanation.original_text}")
    print(f"Canonical MSA : {explanation.canonical_msa_text}")
    print(f"Dialect Stats : {explanation.dialect_tokens_count}/{explanation.total_tokens_count} tokens ({explanation.dialect_density_pct:.1f}%)")

if __name__ == "__main__":
    main()
