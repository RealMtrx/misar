# -*- coding: utf-8 -*-
"""
Example 02: Semantic Discourse RAG Chunking & Negation Binding
Demonstrates discourse-aware text splitting for Arabic vector databases and RAG systems,
ensuring negative particles are never severed from their predicates.
"""

from misar import chunk_document, MisarArabicSplitter

def main() -> None:
    print("=" * 80)
    print("Misar Example 02: Semantic Discourse RAG Chunker")
    print("=" * 80)

    document = """
    أعلنت الهيئة العامة للمنشآت الصغيرة والمتوسطة اعتماد حزمة الدعم الرقمي الجديدة للربع الأول.
    ومن الجدير بالذكر أن المؤشرات المالية السنوية أظهرت نمواً متسارعاً بنسبة تجاوزت خمسة وعشرين بالمائة.
    علاوة على ذلك وافق مجلس الإدارة بالإجماع على ميزانية التحول ولم يعترض أي عضو على المخصصات التقنية.
    بناءً على ذلك سيبدأ تنفيذ المشاريع المعتمدة مطلع الشهر القادم في كافة الفروع.
    """

    # 1. Native Discourse Chunking with exact character offsets & WordPiece token counts
    chunks = chunk_document(document.strip(), max_tokens=30, overlap_tokens=6)

    print(f"\nDocument split into {len(chunks)} semantic discourse chunks:")
    for idx, c in enumerate(chunks, 1):
        print(f"\n--- Chunk #{idx} [{c.start_char}:{c.end_char}] (Tokens: {c.token_count}, Route: {c.route_used}) ---")
        print(f"Text: {c.text}")

    # 2. LangChain Splitter compatibility
    print("\n" + "-" * 80)
    print("LangChain Drop-in Compatibility:")
    print("-" * 80)
    splitter = MisarArabicSplitter(max_tokens=64, overlap_tokens=8)
    docs = splitter.create_documents([document], metadatas=[{"source": "annual_economic_report.pdf"}])
    print(f"Created {len(docs)} LangChain Document objects with metadata: {docs[0]['metadata']}")

if __name__ == "__main__":
    main()
