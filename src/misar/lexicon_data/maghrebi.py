# -*- coding: utf-8 -*-
"""
Maghrebi Curated Lexical Database (اللهجة المغاربية: المغربية، الجزائرية، التونسية).
"""

from typing import List
from ..lexicon import LexicalEntry

MAGHREBI_ENTRIES: List[LexicalEntry] = [
    LexicalEntry(
        word="بزاف",
        lemma_msa="كثيراً وفوق الكفاية",
        root="ج-ز-ف",
        dialect="maghrebi",
        pos="ظرف كمية وتوكيد مغاربي",
        definition="أشهر لفظ في المغرب والجزائر للدلالة على الكثرة المفرطة، من أصل فصيح: بالجزاف.",
        synonyms=["كثيراً", "جداً", "وايد", "كتير"],
        example="المشروع عجبني بزاف ومفيد جداً.",
    ),
    LexicalEntry(
        word="دابا",
        lemma_msa="الآن في هذه اللحظة",
        root="د-أ-ب",
        dialect="maghrebi",
        pos="ظرف زمان مغربي",
        definition="أشهر أداة زمنية في اللهجة المغربية تدل على اللحظة الحاضرة والسرعة، من دأباً.",
        synonyms=["الآن", "دلوقتي", "الحين", "هلق"],
        example="دابا غادي نبداو الاختبارات.",
    ),
    LexicalEntry(
        word="زوين",
        lemma_msa="جميل وحسن ورائع",
        root="ز-ي-ن",
        dialect="maghrebi",
        pos="صفة جمال واستحسان",
        definition="تطلق على الشيء الجميل الأنيق المتناسق، تصغير لكلمة 'زين'.",
        synonyms=["جميل", "رائع", "حسن", "أنيق"],
        example="تصميم الواجهة طالع زوين بزاف.",
    ),
    LexicalEntry(
        word="ديالنا",
        lemma_msa="خاص بنا وملكنا",
        root="ذ-و",
        dialect="maghrebi",
        pos="أداة نسبة وملكية مغربية",
        definition="تفيد نسبة الملكية والاختصاص للجماعة، أصلها من 'ذي' الإشارية والنسبة.",
        synonyms=["خاص بنا", "ملكنا", "حقنا"],
        example="هاد التطبيق ديالنا ومطور بأيدينا.",
    ),
    LexicalEntry(
        word="واش",
        lemma_msa="هل وماذا",
        root="ش-ي-ء",
        dialect="maghrebi",
        pos="أداة استفهام مغاربية",
        definition="أداة استفهام متعددة الاستخدام تفيد السؤال والاستفسار، منحوتة من 'وأي شيء'.",
        synonyms=["هل", "ماذا", "أ", "وش"],
        example="واش مستعدين للإطلاق الرسمي؟",
    ),
    LexicalEntry(
        word="مزيان",
        lemma_msa="جيد وممتاز ومتقن",
        root="ز-ي-ن",
        dialect="maghrebi",
        pos="صفة جودة مغاربية",
        definition="تفيد الجودة والإتقان والرضا عن الأداء في المغرب والجزائر، من زانَ.",
        synonyms=["جيد", "ممتاز", "طيب", "حسن"],
        example="الكود مكتوب بطريقة مزيانة وبدون أخطاء.",
    ),
    LexicalEntry(
        word="برشا",
        lemma_msa="كثيراً وبوفرة",
        root="ب-ر-ش",
        dialect="maghrebi",
        pos="ظرف كثرة تونسي",
        definition="أشهر لفظ في تونس للدلالة على الكثرة والوفرة، مأخوذة من الكثرة والبرشة.",
        synonyms=["كثيراً", "جداً", "بزاف", "وايد"],
        example="يعطيك الصحة عجبني برشا.",
    ),
    LexicalEntry(
        word="شنو",
        lemma_msa="ماذا وما هو",
        root="ش-ي-ء",
        dialect="maghrebi",
        pos="اسم استفهام مغاربي",
        definition="أداة استفهام شائعة في المغرب والجزائر وتونس للسؤال عن الماهية.",
        synonyms=["ماذا", "ما هو", "وش", "شو"],
        example="شنو الجديد في الإصدار القادم؟",
    ),
]
