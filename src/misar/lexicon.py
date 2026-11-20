# -*- coding: utf-8 -*-
"""
Misar Arabic Lexicon & Morphological Dictionary (معجم مِسار الدلالي والصرفي).

Provides high-speed zero-dependency lexical lookup, trilateral/quadrilateral root
extraction, multi-dialect translation to Standard Modern Arabic (MSA), part-of-speech
tagging, semantic definitions, and comprehensive sentence-level vocabulary explanation.

Architecture Note:
    - `misar.lexicon` (singular): High-level semantic dictionary (`lookup_word`),
      morphological root extraction (`extract_root`), and sentence explanation (`explain_text`).
    - `misar.lexicons` (plural): Low-level regex patterns, token filters, and orthographic
      normalization tables used internally by the router pipelines.

Supports dialects across:
- Saudi & Gulf (`saudi_gulf`): Najdi, Hijazi, Southern, Kuwaiti, Emirati
- Egyptian (`egyptian`): Cairene, Sa'idi
- Levantine (`levantine`): Syrian, Lebanese, Jordanian, Palestinian
- Maghrebi (`maghrebi`): Moroccan, Algerian, Tunisian
- Iraqi (`iraqi`): Baghdadi, Southern
- Yemeni (`yemeni`): Sana'ani, Ta'izzi, Adeni
- Modern Standard Arabic (`msa`)
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class LexicalEntry:
    """Full semantic and morphological dictionary entry for an Arabic word."""
    word: str
    lemma_msa: str
    root: str
    dialect: str
    pos: str
    definition: str
    synonyms: List[str] = field(default_factory=list)
    example: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "lemma_msa": self.lemma_msa,
            "root": self.root,
            "dialect": self.dialect,
            "pos": self.pos,
            "definition": self.definition,
            "synonyms": list(self.synonyms),
            "example": self.example,
        }


@dataclass
class TokenExplanation:
    """Detailed lexical breakdown for a single token in context."""
    token: str
    is_dialect: bool
    entry: Optional[LexicalEntry]
    normalized_token: str
    detected_root: str
    suggested_msa: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "is_dialect": self.is_dialect,
            "entry": self.entry.to_dict() if self.entry else None,
            "normalized_token": self.normalized_token,
            "detected_root": self.detected_root,
            "suggested_msa": self.suggested_msa,
        }


@dataclass
class TextExplanation:
    """Comprehensive semantic sentence explanation and dialect analysis."""
    original_text: str
    canonical_msa_text: str
    dialect_tokens_count: int
    total_tokens_count: int
    dialect_density_pct: float
    dominant_dialect: str
    tokens: List[TokenExplanation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "canonical_msa_text": self.canonical_msa_text,
            "dialect_tokens_count": self.dialect_tokens_count,
            "total_tokens_count": self.total_tokens_count,
            "dialect_density_pct": round(self.dialect_density_pct, 2),
            "dominant_dialect": self.dominant_dialect,
            "tokens": [t.to_dict() for t in self.tokens],
        }


# ---------------------------------------------------------------------------
# Morphological Light Stemmer & Root Extractor
# ---------------------------------------------------------------------------

_ARABIC_STRIP_REGEX = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_HAMZA_MAP = str.maketrans("إأآٱ", "اااا")

_COMPOUND_PREFIXES = (
    "وبال", "وفال", "وكال", "فبال", "ولل", "فلل",
    "وال", "فال", "بال", "كال", "لل",
    "ال",
)
_SINGLE_PREFIXES = ("و", "ف")

_COMMON_SUFFIXES = (
    "كما", "كم", "كن", "هما", "هم", "هن",
    "تكم", "تنا", "تم", "كن", "نا", "ني",
    "ها", "ات", "ون", "ين", "ان", "وا",
    "ه", "ي", "ك", "ة", "ت",
)


def normalize_for_morphology(word: str) -> str:
    """Strip diacritics, tatweel, and unify hamzas for morphological matching."""
    if not word:
        return ""
    w = _ARABIC_STRIP_REGEX.sub("", word)
    w = w.translate(_HAMZA_MAP)
    return w.strip()


def light_stem(word: str) -> str:
    """
    High-speed deterministic Arabic light stemmer.
    Strips grammatical proclitics and inflectional enclitics while protecting 3-letter roots.
    """
    clean = normalize_for_morphology(word)
    if len(clean) <= 3:
        return clean

    # 1. Strip compound proclitics (ال, وال, بال, لل, etc.)
    for prefix in _COMPOUND_PREFIXES:
        if clean.startswith(prefix) and len(clean) - len(prefix) >= 3:
            clean = clean[len(prefix):]
            break

    # 2. Strip single conjunction proclitics (و, ف)
    if clean.startswith(_SINGLE_PREFIXES) and len(clean) >= 4:
        # Don't strip if root starts with و (e.g., وصل, وقف, وزن)
        if clean[0] == "و" and clean[1] not in ("ا", "ي", "ت", "ن", "م"):
            pass  # Protect initial waw verbs/nouns unless followed by prefix
        elif clean[0] == "ف" and clean[1] not in ("ا", "ي", "ت", "ن", "م"):
            pass  # Protect initial faa verbs/nouns (فتح, فحص, فصل)
        else:
            clean = clean[1:]

    # 3. Strip suffixes
    for suffix in _COMMON_SUFFIXES:
        if clean.endswith(suffix) and len(clean) - len(suffix) >= 3:
            clean = clean[:-len(suffix)]
            break

    return clean


def extract_root(word: str) -> str:
    """
    Extract best-effort trilateral or quadrilateral Arabic root candidates.
    Uses pattern decomposition without external dictionary binaries.
    """
    stem = light_stem(word)
    if len(stem) == 3:
        return f"{stem[0]}-{stem[1]}-{stem[2]}"
    if len(stem) == 4:
        # Check if first letter is a weak augment (م، ت، ي، ن)
        if stem[0] in ("م", "ت", "ي", "ن", "ا") and stem[1] != stem[2]:
            return f"{stem[1]}-{stem[2]}-{stem[3]}"
        # If 2nd letter is Alef: فاعل (كاتب -> ك-ت-ب)
        if stem[1] == "ا":
            return f"{stem[0]}-{stem[2]}-{stem[3]}"
        # Or quadrilateral root (e.g., زلزل, دحرج)
        return f"{stem[0]}-{stem[1]}-{stem[2]}-{stem[3]}"

    if len(stem) >= 5:
        # e.g., استفعال / مستفعل (استكشاف -> ك-ش-ف, مستوصف -> و-ص-ف)
        if (stem.startswith("است") or stem.startswith("مست")) and len(stem) >= 6:
            r = stem[3:]
            if len(r) == 4 and r[2] == "ا":
                return f"{r[0]}-{r[1]}-{r[3]}"
            if len(r) >= 3:
                return f"{r[0]}-{r[1]}-{r[2]}"
        # e.g., تفعيل (تطبيق -> ط-ب-ق)
        if stem.startswith("ت") and stem[3] == "ي" and len(stem) == 5:
            return f"{stem[1]}-{stem[2]}-{stem[4]}"
        # e.g., مفعول (مكتوب -> ك-ت-ب)
        if stem.startswith("م") and stem[3] == "و" and len(stem) == 5:
            return f"{stem[1]}-{stem[2]}-{stem[4]}"
        # e.g., انفعال (انكسار -> ك-س-ر)
        if stem.startswith("ان") and len(stem) == 5:
            return f"{stem[2]}-{stem[3]}-{stem[4]}"
        if stem.startswith("ان") and len(stem) == 6 and stem[4] == "ا":
            return f"{stem[2]}-{stem[3]}-{stem[5]}"
        # e.g., افتعال (افتتاح -> ف-ت-ح)
        if stem[0] in ("ا", "ي", "ت") and stem[2] == "ت" and len(stem) >= 5:
            if len(stem) == 6 and stem[4] == "ا":
                return f"{stem[1]}-{stem[3]}-{stem[5]}"
            return f"{stem[1]}-{stem[3]}-{stem[4]}"
        # e.g., مفتعل (مشترك -> ش-ر-ك)
        if stem[0] == "م" and stem[2] == "ت" and len(stem) == 5:
            return f"{stem[1]}-{stem[3]}-{stem[4]}"

    # Fallback trilateral truncation
    chars = [c for c in stem if c not in ("ا", "و", "ي", "ت", "م", "ن")]
    if len(chars) >= 3:
        return f"{chars[0]}-{chars[1]}-{chars[2]}"

    letters = list(stem[:3])
    while len(letters) < 3:
        letters.append("ء")
    return f"{letters[0]}-{letters[1]}-{letters[2]}"


# ---------------------------------------------------------------------------
# Comprehensive Multi-Dialect Lexical Knowledge Base
# ---------------------------------------------------------------------------

_CURATED_ENTRIES: List[LexicalEntry] = [
    # ---------------- Saudi & Gulf Dialect (نجدي، حجازي، كويتي، إماراتي، جنوبي) ----------------
    LexicalEntry(
        word="أبخص",
        lemma_msa="أعلم وأدرى",
        root="ب-خ-ص",
        dialect="saudi_gulf",
        pos="اسم تفضيل",
        definition="الأكثر دراية وبصيرة بالشيء وبواطنه؛ تقال في المثل 'أهل مكة أبخص بشعابها'.",
        synonyms=["أعلم", "أدرى", "أخبر", "أعرف"],
        example="أنا أبخص بشغلي ولا تشيل هم.",
    ),
    LexicalEntry(
        word="استبينا",
        lemma_msa="اتفقنا وفهمنا",
        root="ب-ي-ن",
        dialect="saudi_gulf",
        pos="فعل ماض مركب",
        definition="صيغة تفيد الرضا والاتفاق التام ووضوح الأمر بين المتحدثين.",
        synonyms=["اتفقنا", "فهمنا", "خلصنا", "صار واضحاً"],
        example="خلاص نلتقي بكرة بعد المغرب واستبينا.",
    ),
    LexicalEntry(
        word="تكفى",
        lemma_msa="أرجوك والرجاء",
        root="ك-ف-ي",
        dialect="saudi_gulf",
        pos="أسلوب رجاء واستعطاف",
        definition="كلمة نخوة ورجاء حار تُقال لدفع الشخص إلى قضاء حاجة أو المساعدة.",
        synonyms=["أرجوك", "أستحلفك", "من فضلك", "بالله عليك"],
        example="تكفى يا فلان ساعدني في حل هالمشكلة.",
    ),
    LexicalEntry(
        word="سالفة",
        lemma_msa="قصة وموضوع",
        root="س-ل-ف",
        dialect="saudi_gulf",
        pos="اسم",
        definition="الحديث أو الموضوع أو القصة الواقعة مما سلف من الأخبار.",
        synonyms=["قصة", "موضوع", "خبر", "قضية", "شأن"],
        example="عطنا سالفة المشروع الجديد وإيش صار فيه.",
    ),
    LexicalEntry(
        word="دحين",
        lemma_msa="الآن",
        root="ح-ي-ن",
        dialect="saudi_gulf",
        pos="ظرف زمان",
        definition="الوقت الحاضر في اللهجة الحجازية خصوصاً.",
        synonyms=["الآن", "في هذه اللحظة", "حالياً"],
        example="أنا طالع من المكتب دحين.",
    ),
    LexicalEntry(
        word="الحين",
        lemma_msa="الآن",
        root="ح-ي-ن",
        dialect="saudi_gulf",
        pos="ظرف زمان",
        definition="تدل على اللحظة الآنية في نجد والخليج عامة.",
        synonyms=["الآن", "هذه الساعة", "حالياً"],
        example="خلنا نبدأ الحين بدون تأخير.",
    ),
    LexicalEntry(
        word="أبغى",
        lemma_msa="أريد وأطلب",
        root="ب-غ-ي",
        dialect="saudi_gulf",
        pos="فعل مضارع",
        definition="تفيد الرغبة والطلب المباشر في لهجات الجزيرة العربية.",
        synonyms=["أريد", "أرغب", "أطلب", "أود"],
        example="أبغى تقرير الأداء قبل نهاية الدوام.",
    ),
    LexicalEntry(
        word="تبي",
        lemma_msa="تريد",
        root="أ-ب-ي",
        dialect="saudi_gulf",
        pos="فعل مضارع",
        definition="السؤال عن رغبة المخاطب أو طلب الشيء.",
        synonyms=["تريد", "ترغب", "تطلب"],
        example="إذا تبي مساعدة علمنا وحنا بالخدمة.",
    ),
    LexicalEntry(
        word="أرحب",
        lemma_msa="أهلاً وسهلاً",
        root="ر-ح-ب",
        dialect="saudi_gulf",
        pos="عبارة ترحيب",
        definition="تحية ترحيبية عربية أصيلة تعبر عن السعة والحفاوة بالضيف والمقبل.",
        synonyms=["مرحباً", "أهلاً وسهلاً", "حياك الله"],
        example="أرحب يا بو فهد نورت المكان.",
    ),
    LexicalEntry(
        word="بالحيل",
        lemma_msa="كثيراً وجداً",
        root="ح-ي-ل",
        dialect="saudi_gulf",
        pos="ظرف مبالغة",
        definition="تفيد المبالغة والشدة والكثرة، مشتقة من الحَوْل والقوة.",
        synonyms=["كثيراً", "جداً", "للغاية", "بشدة"],
        example="الجو اليوم بارد بالحيل.",
    ),
    LexicalEntry(
        word="هقوتي",
        lemma_msa="ظني واعتقادي",
        root="ه-ق-و",
        dialect="saudi_gulf",
        pos="اسم/ظن",
        definition="الظن والتوقع في اللهجة النجدية والبدوية.",
        synonyms=["ظني", "حسباني", "تقديري", "اعتقادي"],
        example="هقوتي إنه بيوصل قبل العصر.",
    ),
    LexicalEntry(
        word="يبيلها",
        lemma_msa="يلزمها ويحتاج إليها",
        root="ب-ي-ل",
        dialect="saudi_gulf",
        pos="تعبير لزوم",
        definition="تفيد حاجة الأمر إلى وقت أو جهد أو معالجة خاصة.",
        synonyms=["يلزمها", "تتطلب", "تحتاج إلى", "ينبغي لها"],
        example="المسألة يبيلها تركيز وتأني.",
    ),
    LexicalEntry(
        word="ماكو",
        lemma_msa="لا يوجد",
        root="ك-و-ن",
        dialect="saudi_gulf",
        pos="أداة نفي وجود",
        definition="نفي الوجود في اللهجة الكويتية والخليجية والعراقية (مركبة من: ما + يكون).",
        synonyms=["لا يوجد", "منعدم", "ليس هناك"],
        example="ماكو أي مشكلة في النظام اليوم.",
    ),
    LexicalEntry(
        word="مافيه",
        lemma_msa="لا يوجد",
        root="ف-ي-ه",
        dialect="saudi_gulf",
        pos="أداة نفي وجود",
        definition="نفي الوجود العام المستعمل في معظم أرجاء السعودية والخليج.",
        synonyms=["لا يوجد", "ليس هناك", "غير موجود"],
        example="مافيه أي عائق أمام إتمام المعاملة.",
    ),
    LexicalEntry(
        word="طقطقة",
        lemma_msa="مزاح ودعابة",
        root="ط-ق-ق",
        dialect="saudi_gulf",
        pos="اسم/مصدر",
        definition="المزاح الخفيف والمداعبة الكلامية أو السخرية الودية غير المؤذية.",
        synonyms=["مزاح", "دعابة", "ملاطفة", "مداعبة"],
        example="كنا نسولف على سبيل الطقطقة فقط.",
    ),
    LexicalEntry(
        word="خرابيط",
        lemma_msa="ترهات وأباطيل",
        root="خ-ر-ب",
        dialect="saudi_gulf",
        pos="اسم جمع",
        definition="الكلام الذي لا فائدة منه، أو الأشياء غير المنظمة والفوضوية.",
        synonyms=["ترهات", "أباطيل", "فوضى", "توافه"],
        example="اترك هالخرابيط وركز في المهمة الأساسية.",
    ),
    LexicalEntry(
        word="كفو",
        lemma_msa="شهم وكفء",
        root="ك-ف-ء",
        dialect="saudi_gulf",
        pos="صفة مدح",
        definition="لفظ ثناء وتشجيع يقال لمن يتصف بالشهامة والكفاءة وحسن التصرف.",
        synonyms=["شهم", "كفء", "بطل", "جدير بالاحترام"],
        example="كفو والله ما قصرت بيض الله وجهك.",
    ),
    LexicalEntry(
        word="يزين",
        lemma_msa="ما أجمل وأحسن",
        root="ز-ي-ن",
        dialect="saudi_gulf",
        pos="تعبير استحسان",
        definition="صيغة تعجب واستحسان لحسن الشيء وجماله.",
        synonyms=["ما أجمل", "حبذا", "ما أحسن"],
        example="يزين الهدوء والعمل المنظم.",
    ),
    LexicalEntry(
        word="عطني",
        lemma_msa="أعطني وقدم لي",
        root="ع-ط-و",
        dialect="saudi_gulf",
        pos="فعل أمر",
        definition="صيغة أمر دارجة لطلب الشيء أو الحصول على المعلومة.",
        synonyms=["أعطني", "قدم لي", "زودني بـ"],
        example="عطني ملخص الاجتماع السابق.",
    ),
    LexicalEntry(
        word="فزعة",
        lemma_msa="نجدة ومعونة سريعة",
        root="ف-ز-ع",
        dialect="saudi_gulf",
        pos="اسم",
        definition="المبادرة العاجلة لمساعدة المحتاج ونصرته انطلاقاً من المروءة.",
        synonyms=["نجدة", "إغاثة", "معونة", "مساعدة فورية"],
        example="أشكر الجميع على الفزعة الطيبة.",
    ),
    LexicalEntry(
        word="هبد",
        lemma_msa="كلام عشوائي غير موثق",
        root="ه-ب-د",
        dialect="saudi_gulf",
        pos="اسم/مصدر",
        definition="التحدث برأي أو معلومة جازمة بغير علم أو تثبت؛ الارتجال غير المبرر.",
        synonyms=["كلام عشوائي", "تخبيص", "تقول بغير علم", "تكهن"],
        example="المعلومات هذي مجرد هبد ولا أساس لها من الصحة.",
    ),
    LexicalEntry(
        word="قروشة",
        lemma_msa="إزعاج وتشويش",
        root="ق-ر-ش",
        dialect="saudi_gulf",
        pos="اسم",
        definition="حالة الإزعاج والربكة الذهنية أو كثرة الإجراءات المعقدة.",
        synonyms=["إزعاج", "ارتباك", "تشويش", "صداع"],
        example="بلاش قروشة وخلنا نركز في الهدف.",
    ),
    LexicalEntry(
        word="وش",
        lemma_msa="ماذا وما هو",
        root="ش-ي-ء",
        dialect="saudi_gulf",
        pos="أداة استفهام",
        definition="أداة استفهام للاستفسار عن الماهية (مخففة من: وأي شيء).",
        synonyms=["ماذا", "ما هو", "أي شيء"],
        example="وش الموعد المناسب للاجتماع؟",
    ),
    LexicalEntry(
        word="شلونك",
        lemma_msa="كيف حالك",
        root="ل-و-ن",
        dialect="saudi_gulf",
        pos="سؤال وتحية",
        definition="السؤال عن الصحة والحال (أصلها: ما هو لون حالك).",
        synonyms=["كيف حالك", "كيف صحتك", "ما أخبارك"],
        example="شلونك يا خوي عساك بخير؟",
    ),
    LexicalEntry(
        word="عشان",
        lemma_msa="لأجل وبسبب",
        root="ع-ش-ن",
        dialect="saudi_gulf",
        pos="حرف تعليل",
        definition="بيان العلة والسبب (منحوتة من: على شأن).",
        synonyms=["لأجل", "بسبب", "بغية", "كي"],
        example="حضرنا بدري عشان نخلص كل الشغل.",
    ),
    LexicalEntry(
        word="مو",
        lemma_msa="ليس وغير",
        root="ل-ي-س",
        dialect="saudi_gulf",
        pos="أداة نفي",
        definition="نفي الصفة أو الخبر أو الاسم في مختلف اللهجات العربية.",
        synonyms=["ليس", "غير", "لا"],
        example="القرار مو نهائي وقابل للمراجعة.",
    ),
    LexicalEntry(
        word="ودنا",
        lemma_msa="نريد ونرغب",
        root="و-د-د",
        dialect="saudi_gulf",
        pos="فعل رغبة",
        definition="التعبير عن إرادة الجماعة ومحبتهم لحصول الأمر.",
        synonyms=["نريد", "نرغب", "نود", "نتمنى"],
        example="ودنا نبدأ المرحلة الثانية الأسبوع المقبل.",
    ),
    LexicalEntry(
        word="تبغى",
        lemma_msa="تريد وتطلب",
        root="ب-غ-ي",
        dialect="saudi_gulf",
        pos="فعل مضارع",
        definition="تفيد رغبة المخاطب وإرادته في الجزيرة والخليج.",
        synonyms=["تريد", "ترغب", "تطلب", "تود"],
        example="إذا تبغى تبدأ المشروع إحنا مستعدين.",
    ),
    LexicalEntry(
        word="يبغى",
        lemma_msa="يريد ويطلب",
        root="ب-غ-ي",
        dialect="saudi_gulf",
        pos="فعل مضارع",
        definition="تفيد إرادة الغائب وطلبه للشيء.",
        synonyms=["يريد", "يرغب", "يطلب"],
        example="المدير يبغى التقرير النهائي اليوم.",
    ),
    LexicalEntry(
        word="تسوي",
        lemma_msa="تفعل وتصنع",
        root="س-و-ي",
        dialect="saudi_gulf",
        pos="فعل مضارع",
        definition="تفيد إنجاز الفعل والعمل وصنعه في الخليج والجزيرة.",
        synonyms=["تفعل", "تصنع", "تنجز", "تعمل"],
        example="وش ناوي تسوي في المرحلة القادمة؟",
    ),
    LexicalEntry(
        word="يسوي",
        lemma_msa="يفعل ويصنع",
        root="س-و-ي",
        dialect="saudi_gulf",
        pos="فعل مضارع",
        definition="تفيد قيام الغائب بالفعل والصنع والإنجاز.",
        synonyms=["يفعل", "يصنع", "يعمل", "ينجز"],
        example="الفريق يسوي اختبارات الجودة باستمرار.",
    ),
    LexicalEntry(
        word="نسوي",
        lemma_msa="نفعل ونصنع",
        root="س-و-ي",
        dialect="saudi_gulf",
        pos="فعل مضارع",
        definition="تفيد قيام المتكلمين بالعمل والإنجاز.",
        synonyms=["نفعل", "نصنع", "نعمل", "ننجز"],
        example="خلنا نسوي مراجعة شاملة قبل الإطلاق.",
    ),
    LexicalEntry(
        word="أبي",
        lemma_msa="أريد",
        root="أ-ب-ي",
        dialect="saudi_gulf",
        pos="فعل رغبة",
        definition="تفيد إرادة المتكلم وطلبه في نجد والخليج.",
        synonyms=["أريد", "أرغب", "أطلب"],
        example="أبي أتأكد من صحة الحسابات.",
    ),
    LexicalEntry(
        word="نبي",
        lemma_msa="نريد",
        root="أ-ب-ي",
        dialect="saudi_gulf",
        pos="فعل رغبة",
        definition="تفيد إرادة المتكلمين والطلب المشترك.",
        synonyms=["نريد", "نرغب", "نود"],
        example="نبي نطور استيعاب الكلمات في المنظومة.",
    ),
    LexicalEntry(
        word="ترا",
        lemma_msa="اعلم وافهم",
        root="ر-أ-ي",
        dialect="saudi_gulf",
        pos="أداة تنبيه",
        definition="أداة لفت انتباه وتوكيد للمخاطب (أصلها: ترى).",
        synonyms=["اعلم", "انتبه", "تذكر"],
        example="ترا الاجتماع تأجل إلى الغد.",
    ),
    LexicalEntry(
        word="شف",
        lemma_msa="انظر وتأمل",
        root="ش-و-ف",
        dialect="saudi_gulf",
        pos="فعل أمر",
        definition="الحث على المشاهدة والتأمل وفحص الأمر (تخفيف: شُفْ).",
        synonyms=["انظر", "تأمل", "طالع", "افحص"],
        example="شف النتيجة وقول لي رأيك.",
    ),

    # ---------------- Egyptian Dialect (اللهجة المصرية) ----------------
    LexicalEntry(
        word="دلوقتي",
        lemma_msa="الآن",
        root="و-ق-ت",
        dialect="egyptian",
        pos="ظرف زمان",
        definition="الوقت الحاضر في اللهجة المصرية (أصلها: ده الوقت).",
        synonyms=["الآن", "في هذا الوقت", "حالياً"],
        example="الخدمة اشتغلت دلوقتي بدون مشاكل.",
    ),
    LexicalEntry(
        word="عايز",
        lemma_msa="أريد",
        root="ع-و-ز",
        dialect="egyptian",
        pos="اسم فاعل/طلب",
        definition="تفيد الإرادة والحاجة والرغبة في الفعل أو الشيء.",
        synonyms=["أريد", "أحتاج", "أرغب", "أطلب"],
        example="أنا عايز نسخة من العقد الموقع.",
    ),
    LexicalEntry(
        word="ازيك",
        lemma_msa="كيف حالك",
        root="ز-ي-ي",
        dialect="egyptian",
        pos="سؤال وتحية",
        definition="السؤال عن الحال في مصر (أصلها: إيش زيّك، أي كيف هيئتك).",
        synonyms=["كيف حالك", "كيف صحتك", "ما أخبارك"],
        example="ازيك يا باشا عامل ايه في الشغل؟",
    ),
    LexicalEntry(
        word="جدع",
        lemma_msa="شهم وشجاع",
        root="ج-د-ع",
        dialect="egyptian",
        pos="صفة مدح",
        definition="الشخص الشهم صاحب النخوة والمواقف النبيلة الموثوقة.",
        synonyms=["شهم", "صاحب واجب", "شجاع", "أصيل"],
        example="فلان ده راجل جدع ومواقفه تشهد له.",
    ),
    LexicalEntry(
        word="شغال",
        lemma_msa="يعمل وقائم",
        root="ش-غ-ل",
        dialect="egyptian",
        pos="صفة/حال",
        definition="تفيد استمرار العمل أو سريان النظام والخدمة بصورة طبيعية.",
        synonyms=["يعمل", "قائم", "سارٍ", "فعال"],
        example="السيرفر شغال تمام ومفيش أي انقطاع.",
    ),
    LexicalEntry(
        word="مفيش",
        lemma_msa="لا يوجد",
        root="ف-ي-ش",
        dialect="egyptian",
        pos="أداة نفي وجود",
        definition="نفي الوجود (منحوتة من: ما فيه شيء).",
        synonyms=["لا يوجد", "منعدم", "ليس هناك"],
        example="مفيش أي تعارض في البيانات المدخلة.",
    ),
    LexicalEntry(
        word="بتاع",
        lemma_msa="خاص بـ وملك لـ",
        root="ت-و-ع",
        dialect="egyptian",
        pos="أداة ملكية ونسبة",
        definition="تفيد نسبة الشيء إلى مالكه أو اختصاصه.",
        synonyms=["خاص بـ", "عائد إلى", "ملك لـ"],
        example="الملف بتاع الحسابات جاهز للمراجعة.",
    ),
    LexicalEntry(
        word="أوي",
        lemma_msa="جداً وكثيراً",
        root="ق-و-ي",
        dialect="egyptian",
        pos="ظرف مبالغة",
        definition="تفيد شدة الصفة والكثرة (تخفيف لكلمة قوي).",
        synonyms=["جداً", "للغاية", "كثيراً", "بشدة"],
        example="النظام سريع أوي بعد التحديث الأخير.",
    ),
    LexicalEntry(
        word="كده",
        lemma_msa="هكذا ومثل ذلك",
        root="ك-ذ-ا",
        dialect="egyptian",
        pos="اسم إشارة وتشبيه",
        definition="الدلالة على الكيفية أو الهيئة الموصوفة (تخفيف لكلمة كذا).",
        synonyms=["هكذا", "مثل هذا", "بهذه الطريقة"],
        example="التنظيم لما يكون كده يسهل الإنجاز.",
    ),
    LexicalEntry(
        word="معلش",
        lemma_msa="عذراً ولا تثريب",
        root="ع-ل-ش",
        dialect="egyptian",
        pos="تعبير اعتذار ومواساة",
        definition="التخفيف عن الشخص وطلب المسامحة (منحوتة من: ما عليه شيء).",
        synonyms=["عذراً", "سامحني", "لا بأس", "لا عليك"],
        example="معلش على التأخير كان الطريق زحمة.",
    ),
    LexicalEntry(
        word="برضه",
        lemma_msa="أيضاً وكذلك",
        root="ر-ض-ي",
        dialect="egyptian",
        pos="ظرف توكيد",
        definition="تفيد المشاركة وتأكيد الرأي أو الفعل (أصلها: برضاه).",
        synonyms=["أيضاً", "كذلك", "على كل حال"],
        example="برضه لازم نتأكد من مطابقة المعايير.",
    ),
    LexicalEntry(
        word="بلاش",
        lemma_msa="دعك منه ومجاناً",
        root="ل-ا-ش",
        dialect="egyptian",
        pos="نهي واستبعاد",
        definition="النهي عن فعل الشيء أو الدلالة على المجانية (من: بلا شيء).",
        synonyms=["لا داعي", "اترك", "مجاناً", "دعك من"],
        example="بلاش استعجال وخلينا نراجع الخطوات.",
    ),
    LexicalEntry(
        word="النهارده",
        lemma_msa="اليوم",
        root="ن-ه-ر",
        dialect="egyptian",
        pos="ظرف زمان",
        definition="اليوم الحاضر (أصلها: النهار ده).",
        synonyms=["اليوم", "هذا اليوم", "حالياً"],
        example="الاجتماع هيتعقد النهارده الساعة عشرة.",
    ),
    LexicalEntry(
        word="امبارح",
        lemma_msa="أمس والبارحة",
        root="ب-ر-ح",
        dialect="egyptian",
        pos="ظرف زمان",
        definition="اليوم السابق المنصرم (أصلها: البارحة).",
        synonyms=["أمس", "البارحة", "اليوم الماضي"],
        example="خلصنا فحص قواعد البيانات امبارح.",
    ),
    LexicalEntry(
        word="فين",
        lemma_msa="أين",
        root="أ-ي-ن",
        dialect="egyptian",
        pos="أداة استفهام",
        definition="السؤال عن مكان تواجد الشيء أو الشخص (منحوتة من: في أين).",
        synonyms=["أين", "في أي مكان"],
        example="فين التقرير المالي بتاع الربع الأول؟",
    ),

    # ---------------- Levantine Dialect (اللهجة الشامية: سورية، لبنانية، أردنية، فلسطينية) ----------------
    LexicalEntry(
        word="شو",
        lemma_msa="ماذا وما هو",
        root="ش-ي-ء",
        dialect="levantine",
        pos="أداة استفهام",
        definition="الاستفهام عن الشيء والماهية (مخففة من: أي شيء هو).",
        synonyms=["ماذا", "ما هو", "أي شيء"],
        example="شو رأيك بالخطة المقترحة؟",
    ),
    LexicalEntry(
        word="بدي",
        lemma_msa="أريد وأرغب",
        root="و-د-د",
        dialect="levantine",
        pos="فعل رغبة",
        definition="تفيد الطلب والرغبة والإرادة (من: في ودّي).",
        synonyms=["أريد", "أرغب", "أود", "أحتاج"],
        example="بدي أحكي معك بموضوع مهم.",
    ),
    LexicalEntry(
        word="هيك",
        lemma_msa="هكذا ومثل ذلك",
        root="ه-ك-ذ",
        dialect="levantine",
        pos="اسم إشارة وتشبيه",
        definition="الإشارة إلى الكيفية أو الهيئة (مخففة من: هكذا).",
        synonyms=["هكذا", "بهذا الشكل", "مثل ذلك"],
        example="الترتيب هيك ممتاز وما بدو تعديل.",
    ),
    LexicalEntry(
        word="هلق",
        lemma_msa="الآن",
        root="و-ق-ت",
        dialect="levantine",
        pos="ظرف زمان",
        definition="الوقت الحاضر في الشام ولبنان (مخففة من: هذا الوقت / هالوقت).",
        synonyms=["الآن", "في هذه اللحظة", "حالياً"],
        example="رح نبلش الاجتماع هلق بدون تأخير.",
    ),
    LexicalEntry(
        word="منيح",
        lemma_msa="جيد وطيب",
        root="م-ن-ح",
        dialect="levantine",
        pos="صفة",
        definition="تدل على الحسن والجودة والسلامة (مشتقة من المنيحة والنعمة).",
        synonyms=["جيد", "حسن", "طيب", "ممتاز"],
        example="الأداء منيح كتير وتطور عن أول.",
    ),
    LexicalEntry(
        word="زلمة",
        lemma_msa="رجل وشهم",
        root="ز-ل-م",
        dialect="levantine",
        pos="اسم",
        definition="الرجل الشجاع ذو المروءة (فصيحة قديمة: رجل زَلَمة).",
        synonyms=["رجل", "شخص", "شهم", "إنسان"],
        example="هاد زلمة محترم وقد كلمته.",
    ),
    LexicalEntry(
        word="كتير",
        lemma_msa="كثيراً وجداً",
        root="ك-ث-ر",
        dialect="levantine",
        pos="ظرف مبالغة",
        definition="تفيد الكثرة في العدد أو الشدة في الوصف (تخفيف: كثير).",
        synonyms=["كثيراً", "جداً", "للغاية"],
        example="شغلك مرتب وكتير واضح.",
    ),
    LexicalEntry(
        word="عم",
        lemma_msa="في حالة استمرار",
        root="ع-م-م",
        dialect="levantine",
        pos="سابقة استمرار",
        definition="سابقة تدخل على الفعل المضارع لإفادة وقوعه المستمر في الحاضر.",
        synonyms=["مستمر في", "يقوم بـ", "حالياً يفعل"],
        example="نحنا عم نشتغل على التحديث الجديد.",
    ),
    LexicalEntry(
        word="مزبوط",
        lemma_msa="صحيح ودقيق",
        root="ض-ب-ط",
        dialect="levantine",
        pos="صفة",
        definition="الشيء المحكم الصحيح المطابق للواقع (أصلها: مضبوط).",
        synonyms=["صحيح", "دقيق", "محكم", "موافق للحق"],
        example="كلامك مزبوط مية بالمية.",
    ),
    LexicalEntry(
        word="قديش",
        lemma_msa="كم المقدار",
        root="ق-د-ر",
        dialect="levantine",
        pos="أداة استفهام",
        definition="السؤال عن الكمية أو السعر أو المدة (منحوتة من: قدر أي شيء).",
        synonyms=["كم", "ما مقدار", "ما كلفة"],
        example="قديش بياخد وقت التحميل؟",
    ),
    LexicalEntry(
        word="بلش",
        lemma_msa="بدأ وشرع",
        root="ب-ل-ش",
        dialect="levantine",
        pos="فعل ماض",
        definition="الشروع في الفعل والبدء به في بلاد الشام.",
        synonyms=["بدأ", "شرع", "انطلق في"],
        example="الفريق بلش في كتابة الكود.",
    ),
    LexicalEntry(
        word="مشان",
        lemma_msa="لأجل وبسبب",
        root="ش-أ-ن",
        dialect="levantine",
        pos="حرف تعليل",
        definition="بيان العلة والهدف (من: من شأن).",
        synonyms=["لأجل", "بسبب", "في سبيل", "كي"],
        example="سوينا التدقيق مشان نضمن الأمان.",
    ),

    # ---------------- Maghrebi Dialect (اللهجة المغاربية: مغربية، جزائرية، تونسية) ----------------
    LexicalEntry(
        word="ديال",
        lemma_msa="خاص بـ وملك لـ",
        root="د-ي-ل",
        dialect="maghrebi",
        pos="أداة نسبة وتملك",
        definition="أداة إضافة تفيد ملكية الشيء واختصاصه في المغرب والجزائر.",
        synonyms=["خاص بـ", "ملك لـ", "تابع لـ"],
        example="هاد المشروع ديال الشركة ديالنا.",
    ),
    LexicalEntry(
        word="بزاف",
        lemma_msa="كثيراً وفوق الكفاية",
        root="ج-ز-ف",
        dialect="maghrebi",
        pos="ظرف مبالغة",
        definition="تفيد الكثرة الشديدة في الكم أو الكيف (من أصل فصيح: بالجزاف).",
        synonyms=["كثيراً", "جداً", "بإفراط", "للغاية"],
        example="شكراً بزاف على المجهود الرائع.",
    ),
    LexicalEntry(
        word="دابا",
        lemma_msa="الآن وفي الحال",
        root="د-أ-ب",
        dialect="maghrebi",
        pos="ظرف زمان",
        definition="الوقت الحاضر الفوري (أصلها: دأباً أي باستمرار وفوراً).",
        synonyms=["الآن", "فوراً", "في هذا الحين"],
        example="خاصنا نبداو الاجتماع دابا.",
    ),
    LexicalEntry(
        word="زوين",
        lemma_msa="جميل وحسن",
        root="ز-ي-ن",
        dialect="maghrebi",
        pos="صفة مدح",
        definition="تصغير تمليح لـ (زَيْن) وتفيد الجمال والأناقة وحسن المنظر والعمل.",
        synonyms=["جميل", "حسن", "رائع", "مليح"],
        example="التصميم زوين بزاف وعصري.",
    ),
    LexicalEntry(
        word="مزيان",
        lemma_msa="ممتاز وجيد",
        root="ز-ي-ن",
        dialect="maghrebi",
        pos="صفة",
        definition="تدل على الإتقان والرضا والصلاح (أصلها: مزيان/مزين).",
        synonyms=["جيد", "ممتاز", "صالح", "طيب"],
        example="كلشي غادي مزيان وبلا صعوبات.",
    ),
    LexicalEntry(
        word="واش",
        lemma_msa="هل وماذا",
        root="أ-ي-ش",
        dialect="maghrebi",
        pos="أداة استفهام",
        definition="أداة استفهام تستخدم للطلب والتحقق والتصديق (منحوتة من: وأي شيء).",
        synonyms=["هل", "أ", "ماذا"],
        example="واش ساليتي الشغل ديال اليوم؟",
    ),
    LexicalEntry(
        word="علاش",
        lemma_msa="لماذا وما السبب",
        root="ل-ي-ش",
        dialect="maghrebi",
        pos="أداة استفهام",
        definition="السؤال عن السبب والعلة (منحوتة من: على أي شيء).",
        synonyms=["لماذا", "ما السبب", "لأي غرض"],
        example="علاش ما حضر المدير للاجتماع؟",
    ),
    LexicalEntry(
        word="برشا",
        lemma_msa="كثيراً وجداً",
        root="ب-ر-ش",
        dialect="maghrebi",
        pos="ظرف مبالغة",
        definition="الكلمة التونسية الشهيرة الدالة على الكثرة والوفرة.",
        synonyms=["كثيراً", "جداً", "بوفرة"],
        example="فرحانين برشا بالنتيجة الممتازة.",
    ),
    LexicalEntry(
        word="شكون",
        lemma_msa="من هو",
        root="ك-و-ن",
        dialect="maghrebi",
        pos="أداة استفهام",
        definition="السؤال عن الشخص العاقل (منحوتة من: أي شيء يكون).",
        synonyms=["من", "من هو الشخص", "أي إنسان"],
        example="شكون اللي تكلف بكتابة التقرير؟",
    ),
    LexicalEntry(
        word="غادي",
        lemma_msa="سوف وسائر",
        root="غ-د-و",
        dialect="maghrebi",
        pos="أداة استقبال وحال",
        definition="الدلالة على الاستقبال المستقبلي أو الذهاب والسير (من الغدو).",
        synonyms=["سوف", "سـ", "ذاهب إلى"],
        example="غادي نطلقو الإصدار غداً صباحاً.",
    ),
    LexicalEntry(
        word="صافي",
        lemma_msa="انتهى وحسناً",
        root="ص-ف-و",
        dialect="maghrebi",
        pos="كلمة إقرار وتمام",
        definition="تفيد تمام الاتفاق وانتهاء النقاش والرضا بما استقر عليه الأمر.",
        synonyms=["حسناً", "تم الأمر", "كفى", "انتهى"],
        example="صافي تفاهمنا وغادي نبداو العمل.",
    ),

    # ---------------- Iraqi Dialect (اللهجة العراقية) ----------------
    LexicalEntry(
        word="يمعود",
        lemma_msa="يا رجل ويا رعاك الله",
        root="ع-و-د",
        dialect="iraqi",
        pos="نداء وتنبيه",
        definition="نداء عراقي للتعجب أو لفت النظر أو التلطف (من المعوَّذ بالله).",
        synonyms=["يا رجل", "يا صاحبي", "رويدك", "يا هذا"],
        example="يمعود اصبر علينا دقائق ونكمل الشغل.",
    ),
    LexicalEntry(
        word="هواي",
        lemma_msa="كثيراً وجداً",
        root="ه-و-ي",
        dialect="iraqi",
        pos="ظرف مبالغة",
        definition="تفيد الكثرة الزائدة والمبالغة في الكم والعدد بالعراق.",
        synonyms=["كثيراً", "جداً", "بوفرة", "فوق الكفاية"],
        example="استفدنا هواي من ملاحظاتك القيمة.",
    ),
    LexicalEntry(
        word="شكو ماكو",
        lemma_msa="ما الأخبار وما الجديد",
        root="ك-و-ن",
        dialect="iraqi",
        pos="استفهام جامع",
        definition="تحية واستفهام عراقي شائع لتقصي الأخبار (ماذا كائن وماذا غير كائن).",
        synonyms=["ما الأخبار", "ما الجديد", "كيف الأحوال"],
        example="شكو ماكو يا غالي طمنا عنك.",
    ),
    LexicalEntry(
        word="فدوه",
        lemma_msa="فداء وإكرام",
        root="ف-د-ي",
        dialect="iraqi",
        pos="تعبير مودة وتقدير",
        definition="لفظ عراقي دافئ يقال للتعبير عن المحبة والاستعداد للتضحية من أجل المخاطب.",
        synonyms=["فداء لك", "لك العتبى", "أكرمك الله"],
        example="فدوه لعينك ما قصرت ويانا.",
    ),
    LexicalEntry(
        word="لعد",
        lemma_msa="إذن وماذا بعد",
        root="ع-د-د",
        dialect="iraqi",
        pos="رابط استنتاجي",
        definition="أداة استدراك وربط بمعنى 'إذن' أو 'بناء على ذلك' (أصلها: لعاد).",
        synonyms=["إذن", "إذاً", "بناء على ذلك", "ما دام الأمر كذلك"],
        example="لعد ليش تأخرت عن الموعد؟",
    ),
    LexicalEntry(
        word="هسه",
        lemma_msa="الآن وفي اللحظة",
        root="س-ع-ة",
        dialect="iraqi",
        pos="ظرف زمان",
        definition="الدلالة على الوقت الحاضر بالعراق (تخفيف: هالساعة أي هذه الساعة).",
        synonyms=["الآن", "في هذه الساعة", "فوراً"],
        example="هسه نرسل لك الملف الكامل.",
    ),
    LexicalEntry(
        word="دربالك",
        lemma_msa="احذر وانتبه",
        root="د-ر-ي",
        dialect="iraqi",
        pos="أمر وتحذير",
        definition="طلب اليقظة والاحتراز (من: أدر بالك أي وجه انتباهك).",
        synonyms=["احذر", "انتبه", "كن يقظاً"],
        example="دربالك من إهمال التوثيق البرمجي.",
    ),

    # ---------------- Yemeni Dialect (اللهجة اليمنية) ----------------
    LexicalEntry(
        word="ناهي",
        lemma_msa="حسناً وممتاز",
        root="ن-ه-ي",
        dialect="yemeni",
        pos="إقرار ورضا",
        definition="تفيد الموافقة والرضا التام بجودة الأمر وحسنه في اليمن.",
        synonyms=["حسناً", "ممتاز", "طيب", "تم"],
        example="ناهي ولا تشيل هم بنسويه اليوم.",
    ),
    LexicalEntry(
        word="حالي",
        lemma_msa="جميل ولذيذ",
        root="ح-ل-و",
        dialect="yemeni",
        pos="صفة مدح",
        definition="تقال للشيء الجميل المتقن أو الكلام العذب (من الحلاوة والحُسن).",
        synonyms=["جميل", "حلو", "بديع", "متقن"],
        example="التنظيم حالي قوي ومعجب للجميع.",
    ),
    LexicalEntry(
        word="قوي",
        lemma_msa="جداً وللغاية",
        root="ق-و-ي",
        dialect="yemeni",
        pos="ظرف مبالغة",
        definition="تستخدم كظرف مبالغة لتوكيد شدة الصفة في اللهجة اليمنية.",
        synonyms=["جداً", "للغاية", "كثيراً"],
        example="الشغل متعب قوي لكنه يستاهل.",
    ),
    LexicalEntry(
        word="شتي",
        lemma_msa="أريد وأبتغي",
        root="ش-ي-أ",
        dialect="yemeni",
        pos="فعل رغبة",
        definition="تفيد الطلب والإرادة (مشتقة من: أشتي / أشاء وأبتغي).",
        synonyms=["أريد", "أرغب", "أشاء", "أطلب"],
        example="أنا أشتي أتعلم البرمجة من الأساسات.",
    ),
    LexicalEntry(
        word="مليح",
        lemma_msa="جيد وحسن",
        root="م-ل-ح",
        dialect="yemeni",
        pos="صفة",
        definition="تفيد الصلاح والجودة والقبول الحسن (فصيحة قديمة: شيء مليح).",
        synonyms=["جيد", "حسن", "طيب", "لائق"],
        example="الوضع مليح ومستقر تماماً.",
    ),
]


# ---------------------------------------------------------------------------
# Lexicon Manager & Search Engine
# ---------------------------------------------------------------------------

class ArabicLexicon:
    """
    In-memory comprehensive Arabic multi-dialect lexicon and morphology engine.
    Supports microsecond exact & normalized lookups, sentence explanation, and root extraction.
    """

    def __init__(self) -> None:
        self._exact_dict: Dict[str, LexicalEntry] = {}
        self._normalized_dict: Dict[str, LexicalEntry] = {}
        self._split_regex = re.compile(r"[\u0621-\u064A\u0671-\u06D3]+")

        # Populate internal indices
        try:
            from .lexicon_data import load_all_curated_entries
            all_entries = load_all_curated_entries()
        except Exception:
            all_entries = []

        for entry in all_entries:
            self.register(entry)
        for entry in _CURATED_ENTRIES:
            self.register(entry)

    def register(self, entry: LexicalEntry) -> None:
        """Register or override a lexical entry."""
        self._exact_dict[entry.word] = entry
        norm_key = normalize_for_morphology(entry.word)
        if norm_key:
            self._normalized_dict[norm_key] = entry

    @property
    def total_entries(self) -> int:
        return len(self._exact_dict)

    @property
    def entries(self) -> Dict[str, LexicalEntry]:
        """Return a mapping of all registered lexical entries."""
        return dict(self._exact_dict)

    def lookup(self, word: str) -> Optional[LexicalEntry]:
        """
        Lookup a word in the dictionary:
        1. Exact match
        2. Normalized match (without tashkeel/tatweel/hamzas)
        3. Stem-stripped match
        """
        if not word:
            return None
        w = word.strip()
        if w in self._exact_dict:
            return self._exact_dict[w]

        norm = normalize_for_morphology(w)
        if norm in self._normalized_dict:
            return self._normalized_dict[norm]

        stem = light_stem(w)
        stem_norm = normalize_for_morphology(stem)
        if stem_norm in self._normalized_dict:
            return self._normalized_dict[stem_norm]

        return None

    def explain_word(self, word: str) -> TokenExplanation:
        """Generate a complete token explanation for a single word."""
        entry = self.lookup(word)
        norm = normalize_for_morphology(word)
        root = entry.root if entry else extract_root(word)
        suggested_msa = entry.lemma_msa if entry else word

        return TokenExplanation(
            token=word,
            is_dialect=(entry is not None and entry.dialect != "msa"),
            entry=entry,
            normalized_token=norm,
            detected_root=root,
            suggested_msa=suggested_msa,
        )

    def explain_text(self, text: str) -> TextExplanation:
        """
        Deconstruct an entire Arabic text, identify all dialect/vernacular words,
        provide their lexical cards, compute dialect density, and generate a
        reconstructed MSA sentence.
        """
        if not text or not text.strip():
            return TextExplanation(
                original_text="",
                canonical_msa_text="",
                dialect_tokens_count=0,
                total_tokens_count=0,
                dialect_density_pct=0.0,
                dominant_dialect="msa",
                tokens=[],
            )

        words = self._split_regex.findall(text)
        token_explanations: List[TokenExplanation] = []
        dialect_counts: Dict[str, int] = {}
        dialect_hits = 0

        # Replace dialect words with canonical MSA lemmas in-place
        canonical_map: Dict[str, str] = {}

        for w in words:
            expl = self.explain_word(w)
            token_explanations.append(expl)
            if expl.is_dialect and expl.entry:
                dialect_hits += 1
                d = expl.entry.dialect
                dialect_counts[d] = dialect_counts.get(d, 0) + 1
                canonical_map[w] = expl.entry.lemma_msa

        total_words = len(words)
        dialect_density = (dialect_hits / total_words * 100.0) if total_words > 0 else 0.0
        dominant = max(dialect_counts.keys(), key=lambda k: dialect_counts[k]) if dialect_counts else "msa"

        # Construct MSA sentence
        def _sub_word(m: re.Match) -> str:
            w = m.group(0)
            return canonical_map.get(w, w)

        msa_text = self._split_regex.sub(_sub_word, text)
        msa_text = " ".join(msa_text.split())

        return TextExplanation(
            original_text=text,
            canonical_msa_text=msa_text,
            dialect_tokens_count=dialect_hits,
            total_tokens_count=total_words,
            dialect_density_pct=dialect_density,
            dominant_dialect=dominant,
            tokens=token_explanations,
        )


# Global singleton instance
_GLOBAL_LEXICON: Optional[ArabicLexicon] = None


def get_lexicon() -> ArabicLexicon:
    """Retrieve the global singleton ArabicLexicon instance."""
    global _GLOBAL_LEXICON
    if _GLOBAL_LEXICON is None:
        _GLOBAL_LEXICON = ArabicLexicon()
    return _GLOBAL_LEXICON


def lookup_word(word: str) -> Optional[LexicalEntry]:
    """Convenience helper to lookup an Arabic word in the lexicon."""
    return get_lexicon().lookup(word)


def explain_text(text: str) -> TextExplanation:
    """Convenience helper to explain and break down an Arabic sentence."""
    return get_lexicon().explain_text(text)
