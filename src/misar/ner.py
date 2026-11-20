# -*- coding: utf-8 -*-
"""
High-speed zero-dependency Arabic Named Entity Recognition (NER) & Masking for Misar (مِسار).
Extracts and classifies:
- PER: Arabic personal names & composite titles (عبدالله، محمد بن سلمان، د. خالد، المهندس منير)
- ORG: Sovereign entities, ministries, universities, and enterprises (سدايا، موهبة، أرامكو، STC، الراجحي)
- LOC: Saudi regions, governorates, cities, and Arab capitals (عسير، أبها، الرياض، جدة، مكة)
- DATE & MONEY: Temporal expressions and monetary values (100 ريال، 2027 م)
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# Sovereign & Major Organizations in Saudi Arabia and the Arab World
KNOWN_ORGS: Set[str] = {
    "موهبة", "مؤسسة موهبة", "مؤسسة الملك عبدالعزيز ورجاله للموهبة والإبداع",
    "سدايا", "الهيئة السعودية للبيانات والذكاء الاصطناعي",
    "أرامكو", "أرامكو السعودية", "شركة أرامكو",
    "سابك", "الشركة السعودية للصناعات الأساسية",
    "الاتصالات السعودية", "إس تي سي", "stc", "STC",
    "مصرف الراجحي", "بنك الراجحي", "البنك الأهلي", "بنك الرياض", "بنك الإنماء",
    "وزارة التعليم", "وزارة الصحة", "وزارة الداخلية", "وزارة الاتصالات", "وزارة التجارة",
    "وزارة المالية", "وزارة الموارد البشرية", "وزارة العدل", "وزارة الطاقة",
    "جامعة الملك سعود", "جامعة الملك فهد للبترول والمعادن", "جامعة الملك عبدالعزيز",
    "جامعة الملك خالد", "جامعة الأميرة نورة", "جامعة الإمام", "جامعة أم القرى",
    "هيئة الاتصالات والفضاء والتقنية", "هيئة الحكومة الرقمية", "المركز الوطني للذكاء الاصطناعي",
    "مستشفى الملك فيصل التخصصي", "مدينة الملك عبدالعزيز للعلوم والتقنية", "كاوست", "KAUST",
}

# Saudi Regions, Governorates, and Major Arab Cities
KNOWN_LOCS: Set[str] = {
    # Saudi Regions & Cities (Southern Province & Asir focus)
    "عسير", "منطقة عسير", "أبها", "خميس مشيط", "النماص", "تنومة", "بيشة", "سراة عبيدة", "أحد رفيدة", "ظهران الجنوب",
    "جازان", "نجران", "الباحة",
    # Major Saudi Regions & Metros
    "الرياض", "منطقة الرياض", "الدرعية", "الخرج",
    "مكة", "مكة المكرمة", "جدة", "الطائف", "رابغ",
    "المدينة", "المدينة المنورة", "ينبع", "العلا",
    "الدمام", "الخبر", "الظهران", "الجبيل", "الأحساء", "القطيف", "حفر الباطن",
    "القصيم", "بريدة", "عنيزة", "الرس",
    "حائل", "تبوك", "نيوم", "الجوف", "سكاكا", "عرعر",
    # Arab Capitals & Metros
    "القاهرة", "دبي", "أبوظبي", "الدوحة", "المنامة", "الكويت", "مسقط", "عمان", "بيروت",
}

# Personal Honorifics and Prefixes
PERSON_PREFIXES: Tuple[str, ...] = (
    "الملك", "الأمير", "ولي العهد", "الشيخ", "الدكتور", "د.", "المهندس", "م.",
    "الأستاذ", "أ.", "المستشار", "معالي", "سعادة", "الأستاذ الدكتور", "أ.د.",
)

# Common First Names for Composite Name Detection
COMMON_FIRST_NAMES: Set[str] = {
    "محمد", "أحمد", "علي", "عبدالله", "عبدالرحمن", "عبدالعزيز", "عبدالمجيد",
    "سعود", "فيصل", "خالد", "فهد", "سلطان", "منير", "سالم", "صالح", "إبراهيم",
    "يوسف", "عمر", "عثمان", "سلمان", "ناصر", "بدر", "تركي", "مشعل", "ماجد",
    "سارة", "نورة", "فاطمة", "مريم", "ريم", "هند", "أمل", "منى", "شهد", "روان",
}

MONEY_REGEX = re.compile(
    r"\b\d+(?:[\.,]\d+)?\s*(?:ريال(?:\s*سعودي)?|SAR|دولار|USD|درهم|دينار|هللة|مليون ريال|مليار ريال)\b",
    re.UNICODE | re.IGNORECASE,
)

DATE_REGEX = re.compile(
    r"\b(?:\d{1,2}\s+(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر|محرم|صفر|ربيع الأول|ربيع الثاني|جمادى الأولى|جمادى الآخرة|رجب|شعبان|رمضان|شوال|ذو القعدة|ذو الحجة)|(?:عام|سنة)\s+\d{4}|\d{4}\s*(?:م|هـ))\b",
    re.UNICODE,
)


@dataclass
class EntitySpan:
    """Represents a classified Named Entity span."""
    text: str
    category: str      # PER, ORG, LOC, DATE, MONEY
    start_char: int
    end_char: int
    confidence: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NERResult:
    """Detailed Arabic Named Entity Recognition report."""
    original_text: str
    masked_text: str
    entities_count: int
    entities: List[EntitySpan] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "masked_text": self.masked_text,
            "entities_count": self.entities_count,
            "entities": [e.to_dict() for e in self.entities],
        }


def extract_entities(text: str, mask_entities: bool = False) -> NERResult:
    """
    Extracts Arabic Named Entities (PER, ORG, LOC, DATE, MONEY) using gazetteers and context patterns.
    Optionally returns an anonymized masked text for LLM privacy.
    """
    if not text or not text.strip():
        return NERResult(original_text=text, masked_text=text, entities_count=0)

    entities: List[EntitySpan] = []

    # 1. Money Expressions
    for m in MONEY_REGEX.finditer(text):
        entities.append(EntitySpan(text=m.group(), category="MONEY", start_char=m.start(), end_char=m.end(), confidence=0.98))

    # 2. Date Expressions
    for m in DATE_REGEX.finditer(text):
        entities.append(EntitySpan(text=m.group(), category="DATE", start_char=m.start(), end_char=m.end(), confidence=0.95))

    # 3. Known Organizations (Gazetteer exact scan)
    for org in sorted(KNOWN_ORGS, key=len, reverse=True):
        pattern = re.compile(rf"(?:^|[\s،,.:;؟!\(\)\[\]])(?:[وفلبك])?({re.escape(org)})(?=[\s،,.:;؟!\(\)\[\]]|$)", re.UNICODE)
        for m in pattern.finditer(text):
            span_text = m.group(1)
            start = m.start(1)
            end = m.end(1)
            # Avoid overlaps
            if not any(e.start_char <= start < e.end_char or e.start_char < end <= e.end_char for e in entities):
                entities.append(EntitySpan(text=span_text, category="ORG", start_char=start, end_char=end, confidence=0.97))

    # 4. Known Locations (Gazetteer exact scan)
    for loc in sorted(KNOWN_LOCS, key=len, reverse=True):
        pattern = re.compile(rf"(?:^|[\s،,.:;؟!\(\)\[\]])(?:[وفلبك])?({re.escape(loc)})(?=[\s،,.:;؟!\(\)\[\]]|$)", re.UNICODE)
        for m in pattern.finditer(text):
            span_text = m.group(1)
            start = m.start(1)
            end = m.end(1)
            if not any(e.start_char <= start < e.end_char or e.start_char < end <= e.end_char for e in entities):
                entities.append(EntitySpan(text=span_text, category="LOC", start_char=start, end_char=end, confidence=0.96))

    # 5. Person Patterns: Honorific + Name, or "بن / آل" composite names
    # e.g., "المهندس منير الذيب", "الأمير محمد بن سلمان", "د. خالد", "عبدالله السالم"
    prefix_pattern = r"(?:" + "|".join(re.escape(p) for p in PERSON_PREFIXES) + r")\s+[\u0621-\u064A]+(?:\s+بن\s+[\u0621-\u064A]+|\s+آل\s+[\u0621-\u064A]+|\s+[\u0621-\u064A]+){0,3}"
    for m in re.finditer(prefix_pattern, text, re.UNICODE):
        span_text = m.group()
        start = m.start()
        end = m.end()
        if not any(e.start_char <= start < e.end_char or e.start_char < end <= e.end_char for e in entities):
            entities.append(EntitySpan(text=span_text, category="PER", start_char=start, end_char=end, confidence=0.95))

    # Common standalone first names + family names (e.g., "منير الذيب", "سلمان الفرج")
    words = text.split()
    for idx, w in enumerate(words):
        if w in COMMON_FIRST_NAMES and idx + 1 < len(words):
            next_w = words[idx + 1]
            if next_w.startswith("ال") and len(next_w) >= 4 and next_w not in KNOWN_LOCS and next_w not in KNOWN_ORGS:
                full_name = f"{w} {next_w}"
                pos = text.find(full_name)
                if pos != -1:
                    start = pos
                    end = pos + len(full_name)
                    if not any(e.start_char <= start < e.end_char or e.start_char < end <= e.end_char for e in entities):
                        entities.append(EntitySpan(text=full_name, category="PER", start_char=start, end_char=end, confidence=0.90))

    # Sort entities by start_char
    entities.sort(key=lambda e: e.start_char)

    # Build masked text
    masked_text = text
    if mask_entities and entities:
        # Replace from end to start to preserve indices
        for e in sorted(entities, key=lambda x: x.start_char, reverse=True):
            placeholder = f"[{e.category}_محجوب]"
            masked_text = masked_text[:e.start_char] + placeholder + masked_text[e.end_char:]

    return NERResult(
        original_text=text,
        masked_text=masked_text,
        entities_count=len(entities),
        entities=entities,
    )
