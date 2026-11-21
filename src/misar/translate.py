# -*- coding: utf-8 -*-
"""
High-speed Dialect-to-MSA Parallel Translation Engine for Misar (مِسار).
Converts colloquial expressions from Saudi/Gulf, Egyptian, Levantine, and Maghrebi
into canonical Modern Standard Arabic (الفصحى المعيارية) with zero external dependencies.
"""

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Multi-word idioms and phrases ordered from longest to shortest
DIALECT_TO_MSA_PHRASES: List[Tuple[str, str, str]] = [
    # (colloquial_pattern, msa_replacement, dialect_name)
    # Saudi & Gulf
    (r"\bوش تبغى تسوي\b", "ماذا تريد أن تفعل", "saudi_gulf"),
    (r"\bوش تبي تسوي\b", "ماذا تريد أن تفعل", "saudi_gulf"),
    (r"\bوش يبغى يسوي\b", "ماذا يريد أن يفعل", "saudi_gulf"),
    (r"\bوش يبي يسوي\b", "ماذا يريد أن يفعل", "saudi_gulf"),
    (r"\bوش يبغى تسوي\b", "ماذا يريد أن يفعل", "saudi_gulf"),
    (r"\bوش يبي تسوي\b", "ماذا يريد أن يفعل", "saudi_gulf"),
    (r"\bوش يبغى\b", "ماذا يريد", "saudi_gulf"),
    (r"\bوش يبي\b", "ماذا يريد", "saudi_gulf"),
    (r"\bوش يبون\b", "ماذا يريدون", "saudi_gulf"),
    (r"\bوش تبي\b", "ماذا تريد", "saudi_gulf"),
    (r"\bوش تبغى\b", "ماذا تريد", "saudi_gulf"),
    (r"\bوش تبون\b", "ماذا تريدون", "saudi_gulf"),
    (r"\bايش تبغى\b", "ماذا تريد", "saudi_gulf"),
    (r"\bإيش تبغى\b", "ماذا تريد", "saudi_gulf"),
    (r"\bوش السالفة\b", "ما الأمر", "saudi_gulf"),
    (r"\bوش السالفه\b", "ما الأمر", "saudi_gulf"),
    (r"\bشسالفة\b", "ما الأمر", "saudi_gulf"),
    (r"\bشسالفه\b", "ما الأمر", "saudi_gulf"),
    (r"\bشلونك\b", "كيف حالك", "saudi_gulf"),
    (r"\bشلونكم\b", "كيف حالكم", "saudi_gulf"),
    (r"\bيبيلها تحديث\b", "تحتاج إلى تحديث", "saudi_gulf"),
    (r"\bيبيله تحديث\b", "يحتاج إلى تحديث", "saudi_gulf"),
    (r"\bيبيلها\b", "تحتاج إلى", "saudi_gulf"),
    (r"\bيبيله\b", "يحتاج إلى", "saudi_gulf"),
    (r"\bمافيه أي تأخير\b", "لا يوجد أي تأخير", "saudi_gulf"),
    (r"\bمافيه أي\b", "لا يوجد أي", "saudi_gulf"),
    (r"\bمافيه\b", "لا يوجد", "saudi_gulf"),
    (r"\bمافيها\b", "لا توجد فيها", "saudi_gulf"),
    (r"\bماكو مشاكل\b", "لا توجد مشاكل", "saudi_gulf"),
    (r"\bماكو\b", "لا يوجد", "saudi_gulf"),
    (r"\bمو معقدة\b", "ليست معقدة", "saudi_gulf"),
    (r"\bمو معقد\b", "ليس معقداً", "saudi_gulf"),
    (r"\bمو حلو\b", "ليس جميلاً", "saudi_gulf"),
    (r"\bمو زين\b", "ليس جيداً", "saudi_gulf"),
    (r"\bمو بطال\b", "ليس سيئاً", "saudi_gulf"),
    (r"\bمو\b", "ليس", "saudi_gulf"),
    (r"\bمب بطيء\b", "ليس بطيئاً", "saudi_gulf"),
    (r"\bمب زين\b", "ليس جيداً", "saudi_gulf"),
    (r"\bمب\b", "ليس", "saudi_gulf"),
    (r"\bالحين\b", "الآن", "saudi_gulf"),
    (r"\bدحين\b", "الآن", "saudi_gulf"),
    (r"\bعشان كذا\b", "لذلك", "saudi_gulf"),
    (r"\bعشان كذه\b", "لذلك", "saudi_gulf"),
    (r"\bعشان\b", "لأجل", "saudi_gulf"),
    (r"\bعلشان\b", "لأجل", "saudi_gulf"),
    (r"\bيا خوي\b", "يا أخي", "saudi_gulf"),
    (r"\bياخي\b", "يا أخي", "saudi_gulf"),
    (r"\bبالحيل\b", "جداً", "saudi_gulf"),
    (r"\bوايد\b", "كثيراً", "saudi_gulf"),
    (r"\bسالفة\b", "قصة", "saudi_gulf"),
    (r"\bسالفه\b", "قصة", "saudi_gulf"),
    (r"\bشويتين\b", "قليلاً", "saudi_gulf"),
    (r"\bهاذا\b", "هذا", "saudi_gulf"),
    (r"\bهاذي\b", "هذه", "saudi_gulf"),
    (r"\bهذول\b", "هؤلاء", "saudi_gulf"),
    (r"\bهذولي\b", "هؤلاء", "saudi_gulf"),

    # Egyptian
    (r"\bعايز ايه دلوقتي\b", "ماذا تريد الآن", "egyptian"),
    (r"\bعايز إيه دلوقتي\b", "ماذا تريد الآن", "egyptian"),
    (r"\bعاوز ايه دلوقتي\b", "ماذا تريد الآن", "egyptian"),
    (r"\bعايز ايه\b", "ماذا تريد", "egyptian"),
    (r"\bعاوز ايه\b", "ماذا تريد", "egyptian"),
    (r"\bعايز\b", "أريد", "egyptian"),
    (r"\bعاوز\b", "أريد", "egyptian"),
    (r"\bدلوقتي\b", "الآن", "egyptian"),
    (r"\bازيك\b", "كيف حالك", "egyptian"),
    (r"\bإزيك\b", "كيف حالك", "egyptian"),
    (r"\bازيكم\b", "كيف حالكم", "egyptian"),
    (r"\bمش عاجبني\b", "لا يعجبني", "egyptian"),
    (r"\bمش بطال\b", "ليس سيئاً", "egyptian"),
    (r"\bمش بطيء\b", "ليس بطيئاً", "egyptian"),
    (r"\bمش\b", "ليس", "egyptian"),
    (r"\bمفيش مشاكل\b", "لا توجد مشاكل", "egyptian"),
    (r"\bمافيش مشاكل\b", "لا توجد مشاكل", "egyptian"),
    (r"\bمفيش\b", "لا يوجد", "egyptian"),
    (r"\bمافيش\b", "لا يوجد", "egyptian"),
    (r"\bكده\b", "هكذا", "egyptian"),
    (r"\bكدا\b", "هكذا", "egyptian"),
    (r"\bاوي\b", "جداً", "egyptian"),
    (r"\bأوي\b", "جداً", "egyptian"),
    (r"\bيا\s*باشا\b", "يا سيدي", "egyptian"),
    (r"\bياباشا\b", "يا سيدي", "egyptian"),
    (r"\bالنهاردة\b", "اليوم", "egyptian"),
    (r"\bالنهارده\b", "اليوم", "egyptian"),
    (r"\bامبارح\b", "أمس", "egyptian"),

    # Levantine
    (r"\bشو بدك هلأ\b", "ماذا تريد الآن", "levantine"),
    (r"\bشو بدك\b", "ماذا تريد", "levantine"),
    (r"\bشو صار\b", "ماذا حدث", "levantine"),
    (r"\bشو\b", "ماذا", "levantine"),
    (r"\bهلأ\b", "الآن", "levantine"),
    (r"\bهلق\b", "الآن", "levantine"),
    (r"\bهيك\b", "هكذا", "levantine"),
    (r"\bكتير منيح\b", "جيد جداً", "levantine"),
    (r"\bكتير\b", "كثيراً", "levantine"),
    (r"\bمنيح\b", "جيد", "levantine"),
    (r"\bمنيحة\b", "جيدة", "levantine"),
    (r"\bمنيحه\b", "جيدة", "levantine"),
    (r"\bهاد\b", "هذا", "levantine"),
    (r"\bهادا\b", "هذا", "levantine"),
    (r"\bهيدي\b", "هذه", "levantine"),
    (r"\bهيدا\b", "هذا", "levantine"),
    (r"\bبدي\b", "أريد", "levantine"),
    (r"\bبدنا\b", "نريد", "levantine"),

    # Maghrebi
    (r"\bواش كاين دابا\b", "ماذا يوجد الآن", "maghrebi"),
    (r"\bواش كاين\b", "ماذا يوجد", "maghrebi"),
    (r"\bدابا\b", "الآن", "maghrebi"),
    (r"\bمزيانة بزاف\b", "جيد جداً", "maghrebi"),
    (r"\bمزيانه بزاف\b", "جيد جداً", "maghrebi"),
    (r"\bمزيان بزاف\b", "جيد جداً", "maghrebi"),
    (r"\bبزاف\b", "كثيراً جداً", "maghrebi"),
    (r"\bمزيانة\b", "جيدة", "maghrebi"),
    (r"\bمزيانه\b", "جيدة", "maghrebi"),
    (r"\bمزيان\b", "جيد", "maghrebi"),
    (r"\bديالكم\b", "الخاص بـكم", "maghrebi"),
    (r"\bديالنا\b", "الخاص بنا", "maghrebi"),
    (r"\bديالي\b", "الخاص بي", "maghrebi"),
    (r"\bديال\b", "الخاص بـ", "maghrebi"),
    (r"\bباش\b", "لكي", "maghrebi"),
    (r"\bغادي\b", "سوف", "maghrebi"),
    (r"\bخويا\b", "أخي", "maghrebi"),
]


@dataclass
class TranslationResult:
    """Detailed report for Dialect-to-MSA translation."""
    original_text: str
    msa_text: str
    detected_dialect: str
    rules_applied_count: int
    rules: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def translate_to_msa(text: str) -> TranslationResult:
    """
    Translates dialectal colloquial sentences into canonical Modern Standard Arabic (MSA).
    Preserves punctuation, emojis, and technical keywords.
    """
    if not text or not text.strip():
        return TranslationResult(original_text=text, msa_text=text, detected_dialect="msa", rules_applied_count=0)

    msa_text = text
    applied_rules: List[str] = []
    dialects_count: Dict[str, int] = {}

    for pattern, replacement, dialect in DIALECT_TO_MSA_PHRASES:
        regex = re.compile(pattern, re.UNICODE)
        if regex.search(msa_text):
            msa_text = regex.sub(replacement, msa_text)
            applied_rules.append(f"{pattern} -> {replacement}")
            dialects_count[dialect] = dialects_count.get(dialect, 0) + 1

    # Detect dominant dialect
    if dialects_count:
        dominant_dialect = max(dialects_count.items(), key=lambda x: x[1])[0]
    else:
        dominant_dialect = "msa"

    # Cleanup any accidental double spaces
    msa_text = re.sub(r"\s+", " ", msa_text).strip()

    return TranslationResult(
        original_text=text,
        msa_text=msa_text,
        detected_dialect=dominant_dialect,
        rules_applied_count=len(applied_rules),
        rules=applied_rules,
    )
