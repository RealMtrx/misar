# -*- coding: utf-8 -*-
"""
Misar Synthetic Data Augmentation & Arabic LLM Stress-Testing Engine (محرك تعزيز البيانات واختبار المتانة).

Generates realistic typographical, orthographic, dialectal, and Arabizi variations
to stress-test Arabic Transformer language models and quantify token inflation.
"""

import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Mapping of common Arabic keyboard / orthographic confusions
_ORTHO_CONFUSIONS: Dict[str, List[str]] = {
    "ة": ["ه"],
    "ه": ["ة"],
    "أ": ["ا", "إ"],
    "إ": ["ا", "أ"],
    "آ": ["ا"],
    "ا": ["أ", "إ"],
    "ى": ["ي"],
    "ي": ["ى"],
    "ذ": ["د", "ز"],
    "د": ["ذ"],
    "ز": ["ر", "ذ"],
    "ر": ["ز"],
    "س": ["ش", "ص"],
    "ص": ["س", "ض"],
    "ض": ["ص", "ظ"],
    "ظ": ["ط", "ض"],
    "ط": ["ظ", "ت"],
    "ت": ["ط", "ة"],
    "ث": ["ت", "س"],
    "ق": ["غ", "ك"],
    "ك": ["ق"],
}

_ARABIC_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهويىةأإآءئؤ")
_CONNECTABLE_LETTERS = set("بتثجحخسشصضطظعغفقكلمنهيئ")
_HARAKAT = ["\u064E", "\u064F", "\u0650", "\u0652", "\u064B", "\u064C", "\u064D", "\u0651"]

# Reverse Arabizi mappings for generating code-switching noise
_ARABIC_TO_ARABIZI = {
    "مرحبا": "mar7aba",
    "حبيبي": "7abibi",
    "كيفك": "keefak",
    "شلونك": "shlonak",
    "شكرا": "shukran",
    "شكراً": "shokran",
    "عفوا": "3afwan",
    "عفواً": "3afwan",
    "تمام": "tamam",
    "يلا": "yalla",
    "والله": "wallah",
    "مشكور": "mashkour",
    "خير": "kheir",
    "صباح": "9aba7",
    "مساء": "masa2",
    "ليش": "leesh",
    "وين": "wain",
    "إيش": "eish",
    "كثير": "kteer",
    "كويس": "kwayes",
    "خلاص": "5alas",
}


def inject_orthographic_noise(text: str, typo_rate: float = 0.15, seed: Optional[int] = None) -> str:
    """
    Inject realistic human typos (swapping hamzas, taa marbuta/haa, alif maqsura/yaa).
    """
    if not text:
        return ""
    rng = random.Random(seed)
    chars = list(text)

    for i, ch in enumerate(chars):
        if ch in _ORTHO_CONFUSIONS and rng.random() < typo_rate:
            chars[i] = rng.choice(_ORTHO_CONFUSIONS[ch])

    return "".join(chars)


def inject_tatweel_noise(
    text: str,
    rate: float = 0.25,
    max_stretch: int = 5,
    seed: Optional[int] = None,
) -> str:
    """
    Simulate social media and chat elongation by inserting kashida (ـ) after connectable letters.
    """
    if not text:
        return ""
    rng = random.Random(seed)
    res = []

    for ch in text:
        res.append(ch)
        if ch in _CONNECTABLE_LETTERS and rng.random() < rate:
            count = rng.randint(2, max_stretch)
            res.append("ـ" * count)

    return "".join(res)


def inject_diacritics_noise(
    text: str,
    rate: float = 0.35,
    seed: Optional[int] = None,
) -> str:
    """
    Inject random Arabic harakat (tashkeel) across consonants.
    """
    if not text:
        return ""
    rng = random.Random(seed)
    res = []

    for ch in text:
        res.append(ch)
        if ch in _ARABIC_LETTERS and rng.random() < rate:
            res.append(rng.choice(_HARAKAT))

    return "".join(res)


def inject_arabizi_noise(
    text: str,
    replacement_rate: float = 0.4,
    seed: Optional[int] = None,
) -> str:
    """
    Inject Franco-Arabic (Arabizi) code-switching tokens into colloquial Arabic sentences.
    """
    if not text:
        return ""
    rng = random.Random(seed)
    words = text.split()
    out = []

    for w in words:
        clean_w = re.sub(r"[^\w\s]", "", w)
        if clean_w in _ARABIC_TO_ARABIZI and rng.random() < replacement_rate:
            arabizi_equiv = _ARABIC_TO_ARABIZI[clean_w]
            out.append(w.replace(clean_w, arabizi_equiv))
        else:
            out.append(w)

    return " ".join(out)


def generate_stress_suite(
    text: str,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Generate an exhaustive set of 5 noisy perturbations for evaluating LLM robustness.
    """
    return {
        "original_clean": text,
        "orthographic_typos": inject_orthographic_noise(text, typo_rate=0.25, seed=seed),
        "tatweel_elongation": inject_tatweel_noise(text, rate=0.35, max_stretch=6, seed=seed),
        "heavy_diacritics": inject_diacritics_noise(text, rate=0.45, seed=seed),
        "arabizi_code_switching": inject_arabizi_noise(text, replacement_rate=0.7, seed=seed),
        "adversarial_composite": inject_orthographic_noise(
            inject_tatweel_noise(inject_diacritics_noise(text, rate=0.2, seed=seed), rate=0.2, seed=seed),
            typo_rate=0.15,
            seed=seed,
        ),
    }


def evaluate_augmentation_recovery(
    text: str,
    router=None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Stress-tests Misar's ability to clean and normalize synthetic perturbations,
    computing token savings and recovery ratios.
    """
    from .router import MisarRouter
    from .tokenizer import load_default_tokenizer

    r = router or MisarRouter(arabizi_mode=True)
    tok = load_default_tokenizer()

    suite = generate_stress_suite(text, seed=seed)
    results = {}

    for variant_name, variant_text in suite.items():
        raw_tokens = len(tok.tokenize(variant_text))
        cleaned = r.preprocess(variant_text)
        cleaned_tokens = len(tok.tokenize(cleaned))
        tokens_saved = max(0, raw_tokens - cleaned_tokens)
        reduction_pct = round((tokens_saved / (raw_tokens or 1)) * 100, 2)

        results[variant_name] = {
            "variant_text": variant_text,
            "cleaned_text": cleaned,
            "raw_token_count": raw_tokens,
            "cleaned_token_count": cleaned_tokens,
            "tokens_saved": tokens_saved,
            "reduction_pct": reduction_pct,
        }

    return results
