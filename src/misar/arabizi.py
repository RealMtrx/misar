# -*- coding: utf-8 -*-
"""
Misar Arabizi (Franco-Arabic / عربيزي) Decoupler & Transliteration Engine.

Provides high-speed zero-dependency transliteration of Arabizi / Franco-Arabic
(Latin alphabet and Arabic numerals: 2, 3, 5, 7, 8, 9) into canonical Arabic text.
Neutralizes token fragmentation and [UNK] blindness caused by colloquial youth chat.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# High-Frequency Arabizi Vocabulary Dictionary (Direct Fast-Path Mappings)
# ---------------------------------------------------------------------------

ARABIZI_WORD_MAP: Dict[str, str] = {
    # Greetings & Courtesy
    "salam": "سلام",
    "alsalam": "السلام",
    "elsalam": "السلام",
    "3alekom": "عليكم",
    "3aleikum": "عليكم",
    "3alaykum": "عليكم",
    "3alaykom": "عليكم",
    "mar7aba": "مرحبا",
    "marhaba": "مرحبا",
    "ahlan": "أهلاً",
    "ahlan wa sahlan": "أهلاً وسهلاً",
    "shokran": "شكراً",
    "shukran": "شكراً",
    "3afwan": "عفواً",
    "afwan": "عفواً",
    "mashkour": "مشكور",
    "mashkoor": "مشكور",
    "mabrouk": "مبروك",
    "mabrook": "مبروك",
    "sabah": "صباح",
    "9aba7": "صباح",
    "kheir": "خير",
    "kheer": "خير",
    "5air": "خير",
    "5ayr": "خير",
    "masa2": "مساء",
    "mesa2": "مساء",
    "tisba7": "تصبح",
    "tesba7": "تصبح",
    
    # Islamic Expressions
    "inshallah": "إن شاء الله",
    "insha2allah": "إن شاء الله",
    "mshallah": "ما شاء الله",
    "mashallah": "ما شاء الله",
    "masha2allah": "ما شاء الله",
    "al7amdulillah": "الحمد لله",
    "alhamdulillah": "الحمد لله",
    "subhanallah": "سبحان الله",
    "astaghfirullah": "أستغفر الله",
    "wallah": "والله",
    "walah": "والله",

    # Interrogatives & Question Words
    "keefak": "كيفك",
    "kefak": "كيفك",
    "kaifak": "كيفك",
    "keefek": "كيفك",
    "keefik": "كيفك",
    "shlonak": "شلونك",
    "shloonak": "شلونك",
    "shlonik": "شلونك",
    "sh5barak": "شخبارك",
    "shakhbarak": "شخبارك",
    "leesh": "ليش",
    "lesh": "ليش",
    "laih": "ليه",
    "leh": "ليه",
    "wain": "وين",
    "wayn": "وين",
    "feen": "فين",
    "eish": "إيش",
    "aysh": "إيش",
    "shu": "شو",
    "sho": "شو",
    "meen": "مين",
    "mata": "متى",
    "kam": "كم",

    # Affirmations, Negations & Particles
    "ya": "يا",
    "fi": "في",
    "fe": "في",
    "men": "من",
    "min": "من",
    "3an": "عن",
    "ala": "على",
    "3ala": "على",
    "ila": "إلى",
    "el": "ال",
    "al": "ال",
    "hada": "هذا",
    "hatha": "هذا",
    "da": "ده",
    "hadi": "هذه",
    "di": "دي",
    "aywa": "أيوه",
    "eeh": "إيه",
    "na3am": "نعم",
    "la2": "لا",
    "laa": "لا",
    "msh": "مش",
    "mosh": "مش",
    "mesh": "مش",
    "mo": "مو",
    "mub": "مب",
    "mafeesh": "ما فيه",
    "mafeh": "ما فيه",
    "bas": "بس",
    "bass": "بس",
    "kaman": "كمان",
    "bardoo": "برضه",
    "bardo": "برضه",

    # Common Adjectives & Nouns
    "7abibi": "حبيبي",
    "habibi": "حبيبي",
    "7abibti": "حبيبتي",
    "tamam": "تمام",
    "tmm": "تمام",
    "yalla": "يلا",
    "zein": "زين",
    "zain": "زين",
    "kwais": "كويس",
    "kwayes": "كويس",
    "kwayyis": "كويس",
    "7elw": "حلو",
    "helw": "حلو",
    "kteer": "كثير",
    "ktheer": "كثير",
    "wayed": "وايد",
    "marra": "مرة",
    "awwi": "أوي",
    "gedan": "جداً",
    "saree3": "سريع",
    "sa7": "صح",
    "ghalat": "غلط",
    "5alas": "خلاص",
    "khalas": "خلاص",
    "forsa": "فرصة",
    "sa3ida": "سعيدة",
}

# ---------------------------------------------------------------------------
# Phonetic Replacement Rules for Character-Level Transliteration
# ---------------------------------------------------------------------------

_DIGRAPH_RULES: List[Tuple[str, str]] = [
    ("3'", "غ"),
    ("7'", "خ"),
    ("6'", "ظ"),
    ("9'", "ض"),
    ("kh", "خ"),
    ("sh", "ش"),
    ("th", "ث"),
    ("dh", "ذ"),
    ("gh", "غ"),
    ("ch", "تش"),
    ("ph", "ف"),
    ("ee", "ي"),
    ("oo", "و"),
    ("aa", "ا"),
    ("ou", "و"),
]

_CHAR_RULES: Dict[str, str] = {
    # Numerals representing unique Arabic consonants
    "2": "ء",
    "3": "ع",
    "5": "خ",
    "6": "ط",
    "7": "ح",
    "8": "ق",
    "9": "ص",

    # Standard Latin Alphabets
    "a": "ا",
    "b": "ب",
    "c": "ك",
    "d": "د",
    "e": "ي",
    "f": "ف",
    "g": "ج",
    "h": "ه",
    "i": "ي",
    "j": "ج",
    "k": "ك",
    "l": "ل",
    "m": "م",
    "n": "ن",
    "o": "و",
    "p": "ب",
    "q": "ق",
    "r": "ر",
    "s": "س",
    "t": "ت",
    "u": "و",
    "v": "ف",
    "w": "و",
    "x": "كس",
    "y": "ي",
    "z": "ز",
}

# Regex to detect tokens containing digits typically found in Arabizi
_ARABIZI_NUM_TOKEN_RE = re.compile(r"\b[a-zA-Z]*[2356789][a-zA-Z0-9]*\b")
_LATIN_WORD_RE = re.compile(r"\b[a-zA-Z0-9']+\b")

# English technical/system keywords to preserve as Latin (never transliterate)
_PRESERVED_TECH_WORDS: Set[str] = {
    "api", "llm", "ai", "cpu", "gpu", "ram", "id", "url", "http", "https",
    "json", "csv", "sql", "html", "css", "js", "python", "docker", "pypi",
    "github", "git", "rest", "rag", "flops", "token", "tokens", "vllm", "ollama",
    "arabert", "marbert", "camelbert", "allam", "llama", "deepseek", "chatgpt",
    "base64", "utf8", "http2", "http1", "sha256", "mp3", "mp4", "ipv4", "ipv6",
    "html5", "css3", "w3c", "oauth2"
}

_TECH_PATTERN_REGEX = re.compile(r"^(base\d+|utf\d+|http\d*|sha\d+|mp\d+|ipv\d+|html\d+|css\d+|oauth\d*)$")


def is_arabizi_token(token: str) -> bool:
    """Check if an isolated token exhibits strong Arabizi patterns."""
    lower = token.lower()
    if lower in _PRESERVED_TECH_WORDS or _TECH_PATTERN_REGEX.match(lower):
        return False
    if lower in ARABIZI_WORD_MAP:
        return True
    # Word contains Arabizi characteristic numerals (3, 7, 5, 2, 8, 9) mixed with letters
    if any(ch in "2356789" for ch in lower) and any(ch.isalpha() for ch in lower):
        return True
    return False


def is_arabizi_text(text: str) -> bool:
    """
    Detect whether a sentence or document contains meaningful Arabizi content.
    Returns True if at least one Arabizi token or pattern is present.
    """
    if not text:
        return False
    words = _LATIN_WORD_RE.findall(text)
    if not words:
        return False
    
    arabizi_count = sum(1 for w in words if is_arabizi_token(w))
    return arabizi_count > 0


def transliterate_arabizi_word(word: str) -> str:
    """
    Transliterate a single Arabizi token into Arabic script using phonetic heuristics.
    """
    lower = word.lower()
    if lower in _PRESERVED_TECH_WORDS or _TECH_PATTERN_REGEX.match(lower):
        return word
    if lower in ARABIZI_WORD_MAP:
        return ARABIZI_WORD_MAP[lower]

    # Handle definite article prefixes (el-, al-)
    prefix = ""
    stem = lower
    if stem.startswith("el-") or stem.startswith("al-"):
        prefix = "ال"
        stem = stem[3:]
    elif (stem.startswith("el") or stem.startswith("al")) and len(stem) > 4:
        # Check if remainder is a known word
        if stem[2:] in ARABIZI_WORD_MAP:
            return "ال" + ARABIZI_WORD_MAP[stem[2:]]

    # Apply digraph substitutions first
    temp = stem
    for latin_dg, arabic_char in _DIGRAPH_RULES:
        temp = temp.replace(latin_dg, arabic_char)

    # Character-by-character mapping
    res: List[str] = []
    for ch in temp:
        if ch in _CHAR_RULES:
            res.append(_CHAR_RULES[ch])
        else:
            res.append(ch)

    translated = "".join(res)

    # Clean redundant consecutive identical vowels
    translated = re.sub(r"ا{2,}", "ا", translated)
    translated = re.sub(r"ي{3,}", "ي", translated)
    translated = re.sub(r"و{3,}", "و", translated)

    return prefix + translated


class ArabiziDecoder:
    """
    High-throughput decoder for normalizing Arabizi / Franco-Arabic in LLM pipelines.
    """

    def __init__(self, preserve_code_switching: bool = True) -> None:
        self.preserve_code_switching = preserve_code_switching

    def decode(self, text: str) -> str:
        """
        Scan text, identify Arabizi words, and replace them with Arabic transliterations
        while preserving English technical terms, punctuation, and native Arabic tokens.
        """
        if not text:
            return ""

        def _replace_match(match: re.Match) -> str:
            token = match.group(0)
            if is_arabizi_token(token):
                return transliterate_arabizi_word(token)
            return token

        # Substitute matching tokens
        decoded = _LATIN_WORD_RE.sub(_replace_match, text)
        return decoded


# Global default instance
_DEFAULT_DECODER = ArabiziDecoder()


def decode_arabizi(text: str) -> str:
    """
    Convenience function: decode Arabizi / Franco-Arabic in text to Arabic script.
    
    Example:
        >>> decode_arabizi("mar7aba ya 7abibi keefak?")
        'مرحبا يا حبيبي كيفك؟'
    """
    return _DEFAULT_DECODER.decode(text)
