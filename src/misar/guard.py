# -*- coding: utf-8 -*-
"""
Arabic LLM Prompt Guard & PII Sanitizer for Misar.

Provides enterprise-grade prompt sanitization for Arabic Generative AI applications:
1. Adversarial Unicode Attack Neutralization (BiDi Override & Zero-Width characters).
2. Adversarial Tatweel/Kashida Evasion Detection (obfuscated jailbreaks).
3. Arabic Prompt Injection & Jailbreak Signature Detection.
4. Privacy-Preserving Arabic PII Masking with authentic checksum validation
   (Saudi National ID / Iqama, Saudi/Arab IBAN, Arab Mobile Numbers, Email).
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Unicode Security Regexes
# ---------------------------------------------------------------------------

# BiDi Override characters used to disguise prompts or reverse text order
BIDI_OVERRIDE_REGEX = re.compile(r"[\u202A-\u202E\u2066-\u2069]")

# Zero-width spaces, joiners, soft hyphens used to fragment keywords for evasion
ZERO_WIDTH_REGEX = re.compile(r"[\u00AD\u200B-\u200D\u2060\uFEFF]")

# Tatweel / Kashida
TATWEEL_REGEX = re.compile(r"\u0640+")

# Repeated or stacked diacritics
STACKED_DIACRITICS_REGEX = re.compile(r"[\u064B-\u065F\u0670]{2,}")

# ---------------------------------------------------------------------------
# PII Patterns
# ---------------------------------------------------------------------------

# Saudi National ID (starts with 1) / Iqama (starts with 2) - 10 digits
SAUDI_ID_REGEX = re.compile(r"(?<!\d)([12]\d{9})(?!\d)")

# IBAN (Saudi SA + 22 alphanumeric digits, optional whitespace)
IBAN_REGEX = re.compile(r"\b(SA\s*\d{2}(?:\s*[A-Z0-9]){20})\b", re.IGNORECASE)

# Arab mobile phone formats (Saudi, UAE, Egypt, Kuwait)
ARAB_PHONE_REGEX = re.compile(
    r"(?<!\d)(?:(?:\+|00)?9665\d{8}|05\d{8}|(?:\+|00)?9715[024568]\d{7}|(?:\+|00)?201[0125]\d{8}|(?:\+|00)?965[569]\d{7})(?!\d)"
)

# Email
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# ---------------------------------------------------------------------------
# Prompt Injection & Jailbreak Signatures
# ---------------------------------------------------------------------------

INJECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (
        "ignore_instructions_ar",
        re.compile(
            r"(?:تجاهل|تجاوز|انس|انسى|الغ|إلغاء)\s+(?:كل\s+)?(?:التعليمات|الأوامر|القواعد|السياسات|الإرشادات|الشروط)",
            re.IGNORECASE,
        ),
    ),
    (
        "roleplay_jailbreak_ar",
        re.compile(
            r"(?:أنت\s+الآن|تصرف\s+كـ?|تخيل\s+أنك|العب\s+دور)\s+(?:DAN|مساعد\s+متحرر|بلا\s+قيود|مخترق|وضع\s+المطور)",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal_system_prompt_ar",
        re.compile(
            r"(?:ما\s+هي|اطبع|اعرض|اكشف|شارك)\s+(?:تعليمات\s+النظام|البرومبت\s+الأصلي|التعليمات\s+السرية|system\s*prompt)",
            re.IGNORECASE,
        ),
    ),
    (
        "developer_mode_ar",
        re.compile(r"(?:وضع\s+المطور|وضع\s+الحرية|وضع\s+التصحيح\s+الأمني)", re.IGNORECASE),
    ),
    (
        "ignore_instructions_en",
        re.compile(
            r"ignore\s+(?:all\s+)?(?:previous\s+)?(?:instructions|rules|prompts|guidelines)",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak_keyword_en",
        re.compile(r"\b(?:jailbreak|dan\s+mode|developer\s+mode\s+enabled)\b", re.IGNORECASE),
    ),
]


# ---------------------------------------------------------------------------
# Checksum Validation Helpers (Deterministic, zero false-positives)
# ---------------------------------------------------------------------------

def validate_saudi_national_id(id_str: str) -> bool:
    """
    Validate 10-digit Saudi National ID or Iqama using the official Luhn-variant checksum.
    Citizens start with 1, Residents (Iqama) start with 2.
    """
    clean_id = str(id_str).strip()
    if len(clean_id) != 10 or not clean_id.isdigit():
        return False
    if clean_id[0] not in ("1", "2"):
        return False

    total = 0
    for i in range(9):
        digit = int(clean_id[i])
        if i % 2 == 0:  # Odd position (0-indexed 0, 2, 4, 6, 8)
            doubled = digit * 2
            total += (doubled // 10) + (doubled % 10)
        else:
            total += digit

    check_digit = (10 - (total % 10)) % 10
    return check_digit == int(clean_id[9])


def validate_iban(iban_str: str) -> bool:
    """Validate Saudi IBAN checksum according to ISO 7064 Mod 97-10."""
    clean = "".join(iban_str.split()).upper()
    if len(clean) != 24 or not clean.startswith("SA"):
        return False

    reordered = clean[4:] + clean[:4]
    numeric_str = ""
    for ch in reordered:
        if ch.isdigit():
            numeric_str += ch
        elif "A" <= ch <= "Z":
            numeric_str += str(ord(ch) - ord("A") + 10)
        else:
            return False

    return int(numeric_str) % 97 == 1


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    """Telemetry report produced by ArabicPromptGuard."""
    sanitized_text: str
    original_text: str
    is_suspicious: bool
    risk_score: float
    detected_risks: List[str] = field(default_factory=list)
    masked_pii_count: int = 0
    stripped_unicode_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sanitized_text": self.sanitized_text,
            "original_text": self.original_text,
            "is_suspicious": self.is_suspicious,
            "risk_score": round(self.risk_score, 3),
            "detected_risks": list(self.detected_risks),
            "masked_pii_count": self.masked_pii_count,
            "stripped_unicode_count": self.stripped_unicode_count,
        }


# ---------------------------------------------------------------------------
# Arabic Prompt Guard Engine
# ---------------------------------------------------------------------------

class ArabicPromptGuard:
    """
    Security & PII sanitizer for Arabic Generative AI and LLM prompt pipelines.
    """

    def __init__(
        self,
        mask_pii: bool = True,
        detect_injection: bool = True,
        strip_adversarial_unicode: bool = True,
        strict_id_checksum: bool = True,
        id_replacement: str = "[رقم_هوية_محجوب]",
        iban_replacement: str = "[آيبان_محجوب]",
        phone_replacement: str = "[هاتف_محجوب]",
        email_replacement: str = "[بريد_محجوب]",
    ):
        self.mask_pii = mask_pii
        self.detect_injection = detect_injection
        self.strip_adversarial_unicode = strip_adversarial_unicode
        self.strict_id_checksum = strict_id_checksum
        self.id_replacement = id_replacement
        self.iban_replacement = iban_replacement
        self.phone_replacement = phone_replacement
        self.email_replacement = email_replacement

    def sanitize(self, text: str) -> GuardResult:
        """Sanitize an incoming prompt against injections, malicious unicode, and PII."""
        if not text:
            return GuardResult(
                sanitized_text="",
                original_text="",
                is_suspicious=False,
                risk_score=0.0,
            )

        sanitized = text
        detected_risks: List[str] = []
        stripped_unicode_count = 0
        masked_pii_count = 0

        # 1. Neutralize Adversarial Unicode (BiDi Overrides & Invisible Zero-Widths)
        if self.strip_adversarial_unicode:
            bidi_matches = list(BIDI_OVERRIDE_REGEX.finditer(sanitized))
            if bidi_matches:
                detected_risks.append("bidi_override_attack")
                stripped_unicode_count += len(bidi_matches)
                sanitized = BIDI_OVERRIDE_REGEX.sub("", sanitized)

            zw_matches = list(ZERO_WIDTH_REGEX.finditer(sanitized))
            if zw_matches:
                detected_risks.append("zero_width_injection")
                stripped_unicode_count += len(zw_matches)
                sanitized = ZERO_WIDTH_REGEX.sub("", sanitized)

            # Check stacked diacritic noise
            if STACKED_DIACRITICS_REGEX.search(sanitized):
                detected_risks.append("diacritic_camouflage")

        # 2. Check for Adversarial Tatweel Evasion (e.g. 'تــــجــــاهــــل')
        collapsed_tatweel = TATWEEL_REGEX.sub("", sanitized)
        is_tatweel_evading = False
        if len(collapsed_tatweel) < len(sanitized) - 3:
            # Check if collapsing tatweels triggers an injection rule that wasn't visible before
            for risk_name, pat in INJECTION_PATTERNS:
                if pat.search(collapsed_tatweel) and not pat.search(sanitized):
                    is_tatweel_evading = True
                    detected_risks.append("tatweel_evasion")
                    break

        # 3. Prompt Injection & Jailbreak Signatures
        if self.detect_injection:
            text_to_eval = collapsed_tatweel if is_tatweel_evading else sanitized
            for risk_name, pattern in INJECTION_PATTERNS:
                if pattern.search(text_to_eval):
                    detected_risks.append(risk_name)

        # 4. Privacy-Preserving PII Masking
        if self.mask_pii:
            # Email Masking
            emails = list(EMAIL_REGEX.finditer(sanitized))
            if emails:
                masked_pii_count += len(emails)
                detected_risks.append("pii_email")
                sanitized = EMAIL_REGEX.sub(self.email_replacement, sanitized)

            # IBAN Masking
            def _replace_iban(m: re.Match) -> str:
                nonlocal masked_pii_count
                candidate = m.group(0)
                if validate_iban(candidate):
                    masked_pii_count += 1
                    return self.iban_replacement
                return candidate

            new_sanitized = IBAN_REGEX.sub(_replace_iban, sanitized)
            if new_sanitized != sanitized:
                detected_risks.append("pii_iban")
                sanitized = new_sanitized

            # Phone Masking
            phones = list(ARAB_PHONE_REGEX.finditer(sanitized))
            if phones:
                masked_pii_count += len(phones)
                detected_risks.append("pii_phone")
                sanitized = ARAB_PHONE_REGEX.sub(self.phone_replacement, sanitized)

            # Saudi National ID / Iqama Masking
            def _replace_saudi_id(m: re.Match) -> str:
                nonlocal masked_pii_count
                candidate = m.group(1)
                if self.strict_id_checksum:
                    if validate_saudi_national_id(candidate):
                        masked_pii_count += 1
                        return self.id_replacement
                    return candidate
                else:
                    masked_pii_count += 1
                    return self.id_replacement

            new_sanitized = SAUDI_ID_REGEX.sub(_replace_saudi_id, sanitized)
            if new_sanitized != sanitized:
                detected_risks.append("pii_national_id")
                sanitized = new_sanitized

        # Compute risk score (0.0 to 1.0)
        risk_weights = {
            "bidi_override_attack": 0.40,
            "zero_width_injection": 0.35,
            "tatweel_evasion": 0.50,
            "diacritic_camouflage": 0.20,
            "ignore_instructions_ar": 0.60,
            "roleplay_jailbreak_ar": 0.60,
            "reveal_system_prompt_ar": 0.50,
            "developer_mode_ar": 0.50,
            "ignore_instructions_en": 0.60,
            "jailbreak_keyword_en": 0.50,
            "pii_national_id": 0.25,
            "pii_iban": 0.25,
            "pii_phone": 0.15,
            "pii_email": 0.15,
        }
        raw_score = sum(risk_weights.get(r, 0.20) for r in set(detected_risks))
        risk_score = min(1.0, raw_score)
        is_suspicious = (
            risk_score >= 0.40
            or any("injection" in r or "jailbreak" in r or "override" in r or "evasion" in r for r in detected_risks)
        )

        return GuardResult(
            sanitized_text=sanitized,
            original_text=text,
            is_suspicious=is_suspicious,
            risk_score=risk_score,
            detected_risks=sorted(list(set(detected_risks))),
            masked_pii_count=masked_pii_count,
            stripped_unicode_count=stripped_unicode_count,
        )


def sanitize_prompt(
    text: str,
    mask_pii: bool = True,
    detect_injection: bool = True,
) -> GuardResult:
    """Convenience helper to sanitize an Arabic prompt and detect security/PII risks."""
    guard = ArabicPromptGuard(mask_pii=mask_pii, detect_injection=detect_injection)
    return guard.sanitize(text)
