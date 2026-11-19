# -*- coding: utf-8 -*-
"""
LiteLLM Pre-Call Security & Token Optimization Hook for Misar.
"""

from typing import Any, Dict, Optional, Tuple

from ..classifier import canonicalize
from ..guard import sanitize_prompt
from ..router import MisarRouter


def misar_litellm_pre_hook(
    data: Dict[str, Any],
    mask_pii: bool = True,
    detect_injection: bool = True,
    risk_threshold: float = 0.5,
    canonicalize_prompts: bool = True,
    optimize_tokens: bool = True,
    router: Optional[MisarRouter] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Pre-call hook for LiteLLM proxies and gateways.
    Inspects messages in `data['messages']`:
    1. Redacts PII (Saudi ID, IBAN, Phone, Email).
    2. Detects and neutralizes prompt injections & BiDi attacks.
    3. Canonicalizes dialect queries to maximize prompt cache hit rates.
    4. Strips repeated character stretches and social noise to minimize token billing.

    Returns:
        (is_blocked: bool, modified_data: Dict[str, Any])
    """
    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return False, data

    r = router or (MisarRouter() if optimize_tokens else None)
    new_messages = []

    for msg in messages:
        if not isinstance(msg, dict):
            new_messages.append(msg)
            continue

        role = msg.get("role", "")
        content = msg.get("content", "")

        # Only process user and system prompt contents
        if isinstance(content, str) and role in ("user", "system"):
            guard = sanitize_prompt(content, mask_pii=mask_pii, detect_injection=detect_injection)
            has_injection = any(
                "injection" in r or "jailbreak" in r or "override" in r or "evasion" in r or "ignore" in r
                for r in guard.detected_risks
            )
            if has_injection and guard.risk_score >= risk_threshold:
                return True, {"error": "Blocked by Misar Security Guard", "risks": guard.detected_risks}

            text = guard.sanitized_text
            if canonicalize_prompts:
                text = canonicalize(text)
            if optimize_tokens and r:
                text = r.preprocess(text)

            new_msg = dict(msg)
            new_msg["content"] = text
            new_messages.append(new_msg)
        else:
            new_messages.append(msg)

    data_copy = dict(data)
    data_copy["messages"] = new_messages
    return False, data_copy
