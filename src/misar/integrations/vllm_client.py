# -*- coding: utf-8 -*-
"""
vLLM High-Throughput Inference Preprocessing Hook for Misar.
Optimizes incoming prompts for vLLM servers to minimize KV-cache memory and attention FLOPs.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..classifier import canonicalize
from ..guard import sanitize_prompt
from ..router import MisarRouter


def misar_vllm_pre_hook(
    request_data: Dict[str, Any],
    mask_pii: bool = True,
    detect_injection: bool = True,
    canonicalize_prompts: bool = True,
    optimize_tokens: bool = True,
    router: Optional[MisarRouter] = None,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Pre-inference hook for vLLM / OpenAI-compatible local endpoints.
    Accepts raw vLLM request payload with 'prompt' or 'messages',
    neutralizes attacks, redacts Saudi PII, optimizes tokens, and returns sanitized data.

    Returns:
        (is_blocked: bool, modified_request: Dict[str, Any], detected_risks: List[str])
    """
    req_copy = dict(request_data)
    detected_risks: List[str] = []
    r = router or (MisarRouter() if optimize_tokens else None)

    # 1. Single prompt string
    if "prompt" in req_copy and isinstance(req_copy["prompt"], str):
        prompt_str = req_copy["prompt"]
        guard = sanitize_prompt(prompt_str, mask_pii=mask_pii, detect_injection=detect_injection)
        has_injection = any(
            "injection" in r_name or "jailbreak" in r_name or "override" in r_name or "evasion" in r_name or "ignore" in r_name
            for r_name in guard.detected_risks
        )
        if has_injection and guard.risk_score >= 0.5:
            return True, {"error": "Adversarial prompt rejected by Misar Guard", "risks": guard.detected_risks}, guard.detected_risks

        text = guard.sanitized_text
        if canonicalize_prompts:
            text = canonicalize(text)
        if optimize_tokens and r:
            text = r.preprocess(text)

        req_copy["prompt"] = text
        detected_risks.extend(guard.detected_risks)

    # 2. Chat messages list
    elif "messages" in req_copy and isinstance(req_copy["messages"], list):
        clean_msgs = []
        for msg in req_copy["messages"]:
            if not isinstance(msg, dict):
                clean_msgs.append(msg)
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")

            if role in ("user", "system") and isinstance(content, str):
                guard = sanitize_prompt(content, mask_pii=mask_pii, detect_injection=detect_injection)
                has_injection = any(
                    "injection" in r_name or "jailbreak" in r_name or "override" in r_name or "evasion" in r_name or "ignore" in r_name
                    for r_name in guard.detected_risks
                )
                if has_injection and guard.risk_score >= 0.5:
                    return True, {"error": "Adversarial prompt rejected by Misar Guard", "risks": guard.detected_risks}, guard.detected_risks

                text = guard.sanitized_text
                if canonicalize_prompts:
                    text = canonicalize(text)
                if optimize_tokens and r:
                    text = r.preprocess(text)

                clean_msg = dict(msg)
                clean_msg["content"] = text
                clean_msgs.append(clean_msg)
                detected_risks.extend(guard.detected_risks)
            else:
                clean_msgs.append(msg)

        req_copy["messages"] = clean_msgs

    return False, req_copy, sorted(list(set(detected_risks)))
