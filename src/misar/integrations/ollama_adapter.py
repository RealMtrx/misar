# -*- coding: utf-8 -*-
"""
Ollama & Local Arabic LLM Adapter for Misar.
Provides prompt preprocessing, PII redaction, prompt caching canonicalization,
and token optimization for local models like ALLaM, Llama-3-Arabic, and Qwen.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..classifier import canonicalize
from ..guard import sanitize_prompt
from ..router import MisarRouter


class MisarOllamaAdapter:
    """
    Adapter for local Ollama instances running Arabic LLMs (e.g. ALLaM, Llama 3, Qwen).
    Preprocesses messages or prompt strings to reduce token usage and enforce privacy controls.
    """

    def __init__(
        self,
        mask_pii: bool = True,
        detect_injection: bool = True,
        canonicalize_prompts: bool = True,
        optimize_tokens: bool = True,
        router: Optional[MisarRouter] = None,
    ) -> None:
        self.mask_pii = mask_pii
        self.detect_injection = detect_injection
        self.canonicalize_prompts = canonicalize_prompts
        self.optimize_tokens = optimize_tokens
        self.router = router or (MisarRouter() if optimize_tokens else None)

    def process_prompt(self, prompt: str) -> Tuple[bool, str, List[str]]:
        """
        Preprocess a single raw text prompt string before sending to Ollama.
        Returns:
            (is_blocked: bool, clean_prompt: str, detected_risks: List[str])
        """
        if not prompt:
            return False, "", []

        guard = sanitize_prompt(prompt, mask_pii=self.mask_pii, detect_injection=self.detect_injection)
        has_injection = any(
            "injection" in r or "jailbreak" in r or "override" in r or "evasion" in r or "ignore" in r
            for r in guard.detected_risks
        )
        if has_injection and guard.risk_score >= 0.5:
            return True, prompt, guard.detected_risks

        text = guard.sanitized_text
        if self.canonicalize_prompts:
            text = canonicalize(text)
        if self.optimize_tokens and self.router:
            text = self.router.preprocess(text)

        return False, text, guard.detected_risks

    def process_messages(self, messages: List[Dict[str, str]]) -> Tuple[bool, List[Dict[str, str]], List[str]]:
        """
        Preprocess a list of chat message dictionaries ({'role': ..., 'content': ...}).
        """
        cleaned_messages: List[Dict[str, str]] = []
        all_risks: List[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role in ("user", "system") and isinstance(content, str):
                blocked, clean_text, risks = self.process_prompt(content)
                if blocked:
                    return True, messages, risks
                all_risks.extend(risks)
                cleaned_messages.append({"role": role, "content": clean_text})
            else:
                cleaned_messages.append(dict(msg))

        return False, cleaned_messages, all_risks


def prepare_ollama_prompt(
    prompt: str,
    mask_pii: bool = True,
    canonicalize_prompts: bool = True,
    optimize_tokens: bool = True,
) -> str:
    """
    Convenience functional helper to sanitize and optimize an Arabic prompt for Ollama in one line.
    """
    adapter = MisarOllamaAdapter(
        mask_pii=mask_pii,
        canonicalize_prompts=canonicalize_prompts,
        optimize_tokens=optimize_tokens,
    )
    _, clean_prompt, _ = adapter.process_prompt(prompt)
    return clean_prompt
