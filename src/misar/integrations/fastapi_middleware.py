# -*- coding: utf-8 -*-
"""
ASGI & FastAPI Security and Token Optimization Middleware for Misar.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..guard import sanitize_prompt
from ..router import MisarRouter


class MisarSecurityMiddleware:
    """
    Standard ASGI middleware for FastAPI / Starlette applications.
    Inspects incoming JSON request bodies:
    1. Masks Saudi National IDs, IBANs, phone numbers, and emails.
    2. Neutralizes BiDi overrides and Tatweel evasion characters.
    3. Blocks malicious prompt injections with HTTP 400 (if block_injections=True).
    4. Optionally applies Misar token optimization to prompts.
    """

    def __init__(
        self,
        app: Any,
        text_keys: Optional[List[str]] = None,
        mask_pii: bool = True,
        detect_injection: bool = True,
        block_injections: bool = True,
        risk_threshold: float = 0.5,
        optimize_tokens: bool = False,
        router: Optional[MisarRouter] = None,
    ) -> None:
        self.app = app
        self.text_keys: Set[str] = set(text_keys or ["prompt", "text", "query", "message", "content"])
        self.mask_pii = mask_pii
        self.detect_injection = detect_injection
        self.block_injections = block_injections
        self.risk_threshold = risk_threshold
        self.optimize_tokens = optimize_tokens
        self.router = router or (MisarRouter() if optimize_tokens else None)

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Check content type
        headers = dict(scope.get("headers", []))
        content_type = headers.get(b"content-type", b"").decode("latin-1").lower()
        if "application/json" not in content_type:
            await self.app(scope, receive, send)
            return

        # Receive body
        body_chunks = []
        more_body = True
        while more_body:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)

        raw_body = b"".join(body_chunks)
        if not raw_body:
            async def empty_receive():
                return {"type": "http.request", "body": b"", "more_body": False}
            await self.app(scope, empty_receive, send)
            return

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            # Not valid json, pass through
            async def passthrough_receive():
                return {"type": "http.request", "body": raw_body, "more_body": False}
            await self.app(scope, passthrough_receive, send)
            return

        # Process payload recursively
        blocked, block_reason, sanitized_payload = self._process_payload(payload)

        if blocked and self.block_injections:
            response_body = json.dumps({
                "error": "Adversarial prompt injection detected",
                "reason": block_reason,
                "status_code": 400
            }, ensure_ascii=False).encode("utf-8")

            await send({
                "type": "http.response.start",
                "status": 400,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"content-length", str(len(response_body)).encode("ascii")),
                ],
            })
            await send({
                "type": "http.response.body",
                "body": response_body,
                "more_body": False,
            })
            return

        # Forward modified body
        new_body = json.dumps(sanitized_payload, ensure_ascii=False).encode("utf-8")

        async def modified_receive():
            return {"type": "http.request", "body": new_body, "more_body": False}

        # Update content-length header
        new_headers = []
        for k, v in scope.get("headers", []):
            if k.lower() == b"content-length":
                new_headers.append((b"content-length", str(len(new_body)).encode("ascii")))
            else:
                new_headers.append((k, v))
        scope["headers"] = new_headers

        await self.app(scope, modified_receive, send)

    def _process_payload(self, data: Any) -> Tuple[bool, str, Any]:
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                if k in self.text_keys and isinstance(v, str):
                    guard = sanitize_prompt(v, mask_pii=self.mask_pii, detect_injection=self.detect_injection)
                    has_injection = any(
                        "injection" in r or "jailbreak" in r or "override" in r or "evasion" in r or "ignore" in r
                        for r in guard.detected_risks
                    )
                    if self.block_injections and has_injection and guard.risk_score >= self.risk_threshold:
                        return True, f"High-risk injection in '{k}': {', '.join(guard.detected_risks)}", data
                    clean_text = guard.sanitized_text
                    if self.optimize_tokens and self.router:
                        clean_text = self.router.preprocess(clean_text)
                    new_dict[k] = clean_text
                else:
                    blocked, reason, new_val = self._process_payload(v)
                    if blocked:
                        return True, reason, data
                    new_dict[k] = new_val
            return False, "", new_dict

        elif isinstance(data, list):
            new_list = []
            for item in data:
                blocked, reason, new_val = self._process_payload(item)
                if blocked:
                    return True, reason, data
                new_list.append(new_val)
            return False, "", new_list

        return False, "", data
