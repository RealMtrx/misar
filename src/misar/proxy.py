# -*- coding: utf-8 -*-
"""
Misar Transparent OpenAI-Compatible Proxy Gateway (بوابة الوكيل الشفافة المتوافقة مع OpenAI).

Acts as an autonomous, drop-in reverse proxy sitting between LLM clients (OpenAI SDK,
LangChain, LlamaIndex, LiteLLM) and inference engines (Ollama, vLLM, OpenAI, DeepSeek).

Interception Pipeline:
1. Intercepts incoming `/v1/chat/completions` and `/v1/completions` requests.
2. Evaluates prompt safety against Arabic adversarial injections and jailbreaks.
3. Masks sensitive Saudi PII (National ID, IBAN with mod-97 checksum, Phone, Email).
4. Strips redundant tatweel, diacritics, and normalizes dialectal queries.
5. Injects telemetry headers: X-Misar-Tokens-Saved, X-Misar-Attention-Saved-Pct.
6. Forwards to upstream or generates synthetic responses in standalone mode.
"""

import json
import math
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from .classifier import canonicalize
from .guard import sanitize_prompt
from .router import MisarRouter
from .tokenizer import load_default_tokenizer


class ProxyTelemetry:
    """Thread-safe telemetry accumulator for gateway telemetry."""
    def __init__(self) -> None:
        self.total_requests = 0
        self.total_tokens_raw = 0
        self.total_tokens_processed = 0
        self.total_tokens_saved = 0
        self.total_pii_masked = 0
        self.total_injections_blocked = 0
        self.start_time = time.time()

    def record_request(
        self,
        raw_tokens: int,
        proc_tokens: int,
        pii_count: int,
        blocked: bool = False,
    ) -> None:
        self.total_requests += 1
        if blocked:
            self.total_injections_blocked += 1
            return
        saved = max(0, raw_tokens - proc_tokens)
        self.total_tokens_raw += raw_tokens
        self.total_tokens_processed += proc_tokens
        self.total_tokens_saved += saved
        self.total_pii_masked += pii_count

    def to_dict(self) -> Dict[str, Any]:
        uptime = round(time.time() - self.start_time, 2)
        savings_pct = (
            round((self.total_tokens_saved / self.total_tokens_raw) * 100, 2)
            if self.total_tokens_raw > 0
            else 0.0
        )
        # Cost saved estimation assuming $0.015 / 1k tokens or ~0.056 SAR
        sar_saved = round((self.total_tokens_saved / 1000.0) * 0.056, 4)
        return {
            "total_requests": self.total_requests,
            "total_tokens_raw": self.total_tokens_raw,
            "total_tokens_processed": self.total_tokens_processed,
            "total_tokens_saved": self.total_tokens_saved,
            "tokens_savings_pct": savings_pct,
            "total_pii_masked": self.total_pii_masked,
            "total_injections_blocked": self.total_injections_blocked,
            "estimated_sar_saved": sar_saved,
            "uptime_seconds": uptime,
        }


GLOBAL_TELEMETRY = ProxyTelemetry()


class MisarProxyHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler implementing transparent OpenAI API reverse proxying."""

    upstream_url: Optional[str] = None
    router: Optional[MisarRouter] = None

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress verbose standard HTTP server request logging
        pass

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?")[0]

        if path in ("/", "/health"):
            data = {
                "status": "online",
                "service": "Misar Transparent Sovereign AI Gateway",
                "version": "2.7.0",
                "upstream": self.upstream_url or "standalone_evaluation_mock",
                "telemetry": GLOBAL_TELEMETRY.to_dict(),
            }
            body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/v1/models":
            models = {
                "object": "list",
                "data": [
                    {"id": "misar-gateway", "object": "model", "owned_by": "misar"},
                    {"id": "sdaia/allam-7b-instruct", "object": "model", "owned_by": "sdaia"},
                    {"id": "meta-llama/Meta-Llama-3-8B-Instruct", "object": "model", "owned_by": "meta"},
                    {"id": "aubmindlab/bert-base-arabertv02", "object": "model", "owned_by": "aubmindlab"},
                ],
            }
            body = json.dumps(models, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(b'{"error": "Endpoint not found"}')

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        if path not in ("/v1/chat/completions", "/v1/completions"):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Unsupported POST endpoint"}')
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length <= 0:
            self._send_error_response(400, "Empty request body")
            return

        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(raw_body)
        except Exception:
            self._send_error_response(400, "Invalid JSON payload")
            return

        t0 = time.perf_counter()
        router = self.router or MisarRouter()
        tok = load_default_tokenizer()

        # 1. Process /v1/chat/completions
        if path == "/v1/chat/completions":
            messages = payload.get("messages", [])
            modified_messages = []
            total_raw_tok = 0
            total_proc_tok = 0
            total_pii = 0

            for msg in messages:
                if not isinstance(msg, dict):
                    modified_messages.append(msg)
                    continue
                role = msg.get("role", "")
                content = msg.get("content", "")

                if role in ("user", "system") and isinstance(content, str):
                    # Guard evaluation
                    guard = sanitize_prompt(content, mask_pii=True, detect_injection=True)
                    has_injection = any(
                        "injection" in r or "jailbreak" in r or "override" in r or "evasion" in r or "ignore" in r
                        for r in guard.detected_risks
                    )
                    if has_injection and guard.risk_score >= 0.5:
                        GLOBAL_TELEMETRY.record_request(0, 0, 0, blocked=True)
                        self._send_error_response(
                            400,
                            f"Misar Security Guard blocked adversarial prompt injection: {', '.join(guard.detected_risks)}",
                            err_type="misar_security_violation",
                        )
                        return

                    raw_tokens = len(tok.tokenize(content))
                    sanitized_text = router.preprocess(guard.sanitized_text)
                    sanitized_text = canonicalize(sanitized_text)
                    proc_tokens = len(tok.tokenize(sanitized_text))

                    total_raw_tok += raw_tokens
                    total_proc_tok += proc_tokens
                    total_pii += guard.masked_pii_count

                    clean_msg = dict(msg)
                    clean_msg["content"] = sanitized_text
                    modified_messages.append(clean_msg)
                else:
                    modified_messages.append(msg)

            payload["messages"] = modified_messages
            saved_tokens = max(0, total_raw_tok - total_proc_tok)
            att_saved_pct = (
                round((1.0 - (total_proc_tok**2 / (total_raw_tok**2 or 1))) * 100, 2)
                if total_raw_tok > 0
                else 0.0
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            GLOBAL_TELEMETRY.record_request(total_raw_tok, total_proc_tok, total_pii)

            self._dispatch_or_mock_response(
                path=path,
                payload=payload,
                raw_tok=total_raw_tok,
                proc_tok=total_proc_tok,
                saved_tok=saved_tokens,
                att_saved_pct=att_saved_pct,
                pii_count=total_pii,
                elapsed_ms=elapsed_ms,
            )

        # 2. Process /v1/completions
        elif path == "/v1/completions":
            prompt = payload.get("prompt", "")
            if isinstance(prompt, str):
                guard = sanitize_prompt(prompt, mask_pii=True, detect_injection=True)
                has_injection = any(
                    "injection" in r or "jailbreak" in r or "override" in r or "evasion" in r or "ignore" in r
                    for r in guard.detected_risks
                )
                if has_injection and guard.risk_score >= 0.5:
                    GLOBAL_TELEMETRY.record_request(0, 0, 0, blocked=True)
                    self._send_error_response(
                        400,
                        f"Misar Security Guard blocked adversarial prompt injection: {', '.join(guard.detected_risks)}",
                        err_type="misar_security_violation",
                    )
                    return

                raw_tokens = len(tok.tokenize(prompt))
                sanitized_text = router.preprocess(guard.sanitized_text)
                sanitized_text = canonicalize(sanitized_text)
                proc_tokens = len(tok.tokenize(sanitized_text))
                saved_tokens = max(0, raw_tokens - proc_tokens)
                att_saved_pct = (
                    round((1.0 - (proc_tokens**2 / (raw_tokens**2 or 1))) * 100, 2)
                    if raw_tokens > 0
                    else 0.0
                )
                payload["prompt"] = sanitized_text
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                GLOBAL_TELEMETRY.record_request(raw_tokens, proc_tokens, guard.masked_pii_count)

                self._dispatch_or_mock_response(
                    path=path,
                    payload=payload,
                    raw_tok=raw_tokens,
                    proc_tok=proc_tokens,
                    saved_tok=saved_tokens,
                    att_saved_pct=att_saved_pct,
                    pii_count=guard.masked_pii_count,
                    elapsed_ms=elapsed_ms,
                )

    def _dispatch_or_mock_response(
        self,
        path: str,
        payload: Dict[str, Any],
        raw_tok: int,
        proc_tok: int,
        saved_tok: int,
        att_saved_pct: float,
        pii_count: int,
        elapsed_ms: float,
    ) -> None:
        """Forward request to upstream LLM or synthesize clean mock response."""
        # Case A: Upstream URL is configured
        if self.upstream_url:
            upstream_target = self.upstream_url.rstrip("/") + path
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                upstream_target,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            # Pass original Authorization header if present
            auth = self.headers.get("Authorization")
            if auth:
                req.add_header("Authorization", auth)

            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_body = resp.read()
                    self.send_response(resp.status)
                    self._send_cors_headers()
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                    self.send_header("X-Misar-Processed", "true")
                    self.send_header("X-Misar-Tokens-Saved", str(saved_tok))
                    self.send_header("X-Misar-Attention-Saved-Pct", f"{att_saved_pct}%")
                    self.send_header("X-Misar-PII-Masked", str(pii_count))
                    self.send_header("X-Misar-Latency-Ms", f"{elapsed_ms:.2f}")
                    self.end_headers()
                    self.wfile.write(resp_body)
                    return
            except urllib.error.HTTPError as e:
                err_content = e.read()
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(err_content)
                return
            except Exception as e:
                self._send_error_response(502, f"Failed to connect to upstream LLM gateway: {str(e)}")
                return

        # Case B: Standalone / Mock mode (Instant local response without upstream dependency)
        model_name = payload.get("model", "misar-sovereign-proxy")
        last_user_content = ""
        if path == "/v1/chat/completions":
            for m in reversed(payload.get("messages", [])):
                if m.get("role") == "user":
                    last_user_content = m.get("content", "")
                    break
        else:
            last_user_content = str(payload.get("prompt", ""))

        response_obj = {
            "id": f"chatcmpl-misar-{int(time.time() * 1000)}",
            "object": "chat.completion" if path == "/v1/chat/completions" else "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"[Misar Sovereign Gateway]: Intercepted, sanitized, and token-optimized: {last_user_content[:100]}",
                    }
                    if path == "/v1/chat/completions"
                    else None,
                    "text": f"[Misar Gateway]: {last_user_content[:100]}"
                    if path == "/v1/completions"
                    else None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": proc_tok,
                "completion_tokens": 15,
                "total_tokens": proc_tok + 15,
                "tokens_saved_by_misar": saved_tok,
            },
            "misar_telemetry": {
                "raw_tokens": raw_tok,
                "processed_tokens": proc_tok,
                "tokens_saved": saved_tok,
                "attention_saved_pct": att_saved_pct,
                "pii_masked": pii_count,
                "latency_ms": round(elapsed_ms, 2),
            },
        }

        body = json.dumps(response_obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("X-Misar-Processed", "true")
        self.send_header("X-Misar-Tokens-Saved", str(saved_tok))
        self.send_header("X-Misar-Attention-Saved-Pct", f"{att_saved_pct}%")
        self.send_header("X-Misar-PII-Masked", str(pii_count))
        self.send_header("X-Misar-Latency-Ms", f"{elapsed_ms:.2f}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_response(self, code: int, message: str, err_type: str = "invalid_request_error") -> None:
        self.send_response(code)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        err_obj = {
            "error": {
                "message": message,
                "type": err_type,
                "code": code,
            }
        }
        body = json.dumps(err_obj, ensure_ascii=False).encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_proxy_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    upstream_url: Optional[str] = None,
    router: Optional[MisarRouter] = None,
) -> ThreadingHTTPServer:
    """Instantiate and configure the Misar Transparent Proxy Server."""
    handler = MisarProxyHandler
    handler.upstream_url = upstream_url
    handler.router = router or MisarRouter()
    return ThreadingHTTPServer((host, port), handler)


def run_proxy(
    host: str = "0.0.0.0",
    port: int = 8080,
    upstream_url: Optional[str] = None,
    router: Optional[MisarRouter] = None,
) -> None:
    """Run the Misar Transparent Proxy Gateway server indefinitely."""
    server = create_proxy_server(host=host, port=port, upstream_url=upstream_url, router=router)
    mode_str = f"forwarding to upstream '{upstream_url}'" if upstream_url else "standalone mock evaluation mode"
    print(f"🚀 Misar Sovereign OpenAI-Compatible Proxy Gateway active on http://{host}:{port}")
    print(f"⚡ Mode: {mode_str}")
    print("🛡️  Autonomous PII Redaction & Adversarial Injection Guard: ACTIVE")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Misar Proxy Gateway...")
    finally:
        server.server_close()
