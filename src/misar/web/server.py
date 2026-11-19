# -*- coding: utf-8 -*-
"""
Lightweight zero-dependency HTTP server for the Misar (مِسار) Web Workbench & REST API.
"""

import json
import os
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from .. import __version__
from ..pipelines import SUPPORTED_EMOJI_MODES
from ..router import MisarRouter
from ..tokenizer import PRETRAINED_MODEL_CONFIGS, TokenizerProtocol, load_default_tokenizer

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_BODY_BYTES = 1_048_576  # 1 MiB payload cap to protect public server instances
MAX_BATCH_SIZE = 500

STANDARD_SAMPLES: List[Dict[str, str]] = [
    {
        "label": "لهجة مطورين (سعودية + API)",
        "category": "لهجة مطورين سعودية / تقنية (Saudi Dev + Code-Switch)",
        "input": "والله السيرفر الحين صار يجنن واستجابته في API سريعة شويتين بس الكاش يبيله تضبيط",
    },
    {
        "label": "لهجة مطورين (ترميم إملائي + Clitics)",
        "category": "لهجة خليجية وشامية تقنية (Auto-Heal + Clitics + Dialect)",
        "input": "ياخي البرمجه في المنصه هاذي كتير مرتبة وماكو تعقيد في الـ Backend وCache",
    },
    {
        "label": "فصحى تقنية (حفظ الإملاء)",
        "category": "فصحى هندسية وتقنية (MSA Engineering + Transformers)",
        "input": "تعتمد الحوسبة السحابية الحديثة على هندسة النماذج اللغوية الضخمة مثل Transformers لتحسين كفاءة المعالجة وتقليل زمن الاستدلال",
    },
    {
        "label": "تغريدة أعطال وضجيج تقني",
        "category": "تغريدة تقنية هجينة (Noisy Dev Social + Dialect)",
        "input": "السيرفررررر بطييييء مررررة ومافيه استجابة في API أبدددداً 😡😡👎 #هندسة_البرمجيات",
    },
    {
        "label": "نص تراثي مشكول (الخوارزمي)",
        "category": "نص تراثي علمي مشكول (Classical Diacritized)",
        "input": "وَضَعَ الْخَوَارِزْمِيُّ أُسُسَ الْجَبْرِ وَالْحِسَابِ لِيَبْنِيَ مِنْهَا الْعُلَمَاءُ مَنَاهِجَ الْبَرْمَجَةِ الْحَدِيثَةِ",
    },
    {
        "label": "حماية النفي البرمجي والتضعيف",
        "category": "اختبار النفي التقني والتضعيف (Negation & Gemination)",
        "input": "والللله الكود مو معقد والأخطاء متكررررة في السيرفر وما أبي أعيد بناء Docker الحين",
    },
]


def _format_compare_payload(comp: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(comp["misar"])
    for key in (
        "legacy_text",
        "legacy_tokens",
        "legacy_token_count",
        "legacy_unk_count",
        "legacy_tfr",
        "vs_legacy_reduction_pct",
    ):
        out[key] = comp[key]
    return out


def _format_document_payload(active_router: MisarRouter, text: str) -> Dict[str, Any]:
    doc_res = active_router.route_document(text)
    out = doc_res.to_dict()
    profile = active_router.profiler.analyze(text)
    raw_tokens = [tok for seg in doc_res.segments for tok in seg.raw_tokens]
    proc_tokens = [tok for seg in doc_res.segments for tok in seg.processed_tokens]

    legacy_text = active_router._legacy_baseline.process(text, apply_dialect_cleanup=False)
    legacy_tokens = active_router.tokenizer.tokenize(legacy_text)
    legacy_count = len(legacy_tokens)
    if active_router.tokenizer.is_real_vocab:
        legacy_unk = sum(1 for tok in legacy_tokens if tok == "[UNK]" or not active_router.tokenizer.has_token(tok))
    else:
        legacy_unk = legacy_tokens.count("[UNK]")
    legacy_words = max(1, len(legacy_text.split())) if legacy_text else 1
    legacy_tfr = (legacy_count / legacy_words) if legacy_count > 0 else 0.0
    vs_legacy_pct = (
        ((legacy_count - doc_res.processed_token_count) / legacy_count * 100.0)
        if legacy_count > 0
        else 0.0
    )

    raw_words = max(1, profile.word_count) if profile.word_count > 0 else 1
    proc_words = max(1, len(doc_res.processed_text.split())) if doc_res.processed_text else 1
    active_routes = [(k, v) for k, v in doc_res.route_distribution.items() if v > 0]
    dominant = max(active_routes, key=lambda kv: kv[1])[0] if active_routes else "FAST_MSA"
    summary_parts = " · ".join(f"{cnt} {name}" for name, cnt in active_routes) or "0 segments"

    out.update(
        {
            "is_document_mode": True,
            "is_real_tokenizer": active_router.tokenizer.is_real_vocab,
            "route_name": dominant,
            "route_display_ar": "التوجيه متعدد الجمل والفقرات (Clause-Aware)",
            "decision_reason": f"توجيه سياقي على مستوى الجمل ({doc_res.segment_count} مقاطع: {summary_parts})",
            "profile": profile.to_dict(),
            "raw_tokens": raw_tokens,
            "processed_tokens": proc_tokens,
            "tfr_raw": round((doc_res.raw_token_count / raw_words) if doc_res.raw_token_count > 0 else 0.0, 3),
            "tfr_processed": round((doc_res.processed_token_count / proc_words) if doc_res.processed_token_count > 0 else 0.0, 3),
            "legacy_text": legacy_text,
            "legacy_tokens": legacy_tokens,
            "legacy_token_count": legacy_count,
            "legacy_unk_count": legacy_unk,
            "legacy_tfr": round(legacy_tfr, 3),
            "vs_legacy_reduction_pct": round(vs_legacy_pct, 2),
        }
    )
    return out


def make_handler(router: MisarRouter, allow_vocab_download: bool = True):
    cache_lock = threading.Lock()
    tok_cache: Dict[str, TokenizerProtocol] = {router.model_key: router.tokenizer}
    router_cache: Dict[Tuple[str, bool, str, bool], MisarRouter] = {
        (router.model_key, router.auto_heal, router.emoji_mode, router.arabizi_mode): router
    }

    def _resolve_router(
        model_key: str,
        auto_heal: bool,
        emoji_mode: str,
        arabizi_mode: bool,
    ) -> MisarRouter:
        key = (model_key, auto_heal, emoji_mode, arabizi_mode)
        with cache_lock:
            if key in router_cache:
                return router_cache[key]
            tok = tok_cache.get(model_key)
            if tok is None:
                tok = load_default_tokenizer(model_key=model_key, allow_download=allow_vocab_download)
                tok_cache[model_key] = tok
            inst = MisarRouter(
                tokenizer=tok,
                model_key=model_key,
                auto_heal=auto_heal,
                emoji_mode=emoji_mode,
                arabizi_mode=arabizi_mode,
            )
            router_cache[key] = inst
            return inst

    class MisarHTTPHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "SAMEORIGIN")
            self.send_header("Referrer-Policy", "no-referrer")

        def _safe_write(self, data: bytes) -> None:
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            self.end_headers()
            self._safe_write(body)

        def _read_json_object(self) -> Optional[Dict[str, Any]]:
            raw_len = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_len)
            except ValueError:
                self._send_json(400, {"error": "Invalid Content-Length header"})
                return None
            if length < 0:
                self._send_json(400, {"error": "Negative Content-Length header"})
                return None
            if length > MAX_BODY_BYTES:
                self._send_json(413, {"error": f"Payload exceeds {MAX_BODY_BYTES} byte limit"})
                return None
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "Invalid UTF-8 or JSON body"})
                return None
            if not isinstance(payload, dict):
                self._send_json(400, {"error": "JSON payload must be an object"})
                return None
            return payload

        def do_GET(self) -> None:
            route_path = urllib.parse.urlparse(self.path).path
            if route_path in ("/", "/index.html"):
                html_path = os.path.join(WEB_DIR, "index.html")
                if not os.path.exists(html_path):
                    self._send_json(404, {"error": "index.html not found"})
                    return
                with open(html_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self._send_security_headers()
                self.end_headers()
                self._safe_write(data)
                return

            if route_path == "/api/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "version": __version__,
                        "model_key": router.model_key,
                        "tokenizer_backend": router.tokenizer.backend_name,
                        "is_real_tokenizer": router.tokenizer.is_real_vocab,
                        "available_models": [
                            {"key": k, "display_name": cfg["display_name"]}
                            for k, cfg in PRETRAINED_MODEL_CONFIGS.items()
                        ],
                    },
                )
                return

            if route_path == "/api/suite":
                samples = []
                for idx, item in enumerate(STANDARD_SAMPLES, 1):
                    row = _format_compare_payload(router.compare_baselines(item["input"]))
                    row["idx"] = idx
                    row["category"] = item["category"]
                    row["label"] = item["label"]
                    samples.append(row)

                avg_lat = sum(r["latency_ms"] for r in samples) / len(samples)
                avg_red = sum(r["token_reduction_pct"] for r in samples) / len(samples)
                avg_tfr_raw = sum(r["tfr_raw"] for r in samples) / len(samples)
                avg_tfr_proc = sum(r["tfr_processed"] for r in samples) / len(samples)
                avg_tfr_leg = sum(r["legacy_tfr"] for r in samples) / len(samples)
                avg_vs_leg = sum(r["vs_legacy_reduction_pct"] for r in samples) / len(samples)

                self._send_json(
                    200,
                    {
                        "samples": samples,
                        "summary": {
                            "avg_latency_ms": round(avg_lat, 4),
                            "avg_reduction_pct": round(avg_red, 2),
                            "avg_vs_legacy_reduction_pct": round(avg_vs_leg, 2),
                            "avg_tfr_raw": round(avg_tfr_raw, 3),
                            "avg_tfr_legacy": round(avg_tfr_leg, 3),
                            "avg_tfr_processed": round(avg_tfr_proc, 3),
                            "tokenizer_backend": router.tokenizer.backend_name,
                            "is_real_tokenizer": router.tokenizer.is_real_vocab,
                        },
                    },
                )
                return

            self._send_json(404, {"error": "Not found"})

        def _handle_route_post(self, force_document: bool = False) -> None:
            payload = self._read_json_object()
            if payload is None:
                return
            text = payload.get("text", "")
            if not isinstance(text, str):
                self._send_json(400, {"error": "'text' must be a string"})
                return

            model_key = payload.get("model_key", router.model_key)
            emoji_mode = payload.get("emoji_mode", router.emoji_mode)
            auto_heal = payload.get("auto_heal", router.auto_heal)
            arabizi_mode = payload.get("arabizi_mode", router.arabizi_mode)
            document_mode = payload.get("document_mode", force_document)

            if model_key not in PRETRAINED_MODEL_CONFIGS:
                self._send_json(400, {"error": f"Unknown model_key '{model_key}'"})
                return
            if emoji_mode not in SUPPORTED_EMOJI_MODES:
                self._send_json(400, {"error": f"Invalid emoji_mode '{emoji_mode}'"})
                return
            if not isinstance(auto_heal, bool) or not isinstance(arabizi_mode, bool) or not isinstance(document_mode, bool):
                self._send_json(400, {"error": "'auto_heal', 'arabizi_mode', and 'document_mode' must be booleans"})
                return

            try:
                active_router = _resolve_router(model_key, auto_heal, emoji_mode, arabizi_mode)
                if force_document or document_mode:
                    self._send_json(200, _format_document_payload(active_router, text))
                else:
                    self._send_json(200, _format_compare_payload(active_router.compare_baselines(text)))
            except Exception:
                self._send_json(500, {"error": "Internal routing error"})

        def do_POST(self) -> None:
            route_path = urllib.parse.urlparse(self.path).path
            if route_path == "/api/route":
                self._handle_route_post(force_document=False)
                return

            if route_path == "/api/document":
                self._handle_route_post(force_document=True)
                return

            if route_path == "/api/batch":
                payload = self._read_json_object()
                if payload is None:
                    return
                texts = payload.get("texts", [])
                if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                    self._send_json(400, {"error": "'texts' must be a list of strings"})
                    return
                if len(texts) > MAX_BATCH_SIZE:
                    self._send_json(400, {"error": f"Batch size exceeds {MAX_BATCH_SIZE} limit"})
                    return
                try:
                    results = [r.to_dict() for r in router.route_batch(texts)]
                    self._send_json(200, {"results": results})
                except Exception:
                    self._send_json(500, {"error": "Internal batch routing error"})
                return

            self._send_json(404, {"error": "Not found"})

    return MisarHTTPHandler


def bind_server(
    router: MisarRouter,
    host: str = "127.0.0.1",
    preferred_port: int = 8050,
    allow_vocab_download: bool = True,
) -> ThreadingHTTPServer:
    handler = make_handler(router, allow_vocab_download=allow_vocab_download)
    for port in range(preferred_port, preferred_port + 11):
        try:
            return ThreadingHTTPServer((host, port), handler)
        except OSError:
            continue
    return ThreadingHTTPServer((host, 0), handler)


def run_server(
    host: str = "127.0.0.1",
    port: int = 8050,
    model_key: str = "arabertv02",
    open_browser: bool = True,
    test_once: bool = False,
    offline: bool = False,
) -> None:
    router = MisarRouter(model_key=model_key, allow_vocab_download=not offline)
    server = bind_server(router, host=host, preferred_port=port, allow_vocab_download=not offline)
    bound_port = server.server_address[1]
    url = f"http://{host}:{bound_port}"

    if test_once:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=5) as r:
                health = json.loads(r.read().decode("utf-8"))
                assert r.status == 200 and health["status"] == "ok"

            with urllib.request.urlopen(f"{url}/", timeout=5) as r:
                html = r.read().decode("utf-8")
                assert r.status == 200 and "مِسار" in html

            req_data = json.dumps(
                {"text": "السيرفررررر بطييييء مررررة ومافيه استجابة في API 😡 #هندسة_البرمجيات"},
                ensure_ascii=False,
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{url}/api/route",
                data=req_data,
                headers={"Content-Type": "application/json; charset=utf-8"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                out = json.loads(r.read().decode("utf-8"))
                assert r.status == 200 and out["route_name"] == "NOISY_SOCIAL"
                assert out["processed_token_count"] < out["raw_token_count"]

            with urllib.request.urlopen(f"{url}/api/suite", timeout=5) as r:
                suite = json.loads(r.read().decode("utf-8"))
                assert r.status == 200 and len(suite["samples"]) == 6

            print(f"MISAR SERVER SELF-TEST PASSED on {url}:")
            print(f"  Tokenizer : {health['tokenizer_backend']} (real={health['is_real_tokenizer']})")
            print(f"  Route API : {out['route_name']} | tokens {out['raw_token_count']} -> {out['processed_token_count']} ({out['token_reduction_pct']:+.1f}%)")
            print(f"  Suite API : {len(suite['samples'])} samples | avg_tfr {suite['summary']['avg_tfr_raw']} -> {suite['summary']['avg_tfr_processed']}")
        finally:
            server.shutdown()
            server.server_close()
        return

    print("=" * 68)
    print(f"  Misar (مِسار) v{__version__} — Official Arabic NLP Workbench & API")
    print(f"  Tokenizer : {router.tokenizer.backend_name}")
    print(f"  Web UI    : {url}/")
    print(f"  REST API  : POST {url}/api/route | POST {url}/api/document | POST {url}/api/batch")
    print("  Press Ctrl+C to stop.")
    print("=" * 68)

    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(f"{url}/")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Misar server...")
    finally:
        server.shutdown()
        server.server_close()
