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
from typing import Any, Dict, List, Optional

from .. import __version__
from ..router import MisarRouter

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_BODY_BYTES = 1_048_576  # 1 MiB payload cap to protect public server instances
MAX_BATCH_SIZE = 500

STANDARD_SAMPLES: List[Dict[str, str]] = [
    {
        "label": "لهجة سعودية / خليجية",
        "category": "لهجة سعودية / خليجية (Saudi/Gulf Dialect)",
        "input": "والله الفندق مره يجنن وتعامله فخم وما عليه كلام بس السعر غالي شويتين",
    },
    {
        "label": "لهجة مصرية دارجة",
        "category": "لهجة مصرية دارجة (Egyptian Dialect)",
        "input": "الفكرة مش بطالة خالص بس التطبيق كان محتاج شوية تظبيط ودعم فني أسرع",
    },
    {
        "label": "فصحى رسمية (حفظ الإملاء)",
        "category": "فصحى معيارية / بيان رسمي (MSA)",
        "input": "أعلنت الهيئة السعودية للبيانات والذكاء الاصطناعي سدايا عن إطلاق حزمة جديدة من المبادرات الوطنية لتعزيز الابتكار وريادة الأعمال في قطاع التقنية",
    },
    {
        "label": "تغريدة مشوبة بالضجيج واللهجة",
        "category": "تغريدة منصات تواصل هجينة (Noisy + Dialect)",
        "input": "المطعممممم سيييييء جداااااا ومفيش أسلوب خالصصصصص 😡😡😡👎 #تجربة_فاشلة",
    },
    {
        "label": "نص فصيح مشكول",
        "category": "نص فصيح وتراثي مشكول (Classical Diacritized)",
        "input": "إِنَّ الَّذِي مَلَأَ اللُّغَاتِ مَحَاسِناً، جَعَلَ الجَمَالَ وَسِرَّهُ فِي الضَّادِ",
    },
    {
        "label": "حماية النفي العامي + التضعيف",
        "category": "اختبار النفي اللهجي والتضعيف (Negation & Gemination)",
        "input": "والللله المكان مو حلو أبداً والخدمة متكررررة وما أبي أرجع الحين",
    },
]


def make_handler(router: MisarRouter):
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
                        "tokenizer_backend": router.tokenizer.backend_name,
                        "is_real_tokenizer": router.tokenizer.is_real_vocab,
                    },
                )
                return

            if route_path == "/api/suite":
                samples = []
                for idx, item in enumerate(STANDARD_SAMPLES, 1):
                    res = router.route(item["input"])
                    row = res.to_dict()
                    row["idx"] = idx
                    row["category"] = item["category"]
                    row["label"] = item["label"]
                    samples.append(row)

                avg_lat = sum(r["latency_ms"] for r in samples) / len(samples)
                avg_red = sum(r["token_reduction_pct"] for r in samples) / len(samples)
                avg_tfr_raw = sum(r["tfr_raw"] for r in samples) / len(samples)
                avg_tfr_proc = sum(r["tfr_processed"] for r in samples) / len(samples)

                self._send_json(
                    200,
                    {
                        "samples": samples,
                        "summary": {
                            "avg_latency_ms": round(avg_lat, 4),
                            "avg_reduction_pct": round(avg_red, 2),
                            "avg_tfr_raw": round(avg_tfr_raw, 3),
                            "avg_tfr_processed": round(avg_tfr_proc, 3),
                            "tokenizer_backend": router.tokenizer.backend_name,
                            "is_real_tokenizer": router.tokenizer.is_real_vocab,
                        },
                    },
                )
                return

            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            route_path = urllib.parse.urlparse(self.path).path
            if route_path == "/api/route":
                payload = self._read_json_object()
                if payload is None:
                    return
                text = payload.get("text", "")
                if not isinstance(text, str):
                    self._send_json(400, {"error": "'text' must be a string"})
                    return
                try:
                    res = router.route(text)
                    self._send_json(200, res.to_dict())
                except Exception:
                    self._send_json(500, {"error": "Internal routing error"})
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


def bind_server(router: MisarRouter, host: str = "127.0.0.1", preferred_port: int = 8050) -> ThreadingHTTPServer:
    handler = make_handler(router)
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
    server = bind_server(router, host=host, preferred_port=port)
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
                {"text": "المطعممممم سيييييء جداااااا ومفيش أسلوب 😡 #تجربة_فاشلة"},
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
    print(f"  REST API  : POST {url}/api/route | POST {url}/api/batch")
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
