# -*- coding: utf-8 -*-
"""
Command-line interface for Misar (مِسار).
Usage:
  misar demo              Run standard Arabic samples with real AraBERTv2 tokenization
  misar route "<text>"    Route a single Arabic text and output JSON or summary
  misar serve             Start local Misar Web Workbench & REST API
  misar bench             Run reproducible WordPiece fragmentation & latency benchmark
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .benchmark import run_benchmark
from .router import MisarRouter
from .web.server import STANDARD_SAMPLES, run_server


def _ensure_utf8_stdio() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def cmd_demo(args: argparse.Namespace) -> int:
    router = MisarRouter(model_key=args.model, allow_vocab_download=not args.offline)
    print(f"Misar v{__version__} | Tokenizer: {router.tokenizer.backend_name}")
    print("=" * 84)
    for idx, sample in enumerate(STANDARD_SAMPLES, 1):
        res = router.route(sample["input"])
        print(f"[{idx}] {sample['category']}")
        print(f"    Route  : {res.route_name} ({res.latency_ms:.3f} ms)")
        print(f"    Profile: P_dial={res.profile.dialect_ratio:.2f} | D_diac={res.profile.diacritic_density:.2f} | E_noise={res.profile.elongation_noise_score:.2f} | H={res.profile.shannon_entropy:.2f}")
        print(f"    Output : {res.processed_text}")
        print(f"    Tokens : {res.raw_token_count} -> {res.processed_token_count} (TFR {res.tfr_raw:.2f} -> {res.tfr_processed:.2f}, reduction {res.token_reduction_pct:+.1f}%)")
        print("-" * 84)
    return 0


def cmd_route(args: argparse.Namespace) -> int:
    router = MisarRouter(model_key=args.model, allow_vocab_download=not args.offline)
    res = router.route(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"Route    : {res.route_name} - {res.route_display_ar}")
        print(f"Reason   : {res.decision_reason}")
        print(f"Output   : {res.processed_text}")
        print(f"Tokenizer: {res.tokenizer_backend}")
        print(f"Tokens   : {res.raw_token_count} -> {res.processed_token_count} (TFR {res.tfr_raw:.2f} -> {res.tfr_processed:.2f}, {res.token_reduction_pct:+.1f}%)")
        print(f"Latency  : {res.latency_ms:.3f} ms | Entropy: {res.profile.shannon_entropy:.2f} bits")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    run_server(
        host=args.host,
        port=args.port,
        model_key=args.model,
        open_browser=not args.no_browser,
        test_once=args.test_once,
        offline=args.offline,
    )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    save_path = Path(args.save_report) if args.save_report else None
    run_benchmark(model_key=args.model, offline=args.offline, save_report_path=save_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="misar",
        description="Misar (مِسار): Adaptive Context-Aware Arabic NLP Preprocessing Router",
    )
    parser.add_argument("--version", action="version", version=f"misar {__version__}")

    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo", help="Run standard evaluation samples")
    p_demo.add_argument("--model", default="arabertv02", choices=["arabertv02", "camelbert-mix"])
    p_demo.add_argument("--offline", action="store_true", help="Disable vocab download")
    p_demo.set_defaults(func=cmd_demo)

    p_route = sub.add_parser("route", help="Route a single Arabic text")
    p_route.add_argument("text", help="Arabic text to analyze and preprocess")
    p_route.add_argument("--json", action="store_true", help="Output structured JSON")
    p_route.add_argument("--model", default="arabertv02", choices=["arabertv02", "camelbert-mix"])
    p_route.add_argument("--offline", action="store_true")
    p_route.set_defaults(func=cmd_route)

    p_serve = sub.add_parser("serve", help="Start local Misar Web Workbench & REST API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8050)
    p_serve.add_argument("--model", default="arabertv02", choices=["arabertv02", "camelbert-mix"])
    p_serve.add_argument("--no-browser", action="store_true")
    p_serve.add_argument("--test-once", action="store_true")
    p_serve.add_argument("--offline", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_bench = sub.add_parser("bench", help="Run reproducible 15-sample benchmark")
    p_bench.add_argument("--model", default="arabertv02", choices=["arabertv02", "camelbert-mix"])
    p_bench.add_argument("--offline", action="store_true")
    p_bench.add_argument("--save-report", default="", help="Optional JSON output path")
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    sys.exit(args.func(args))
