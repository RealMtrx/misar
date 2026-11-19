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
from .evaluation import run_evaluation
from .pipelines import SUPPORTED_EMOJI_MODES
from .router import MisarRouter
from .tokenizer import PRETRAINED_MODEL_CONFIGS
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
    router = MisarRouter(
        model_key=args.model,
        allow_vocab_download=not args.offline,
        auto_heal=not args.no_auto_heal,
        emoji_mode=args.emoji_mode,
        arabizi_mode=args.arabizi,
    )
    if args.compare:
        comp = router.compare_baselines(args.text)
        if args.json:
            print(json.dumps(comp, ensure_ascii=False, indent=2))
            return 0
        res_m = comp["misar"]
        print(f"Tokenizer : {res_m['tokenizer_backend']}")
        print(f"Raw       : ({res_m['raw_token_count']} tokens, UNK={res_m['unk_count_raw']}) -> {res_m['original_text']}")
        print(f"Legacy    : ({comp['legacy_token_count']} tokens, UNK={comp['legacy_unk_count']}) -> {comp['legacy_text']}")
        print(f"Misar v2  : ({res_m['processed_token_count']} tokens, UNK={res_m['unk_count_processed']}) -> {res_m['processed_text']}")
        print(f"Gain      : vs Raw {res_m['token_reduction_pct']:+.1f}% | vs Legacy {comp['vs_legacy_reduction_pct']:+.1f}% | AutoHeal={res_m['auto_healed_count']} | Clitics={res_m['clitic_fixed_count']}")
        return 0

    res = router.route(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"Route    : {res.route_name} - {res.route_display_ar}")
        print(f"Reason   : {res.decision_reason}")
        print(f"Output   : {res.processed_text}")
        print(f"Tokenizer: {res.tokenizer_backend}")
        print(f"Tokens   : {res.raw_token_count} -> {res.processed_token_count} (TFR {res.tfr_raw:.2f} -> {res.tfr_processed:.2f}, {res.token_reduction_pct:+.1f}%)")
        print(f"V2 Stats : AutoHeal={res.auto_healed_count} | Clitics={res.clitic_fixed_count} | Latency={res.latency_ms:.3f} ms | Entropy={res.profile.shannon_entropy:.2f} bits")
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


def cmd_eval(args: argparse.Namespace) -> int:
    keys = sorted(PRETRAINED_MODEL_CONFIGS.keys()) if args.all_models else [args.model]
    run_evaluation(model_keys=keys, offline=args.offline, json_output=args.json)
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    input_path = Path(args.source)
    if input_path.exists() and input_path.is_file():
        text = input_path.read_text(encoding="utf-8", errors="replace")
    else:
        text = args.source

    from .rag import ArabicDiscourseChunker
    chunker = ArabicDiscourseChunker(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        target_model=args.model,
        auto_preprocess=not args.raw,
    )
    chunks = chunker.chunk(text)

    if args.json:
        data = {
            "total_chunks": len(chunks),
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap,
            "chunks": [c.to_dict() for c in chunks],
        }
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    print(f"Misar RAG Semantic Chunker v{__version__} | Total Chunks: {len(chunks)} | Target Model: {args.model}")
    print("=" * 84)
    for c in chunks:
        print(f"Chunk #{c.chunk_index} [{c.start_char}:{c.end_char}] | Tokens: {c.token_count} | Route: {c.route_used}")
        print(f"  Text: {c.text}")
        print("-" * 84)
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    from .guard import ArabicPromptGuard
    guard = ArabicPromptGuard(
        mask_pii=not args.no_mask_pii,
        detect_injection=not args.no_detect_injection,
    )
    res = guard.sanitize(args.text)

    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0

    status_tag = "⚠️ SUSPICIOUS / RISKY" if res.is_suspicious else "✅ CLEAN / SAFE"
    print(f"Misar LLM Prompt Guard v{__version__} | Status: {status_tag} | Risk Score: {res.risk_score:.2f}")
    print("=" * 84)
    print(f"Detected Risks : {', '.join(res.detected_risks) if res.detected_risks else 'None'}")
    print(f"Masked PII     : {res.masked_pii_count} item(s)")
    print(f"Stripped Chars : {res.stripped_unicode_count} invisible/bidi char(s)")
    print(f"Sanitized Text :\n{res.sanitized_text}")
    print("=" * 84)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="misar",
        description="Misar (مِسار): Adaptive Context-Aware Arabic NLP Preprocessing Router",
    )
    parser.add_argument("--version", action="version", version=f"misar {__version__}")

    model_choices = sorted(PRETRAINED_MODEL_CONFIGS.keys())
    sub = parser.add_subparsers(dest="command")

    p_demo = sub.add_parser("demo", help="Run standard evaluation samples")
    p_demo.add_argument("--model", default="arabertv02", choices=model_choices)
    p_demo.add_argument("--offline", action="store_true", help="Disable vocab download")
    p_demo.set_defaults(func=cmd_demo)

    p_route = sub.add_parser("route", help="Route a single Arabic text")
    p_route.add_argument("text", help="Arabic text to analyze and preprocess")
    p_route.add_argument("--json", action="store_true", help="Output structured JSON")
    p_route.add_argument("--compare", action="store_true", help="Compare Raw vs Legacy Static vs Misar")
    p_route.add_argument("--model", default="arabertv02", choices=model_choices)
    p_route.add_argument("--emoji-mode", default="strip", choices=list(SUPPORTED_EMOJI_MODES))
    p_route.add_argument("--arabizi", action="store_true", help="Enable opt-in Arabizi transliteration")
    p_route.add_argument("--no-auto-heal", action="store_true", help="Disable vocab-verified Auto-Heal")
    p_route.add_argument("--offline", action="store_true")
    p_route.set_defaults(func=cmd_route)

    p_serve = sub.add_parser("serve", help="Start local Misar Web Workbench & REST API")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8050)
    p_serve.add_argument("--model", default="arabertv02", choices=model_choices)
    p_serve.add_argument("--no-browser", action="store_true")
    p_serve.add_argument("--test-once", action="store_true")
    p_serve.add_argument("--offline", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_bench = sub.add_parser("bench", help="Run reproducible 15-sample benchmark")
    p_bench.add_argument("--model", default="arabertv02", choices=model_choices)
    p_bench.add_argument("--offline", action="store_true")
    p_bench.add_argument("--save-report", default="", help="Optional JSON output path")
    p_bench.set_defaults(func=cmd_bench)

    p_eval = sub.add_parser("eval", help="Run semantic disambiguation, negation & cross-model evaluation")
    p_eval.add_argument("--model", default="arabertv02", choices=model_choices)
    p_eval.add_argument("--all-models", action="store_true", help="Evaluate across all 5 Arabic Transformer vocabs")
    p_eval.add_argument("--json", action="store_true", help="Output structured JSON")
    p_eval.add_argument("--offline", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_chunk = sub.add_parser("chunk", help="Chunk Arabic documents with discourse & negation preservation")
    p_chunk.add_argument("source", help="File path or raw Arabic string to chunk")
    p_chunk.add_argument("--max-tokens", type=int, default=256, help="Maximum WordPiece tokens per chunk")
    p_chunk.add_argument("--overlap", type=int, default=32, help="Overlap WordPiece tokens between chunks")
    p_chunk.add_argument("--model", default="arabertv02", choices=model_choices)
    p_chunk.add_argument("--raw", action="store_true", help="Do not auto-preprocess with Misar")
    p_chunk.add_argument("--json", action="store_true", help="Output structured JSON")
    p_chunk.set_defaults(func=cmd_chunk)

    p_guard = sub.add_parser("guard", help="Sanitize prompts against injection, jailbreaks, and mask PII")
    p_guard.add_argument("text", help="Arabic prompt to inspect and sanitize")
    p_guard.add_argument("--no-mask-pii", action="store_true", help="Disable PII masking")
    p_guard.add_argument("--no-detect-injection", action="store_true", help="Disable injection detection")
    p_guard.add_argument("--json", action="store_true", help="Output structured JSON")
    p_guard.set_defaults(func=cmd_guard)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    sys.exit(args.func(args))
