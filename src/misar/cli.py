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

    if getattr(args, "correct", False) or getattr(args, "mask_pii", False) or getattr(args, "mask_ner", False) or getattr(args, "translate_msa", False):
        res_u = router.process(
            args.text,
            correct=getattr(args, "correct", False),
            mask_pii=getattr(args, "mask_pii", False),
            mask_ner=getattr(args, "mask_ner", False),
            translate_msa=getattr(args, "translate_msa", False),
            return_telemetry=True,
        )
        if args.json:
            print(json.dumps(res_u.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"Unified Route : {res_u.route_name} - {res_u.route_display_ar}")
            print(f"Output        : {res_u.processed_text}")
            print(f"Telemetry     : Corrections={res_u.corrections_count} | PII Risks={len(res_u.pii_risks)} | Entities={res_u.entities_count} | Dialect={res_u.detected_dialect}")
            print(f"Tokens        : {res_u.tokens_raw} -> {res_u.tokens_processed} ({res_u.token_reduction_pct:+.1f}%) | Latency: {res_u.latency_ms:.3f} ms")
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
    if getattr(args, "text", None):
        from .benchmark import benchmark_text
        res = benchmark_text(args.text, offline=args.offline)
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=2))
            return 0
        print(f"Misar Multi-Model Benchmark for text: '{args.text}'")
        print("=" * 96)
        print(f"{'Model':<20} | {'Raw':>4} | {'Proc':>4} | {'Saved':>5} | {'Reduction %':>11} | {'Attn Savings %':>14} | {'Latency':>8}")
        print("-" * 96)
        for r in res["results"]:
            if "reduction_pct" in r:
                print(f"{r['model_key']:<20} | {r['raw_tokens']:>4} | {r['processed_tokens']:>4} | {r['saved_tokens']:>5} | {r['reduction_pct']:>10.1f}% | {r['attention_savings_pct']:>13.1f}% | {r['latency_ms']:>6.2f}ms")
            else:
                print(f"{r['model_key']:<20} | ERROR: {r.get('error', 'Unknown')}")
        print("=" * 96)
        print(f"Average Token Reduction: {res['avg_token_reduction_pct']}% | Quadratic Attention FLOPs Saved: {res['avg_attention_savings_pct']}%")
        return 0

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


def cmd_classify(args: argparse.Namespace) -> int:
    from .classifier import classify_dialect
    res = classify_dialect(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Dialect Classifier v{__version__}")
    print("=" * 84)
    print(f"Primary Dialect : {res.primary_dialect} (confidence {res.confidence:.1%})")
    print(f"Register        : {res.register} (confidence {res.register_confidence:.1%})")
    print(f"Dialect Scores  : {res.dialect_scores}")
    print(f"Detected Markers: {', '.join(res.detected_markers) if res.detected_markers else 'None'}")
    print("=" * 84)
    return 0


def cmd_canonicalize(args: argparse.Namespace) -> int:
    from .classifier import canonicalize
    res = canonicalize(args.text)
    if args.json:
        print(json.dumps({"original": args.text, "canonical": res}, ensure_ascii=False, indent=2))
    else:
        print(res)
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    from .parallel import process_dataset_file
    print(f"Misar Multi-Core Engine v{__version__} | Input: {args.input} | Workers: {args.workers or 'auto'}")
    summary = process_dataset_file(
        input_file=args.input,
        output_file=args.output,
        text_key=args.text_key,
        num_workers=args.workers,
        chunk_size=args.chunk_size,
        model_key=args.model,
    )
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Processed {summary['total_records']} records in {summary['elapsed_seconds']:.2f}s ({summary['throughput_per_sec']:.1f} texts/sec).")
        print(f"Saved to: {summary['output_file']}")
    return 0


def cmd_dict(args: argparse.Namespace) -> int:
    from .lexicon import lookup_word, extract_root
    entry = lookup_word(args.word)
    if args.json:
        payload = entry.to_dict() if entry else {"word": args.word, "found": False, "root": extract_root(args.word)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if entry:
        print(f"Misar Arabic Lexicon v{__version__} | Word: {entry.word}")
        print("=" * 84)
        print(f"Lemma (الفصيح) : {entry.lemma_msa}")
        print(f"Root (الجذر)   : {entry.root}")
        print(f"Dialect (اللهجة): {entry.dialect}")
        print(f"POS (النوع)    : {entry.pos}")
        print(f"Meaning (المعنى): {entry.definition}")
        if entry.synonyms:
            print(f"Synonyms       : {', '.join(entry.synonyms)}")
        if entry.example:
            print(f"Example        : {entry.example}")
        print("=" * 84)
    else:
        root = extract_root(args.word)
        print(f"Misar Arabic Lexicon v{__version__}")
        print("=" * 84)
        print(f"Word '{args.word}' not in curated dialect entries.")
        print(f"Extracted Morphological Root: {root}")
        print("=" * 84)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from .lexicon import explain_text
    res = explain_text(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"Misar Sentence Explainer v{__version__}")
    print("=" * 84)
    print(f"Original Text  : {res.original_text}")
    print(f"Canonical MSA  : {res.canonical_msa_text}")
    print(f"Dominant Reg   : {res.dominant_dialect} (Dialect Density: {res.dialect_density_pct:.1f}%)")
    print("-" * 84)
    for tok in res.tokens:
        if tok.is_dialect and tok.entry:
            print(f"• [{tok.token}] -> {tok.entry.lemma_msa} (Root: {tok.entry.root} | {tok.entry.dialect}): {tok.entry.definition}")
        else:
            print(f"• [{tok.token}] -> Root: {tok.detected_root}")
    print("=" * 84)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from .audit import audit_dataset_file, generate_html_report
    res = audit_dataset_file(
        input_file=args.input,
        text_key=args.text_key,
        cost_per_1k_tokens=args.cost_per_1k,
        currency=args.currency,
        model_key=args.model,
        max_records=args.max_records,
    )
    if args.output:
        report_html = generate_html_report(res)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report_html)
        print(f"Audit report saved to: {args.output}")

    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"Misar Executive Audit Report v{__version__}")
    print("=" * 84)
    print(f"Records Evaluated : {res.total_records:,} in {res.elapsed_seconds:.2f}s ({res.records_per_second:.1f} texts/sec)")
    print(f"Raw Tokens        : {res.raw_token_count:,} -> Processed: {res.processed_token_count:,}")
    print(f"Tokens Saved      : {res.tokens_saved:,} (-{res.token_reduction_pct:.1f}%)")
    print(f"Attention O(L^2)  : -{res.attention_flops_savings_pct:.1f}% FLOPs reduction")
    print(f"Financial Impact  : Raw {res.raw_cost:.4f} {res.currency} -> Proc {res.processed_cost:.4f} {res.currency} (Saved: {res.money_saved:.4f} {res.currency})")
    print(f"Security & PII    : {res.masked_pii_count} PII masked, {res.adversarial_risks_count} threats neutralized")
    if res.detected_risks:
        print(f"Threats Breakdown : {', '.join(f'{k}:{v}' for k, v in res.detected_risks.items())}")
    if res.dialect_distribution:
        print(f"Dialect Breakdown : {', '.join(f'{k}:{v}' for k, v in res.dialect_distribution.items())}")
    print("=" * 84)
    return 0


def cmd_proxy(args: argparse.Namespace) -> int:
    from .proxy import run_proxy
    run_proxy(host=args.host, port=args.port, upstream_url=args.upstream)
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    from .paper import generate_academic_paper, generate_markdown_summary
    if args.format == "markdown":
        content = generate_markdown_summary(args.output)
        fmt_name = "Markdown Research Dossier"
    else:
        content = generate_academic_paper(args.output)
        fmt_name = "IEEE/ACL LaTeX Research Preprint"

    if args.output:
        print(f"Generated {fmt_name} saved to: {args.output}")
    else:
        print(content)
    return 0


def cmd_arabizi(args: argparse.Namespace) -> int:
    from .arabizi import decode_arabizi
    result = decode_arabizi(args.text)
    if args.json:
        print(json.dumps({"original": args.text, "decoded_arabic": result}, ensure_ascii=False, indent=2))
    else:
        print(result)
    return 0


def cmd_augment(args: argparse.Namespace) -> int:
    from .augment import generate_stress_suite
    suite = generate_stress_suite(args.text)
    if args.json:
        print(json.dumps(suite, ensure_ascii=False, indent=2))
    else:
        print("Misar Synthetic Stress-Testing Suite:")
        print("=" * 80)
        for name, text in suite.items():
            print(f"• [{name}]:\n  {text}\n")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .search import BM25SearchEngine
    engine = BM25SearchEngine(use_roots=not args.no_roots)
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            return 1
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f, 1):
                clean = line.strip()
                if clean:
                    engine.add_document(f"line_{idx}", clean)
    elif args.docs:
        for idx, doc in enumerate(args.docs, 1):
            engine.add_document(f"doc_{idx}", doc)
    else:
        print("Error: Provide documents via --docs or --file", file=sys.stderr)
        return 1

    results = engine.search(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2))
        return 0

    print(f"Misar Arabic BM25 Search Results for: '{args.query}' (Total Indexed: {engine.total_docs})")
    print("=" * 84)
    if not results:
        print("No matching documents found.")
    for r in results:
        print(f"#{r.rank} [{r.doc_id}] Score: {r.score:.4f} | Matched: {', '.join(r.matched_terms)}")
        print(f"  Snippet: {r.snippet}")
        print("-" * 84)
    return 0


def cmd_sentiment(args: argparse.Namespace) -> int:
    from .sentiment import analyze_sentiment
    res = analyze_sentiment(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print(f"Misar Arabic Sentiment & Emotion Analyzer v{__version__}")
    print("=" * 70)
    print(f"Text       : {args.text}")
    print(f"Sentiment  : {res.label} (Score: {res.score:+.3f}, Confidence: {res.confidence * 100:.0f}%)")
    print(f"Emotion    : {res.emotion}")
    if res.positive_tokens:
        print(f"Positive   : {', '.join(res.positive_tokens)}")
    if res.negative_tokens:
        print(f"Negative   : {', '.join(res.negative_tokens)}")
    if res.negations_detected:
        print(f"Negations  : {', '.join(res.negations_detected)}")
    print("=" * 70)
    return 0


def cmd_correct(args: argparse.Namespace) -> int:
    from .correct import correct_arabic_text
    res = correct_arabic_text(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Arabic Grammar & Spell Correction v{__version__}")
    print("=" * 70)
    print(f"Original  : {res.original_text}")
    print(f"Corrected : {res.corrected_text}")
    print(f"Changes   : {res.changes_count} corrections applied (Confidence: {res.confidence * 100:.0f}%)")
    for c in res.changes:
        print(f"  • [{c.rule}] '{c.original}' -> '{c.corrected}'")
    print("=" * 70)
    return 0


def cmd_ner(args: argparse.Namespace) -> int:
    from .ner import extract_entities
    res = extract_entities(args.text, mask_entities=args.mask)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Arabic Named Entity Recognition (NER) v{__version__}")
    print("=" * 70)
    print(f"Input    : {res.original_text}")
    if args.mask:
        print(f"Masked   : {res.masked_text}")
    print(f"Entities : {res.entities_count} detected")
    for e in res.entities:
        print(f"  • [{e.category}] '{e.text}' (chars {e.start_char}:{e.end_char})")
    print("=" * 70)
    return 0


def cmd_translate(args: argparse.Namespace) -> int:
    from .translate import translate_to_msa
    res = translate_to_msa(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Dialect-to-MSA Parallel Translation v{__version__}")
    print("=" * 70)
    print(f"Dialect   : {res.detected_dialect.upper()}")
    print(f"Original  : {res.original_text}")
    print(f"MSA       : {res.msa_text}")
    print(f"Replaced  : {res.rules_applied_count} colloquial expressions")
    print("=" * 70)
    return 0


def cmd_diacritize(args: argparse.Namespace) -> int:
    from .diacritize import diacritize_text
    res = diacritize_text(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Arabic Diacritizer & Tashkeel Recovery v{__version__}")
    print("=" * 70)
    print(f"Original    : {res.original_text}")
    print(f"Diacritized : {res.diacritized_text}")
    print(f"Marks Added : {res.vowels_added_count} mark(s) | Density: {res.diacritic_density:.2f}")
    print(f"Confidence  : {res.confidence * 100:.0f}%")
    print("=" * 70)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    from .summarize import summarize_text
    res = summarize_text(args.text, num_sentences=args.sentences, max_ratio=args.ratio)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Arabic Text Summarizer & Keyphrase Extractor v{__version__}")
    print("=" * 70)
    print(f"Summary:\n{res.summary_text}")
    print("-" * 70)
    print(f"Compression : {res.compression_ratio * 100:.1f}% ({res.summary_sentence_count}/{res.original_sentence_count} sentences)")
    if res.keyphrases:
        print(f"Keyphrases  : {', '.join(k.phrase for k in res.keyphrases)}")
    print(f"Latency     : {res.latency_ms:.2f} ms")
    print("=" * 70)
    return 0


def cmd_readability(args: argparse.Namespace) -> int:
    from .readability import analyze_readability
    res = analyze_readability(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Arabic Readability & Linguistic Complexity Engine v{__version__}")
    print("=" * 70)
    print(f"Osman Index      : {res.osman_index:.2f} ({res.reading_ease_category})")
    print(f"Stage            : {res.educational_stage_ar} ({res.educational_stage})")
    print(f"Stats            : {res.word_count} words | {res.sentence_count} sentences | {res.avg_word_length:.1f} chars/word")
    print(f"Syllables / Word : {res.estimated_syllables_per_word:.2f}")
    print(f"Lexical Richness : TTR {res.type_token_ratio:.2f} | Hapax {res.hapax_legomena_ratio:.2f}")
    print(f"Reading Time     : {res.reading_time_seconds:.1f} sec (~{res.reading_time_seconds/60:.1f} min)")
    print("=" * 70)
    return 0


def cmd_similarity(args: argparse.Namespace) -> int:
    from .similarity import compute_similarity
    res = compute_similarity(args.text1, args.text2)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Arabic Semantic Similarity & Plagiarism Detector v{__version__}")
    print("=" * 70)
    print(f"Overall Similarity : {res.overall_similarity * 100:.1f}%")
    print(f"Root Similarity    : {res.root_similarity * 100:.1f}%")
    print(f"Lexical Jaccard    : {res.lexical_similarity * 100:.1f}%")
    print(f"LCS Sequence Ratio : {res.lcs_ratio * 100:.1f}%")
    print(f"Is Paraphrase      : {'نعم (إعادة صياغة)' if res.is_paraphrase else 'لا'}")
    print(f"Is Plagiarism      : {'نعم (انتحال / تطابق كبير)' if res.is_plagiarism else 'لا'}")
    if res.matched_spans:
        print("-" * 70)
        print(f"Verbatim Matches ({len(res.matched_spans)} spans):")
        for s in res.matched_spans:
            print(f"  • '{s.text}' ({s.word_count} words)")
    print("=" * 70)
    return 0


def cmd_intent(args: argparse.Namespace) -> int:
    from .intent import classify_intent
    res = classify_intent(args.text)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Arabic Customer Intent & Aspect-Based Sentiment v{__version__}")
    print("=" * 70)
    print(f"Primary Intent     : {res.primary_intent} ({res.primary_intent_ar})")
    print(f"Confidence         : {res.confidence * 100:.0f}%")
    print(f"Overall Polarity   : {res.overall_sentiment_polarity.upper()}")
    if res.aspects:
        print("-" * 70)
        print("Detected Aspects (ABSA):")
        for a in res.aspects:
            pol_symbol = "🟢" if a.polarity == "positive" else ("🔴" if a.polarity == "negative" else "⚪")
            print(f"  {pol_symbol} [{a.aspect_display_ar}] {a.polarity} (score: {a.score:+.2f}) -> \"{a.evidence_snippet}\"")
    print("=" * 70)
    return 0


def cmd_phonemize(args: argparse.Namespace) -> int:
    from .phonemize import phonemize_arabic
    res = phonemize_arabic(args.text, auto_diacritize=not args.no_diacritize)
    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print(f"Misar Arabic Grapheme-to-Phoneme (G2P) & Speech Engine v{__version__}")
    print("=" * 70)
    print(f"Input Text    : {res.original_text}")
    print(f"Diacritized   : {res.diacritized_text}")
    print(f"IPA Phonemes  : /{res.ipa_transcription}/")
    print(f"Syllable Count: {res.syllable_count} | Est. TTS Audio: {res.estimated_audio_duration_sec:.2f}s")
    print(f"Sun Letters   : {res.sun_letter_assimilations_count} assimilation(s)")
    print("=" * 70)
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
    p_route.add_argument("--correct", action="store_true", help="Apply context-aware grammar and spell correction")
    p_route.add_argument("--mask-pii", action="store_true", help="Redact Saudi National ID, IBAN, and Phone numbers")
    p_route.add_argument("--mask-ner", action="store_true", help="Redact Named Entities (PER, ORG, LOC)")
    p_route.add_argument("--translate-msa", action="store_true", help="Translate colloquial dialect phrases into Modern Standard Arabic")
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

    p_bench = sub.add_parser("bench", help="Run reproducible WordPiece fragmentation & latency benchmark")
    p_bench.add_argument("--model", default="arabertv02", choices=model_choices)
    p_bench.add_argument("--text", default="", help="Custom Arabic text to benchmark across all 7 models")
    p_bench.add_argument("--json", action="store_true", help="Output benchmark in structured JSON")
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

    p_classify = sub.add_parser("classify", help="Classify Arabic dialect and text register")
    p_classify.add_argument("text", help="Arabic text to classify")
    p_classify.add_argument("--json", action="store_true", help="Output structured JSON")
    p_classify.set_defaults(func=cmd_classify)

    p_canonicalize = sub.add_parser("canonicalize", help="Canonicalize dialect queries for LLM prompt caching")
    p_canonicalize.add_argument("text", help="Arabic text to canonicalize")
    p_canonicalize.add_argument("--json", action="store_true", help="Output structured JSON")
    p_canonicalize.set_defaults(func=cmd_canonicalize)

    p_process = sub.add_parser("process", help="Process large datasets in parallel across CPU cores")
    p_process.add_argument("input", help="Path to input dataset file (.jsonl, .csv, .tsv, .txt)")
    p_process.add_argument("--output", "-o", required=True, help="Path to save processed dataset")
    p_process.add_argument("--text-key", default="text", help="Key or column name for text in JSONL/CSV")
    p_process.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    p_process.add_argument("--chunk-size", type=int, default=500, help="Batch chunk size per worker")
    p_process.add_argument("--model", default="arabertv02", choices=model_choices)
    p_process.add_argument("--json", action="store_true", help="Output summary in JSON format")
    p_process.set_defaults(func=cmd_process)

    p_dict = sub.add_parser("dict", help="Lookup an Arabic word in the multi-dialect semantic lexicon")
    p_dict.add_argument("word", help="Arabic word to lookup")
    p_dict.add_argument("--json", action="store_true", help="Output structured JSON")
    p_dict.set_defaults(func=cmd_dict)

    p_explain = sub.add_parser("explain", help="Deconstruct a sentence, explain dialect words, and output MSA")
    p_explain.add_argument("text", help="Arabic sentence to analyze and explain")
    p_explain.add_argument("--json", action="store_true", help="Output structured JSON")
    p_explain.set_defaults(func=cmd_explain)

    p_audit = sub.add_parser("audit", help="Audit dataset financial ROI, token reduction, and security compliance")
    p_audit.add_argument("input", help="Path to input dataset file (.jsonl, .csv, .tsv, .txt)")
    p_audit.add_argument("--output", "-o", default="", help="Path to save executive HTML report")
    p_audit.add_argument("--cost-per-1k", type=float, default=0.005, help="Estimated API cost per 1k tokens")
    p_audit.add_argument("--currency", default="SAR", help="Currency symbol (default: SAR)")
    p_audit.add_argument("--text-key", default="text", help="Field name for text in JSONL/CSV")
    p_audit.add_argument("--max-records", type=int, default=None, help="Limit number of records evaluated")
    p_audit.add_argument("--model", default="arabertv02", choices=model_choices)
    p_audit.add_argument("--json", action="store_true", help="Output summary in JSON format")
    p_audit.set_defaults(func=cmd_audit)

    p_proxy = sub.add_parser("proxy", help="Start transparent OpenAI-compatible reverse proxy gateway")
    p_proxy.add_argument("--host", default="0.0.0.0", help="Host address to bind (default: 0.0.0.0)")
    p_proxy.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    p_proxy.add_argument("--upstream", "-u", default=None, help="Upstream LLM base URL (e.g., http://localhost:11434/v1)")
    p_proxy.set_defaults(func=cmd_proxy)

    p_paper = sub.add_parser("paper", help="Generate publication-ready academic LaTeX paper preprint or research summary")
    p_paper.add_argument("--output", "-o", default="", help="Path to save output paper file (.tex or .md)")
    p_paper.add_argument("--format", choices=["latex", "markdown"], default="latex", help="Output format (default: latex)")
    p_paper.set_defaults(func=cmd_paper)

    p_arabizi = sub.add_parser("arabizi", help="Transliterate Franco-Arabic / Arabizi youth chat into canonical Arabic")
    p_arabizi.add_argument("text", help="Arabizi text to transliterate (e.g. 'mar7aba keefak 7abibi')")
    p_arabizi.add_argument("--json", action="store_true", help="Output structured JSON")
    p_arabizi.set_defaults(func=cmd_arabizi)

    p_augment = sub.add_parser("augment", help="Generate synthetic typographical, diacritical, and dialect perturbations")
    p_augment.add_argument("text", help="Arabic seed text to augment")
    p_augment.add_argument("--json", action="store_true", help="Output structured JSON")
    p_augment.set_defaults(func=cmd_augment)

    p_search = sub.add_parser("search", help="Perform in-memory BM25 semantic search across Arabic documents")
    p_search.add_argument("--query", "-q", required=True, help="Arabic query to search for")
    p_search.add_argument("--docs", "-d", nargs="+", help="List of Arabic document strings to search")
    p_search.add_argument("--file", "-f", help="Text file containing one document per line")
    p_search.add_argument("--top-k", "-k", type=int, default=5, help="Number of top results (default: 5)")
    p_search.add_argument("--no-roots", action="store_true", help="Disable morphological root matching")
    p_search.add_argument("--json", action="store_true", help="Output structured JSON")
    p_search.set_defaults(func=cmd_search)

    p_sentiment = sub.add_parser("sentiment", help="Analyze Arabic sentiment, dialectal negations, and emotion")
    p_sentiment.add_argument("text", help="Arabic text to analyze for sentiment")
    p_sentiment.add_argument("--json", action="store_true", help="Output structured JSON")
    p_sentiment.set_defaults(func=cmd_sentiment)

    p_correct = sub.add_parser("correct", help="Perform context-aware Arabic spelling, hamza, and grammar correction")
    p_correct.add_argument("text", help="Arabic text to correct")
    p_correct.add_argument("--json", action="store_true", help="Output structured JSON")
    p_correct.set_defaults(func=cmd_correct)

    p_ner = sub.add_parser("ner", help="Extract and classify Arabic Named Entities (PER, ORG, LOC, DATE, MONEY)")
    p_ner.add_argument("text", help="Arabic text to extract entities from")
    p_ner.add_argument("--mask", action="store_true", help="Return anonymized masked text")
    p_ner.add_argument("--json", action="store_true", help="Output structured JSON")
    p_ner.set_defaults(func=cmd_ner)

    p_translate = sub.add_parser("translate", help="Translate colloquial dialect sentences into canonical Modern Standard Arabic (MSA)")
    p_translate.add_argument("text", help="Dialect text to translate into MSA")
    p_translate.add_argument("--json", action="store_true", help="Output structured JSON")
    p_translate.set_defaults(func=cmd_translate)

    p_diacritize = sub.add_parser("diacritize", help="Automatic Arabic diacritization and tashkeel restoration")
    p_diacritize.add_argument("text", help="Arabic text to diacritize")
    p_diacritize.add_argument("--json", action="store_true", help="Output structured JSON")
    p_diacritize.set_defaults(func=cmd_diacritize)

    p_summarize = sub.add_parser("summarize", help="Extractive Arabic text summarization and keyphrase extraction")
    p_summarize.add_argument("text", help="Arabic text or document to summarize")
    p_summarize.add_argument("--sentences", "-s", type=int, default=3, help="Target number of summary sentences")
    p_summarize.add_argument("--ratio", "-r", type=float, default=0.4, help="Maximum compression ratio")
    p_summarize.add_argument("--json", action="store_true", help="Output structured JSON")
    p_summarize.set_defaults(func=cmd_summarize)

    p_readability = sub.add_parser("readability", help="Calculate Osman Readability Index and linguistic complexity")
    p_readability.add_argument("text", help="Arabic text to analyze for readability")
    p_readability.add_argument("--json", action="store_true", help="Output structured JSON")
    p_readability.set_defaults(func=cmd_readability)

    p_similarity = sub.add_parser("similarity", help="Compare semantic similarity, plagiarism, and paraphrase between two Arabic texts")
    p_similarity.add_argument("text1", help="First Arabic text")
    p_similarity.add_argument("text2", help="Second Arabic text to compare")
    p_similarity.add_argument("--json", action="store_true", help="Output structured JSON")
    p_similarity.set_defaults(func=cmd_similarity)

    p_intent = sub.add_parser("intent", help="Classify customer intent and extract Aspect-Based Sentiments (ABSA)")
    p_intent.add_argument("text", help="Arabic text or customer feedback to analyze")
    p_intent.add_argument("--json", action="store_true", help="Output structured JSON")
    p_intent.set_defaults(func=cmd_intent)

    p_phonemize = sub.add_parser("phonemize", help="Transcribe Arabic text into International Phonetic Alphabet (IPA) for TTS")
    p_phonemize.add_argument("text", help="Arabic text to phonemize")
    p_phonemize.add_argument("--no-diacritize", action="store_true", help="Disable automatic diacritics restoration")
    p_phonemize.add_argument("--json", action="store_true", help="Output structured JSON")
    p_phonemize.set_defaults(func=cmd_phonemize)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    sys.exit(args.func(args))
