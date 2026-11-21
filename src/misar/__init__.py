# -*- coding: utf-8 -*-
"""
Misar (مِسار): Adaptive Context-Aware Arabic NLP Preprocessing Router.
"""

__version__ = "2.9.0"
__author__ = "Muneer Jarab Matrak Al-Theeb"

from .arabizi import (
    ArabiziDecoder,
    decode_arabizi,
    is_arabizi_text,
    is_arabizi_token,
)
from .audit import (
    AuditResult,
    audit_dataset_file,
    audit_texts,
    generate_html_report,
)
from .augment import (
    evaluate_augmentation_recovery,
    generate_stress_suite,
    inject_arabizi_noise,
    inject_diacritics_noise,
    inject_orthographic_noise,
    inject_tatweel_noise,
)
from .benchmark import benchmark_text
from .classifier import (
    DialectClassification,
    DialectClassifier,
    canonicalize,
    classify_dialect,
)
from .correct import (
    CorrectionChange,
    CorrectionResult,
    correct_arabic_text,
)
from .evaluation import evaluate_system, run_evaluation
from .guard import (
    ArabicPromptGuard,
    GuardResult,
    sanitize_prompt,
    validate_iban,
    validate_saudi_national_id,
)
from .integrations import (
    MisarNodeParser,
    MisarOllamaAdapter,
    MisarSecurityMiddleware,
    misar_litellm_pre_hook,
    misar_vllm_pre_hook,
    prepare_ollama_prompt,
)
from .lexicon import (
    ArabicLexicon,
    LexicalEntry,
    TextExplanation,
    TokenExplanation,
    explain_text,
    extract_root,
    get_lexicon,
    light_stem,
    lookup_word,
)
from .ner import (
    EntitySpan,
    NERResult,
    extract_entities,
)
from .paper import (
    generate_academic_paper,
    generate_markdown_summary,
)
from .parallel import (
    process_dataset_file,
    route_parallel,
)
from .pipelines import DialectPreservePipeline, MSALightPipeline, NoisySocialPipeline
from .profiler import LinguisticProfile, LinguisticProfiler, compute_shannon_entropy
from .proxy import (
    ProxyTelemetry,
    create_proxy_server,
    run_proxy,
)
from .rag import (
    ArabicDiscourseChunker,
    MisarArabicSplitter,
    SemanticChunk,
    chunk_document,
)
from .router import (
    DocumentRoutingResult,
    MisarRouter,
    RoutingResult,
    UnifiedProcessResult,
    classify,
    clear_cache,
    get_cache_stats,
    preprocess,
    preprocess_batch,
    process,
    route,
)
from .search import (
    BM25SearchEngine,
    SearchResult,
    search_arabic,
)
from .sentiment import (
    SentimentResult,
    analyze_sentiment,
)
from .translate import (
    TranslationResult,
    translate_to_msa,
)
from .tokenizer import (
    PRETRAINED_MODEL_CONFIGS,
    PRETRAINED_VOCAB_URLS,
    HeuristicSubwordTokenizer,
    HuggingFaceTokenizerAdapter,
    WordPieceTokenizer,
    load_default_tokenizer,
)

__all__ = [
    "__version__",
    "__author__",
    "MisarRouter",
    "RoutingResult",
    "DocumentRoutingResult",
    "UnifiedProcessResult",
    "LinguisticProfiler",
    "LinguisticProfile",
    "compute_shannon_entropy",
    "MSALightPipeline",
    "DialectPreservePipeline",
    "NoisySocialPipeline",
    "WordPieceTokenizer",
    "HuggingFaceTokenizerAdapter",
    "HeuristicSubwordTokenizer",
    "PRETRAINED_MODEL_CONFIGS",
    "PRETRAINED_VOCAB_URLS",
    "load_default_tokenizer",
    "route",
    "process",
    "preprocess",
    "preprocess_batch",
    "clear_cache",
    "get_cache_stats",
    "evaluate_system",
    "run_evaluation",
    "ArabicDiscourseChunker",
    "SemanticChunk",
    "chunk_document",
    "MisarArabicSplitter",
    "ArabicPromptGuard",
    "GuardResult",
    "sanitize_prompt",
    "validate_saudi_national_id",
    "validate_iban",
    "DialectClassifier",
    "DialectClassification",
    "classify_dialect",
    "classify",
    "canonicalize",
    "route_parallel",
    "process_dataset_file",
    "ArabicLexicon",
    "LexicalEntry",
    "TextExplanation",
    "TokenExplanation",
    "lookup_word",
    "explain_text",
    "extract_root",
    "light_stem",
    "get_lexicon",
    "AuditResult",
    "audit_texts",
    "audit_dataset_file",
    "generate_html_report",
    "MisarSecurityMiddleware",
    "MisarNodeParser",
    "misar_litellm_pre_hook",
    "MisarOllamaAdapter",
    "prepare_ollama_prompt",
    "misar_vllm_pre_hook",
    "decode_arabizi",
    "is_arabizi_text",
    "is_arabizi_token",
    "ArabiziDecoder",
    "create_proxy_server",
    "run_proxy",
    "ProxyTelemetry",
    "inject_orthographic_noise",
    "inject_tatweel_noise",
    "inject_diacritics_noise",
    "inject_arabizi_noise",
    "generate_stress_suite",
    "evaluate_augmentation_recovery",
    "generate_academic_paper",
    "generate_markdown_summary",
    "BM25SearchEngine",
    "SearchResult",
    "search_arabic",
    "SentimentResult",
    "analyze_sentiment",
    "benchmark_text",
    "CorrectionChange",
    "CorrectionResult",
    "correct_arabic_text",
    "EntitySpan",
    "NERResult",
    "extract_entities",
    "TranslationResult",
    "translate_to_msa",
]
