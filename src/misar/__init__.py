# -*- coding: utf-8 -*-
"""
Misar (مِسار): Adaptive Context-Aware Arabic NLP Preprocessing Router.
"""

__version__ = "2.2.0"
__author__ = "Muneer Jarab Matrak Al-Theeb"

from .evaluation import evaluate_system, run_evaluation
from .guard import (
    ArabicPromptGuard,
    GuardResult,
    sanitize_prompt,
    validate_iban,
    validate_saudi_national_id,
)
from .pipelines import DialectPreservePipeline, MSALightPipeline, NoisySocialPipeline
from .profiler import LinguisticProfile, LinguisticProfiler, compute_shannon_entropy
from .rag import (
    ArabicDiscourseChunker,
    MisarArabicSplitter,
    SemanticChunk,
    chunk_document,
)
from .router import DocumentRoutingResult, MisarRouter, RoutingResult, preprocess, preprocess_batch, route
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
    "preprocess",
    "preprocess_batch",
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
]
