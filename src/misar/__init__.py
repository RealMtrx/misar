# -*- coding: utf-8 -*-
"""
Misar (مِسار): Adaptive Context-Aware Arabic NLP Preprocessing Router.
"""

__version__ = "2.1.0"
__author__ = "Muneer Jarab Matrak Al-Theeb"

from .evaluation import evaluate_system, run_evaluation
from .pipelines import DialectPreservePipeline, MSALightPipeline, NoisySocialPipeline
from .profiler import LinguisticProfile, LinguisticProfiler, compute_shannon_entropy
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
]
