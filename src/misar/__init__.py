# -*- coding: utf-8 -*-
"""
Misar (مِسار): Adaptive Context-Aware Arabic NLP Preprocessing Router.
"""

from .pipelines import DialectPreservePipeline, MSALightPipeline, NoisySocialPipeline
from .profiler import LinguisticProfile, LinguisticProfiler, compute_shannon_entropy
from .router import MisarRouter, RoutingResult, preprocess, route
from .tokenizer import (
    HeuristicSubwordTokenizer,
    HuggingFaceTokenizerAdapter,
    WordPieceTokenizer,
    load_default_tokenizer,
)

__version__ = "1.0.0"
__author__ = "Muneer Jarab Matrak Al-Theeb"

__all__ = [
    "__version__",
    "__author__",
    "MisarRouter",
    "RoutingResult",
    "LinguisticProfiler",
    "LinguisticProfile",
    "compute_shannon_entropy",
    "MSALightPipeline",
    "DialectPreservePipeline",
    "NoisySocialPipeline",
    "WordPieceTokenizer",
    "HuggingFaceTokenizerAdapter",
    "HeuristicSubwordTokenizer",
    "load_default_tokenizer",
    "route",
    "preprocess",
]
