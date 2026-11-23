# -*- coding: utf-8 -*-
"""
High-performance Arabic Text Summarization & Keyphrase Extraction Engine for Misar (مِسار).
Features:
- Extractive TextRank / LexRank graph centrality tuned for Arabic discourse
- Morphological root & stem matching for lexical overlap without vector DB overhead
- Discourse connector and structural position weighting
- Arabic multi-word keyphrase and concept extraction
- Zero external dependencies (Pure Python Standard Library)
"""

import math
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .lexicon import extract_root, light_stem
from .search import RAW_STOPWORDS

SENTENCE_SPLIT_REGEX = re.compile(
    r"(?<=[.!?؛؟\n])\s+|(?<=\n)\s*|(?<=[.!?؛؟])$",
    flags=re.UNICODE,
)

WORD_TOKEN_REGEX = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


@dataclass
class KeyphraseSpan:
    """Represents an extracted Arabic keyphrase with salience metrics."""
    phrase: str
    score: float
    frequency: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SummarizationResult:
    """Detailed report for Arabic document summarization and keyphrase extraction."""
    original_text: str
    summary_text: str
    extracted_sentences: List[str]
    keyphrases: List[KeyphraseSpan]
    compression_ratio: float
    original_sentence_count: int
    summary_sentence_count: int
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "summary_text": self.summary_text,
            "extracted_sentences": list(self.extracted_sentences),
            "keyphrases": [k.to_dict() for k in self.keyphrases],
            "compression_ratio": round(self.compression_ratio, 3),
            "original_sentence_count": self.original_sentence_count,
            "summary_sentence_count": self.summary_sentence_count,
            "latency_ms": round(self.latency_ms, 3),
        }


def _clean_token(w: str) -> str:
    """Normalizes an Arabic word token for semantic similarity."""
    t = re.sub(r"[\u064B-\u0652\u0670ـ]", "", w)
    t = re.sub(r"[إأآا]", "ا", t)
    t = re.sub(r"ة", "ه", t)
    t = re.sub(r"ى", "ي", t)
    return t


def _sentence_tokens(sentence: str) -> List[str]:
    words = WORD_TOKEN_REGEX.findall(sentence)
    tokens: List[str] = []
    for w in words:
        norm = _clean_token(w)
        if norm and norm not in RAW_STOPWORDS and len(norm) >= 2:
            root = extract_root(w)
            tokens.append(root if root else norm)
    return tokens


def _compute_sentence_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    set1 = Counter(tokens1)
    set2 = Counter(tokens2)
    common = set(set1.keys()) & set(set2.keys())
    if not common:
        return 0.0
    dot = sum(set1[w] * set2[w] for w in common)
    denom = (math.sqrt(sum(v * v for v in set1.values())) * math.sqrt(sum(v * v for v in set2.values())))
    return dot / denom if denom > 0 else 0.0


def extract_keyphrases(text: str, top_k: int = 5) -> List[KeyphraseSpan]:
    """
    Extracts salient Arabic multi-word keyphrases (1-3 words) based on
    term frequency, word length, and morphological centrality.
    """
    if not text or not text.strip():
        return []

    words = WORD_TOKEN_REGEX.findall(text)
    n = len(words)
    candidates: List[str] = []

    # 1. Unigrams
    for w in words:
        norm = _clean_token(w)
        if norm not in RAW_STOPWORDS and len(norm) >= 3 and not norm.isdigit():
            candidates.append(w)

    # 2. Bigrams
    for i in range(n - 1):
        w1, w2 = words[i], words[i + 1]
        n1, n2 = _clean_token(w1), _clean_token(w2)
        if n1 not in RAW_STOPWORDS and n2 not in RAW_STOPWORDS and len(n1) >= 3 and len(n2) >= 3:
            candidates.append(f"{w1} {w2}")

    # 3. Trigrams
    for i in range(n - 2):
        w1, w2, w3 = words[i], words[i + 1], words[i + 2]
        n1, n3 = _clean_token(w1), _clean_token(w3)
        if n1 not in RAW_STOPWORDS and n3 not in RAW_STOPWORDS and len(n1) >= 3 and len(n3) >= 3:
            candidates.append(f"{w1} {w2} {w3}")

    counts = Counter(candidates)
    scored: List[KeyphraseSpan] = []

    for phrase, freq in counts.items():
        parts = phrase.split()
        length_boost = 1.0 + (len(parts) - 1) * 0.4
        # Boost phrases with important entity or technical terms
        entity_boost = 1.3 if any(p in ("سدايا", "موهبة", "الذكاء", "البيانات", "السعودية", "التقنية", "النماذج", "الحوسبة") for p in parts) else 1.0
        score = freq * length_boost * entity_boost
        scored.append(KeyphraseSpan(phrase=phrase, score=round(score, 2), frequency=freq))

    # Sort descending by score, take top_k without overlapping sub-phrases
    scored.sort(key=lambda x: x.score, reverse=True)
    selected: List[KeyphraseSpan] = []
    seen: Set[str] = set()

    for item in scored:
        if item.phrase not in seen and not any(item.phrase in s for s in seen):
            selected.append(item)
            seen.add(item.phrase)
            if len(selected) >= top_k:
                break

    return selected


def summarize_text(
    text: str,
    num_sentences: int = 3,
    max_ratio: float = 0.5,
) -> SummarizationResult:
    """
    Produces an extractive summary of Arabic text using TextRank graph centrality
    and morphological root matching with discourse position priors.
    """
    t0 = time.perf_counter()
    if not text or not text.strip():
        return SummarizationResult(
            original_text=text,
            summary_text="",
            extracted_sentences=[],
            keyphrases=[],
            compression_ratio=0.0,
            original_sentence_count=0,
            summary_sentence_count=0,
            latency_ms=0.0,
        )

    # 1. Segment sentences
    raw_sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [text.strip()]

    n_sentences = len(raw_sentences)

    # If the text is already short, return it directly
    if n_sentences <= num_sentences:
        keyphrases = extract_keyphrases(text, top_k=5)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return SummarizationResult(
            original_text=text,
            summary_text=text.strip(),
            extracted_sentences=raw_sentences,
            keyphrases=keyphrases,
            compression_ratio=1.0,
            original_sentence_count=n_sentences,
            summary_sentence_count=n_sentences,
            latency_ms=elapsed,
        )

    # 2. Tokenize sentences
    sent_tokens = [_sentence_tokens(s) for s in raw_sentences]

    # 3. Build Adjacency Matrix (Similarity Graph)
    sim_matrix: List[List[float]] = [[0.0] * n_sentences for _ in range(n_sentences)]
    for i in range(n_sentences):
        for j in range(i + 1, n_sentences):
            sim = _compute_sentence_similarity(sent_tokens[i], sent_tokens[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # 4. TextRank Power Iteration
    d = 0.85
    scores = [1.0 / n_sentences] * n_sentences
    max_iter = 30
    tol = 1e-4

    for _ in range(max_iter):
        next_scores = [0.0] * n_sentences
        for i in range(n_sentences):
            sum_links = 0.0
            for j in range(n_sentences):
                if i != j and sim_matrix[i][j] > 0:
                    denom = sum(sim_matrix[j])
                    if denom > 0:
                        sum_links += (sim_matrix[i][j] / denom) * scores[j]
            next_scores[i] = (1 - d) / n_sentences + d * sum_links

        diff = sum(abs(next_scores[i] - scores[i]) for i in range(n_sentences))
        scores = next_scores
        if diff < tol:
            break

    # 5. Position & Factual Priors
    for idx in range(n_sentences):
        # Lead sentence boost (Headline / thesis statement)
        if idx == 0:
            scores[idx] *= 1.35
        # Second sentence boost
        elif idx == 1:
            scores[idx] *= 1.15
        # Conclusion sentence boost
        elif idx == n_sentences - 1:
            scores[idx] *= 1.20

        # Numbers or sovereign entity boost
        s = raw_sentences[idx]
        if any(e in s for e in ("سدايا", "موهبة", "المملكة", "الرؤية", "أرامكو")):
            scores[idx] *= 1.25
        if re.search(r"\d+", s):
            scores[idx] *= 1.10

    # 6. Rank and pick top sentences, maintaining original document flow
    target_count = min(num_sentences, max(1, math.ceil(n_sentences * max_ratio)))
    ranked_indices = sorted(range(n_sentences), key=lambda i: scores[i], reverse=True)[:target_count]
    selected_indices = sorted(ranked_indices)

    extracted = [raw_sentences[i] for i in selected_indices]
    summary_text = " ".join(extracted)

    keyphrases = extract_keyphrases(text, top_k=5)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    orig_words = len(text.split())
    summ_words = len(summary_text.split())
    comp_ratio = (summ_words / max(1, orig_words))

    return SummarizationResult(
        original_text=text,
        summary_text=summary_text,
        extracted_sentences=extracted,
        keyphrases=keyphrases,
        compression_ratio=round(comp_ratio, 3),
        original_sentence_count=n_sentences,
        summary_sentence_count=len(extracted),
        latency_ms=elapsed_ms,
    )
