# -*- coding: utf-8 -*-
"""
High-performance in-memory Arabic BM25 & Semantic Search Engine for Misar (مِسار).
Implements Okapi BM25 with Arabic-aware tokenization, morphological stem/root matching,
and diacritic-tolerant scoring with zero external dependencies.
"""

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .lexicon import extract_root
from .pipelines import (
    LEGACY_ALEF_REGEX,
    LEGACY_TAA_REGEX,
    LEGACY_YAA_REGEX,
    TASHKEEL_REGEX,
    TATWEEL_REGEX,
    normalize_unicode_safety,
)

RAW_STOPWORDS: Set[str] = {
    "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "التي",
    "الذي", "الذين", "اللذين", "اللواتي", "اللاتي", "هو", "هي", "هم", "هن",
    "أنت", "أنتم", "أنتما", "نحن", "أنا", "كان", "كانت", "يكون", "تكون",
    "أن", "إن", "إنما", "لكن", "لكنما", "غير", "سوى", "إلا", "حيث", "حين",
    "كل", "جميع", "كلا", "كلتا", "قد", "لقد", "ثم", "أو", "أم", "بل",
    "لو", "إذا", "إذ", "كي", "لكي", "حتى", "ماذا", "لماذا", "كيف", "أين",
    "متى", "كم", "أي", "أيان", "هل", "بين", "فوق", "تحت", "أمام", "خلف",
    "بعد", "قبل", "عند", "لدى", "دون", "نحو", "حول", "تجاه",
}

WORD_PATTERN = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def _normalize_text(text: str) -> str:
    """Canonicalize Arabic string for lexical search indexing."""
    t = normalize_unicode_safety(text)
    t = TASHKEEL_REGEX.sub("", t)
    t = TATWEEL_REGEX.sub("", t)
    t = LEGACY_ALEF_REGEX.sub("ا", t)
    t = LEGACY_TAA_REGEX.sub("ه", t)
    t = LEGACY_YAA_REGEX.sub("ي", t)
    return t


ARABIC_STOPWORDS: Set[str] = {_normalize_text(w) for w in RAW_STOPWORDS} | RAW_STOPWORDS


def _tokenize_arabic(
    text: str,
    remove_stopwords: bool = True,
    use_roots: bool = True,
) -> List[Tuple[str, Set[str]]]:
    """
    Tokenizes Arabic text into normalized surface forms and candidate morphological roots.
    Returns a list of tuples: (surface_normalized, set_of_roots).
    """
    normalized = _normalize_text(text)
    tokens: List[Tuple[str, Set[str]]] = []
    raw_words = WORD_PATTERN.findall(normalized)
    for w in raw_words:
        w_lower = w.lower()
        if remove_stopwords and (w_lower in ARABIC_STOPWORDS or w in ARABIC_STOPWORDS):
            continue
        if len(w_lower) <= 1 and not w_lower.isalnum():
            continue

        roots: Set[str] = set()
        if use_roots:
            r = extract_root(w_lower)
            if r:
                roots.add(r)
            # Handle conjunction/preposition proclitics (و، ف، ب، ك، ل)
            if w_lower.startswith(("و", "ف", "ب", "ك", "ل")) and len(w_lower) >= 4:
                r_clitic = extract_root(w_lower[1:])
                if r_clitic:
                    roots.add(r_clitic)
            # Handle definite article (ال)
            if w_lower.startswith("ال") and len(w_lower) >= 5:
                r_def = extract_root(w_lower[2:])
                if r_def:
                    roots.add(r_def)

        tokens.append((w_lower, roots))
    return tokens


@dataclass
class SearchResult:
    """Represents a scored document search result."""
    doc_id: str
    score: float
    text: str
    rank: int = 1
    matched_terms: List[str] = field(default_factory=list)
    snippet: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BM25SearchEngine:
    """
    Okapi BM25 implementation optimized for Arabic texts.
    Features:
    - Native Arabic morphological normalization and root matching
    - Diacritic-resilient and alef-normalized inverted index
    - Configurable k1 (term saturation) and b (length penalization) parameters
    - Zero external dependencies (pure Python standard library)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        use_roots: bool = True,
        remove_stopwords: bool = True,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.use_roots = use_roots
        self.remove_stopwords = remove_stopwords

        # Inverted index: term -> {doc_id: term_frequency}
        self.index: Dict[str, Dict[str, int]] = {}
        # Root index: root -> {doc_id: term_frequency}
        self.root_index: Dict[str, Dict[str, int]] = {}
        # Document storage: doc_id -> (text, metadata, length)
        self.docs: Dict[str, Dict[str, Any]] = {}
        self.total_docs: int = 0
        self.total_length: int = 0
        self.avg_doc_length: float = 0.0

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single document to the search index."""
        tokens = _tokenize_arabic(
            text,
            remove_stopwords=self.remove_stopwords,
            use_roots=self.use_roots,
        )
        doc_len = len(tokens)
        if doc_len == 0:
            doc_len = 1

        # If updating an existing doc, subtract old length
        if doc_id in self.docs:
            self.total_length -= self.docs[doc_id]["length"]
            self.total_docs -= 1

        self.docs[doc_id] = {
            "text": text,
            "metadata": metadata or {},
            "length": doc_len,
        }
        self.total_docs += 1
        self.total_length += doc_len
        self.avg_doc_length = self.total_length / max(1, self.total_docs)

        # Index tokens and roots
        for surface, roots in tokens:
            # Surface index
            if surface not in self.index:
                self.index[surface] = {}
            self.index[surface][doc_id] = self.index[surface].get(doc_id, 0) + 1

            # Root index
            if self.use_roots:
                for root in roots:
                    if root not in self.root_index:
                        self.root_index[root] = {}
                    self.root_index[root][doc_id] = self.root_index[root].get(doc_id, 0) + 1

    def add_documents(
        self,
        documents: Union[Sequence[str], Sequence[Dict[str, Any]]],
    ) -> None:
        """Add multiple documents in batch."""
        for idx, item in enumerate(documents):
            if isinstance(item, str):
                self.add_document(f"doc_{idx + 1}", item)
            elif isinstance(item, dict):
                doc_id = str(item.get("id", f"doc_{idx + 1}"))
                text = str(item.get("text", item.get("content", "")))
                metadata = item.get("metadata", {})
                self.add_document(doc_id, text, metadata)

    def _compute_idf(self, doc_freq: int) -> float:
        """Compute Robertson-Spärck Jones IDF with smoothing."""
        numerator = self.total_docs - doc_freq + 0.5
        denominator = doc_freq + 0.5
        return math.log(max(0.0001, (numerator / denominator) + 1.0))

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search indexed documents using Okapi BM25 with Arabic root & surface boosts.
        """
        if not query or not query.strip() or self.total_docs == 0:
            return []

        query_tokens = _tokenize_arabic(
            query,
            remove_stopwords=self.remove_stopwords,
            use_roots=self.use_roots,
        )
        if not query_tokens:
            return []

        doc_scores: Dict[str, float] = {}
        doc_matches: Dict[str, Set[str]] = {}

        for surface, roots in query_tokens:
            # 1. Exact surface match scoring
            if surface in self.index:
                doc_freq = len(self.index[surface])
                idf = self._compute_idf(doc_freq)
                for doc_id, tf in self.index[surface].items():
                    doc_len = self.docs[doc_id]["length"]
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_length)))
                    score = idf * ((tf * (self.k1 + 1.0)) / max(0.0001, denom))

                    doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                    doc_matches.setdefault(doc_id, set()).add(surface)

            # 2. Morphological root match scoring (weighted at 0.6x to favor exact terms)
            if self.use_roots:
                for root in roots:
                    if root in self.root_index:
                        doc_freq = len(self.root_index[root])
                        idf = self._compute_idf(doc_freq) * 0.6
                        for doc_id, tf in self.root_index[root].items():
                            if surface in self.index and doc_id in self.index[surface]:
                                continue  # already counted with full surface weight
                            doc_len = self.docs[doc_id]["length"]
                            denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_length)))
                            score = idf * ((tf * (self.k1 + 1.0)) / max(0.0001, denom))

                            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
                            doc_matches.setdefault(doc_id, set()).add(f"جذر:{root}")

        if not doc_scores:
            return []

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: List[SearchResult] = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            doc_data = self.docs[doc_id]
            full_text = doc_data["text"]
            snippet = full_text[:180] + ("..." if len(full_text) > 180 else "")
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=round(score, 4),
                    text=full_text,
                    rank=rank,
                    matched_terms=sorted(list(doc_matches.get(doc_id, set()))),
                    snippet=snippet,
                    metadata=doc_data["metadata"],
                )
            )

        return results


def search_arabic(
    query: str,
    documents: Union[Sequence[str], Sequence[Dict[str, Any]]],
    top_k: int = 5,
    use_roots: bool = True,
) -> List[SearchResult]:
    """
    Convenience helper function to index a list of documents and search them in one call.
    """
    engine = BM25SearchEngine(use_roots=use_roots)
    engine.add_documents(documents)
    return engine.search(query, top_k=top_k)
