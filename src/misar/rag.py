# -*- coding: utf-8 -*-
"""
Arabic Semantic & Discourse-Aware RAG Chunker for Misar.

Provides intelligent semantic chunking for Arabic Retrieval-Augmented Generation (RAG)
and Vector Databases. Respects Arabic discourse transition markers, sentence boundaries,
and strictly enforces negation/clitic bonding so negation particles are never severed
from their predicates across chunk boundaries.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union

from .profiler import LinguisticProfiler, LinguisticProfile
from .router import MisarRouter, RoutingResult
from .tokenizer import TokenizerProtocol, load_default_tokenizer

# Primary discourse transition markers indicating authentic semantic boundaries in Arabic
DISCOURSE_MARKERS: Sequence[str] = (
    "علاوة على ذلك",
    "بالإضافة إلى ذلك",
    "فضلاً عن ذلك",
    "ومن الجدير بالذكر",
    "كما تجدر الإشارة إلى أن",
    "بناءً على ذلك",
    "ونتيجة لذلك",
    "وعلى هذا الأساس",
    "وعلى صعيد متصل",
    "وفي هذا السياق",
    "ومع ذلك",
    "على الرغم من ذلك",
    "إلا أن",
    "غير أن",
    "في المقابل",
    "من جهة أخرى",
    "من ناحية أخرى",
    "على النقيض من ذلك",
    "حيث أن",
    "نظراً لأن",
    "الأمر الذي أدى إلى",
    "وختاماً",
    "وصفوة القول",
    "وفي نهاية المطاف",
    "بناءً على ما تقدم",
)

# Arabic negation particles that must NEVER be stranded at the end of a chunk
STRANDED_NEGATION_ENDINGS: Sequence[str] = (
    "لم", "لن", "لا", "ما", "ليس", "ليست", "ليسوا",
    "غير", "سوى", "لات", "دون", "بدون",
    "مش", "مو", "ماكو", "مب", "ماني", "ماناش",
)

# Discourse marker split regex (lookahead-aware to preserve transition marker with following segment)
_DISCOURSE_PATTERN = "|".join(re.escape(m) for m in sorted(DISCOURSE_MARKERS, key=len, reverse=True))
DISCOURSE_SPLIT_REGEX = re.compile(rf"(?<=[.!?؟;\n])\s+|(?=\s+(?:{_DISCOURSE_PATTERN})\s+)")

# Standard sentence and paragraph boundaries
PARAGRAPH_REGEX = re.compile(r"\n{2,}")
SENTENCE_REGEX = re.compile(r"([.!?؟;\n]+)\s*")


@dataclass
class SemanticChunk:
    """A semantic document chunk optimized for Arabic RAG and embedding."""
    chunk_index: int
    text: str
    raw_text: str
    start_char: int
    end_char: int
    token_count: int
    route_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "text": self.text,
            "raw_text": self.raw_text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "token_count": self.token_count,
            "route_used": self.route_used,
            "metadata": self.metadata,
        }


class ArabicDiscourseChunker:
    """
    Semantic RAG chunker for Arabic documents.
    
    Segments documents along authentic Arabic discourse transition markers and sentence
    boundaries while enforcing negation-predicate bonding and optional dynamic Misar
    preprocessing per chunk.
    """

    def __init__(
        self,
        max_tokens: int = 256,
        overlap_tokens: int = 32,
        target_model: str = "arabertv02",
        auto_preprocess: bool = True,
        router: Optional[MisarRouter] = None,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be non-negative and less than max_tokens")

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.target_model = target_model
        self.auto_preprocess = auto_preprocess
        self.router = router or MisarRouter()
        self.tokenizer = load_default_tokenizer(target_model, allow_download=False)

    def count_tokens(self, text: str) -> int:
        """Count tokens using authentic Arabic WordPiece vocabulary."""
        if not text:
            return 0
        return len(self.tokenizer.tokenize(text))

    def _split_into_clauses(self, text: str) -> List[Dict[str, Any]]:
        """Split raw text into fine-grained atomic clauses with exact start and end offsets."""
        clauses: List[Dict[str, Any]] = []
        if not text:
            return clauses

        # Primary pass: split on paragraph breaks
        para_offsets: List[Tuple[int, int, str]] = []
        last_end = 0
        for m in PARAGRAPH_REGEX.finditer(text):
            p_text = text[last_end:m.start()].strip()
            if p_text:
                clauses.extend(self._segment_paragraph(text, last_end, m.start()))
            last_end = m.end()

        if last_end < len(text):
            remainder = text[last_end:].strip()
            if remainder:
                clauses.extend(self._segment_paragraph(text, last_end, len(text)))

        # Fallback if no clauses produced
        if not clauses and text.strip():
            clauses.append({
                "raw": text,
                "start": 0,
                "end": len(text),
                "tokens": self.count_tokens(text),
            })

        return self._fuse_stranded_negations(clauses, text)

    def _segment_paragraph(self, full_text: str, p_start: int, p_end: int) -> List[Dict[str, Any]]:
        """Segment a paragraph using discourse transitions and sentence boundaries."""
        results: List[Dict[str, Any]] = []
        para_slice = full_text[p_start:p_end]
        
        # Split on sentence boundaries and discourse transitions
        splits = [s for s in DISCOURSE_SPLIT_REGEX.split(para_slice) if s and s.strip()]
        if not splits:
            splits = [para_slice]

        current_idx = p_start
        for segment in splits:
            # Locate exact offset in full_text
            seg_start = full_text.find(segment, current_idx, p_end)
            if seg_start == -1:
                seg_start = current_idx
            seg_end = seg_start + len(segment)
            current_idx = seg_end

            trimmed = segment.strip()
            if not trimmed:
                continue

            results.append({
                "raw": trimmed,
                "start": seg_start,
                "end": seg_end,
                "tokens": self.count_tokens(trimmed),
            })

        return results

    def _fuse_stranded_negations(self, clauses: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
        """
        Enforce Arabic Negation & Clitic Invariant:
        If a clause ends with a negation particle (e.g. 'لم', 'لن', 'لا', 'مش', 'مو'),
        merge it with the following clause so it is never severed across a chunk boundary.
        """
        if len(clauses) <= 1:
            return clauses

        fused: List[Dict[str, Any]] = []
        skip_next = False

        for i in range(len(clauses)):
            if skip_next:
                skip_next = False
                continue

            c = clauses[i]
            words = c["raw"].split()
            last_word = words[-1] if words else ""
            
            # Clean punctuation from last word to check negation
            last_clean = re.sub(r"[^\u0621-\u064A\u0671-\u06D3]", "", last_word)

            if i < len(clauses) - 1 and last_clean in STRANDED_NEGATION_ENDINGS:
                next_c = clauses[i + 1]
                # Merge into single protected clause
                merged_start = c["start"]
                merged_end = next_c["end"]
                merged_raw = full_text[merged_start:merged_end].strip()
                fused.append({
                    "raw": merged_raw,
                    "start": merged_start,
                    "end": merged_end,
                    "tokens": self.count_tokens(merged_raw),
                })
                skip_next = True
            else:
                fused.append(c)

        return fused

    def chunk(self, text: str, extra_metadata: Optional[Dict[str, Any]] = None) -> List[SemanticChunk]:
        """
        Chunk text into semantic chunks with token limits, overlap, and Misar optimization.
        """
        if not text or not text.strip():
            return []

        clauses = self._split_into_clauses(text)
        if not clauses:
            return []

        chunks: List[SemanticChunk] = []
        current_clauses: List[Dict[str, Any]] = []
        current_token_count = 0

        i = 0
        while i < len(clauses):
            c = clauses[i]
            c_tokens = c["tokens"]

            # If a single clause exceeds max_tokens on its own, handle it
            if c_tokens > self.max_tokens and not current_clauses:
                chunk_obj = self._build_chunk(
                    chunks_len=len(chunks),
                    full_text=text,
                    c_list=[c],
                    extra_meta=extra_metadata,
                )
                chunks.append(chunk_obj)
                i += 1
                continue

            if current_token_count + c_tokens <= self.max_tokens or not current_clauses:
                current_clauses.append(c)
                current_token_count += c_tokens
                i += 1
            else:
                # Flush current chunk
                chunk_obj = self._build_chunk(
                    chunks_len=len(chunks),
                    full_text=text,
                    c_list=current_clauses,
                    extra_meta=extra_metadata,
                )
                chunks.append(chunk_obj)

                # Compute overlap
                if self.overlap_tokens > 0:
                    overlap_acc: List[Dict[str, Any]] = []
                    overlap_cnt = 0
                    for rev_c in reversed(current_clauses):
                        if overlap_cnt + rev_c["tokens"] <= self.overlap_tokens:
                            overlap_acc.insert(0, rev_c)
                            overlap_cnt += rev_c["tokens"]
                        else:
                            break
                    current_clauses = overlap_acc
                    current_token_count = overlap_cnt
                else:
                    current_clauses = []
                    current_token_count = 0

        # Flush remaining clauses
        if current_clauses:
            chunk_obj = self._build_chunk(
                chunks_len=len(chunks),
                full_text=text,
                c_list=current_clauses,
                extra_meta=extra_metadata,
            )
            chunks.append(chunk_obj)

        return chunks

    def _build_chunk(
        self,
        chunks_len: int,
        full_text: str,
        c_list: List[Dict[str, Any]],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> SemanticChunk:
        """Construct, preprocess, and finalize a SemanticChunk."""
        start_char = c_list[0]["start"]
        end_char = c_list[-1]["end"]
        raw_span = full_text[start_char:end_char].strip()

        if self.auto_preprocess:
            route_res = self.router.route(raw_span)
            processed_text = route_res.processed_text
            route_used = route_res.route_name
            tokens = route_res.processed_token_count
        else:
            processed_text = raw_span
            route_used = "RAW"
            tokens = self.count_tokens(raw_span)

        meta = {
            "model": self.target_model,
            "clauses_count": len(c_list),
            **(extra_meta or {}),
        }

        return SemanticChunk(
            chunk_index=chunks_len,
            text=processed_text,
            raw_text=raw_span,
            start_char=start_char,
            end_char=end_char,
            token_count=tokens,
            route_used=route_used,
            metadata=meta,
        )


def chunk_document(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    target_model: str = "arabertv02",
    auto_preprocess: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[SemanticChunk]:
    """Convenience helper to chunk an Arabic document into semantic RAG chunks."""
    chunker = ArabicDiscourseChunker(
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        target_model=target_model,
        auto_preprocess=auto_preprocess,
    )
    return chunker.chunk(text, extra_metadata=metadata)


# ---------------------------------------------------------------------------
# LangChain & LlamaIndex Drop-in Compatibility Wrappers
# ---------------------------------------------------------------------------

class MisarArabicSplitter:
    """
    Drop-in text splitter for LangChain applications.
    
    Compatible with LangChain's Document schema:
        from misar.rag import MisarArabicSplitter
        splitter = MisarArabicSplitter(max_tokens=256, overlap_tokens=32)
        docs = splitter.create_documents(["نص عربي طويل..."])
    """

    def __init__(
        self,
        max_tokens: int = 256,
        overlap_tokens: int = 32,
        target_model: str = "arabertv02",
        auto_preprocess: bool = True,
    ):
        self.chunker = ArabicDiscourseChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            target_model=target_model,
            auto_preprocess=auto_preprocess,
        )

    def split_text(self, text: str) -> List[str]:
        """Split input string into preprocessed chunk strings."""
        return [c.text for c in self.chunker.chunk(text)]

    def create_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Create LangChain-like Document dictionaries with page_content and metadata."""
        docs: List[Dict[str, Any]] = []
        for i, text in enumerate(texts):
            base_meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            chunks = self.chunker.chunk(text, extra_metadata=base_meta)
            for c in chunks:
                docs.append({
                    "page_content": c.text,
                    "metadata": {
                        **c.metadata,
                        "chunk_index": c.chunk_index,
                        "start_char": c.start_char,
                        "end_char": c.end_char,
                        "token_count": c.token_count,
                        "route_used": c.route_used,
                    },
                })
        return docs
