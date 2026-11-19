# -*- coding: utf-8 -*-
"""
LlamaIndex Node Parser & Arabic Splitter Integration for Misar.
"""

from typing import Any, List, Optional, Sequence

from ..rag import ArabicDiscourseChunker, SemanticChunk


class MisarNodeParser:
    """
    Drop-in LlamaIndex node parser integrating Misar Arabic discourse-aware chunking.
    Preserves Arabic negation particles (لم, لن, ما, لا) and splits at rhetorical discourse markers.
    """

    def __init__(
        self,
        max_tokens: int = 256,
        overlap_tokens: int = 32,
        target_model: str = "arabertv02",
        auto_preprocess: bool = True,
        include_metadata: bool = True,
    ) -> None:
        self.chunker = ArabicDiscourseChunker(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            target_model=target_model,
            auto_preprocess=auto_preprocess,
        )
        self.include_metadata = include_metadata

    def split_text(self, text: str) -> List[str]:
        """Split raw Arabic text into discourse-aware chunks."""
        chunks = self.chunker.chunk(text)
        return [c.text for c in chunks]

    def get_nodes_from_documents(self, documents: Sequence[Any], **kwargs: Any) -> List[Any]:
        """
        LlamaIndex BaseNodeParser compliant method: converts Document objects into TextNode objects.
        Works seamlessly whether llama_index is installed or via mock duck-typing.
        """
        try:
            from llama_index.core.schema import TextNode
        except ImportError:
            class TextNode(dict):
                def __init__(self, text: str, metadata: dict):
                    super().__init__(text=text, metadata=metadata)
                    self.text = text
                    self.metadata = metadata

        nodes: List[Any] = []
        for doc in documents:
            doc_text = getattr(doc, "text", "") or (doc.get("text", "") if isinstance(doc, dict) else str(doc))
            doc_meta = getattr(doc, "metadata", {}) or (doc.get("metadata", {}) if isinstance(doc, dict) else {})

            chunks = self.chunker.chunk(doc_text)
            for chunk in chunks:
                meta = dict(doc_meta)
                if self.include_metadata:
                    meta.update({
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "route_used": chunk.route_used,
                    })
                nodes.append(TextNode(text=chunk.text, metadata=meta))

        return nodes
