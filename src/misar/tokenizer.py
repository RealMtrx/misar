# -*- coding: utf-8 -*-
"""
Real WordPiece Tokenizer (AraBERTv2 / CAMeLBERT compatible) with zero mandatory dependencies.
Supports Hugging Face vocab.txt download/caching, external HF AutoTokenizer wrapping, and
honest offline heuristic fallback when no vocabulary file is cached.
"""

import os
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Set, Tuple

from .lexicons import (
    DECORATION_MARKS_REGEX,
    EMOJI_REGEX,
    PUNCT_REPEAT_REGEX,
    REPEATED_CHARS_REGEX,
)

PRETRAINED_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "arabertv02": {
        "repo": "aubmindlab/bert-base-arabertv02",
        "vocab_url": "https://huggingface.co/aubmindlab/bert-base-arabertv02/resolve/main/vocab.txt",
        "do_lower_case": False,
        "display_name": "AraBERTv02 (64k MSA/Multi)",
    },
    "camelbert-mix": {
        "repo": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
        "vocab_url": "https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix/resolve/main/vocab.txt",
        "do_lower_case": False,
        "display_name": "CAMeLBERT-Mix (30k Dialect/MSA)",
    },
    "marbertv2": {
        "repo": "UBC-NLP/MARBERTv2",
        "vocab_url": "https://huggingface.co/UBC-NLP/MARBERTv2/resolve/main/vocab.txt",
        "do_lower_case": True,
        "display_name": "MARBERTv2 (100k Social/Dialect)",
    },
    "arbertv2": {
        "repo": "UBC-NLP/ARBERTv2",
        "vocab_url": "https://huggingface.co/UBC-NLP/ARBERTv2/resolve/main/vocab.txt",
        "do_lower_case": True,
        "display_name": "ARBERTv2 (100k MSA/Books/News)",
    },
    "qarib": {
        "repo": "qarib/bert-base-qarib",
        "vocab_url": "https://huggingface.co/qarib/bert-base-qarib/resolve/main/vocab.txt",
        "do_lower_case": False,
        "display_name": "QARiB (64k Tweets/Dialect)",
    },
    "allam-7b": {
        "repo": "sdaia/allam-7b-instruct",
        "vocab_url": "https://huggingface.co/sdaia/allam-7b-instruct/resolve/main/vocab.txt",
        "do_lower_case": False,
        "display_name": "ALLaM-7B (SDAIA Sovereign Arabic LLM)",
    },
    "llama3-arabic": {
        "repo": "meta-llama/Meta-Llama-3-8B-Instruct",
        "vocab_url": "https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/resolve/main/vocab.txt",
        "do_lower_case": False,
        "display_name": "Llama-3-Arabic (128k Multi-Lingual)",
    },
}

PRETRAINED_VOCAB_URLS: Dict[str, Tuple[str, str]] = {
    key: (cfg["repo"], cfg["vocab_url"]) for key, cfg in PRETRAINED_MODEL_CONFIGS.items()
}

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "misar"


class TokenizerProtocol(Protocol):
    backend_name: str
    is_real_vocab: bool

    def tokenize(self, text: str) -> List[str]: ...
    def has_token(self, token: str) -> bool: ...


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("C")


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    cat = unicodedata.category(ch)
    return cat.startswith("P")


class WordPieceTokenizer:
    """
    Pure-Python BERT BasicTokenizer + WordPiece longest-match-first implementation
    matching `AraBERTv2`, `CAMeLBERT`, `MARBERTv2`, `ARBERTv2`, and `QARiB`
    configurations (`do_lower_case` aware, `strip_accents=None`).
    """

    def __init__(
        self,
        vocab: Dict[str, int],
        backend_name: str = "wordpiece-custom",
        unk_token: str = "[UNK]",
        never_split: Sequence[str] = ("[بريد]", "[مستخدم]", "[رابط]", "[إيجابي]", "[سلبي]"),
        max_input_chars_per_word: int = 100,
        do_lower_case: bool = False,
    ):
        self.vocab: Dict[str, int] = vocab
        self.vocab_set: Set[str] = set(vocab.keys())
        self.backend_name = backend_name
        self.is_real_vocab = True
        self.unk_token = unk_token
        self.never_split: Set[str] = set(never_split)
        self.max_input_chars_per_word = max_input_chars_per_word
        self.do_lower_case = do_lower_case

    @classmethod
    def from_vocab_file(
        cls,
        vocab_path: os.PathLike,
        backend_name: str = "wordpiece-file",
        do_lower_case: bool = False,
    ) -> "WordPieceTokenizer":
        vocab: Dict[str, int] = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\r\n")
                if token:
                    vocab[token] = idx
        if not vocab:
            raise ValueError(f"Vocabulary file is empty: {vocab_path}")
        return cls(vocab=vocab, backend_name=backend_name, do_lower_case=do_lower_case)

    @classmethod
    def from_pretrained(
        cls,
        model_key: str = "arabertv02",
        cache_dir: Optional[os.PathLike] = None,
        timeout: int = 15,
    ) -> "WordPieceTokenizer":
        if model_key not in PRETRAINED_MODEL_CONFIGS:
            supported = ", ".join(PRETRAINED_MODEL_CONFIGS.keys())
            raise ValueError(f"Unknown pretrained key '{model_key}'. Supported: {supported}")

        cfg = PRETRAINED_MODEL_CONFIGS[model_key]
        hf_repo = cfg["repo"]
        url = cfg["vocab_url"]
        do_lower = bool(cfg["do_lower_case"])
        cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cache_root.mkdir(parents=True, exist_ok=True)
        local_vocab = cache_root / f"{model_key}_vocab.txt"

        if not local_vocab.exists() or local_vocab.stat().st_size == 0:
            req = urllib.request.Request(url, headers={"User-Agent": "misar-nlp/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            tmp_file = local_vocab.with_suffix(".tmp")
            tmp_file.write_bytes(data)
            tmp_file.replace(local_vocab)

        return cls.from_vocab_file(local_vocab, backend_name=f"{hf_repo} (WordPiece)", do_lower_case=do_lower)

    def has_token(self, token: str) -> bool:
        if token in self.vocab_set:
            return True
        if self.do_lower_case:
            return token.lower() in self.vocab_set
        return False

    def _clean_text(self, text: str) -> str:
        out: List[str] = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            else:
                out.append(ch)
        return "".join(out)

    def _split_on_punc(self, token: str) -> List[str]:
        if token in self.never_split:
            return [token]
        output: List[List[str]] = []
        start_new_word = True
        for ch in token:
            if _is_punctuation(ch):
                output.append([ch])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                    start_new_word = False
                output[-1].append(ch)
        return ["".join(chunk) for chunk in output]

    def _wordpiece(self, token: str) -> List[str]:
        if token in self.never_split:
            return [token]
        if len(token) > self.max_input_chars_per_word:
            return [self.unk_token]

        sub_tokens: List[str] = []
        start = 0
        length = len(token)
        while start < length:
            end = length
            cur_substr: Optional[str] = None
            while start < end:
                substr = token[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab_set:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                return [self.unk_token]
            sub_tokens.append(cur_substr)
            start = end
        return sub_tokens

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        cleaned = self._clean_text(text)
        basic_tokens: List[str] = []
        for raw_tok in cleaned.strip().split():
            if raw_tok in self.never_split:
                basic_tokens.append(raw_tok)
            else:
                norm_tok = raw_tok.lower() if self.do_lower_case else raw_tok
                basic_tokens.extend(self._split_on_punc(norm_tok))

        output: List[str] = []
        for tok in basic_tokens:
            if not tok:
                continue
            marks = DECORATION_MARKS_REGEX.findall(tok)
            if marks:
                base = DECORATION_MARKS_REGEX.sub("", tok)
                if base:
                    output.extend(self._wordpiece(base))
                output.extend(marks)
            else:
                output.extend(self._wordpiece(tok))
        return output


class HuggingFaceTokenizerAdapter:
    """Adapter allowing any loaded Hugging Face AutoTokenizer to plug into MisarRouter."""

    def __init__(self, hf_tokenizer, name: Optional[str] = None):
        self._tok = hf_tokenizer
        self.backend_name = name or getattr(hf_tokenizer, "name_or_path", "huggingface-tokenizer")
        self.is_real_vocab = True
        self._vocab_set: Set[str] = set(hf_tokenizer.get_vocab().keys()) if hasattr(hf_tokenizer, "get_vocab") else set()

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        return list(self._tok.tokenize(text))

    def has_token(self, token: str) -> bool:
        return token in self._vocab_set


class HeuristicSubwordTokenizer:
    """Explicit offline fallback tokenizer used only when no vocabulary is cached and offline."""

    backend_name = "heuristic-offline"
    is_real_vocab = False

    def has_token(self, token: str) -> bool:
        return False

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        tokens: List[str] = []
        for word in text.strip().split():
            emojis = EMOJI_REGEX.findall(word)
            clean_part = EMOJI_REGEX.sub("", word)
            if clean_part:
                if clean_part.startswith("#"):
                    tokens.append("#")
                    for sub in clean_part[1:].split("_"):
                        if sub:
                            tokens.append(f"##{sub}" if len(sub) <= 5 else sub)
                else:
                    marks = DECORATION_MARKS_REGEX.findall(clean_part)
                    base_word = DECORATION_MARKS_REGEX.sub("", clean_part)
                    if base_word:
                        if REPEATED_CHARS_REGEX.search(base_word):
                            pieces: List[str] = []
                            buf = ""
                            i = 0
                            while i < len(base_word):
                                ch = base_word[i]
                                run = 1
                                while i + run < len(base_word) and base_word[i + run] == ch:
                                    run += 1
                                if run >= 3:
                                    if buf:
                                        pieces.append(buf)
                                        buf = ""
                                    pieces.append("##" + base_word[i : i + run])
                                else:
                                    buf += ch * run
                                i += run
                            if buf:
                                pieces.append("##" + buf if pieces else buf)
                            tokens.extend(pieces if pieces else [base_word])
                        elif len(base_word) > 8 or PUNCT_REPEAT_REGEX.search(base_word):
                            mid = max(4, len(base_word) // 2)
                            tokens.extend([base_word[:mid], "##" + base_word[mid:]])
                        else:
                            tokens.append(base_word)
                    tokens.extend(marks)
            tokens.extend(emojis)
        return tokens


def load_default_tokenizer(
    model_key: str = "arabertv02",
    allow_download: bool = True,
    cache_dir: Optional[os.PathLike] = None,
) -> TokenizerProtocol:
    """
    Loads the cached or downloaded Arabic Transformer WordPiece tokenizer.
    Falls back explicitly to HeuristicSubwordTokenizer if offline and uncached.
    """
    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    local_vocab = cache_root / f"{model_key}_vocab.txt"

    if local_vocab.exists() and local_vocab.stat().st_size > 0:
        cfg = PRETRAINED_MODEL_CONFIGS.get(model_key, {})
        hf_repo = cfg.get("repo", model_key)
        do_lower = bool(cfg.get("do_lower_case", False))
        return WordPieceTokenizer.from_vocab_file(
            local_vocab,
            backend_name=f"{hf_repo} (WordPiece)",
            do_lower_case=do_lower,
        )

    if allow_download:
        try:
            return WordPieceTokenizer.from_pretrained(model_key=model_key, cache_dir=cache_dir)
        except Exception:
            pass

    return HeuristicSubwordTokenizer()
