# Contributing to Misar (مِسار)

Contributions, dialect lexicon refinements, benchmark corpora, and Transformer vocabulary adapters are warmly welcome.

## Development Setup

```bash
git clone https://github.com/RealMtrx/misar.git
cd misar
pip install -e ".[dev]"
```

## Engineering Standards

1. **Zero Mandatory Runtime Dependencies:** Keep core `src/misar/` pure Python (`>=3.9`) using only the Python Standard Library.
2. **MSA Safety First:** Never introduce dialect lexicon entries that collide with common Modern Standard Arabic (MSA) words (verify against news, classical, and formal corpora).
3. **Run Verification Before Submitting a PR:**

```bash
python -m unittest discover -s tests -v
python -m misar bench
python -m misar serve --test-once
```

4. Keep any changes to offline heuristics in `src/misar/` synchronized with the browser offline fallback inside `src/misar/web/index.html`.
