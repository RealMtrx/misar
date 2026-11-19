# Changelog

All notable releases and engineering updates to **Misar (مِسار)** are documented in this file.

## [1.0.0] — 2026-11-19

### Added
- **Linear-Time `O(n)` Linguistic Profiler (`LinguisticProfiler`):** Extracts 5 structural & sociolinguistic signals in a single pass (< 0.25 ms CPU): Dialect Ratio (`P_dial`), Diacritic Density (`D_diac`), Elongation & Social Noise Score (`E_noise`), Character Shannon Entropy (`H_char`), and Code-Switching Ratio (`R_latin`).
- **Context-Aware Adaptive Router (`MisarRouter`):** Dispatches each input dynamically across `FAST_MSA`, `DIALECT_PRESERVE`, and `NOISY_SOCIAL` (with hybrid dialect preservation) without any model retraining (`Zero Parameter Update`).
- **Real Vocabulary WordPiece Tokenizer (`WordPieceTokenizer`):** Pure-Python tokenizer compatible with `aubmindlab/bert-base-arabertv02` and `CAMeL-Lab/bert-base-arabic-camelbert-mix`, with diacritic/tatweel segmentation and an offline heuristic fallback (`HeuristicSubwordTokenizer`).
- **Hugging Face Drop-in Integration (`wrap_hf_pipeline`):** One-line preprocessing wrapper for any Transformers pipeline.
- **CLI & Hardened Web Workbench (`misar` + REST API):** `misar demo`, `misar route`, `misar bench`, and `misar serve` with a responsive Arabic RTL interactive workbench, payload bounds (`1 MiB`), and XSS-safe DOM rendering.
- **Automated Verification Suite:** 23 unit, security, server, and real-vocabulary tests plus a reproducible 15-sample tech & Arabic benchmark (`-46.82%` tokens vs Raw, `-25.74%` vs Legacy Static, `156 -> 0` `[UNK]` recovery).
