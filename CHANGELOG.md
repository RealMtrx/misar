# Changelog

All notable releases and engineering updates to **Misar (مِسار)** are documented in this file.

## [2.0.1] — 2026-11-19

### Fixed
- **Emoji Stripping & Polarity Support on Long MSA Inputs (`FAST_MSA`):** `MSALightPipeline` now runs `process_emojis` across all emoji modes (`strip`, `keep`, `sentiment`) so long diacritized prose containing scattered emojis (`🌍`, `✨`, `⏳`, `⭐`) removes all emojis in default `strip` mode with **`0 [UNK]`** instead of leaving stray emoji `[UNK]` tokens.
- **Expanded BMP Emoji Regex Coverage:** Extended `EMOJI_REGEX` and `EMOJI_RUN_REGEX` (Python & offline JS) to strip/process Miscellaneous Technical (`\u2300-\u23FF`, e.g., `⏳`, `⏰`) and Stars/Circles (`\u2B00-\u2BFF`, e.g., `⭐`).
- **Workbench RTL Negative Formatting & Fallback Guard:** Wrapped negative token delta values (`-8.3%`) cleanly inside `.ltr` spans and added client-side legacy calculation fallback if a browser reaches a stale backend endpoint.

## [2.0.0] — 2026-11-19

### Added & Improved
- **Vocab-Verified Orthography Auto-Heal (`heal_orthography_with_vocab`):** O(1) vocabulary-lookup repairs dropped final Taa Marbuta (`البرمجه -> البرمجة`, `المنصه -> المنصة`) when raw surface fails single-token match and feminine noun exists in loaded model vocab—with strict possessive proclitic & masculine pronoun safety (`بيته`, `ملفه`, `ومديره` remain untouched).
- **5-Model Authentic Tokenizer Support (`PRETRAINED_MODEL_CONFIGS`):** Added native WordPiece configurations with accurate `do_lower_case` parity for `UBC-NLP/MARBERTv2` (`do_lower_case=True`), `UBC-NLP/ARBERTv2` (`do_lower_case=True`), and `qarib/bert-base-qarib` alongside `arabertv02` and `camelbert-mix`.
- **Code-Switching Clitic Hygiene & Opt-In Arabizi:** Resolves Arabic prefix clitics glued with kashida before Latin tech words (`الـ Backend -> الBackend`, `بـ Docker -> بDocker`), plus tech-safe opt-in Franco-Arabic transliteration (`arabizi_mode=True`) protecting identifiers (`base64`, `utf8`, `http2`).
- **Sentiment-Aware Emoji Modes (`emoji_mode`):** Supports `strip` (default), `keep` (deduplicated spacing), and `sentiment` (emits single real vocab tokens `سلبي` / `إيجابي` to preserve emotional polarity with 0 `[UNK]`).
- **High-Throughput ETL & Hugging Face Datasets Integration:** Added `preprocess_batch`, memory-constant `preprocess_stream`, `hf_dataset_map(..., batched=True)`, and `compare_baselines()` (3-way Raw vs Fair Legacy Static vs Misar v2 comparison).
- **3-Way Scientific Web Workbench & CLI Upgrades:** Redesigned the RTL Web Workbench to display **Raw vs. Legacy Static vs. Misar v2.0.0** side-by-side with live Model / Emoji-Mode / Auto-Heal / Arabizi controls, expanded REST endpoints (`/api/health`, `/api/route`, `/api/suite`), and added `--compare`, `--emoji-mode`, `--arabizi`, `--no-auto-heal` CLI flags.
- **Expanded Verification & Benchmarks:** **32 automated unit/server/real-vocab tests** passing; fair benchmark against denoising Legacy Static (`NoisySocialPipeline(preserve_orthography=False)`) achieves **-46.91%** tokens vs Raw (`469 -> 249`), **-16.44%** vs Legacy Static (`298 -> 249`), **`157 -> 0` `[UNK]`** recovery, and **4,763 texts/sec** CPU throughput.

## [1.0.0] — 2026-11-19

### Added
- **Linear-Time `O(n)` Linguistic Profiler (`LinguisticProfiler`):** Extracts 5 structural & sociolinguistic signals in a single pass (< 0.25 ms CPU): Dialect Ratio (`P_dial`), Diacritic Density (`D_diac`), Elongation & Social Noise Score (`E_noise`), Character Shannon Entropy (`H_char`), and Code-Switching Ratio (`R_latin`).
- **Context-Aware Adaptive Router (`MisarRouter`):** Dispatches each input dynamically across `FAST_MSA`, `DIALECT_PRESERVE`, and `NOISY_SOCIAL` (with hybrid dialect preservation) without any model retraining (`Zero Parameter Update`).
- **Real Vocabulary WordPiece Tokenizer (`WordPieceTokenizer`):** Pure-Python tokenizer compatible with `aubmindlab/bert-base-arabertv02` and `CAMeL-Lab/bert-base-arabic-camelbert-mix`, with diacritic/tatweel segmentation and an offline heuristic fallback (`HeuristicSubwordTokenizer`).
- **Hugging Face Drop-in Integration (`wrap_hf_pipeline`):** One-line preprocessing wrapper for any Transformers pipeline.
- **CLI & Hardened Web Workbench (`misar` + REST API):** `misar demo`, `misar route`, `misar bench`, and `misar serve` with a responsive Arabic RTL interactive workbench, payload bounds (`1 MiB`), and XSS-safe DOM rendering.
- **Automated Verification Suite:** 23 unit, security, server, and real-vocabulary tests plus a reproducible 15-sample tech & Arabic benchmark (`-46.57%` tokens vs Raw, `-26.33%` vs Legacy Static, `156 -> 0` `[UNK]` recovery).
