# Changelog

All notable releases and engineering updates to **Misar (مِسار)** are documented in this file.

## [2.3.0] — 2026-11-19

### Added & Improved
- **Microsecond Arabic Regional Dialect & Register Classifier (`misar.classifier` & `DialectClassifier`):**
  - Ultra-fast deterministic classification (< 0.1 ms on CPU) across major Arabic dialect families:
    - **Saudi & Gulf (`saudi_gulf`):** Najdi, Hijazi, Emirati, Kuwaiti markers (`وش`, `أبغى`, `الحين`, `سالفة`, `تكفى`).
    - **Levantine (`levantine`):** Syrian, Lebanese, Jordanian, Palestinian markers (`شو بدك`, `منيح`, `هيك`, `هلق`, `زلمة`).
    - **Egyptian (`egyptian`):** Cairene and Sa'idi markers (`عايز كده`, `دلوقتي`, `أوي`, `ازيك`, `بتاع`).
    - **Maghrebi (`maghrebi`):** Moroccan, Algerian, and Tunisian markers (`ديالنا`, `بزاف`, `دابا`, `زوين`, `مزيان`).
    - **Modern Standard Arabic (`msa`):** Formal and classical MSA expressions.
  - Automatic discourse register identification: `classical_heritage`, `tech_dev`, `formal_official`, and `social_slang`.
- **Semantic Prompt Cache Gateway (`misar.canonicalize` & `canonicalize`):**
  - Normalizes dialectal interrogatives and temporal variants to canonical MSA representations (`وش تبغى الحين؟` → `ما تبغى الآن؟`, `عايز ايه دلوقتي؟` → `أريد ما الآن؟`).
  - Maximizes prefix-matching hit rates on LLM prompt caching layers (Anthropic, OpenAI, ALLaM, vLLM).
- **High-Throughput Multi-Core Parallel Processing Engine (`misar.parallel` & `route_parallel`):**
  - Zero-lock multiprocessing pool architecture for batch processing massive datasets.
  - High-throughput parallel ingestion (`process_dataset_file`) for JSONL, CSV, TSV, and plain text files with 15,000–25,000+ texts/sec.
- **CLI & REST API Enhancements:**
  - Added CLI commands: `misar classify "<text>" [--json]`, `misar canonicalize "<text>"`, and `misar process <input_file> --output <output_file> [--workers N]`.
  - Added REST API endpoints: `POST /api/classify` and `POST /api/canonicalize`.
  - Updated `GET /api/health` feature flags to include `dialect_classifier`, `prompt_cache_canonicalizer`, and `parallel_engine`.
- **Automated Verification:**
  - Expanded test suite from 49 to **60 comprehensive automated tests** with 100% pass rate.

### Added & Improved
- **Arabic Semantic & Discourse-Aware RAG Chunker (`misar.rag` & `ArabicDiscourseChunker`):**
  - Segment documents along authentic Arabic discourse transition markers (`علاوة على ذلك`, `بناءً على ذلك`, `ومن الجدير بالذكر`, `في المقابل`, `حيث أن`) and sentence boundaries.
  - Enforces strict Negation & Clitic Bonding invariant: negative particles (`لم`, `لن`, `لا`, `ما`, `ليس`, `مو`, `مش`, `ماكو`, `مب`) are never separated from their predicates across chunk boundaries.
  - Generates token-accurate metadata with start/end character offsets, target WordPiece counts, and automatic Misar routing per chunk.
  - Drop-in LangChain `MisarArabicSplitter` and LlamaIndex compatibility.
- **Enterprise Arabic LLM Prompt Guard & PII Sanitizer (`misar.guard` & `ArabicPromptGuard`):**
  - Neutralizes adversarial BiDi override characters (`\u202E`, `\u202D`) and invisible zero-width characters (`\u200B`, `\uFEFF`).
  - Detects tatweel-obfuscated jailbreak attempts (`تــــجــــاهــــل الــــتــــعــــلــــيــــمــــات`) through dual-pass normalized analysis.
  - Validates and redacts authentic Saudi National ID / Iqama numbers using official Luhn-variant checksum algorithm (`[رقم_هوية_محجوب]`).
  - Validates and redacts Saudi IBAN numbers using ISO 7064 Mod 97-10 checksum (`[آيبان_محجوب]`).
  - Redacts Arab mobile phone numbers and email addresses.
- **REST API & CLI Enhancements:**
  - Added `POST /api/chunk` and `POST /api/guard` endpoints with comprehensive payload validation.
  - Added CLI commands: `misar chunk <source>` and `misar guard "<text>"`.
- **Automated Verification:**
  - Expanded test suite from 39 to **49 automated unit, API, and Transformer WordPiece vocab tests**, all passing with 100% success.

## [2.1.1] — 2026-11-19

### Changed
- **Documentation & PyPI Architecture:** Replaced raw client-side Mermaid flowchart block with high-resolution responsive vector-flow infographic (`docs/misar-architecture-flow.png`) for seamless cross-platform rendering across PyPI package descriptions, mobile GitHub views, and dark/light themes.

## [2.1.0] — 2026-11-19

### Added & Improved
- **Unicode Presentation Forms & Keyboard Lookalike Defense (`normalize_unicode_safety`):** Recovers PDF Arabic Presentation Forms-A/B (`NFKC`: `ﺍﻟﺒﺮﻣﺠﺔ → البرمجة`, `ﻻ → لا`) without collapsing hamzas or taa marbuta, normalizes Persian/Urdu lookalikes (`ک → ك`, `ی/ے → ي`, `ۀ → ة`, `ہ/ھ → ه`), strips soft hyphens (`\u00AD`), and cleans smart typography/NBSP to eliminate `[UNK]` blindness on copied documents and multi-locale keyboards.
- **Long-Essay Social Artifact Routing (`social_artifact_count`):** Inputs with explicit URLs, `@mentions`, or inline `#hashtags` (`hashtags + urls_mentions >= 1`) now dispatch cleanly through `NOISY_SOCIAL` (`preserve_orthography=True`) even on long MSA essays—stripping tracking links while respecting Markdown `# Heading` syntax.
- **Clause-Aware Multi-Route Document Engine (`route_document` & `DocumentRoutingResult`):** Splits multi-paragraph / multi-sentence documents on line and punctuation boundaries (`SENTENCE_BOUNDARY_REGEX`), routes each clause independently (`FAST_MSA`, `DIALECT_PRESERVE`, `NOISY_SOCIAL`), and aggregates token, UNK, and route-distribution telemetry across the entire document.
- **Quadratic Transformer Attention Economics (`attention_savings_pct`):** Computes $O(L^2)$ self-attention compute savings (`1 - (L_proc / L_raw)^2`) across `RoutingResult`, `DocumentRoutingResult`, REST API payloads, Web UI badges, and evaluation reports (saving **67.8%–74.3%** attention FLOPs across all 5 Arabic Transformer models).
- **Scientific Cross-Model Evaluation Suite (`evaluation.py` & `misar eval`):** Added reproducible empirical verification across `arabertv02`, `arbertv2`, `camelbert-mix`, `marbertv2`, and `qarib` measuring Minimal-Pair Semantic Disambiguation (`عالمة/عالمه`, `مديرة/مديره`: **100% Misar vs 0% Legacy**), Dialectal Negation Preservation (**100%**), Sentiment Polarity Retention (**100%**), and Token/`O(L^2)` gains.
- **Web Workbench & REST API (`v2.1.0`):** Added `POST /api/document`, optional `"document_mode": true` on `POST /api/route`, Clause-Aware per-segment badges (`#segmentsBar`), offline JS `normalizeUnicodeJS` + `routeDocumentJS` fallback, and quadratic `O(L²)` savings telemetry across **39/39** automated tests.

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
