# Changelog

All notable releases and engineering updates to **Misar (مِسار)** are documented in this file.

## [2.7.0] — 2026-11-20

### Added & Improved
- **Transparent OpenAI-Compatible Reverse Proxy Gateway (`misar.proxy` & CLI `misar proxy`):**
  - Drop-in reverse proxy server compatible with `/v1/chat/completions`, `/v1/completions`, `/v1/models`, and `/health`.
  - Seamlessly intercepts prompts, blocks adversarial prompt injections with HTTP 400 (`misar_security_violation`), masks Saudi PII (National ID, IBAN, Phone), and collapses redundant tokens.
  - Injects live telemetry headers (`X-Misar-Processed`, `X-Misar-Tokens-Saved`, `X-Misar-Attention-Saved-Pct`, `X-Misar-PII-Masked`, `X-Misar-Latency-Ms`).
  - Supports zero-config standalone mock evaluation mode or transparent forwarding to any upstream LLM (Ollama, vLLM, OpenAI).
- **Arabizi (Franco-Arabic / عربيزي) Decoupler & Transliteration Engine (`misar.arabizi` & CLI `misar arabizi`):**
  - High-precision dictionary and phonetic transliterator converting Arabizi youth chat numerals (`3=ع`, `7=ح`, `5=خ`, `2=ء`, `6=ط`, `8=ق`, `9=ص`) into canonical Arabic script.
  - Integrated into `MisarRouter(arabizi_mode=True)` pipeline while strictly safeguarding technical terms (`API`, `Python`, `Docker`, `base64`, `utf8`, `http2`).
- **Synthetic Data Augmentation & Stress-Testing Engine (`misar.augment` & CLI `misar augment`):**
  - Synthesizes realistic typographical noise (hamza, taa marbuta/haa, yaa/alif maqsura confusions), tatweel/kashida stretching, and random tashkeel (harakat).
  - Evaluates and quantifies LLM token inflation and Misar recovery efficiency.
- **Automated Academic Paper & LaTeX Preprint Generator (`misar.paper` & CLI `misar paper`):**
  - Compiles full IEEE/ACL publication-ready LaTeX research preprint (`misar paper --format latex`) with formal mathematical equations, multi-model benchmark tables, and BibTeX citations.
  - Generates Markdown research dossier (`misar paper --format markdown`).
- **Reference Example & Test Coverage:**
  - Added `examples/08_transparent_proxy_and_arabizi.py`.
  - Added `tests/test_arabizi.py`, `tests/test_proxy.py`, `tests/test_augment.py`, and `tests/test_paper.py`.
  - Scaled automated test suite to **105 comprehensive unit tests** passing with 100% success rate.

## [2.6.0] — 2026-11-20

### Added & Improved
- **Sovereign Arabic AI Gateway & Local Model Support:**
  - Added native tokenizer configs and token optimization profiles for **SDAIA ALLaM 7B (`allam-7b`)** and **LLaMA-3 Arabic (`llama3-arabic`)**.
  - `MisarOllamaAdapter`: Drop-in adapter for local Ollama instances running sovereign models, providing prompt sanitization, privacy PII masking, dialect query canonicalization, and adversarial prompt injection blocking.
  - `prepare_ollama_prompt()`: High-level functional helper for single-line Ollama prompt preprocessing.
  - `misar_vllm_pre_hook()`: High-throughput inference middleware for vLLM and OpenAI-compatible local servers, sanitizing prompt and chat messages payloads while neutralizing threats.
- **Specialized Industry Domain Lexicons (`misar.lexicon_data`):**
  - Added FinTech & Banking vocabulary (`fintech.py`): 20+ Saudi financial terms (`آيبان`, `سمة`, `سداد`, `مدى`, `مرابحة`, `تورق`, `استقطاع`, `تمويل_شخصي`).
  - Added GovTech & Citizen Platforms vocabulary (`govtech.py`): 18+ public platform terms (`أبشر`, `ناجز`, `توكلنا`, `صحتي`, `بلدي`, `قوى`, `مدد`, `اعتماد`, `إحكام`).
  - Added HealthTech & Clinical vocabulary (`healthtech.py`): 15+ symptoms and clinical terms (`سخونة`, `كحة`, `دوخة`, `مستوصف`, `إسعاف`, `تطعيم`, `روشتة`).
  - Scaled total curated vocabulary entries from 155 to **208+ entries** across 9 regional and domain submodules.
  - Enhanced morphological root extractor `extract_root()` with support for `مستفعل` patterns (e.g. `مستوصف` -> `و-ص-ف`).
- **Interactive Web Dataset Auditor & ROI Calculator (Workbench Section 8 & API `/api/audit`):**
  - Added executive web auditor UI in the local dashboard with drag-and-drop file ingestion (CSV, JSONL, TXT).
  - Preset domain batches: FinTech, GovTech, HealthTech, and Security/PII.
  - Real-time KPI cards for Token Reduction %, Estimated Financial Savings (SAR/USD), Attention FLOPs Savings %, Masked PII Count, and Risk Telemetry.
  - One-click standalone executive HTML report download from the web interface.
  - Added `/api/audit` REST endpoint to `misar.web.server` with batch safety limits.
- **Cloud & Container Production Deployment:**
  - Added production multi-stage `Dockerfile` (Python 3.11-slim, non-root security user, built-in healthcheck).
  - Added `docker-compose.yml` for instant zero-config sovereign gateway deployment on port 8000.
  - Added `.dockerignore` for minimal image footprints.
- **Test Suites & Reference Examples:**
  - Added `tests/test_domain_lexicons.py` and `tests/test_local_models.py`.
  - Added `examples/07_sovereign_ai_gateway_and_allam.py`.
  - Scaled automated test suite to **89 comprehensive unit tests** passing with 100% success rate.

## [2.5.0] — 2026-11-20

### Added & Improved
- **Deep Multi-Dialect Lexicon Expansion (`misar.lexicon_data`):**
  - Modularized dialect vocabulary into dedicated geographic modules:
    - `saudi_gulf.py`: 55+ curated entries across Najdi, Hijazi, Southern, and Eastern Gulf dialects (`أبخص`, `أزهله`, `دندون`, `دحين`, `تكفى`, `مروح`, `هروج`, etc.).
    - `egyptian.py`: 15+ rich Egyptian colloquial terms (`دلوقتي`, `كده`, `أوي`, `عيش`, `جدع`, etc.).
    - `levantine.py`: 10+ Levantine terms (`شو`, `هيك`, `هلق`, `كتير`, `زلمة`, etc.).
    - `maghrebi.py`: 8+ Moroccan, Algerian, and Tunisian terms (`بزاف`, `دابا`, `ديال`, `زوين`, etc.).
    - `iraqi_yemeni.py`: 9+ Iraqi and Yemeni regional markers (`يمعود`, `ماكو`, `ناهي`, `قاحي`, etc.).
    - `msa_roots.py`: Standard Modern Arabic lexical anchors and roots.
  - Scaled total curated vocabulary entries from 86 to **155+ entries**, dynamically indexed by `ArabicLexicon`.
- **Enterprise ROI & Security Audit Engine (`misar.audit` & CLI `misar audit`):**
  - Added `AuditResult`, `audit_texts()`, and `audit_dataset_file()` supporting JSONL, CSV, TSV, and plain TXT ingestion.
  - Computes exact token count reduction, quadratic attention FLOPs savings ($O(L^2)$), and financial ROI in SAR or USD based on configurable pricing models.
  - Comprehensive PII redaction and adversarial threat quantification.
  - Added `generate_html_report()` creating standalone, responsive, executive HTML dashboard reports with KPI cards and breakdowns.
  - Added CLI command: `misar audit <dataset_path> --out <report.html> --currency SAR --price 0.015`.
- **Enterprise Framework Integrations (`misar.integrations`):**
  - `MisarSecurityMiddleware`: Zero-dependency ASGI middleware for FastAPI / Starlette, blocking prompt injection attacks and redacting sensitive PII in real-time.
  - `MisarNodeParser`: Drop-in LlamaIndex node parser integrating discourse-aware RAG chunking with metadata propagation.
  - `misar_litellm_pre_hook`: Pre-call hook for LiteLLM proxies and gateways to canonicalize dialect queries, maximize prompt cache hit rates, and cut token costs.
- **Reference Example & Test Coverage:**
  - Added `examples/06_enterprise_audit_and_middleware.py` demonstrating enterprise audits and framework integrations.
  - Added `tests/test_audit.py` and `tests/test_integrations.py`, scaling test coverage to **75 comprehensive unit tests** passing at 100%.

## [2.4.1] — 2026-11-20

### Added & Improved
- **Interactive Web Workbench v2.4.1 (`web/index.html`):**
  - Added dedicated **Section 6: Arabic Lexicon & Morphology Explorer**:
    - Real-time lexical lookup with visual cards displaying linguistic roots, dialect classification, MSA lemma, Part-of-Speech, definitions, synonyms, and contextual examples.
    - Full sentence deconstruction and canonical MSA re-synthesis with dialect density percentages.
  - Added dedicated **Section 7: Prompt Guard & PII Security Inspector**:
    - Real-time adversarial risk scoring with visual progress bars.
    - Automated detection tags for Tatweel evasion, BiDi overrides, zero-width characters, and instruction injection.
    - Live sanitization demonstrating automated redaction of Saudi National IDs and IBANs.
- **Packaging & Typing Integrity:**
  - Full PEP 561 compliance with packaged `py.typed` marker for downstream IDE type hint resolution.
  - Enterprise `examples/` directory containing verified standalone reference implementations.
  - Fully modular test architecture with 8 domain suites.

## [2.4.0] — 2026-11-19

### Architecture & Project Structure
- **Modularized Test Suite (`tests/`):**
  - Replaced monolithic 1000-line test file with 8 dedicated domain test suites:
    - `test_router.py`: Linguistic profiler & adaptive pipeline routing (28 tests).
    - `test_tokenizer.py`: WordPiece tokenizer & AraBERTv2 vocab tests (6 tests).
    - `test_server.py`: Hardened HTTP REST API & web server tests (8 tests).
    - `test_rag.py`: Semantic discourse chunker & LangChain tests (4 tests).
    - `test_guard.py`: Prompt guard, BiDi & Saudi PII redaction tests (5 tests).
    - `test_classifier.py`: Sub-dialect classification & prompt cache tests (8 tests).
    - `test_parallel.py`: Multi-core dataset parallel processing tests (2 tests).
    - `test_lexicon.py`: Multi-dialect dictionary & morphological root tests (4 tests).
- **Executable Reference Examples (`examples/`):**
  - Added 5 standalone, enterprise-grade runnable scripts covering all core capabilities:
    - `01_quickstart.py`: Adaptive routing & 3-way baseline comparison.
    - `02_semantic_rag_chunking.py`: Discourse-aware semantic chunking & LangChain integration.
    - `03_prompt_guard_and_pii.py`: Adversarial prompt inspection & Saudi ID/IBAN PII redaction.
    - `04_dialect_and_morphology.py`: Multi-dialect dictionary lookups, root extraction, sentence explanation.
    - `05_parallel_dataset_cleaning.py`: Multi-core parallel batch processing.
- **PEP 561 Type Marker (`py.typed`):**
  - Included `py.typed` marker in package data (`pyproject.toml`) ensuring full type-hinting support in IDEs and type checkers.
- **Module Roles Clarification:**
  - Added architectural docstrings distinguishing low-level regex patterns (`misar.lexicons`) from high-level semantic dictionaries (`misar.lexicon`).

### Added & Improved
- **Arabic Semantic Lexicon & Multi-Dialect Dictionary (`misar.lexicon` & `ArabicLexicon`):**
  - High-speed zero-dependency in-memory dictionary linking Arabic words directly to:
    - Standard Modern Arabic lemmas (`lemma_msa`).
    - Authentic trilateral/quadrilateral linguistic roots (`root`).
    - Dialect and regional origin (`dialect`): Saudi/Gulf, Egyptian, Levantine, Maghrebi, Iraqi, Yemeni, and MSA.
    - Part of Speech (`pos`): verb, noun, interrogative, negation, adverb, idiom.
    - Deep semantic definitions (`definition`), synonyms, and contextual examples.
- **Morphological Light Stemmer & Root Extractor (`extract_root` & `light_stem`):**
  - Zero-dependency rule-based morphological decomposition stripping compound and single proclitics (`ال`, `وال`, `بال`, `لل`, `كال`) and inflectional enclitics (`ها`, `هم`, `كم`, `نا`, `ني`, `ات`, `ين`, `ون`).
  - Matches trilateral and quadrilateral roots across major verbal and nominal patterns (`استفعال`, `تفعيل`, `مفعول`, `فاعل`, `افتعال`, `انفعال`, `مفاعيل`).
- **Interactive Sentence Explainer (`explain_text` & `TextExplanation`):**
  - Token-by-token contextual explanation deconstructing dialect expressions.
  - Automatically produces normalized canonical MSA sentences and computes regional dialect density percentages.
- **CLI & REST API Enhancements:**
  - Added CLI commands: `misar dict <word> [--json]` and `misar explain "<text>" [--json]`.
  - Added REST API endpoints: `POST /api/lexicon/lookup` and `POST /api/lexicon/explain`.
  - Updated `GET /api/health` feature flags to include `arabic_lexicon`, `morphological_root_extractor`, and `sentence_explainer`.
- **Automated Verification:**
  - Expanded test suite from 60 to **65 comprehensive automated tests** with 100% pass rate.

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
