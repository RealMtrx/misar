# Changelog

All notable releases and engineering updates to **Misar (مِسار)** are documented in this file.

## [3.1.0] — 2026-11-26

### The Enterprise & Speech Expansion Release — Semantic Similarity, Intent ABSA, and G2P Phonetics

#### 1. Arabic Semantic Similarity & Plagiarism Detector (`misar.similarity` & CLI `misar similarity`)
- Zero-dependency, multi-dimensional semantic similarity engine combining:
  - Lexical Jaccard similarity across normalized Arabic tokens.
  - Morphological tri-literal root overlap (`misar.lexicon.extract_root`) providing semantic invariance under paraphrasing.
  - Character tri-gram (3-gram) overlap capturing sub-word and orthographic similarity.
  - Longest Common Subsequence (LCS) ratio for word order and syntactic sequence preservation.
- Verbatim copied span extraction with exact start/end character offsets for academic research and content moderation.
- Arabic paraphrase detection identifying morphological root transformations with altered surface forms.
- Corpus-level plagiarism scanning (`check_plagiarism`) ranking candidate texts against multi-document corpora.
- Top-level export `compute_similarity`, `check_plagiarism`, `SimilarityResult`, `MatchedSpan`, and router integration `router.similarity()`.
- CLI command: `misar similarity "<text1>" "<text2>" [--corpus] [--threshold 0.5] [--json]`.
- REST API endpoint: POST `/api/similarity`.

#### 2. Customer Intent & Aspect-Based Sentiment Analysis (ABSA) (`misar.intent` & CLI `misar intent`)
- Multi-dialect intent classification covering 6 core commercial categories:
  - `COMPLAINT` (شكوى وبلاغ استياء)
  - `INQUIRY` (استفسار وسؤال عام)
  - `SUPPORT_REQUEST` (طلب دعم فني ومساعدة)
  - `PURCHASE_INTENT` (طلب شراء واشتراك جديد)
  - `CANCELLATION` (طلب إلغاء واسترجاع أموال)
  - `PRAISE` (مدح وشكر وثناء)
- Multi-dialect colloquial markers across Saudi/Gulf, Egyptian, Levantine, Maghrebi, and MSA.
- Fine-grained Aspect-Based Sentiment Analysis (ABSA) extracting polarity and evidence snippets across 6 product dimensions:
  - `speed` (السرعة والأداء)
  - `price` (السعر والتكلفة)
  - `support` (الدعم الفني والخدمة)
  - `quality` (الجودة والاعتمادية)
  - `usability` (سهولة الاستخدام)
  - `delivery` (التوصيل والشحن)
- Top-level export `classify_intent`, `IntentResult`, `AspectSentiment`, and router integration `router.classify_intent()`.
- CLI command: `misar intent "<text>" [--json]`.
- REST API endpoint: POST `/api/intent`.

#### 3. Arabic Grapheme-to-Phoneme (G2P) & Speech Synthesis Engine (`misar.phonemize` & CLI `misar phonemize`)
- High-precision International Phonetic Alphabet (IPA) conversion calibrated for Arabic speech engines (TTS/STT).
- Solar letter assimilation (اللام الشمسية) mapping nouns following `الـ` with the 14 sun letters (`ت، ث، د، ذ، ر، ز، س، ش، ص، ض، ط، ظ، ل، ن`) to assimilated geminated consonants.
- Lunar letter preservation (اللام القمرية) retaining `ʔal` prefix articulation.
- Pharyngealized emphatic consonant realizations (`ص` $\to$ `/sˤ/`, `ض` $\to$ `/dˤ/`, `ط` $\to$ `/tˤ/`, `ظ` $\to$ `/ðˤ/`, `ع` $\to$ `/ʕ/`, `خ` $\to$ `/x/`, `غ` $\to$ `/ɣ/`, `ح` $\to$ `/ħ/`, `ء` $\to$ `/ʔ/`).
- Short and long vowel duration acoustic modeling (`a`, `u`, `i`, `aː`, `uː`, `iː`, `an`, `un`, `in`).
- Syllabic meter estimator and acoustic speech duration estimator calibrated for natural Arabic cadence.
- Optional automatic short-vowel restoration integration via `misar.diacritize`.
- Top-level export `phonemize_arabic`, `PhonemizationResult`, and router integration `router.phonemize()`.
- CLI command: `misar phonemize "<text>" [--no-diacritize] [--json]`.
- REST API endpoint: POST `/api/phonemize`.

#### 4. Web Workbench v3.1 & Sovereign AI Gateway Expansion
- Section 20: Arabic Semantic Similarity, Plagiarism & Paraphrase Detector with visual similarity gauge and verbatim copied spans.
- Section 21: Customer Intent & Aspect-Based Sentiment Analyzer (ABSA) with intent score progress bars and aspect polarity cards.
- Section 22: Arabic G2P & Speech Phonemizer with IPA transcription, phoneme token tags, syllable counter, and speech duration estimator.
- Live system status badge bumped to `v3.1.0`.

#### 5. Engineering Reliability & Empirical Testing
- Expanded test suite from 154 to **176 comprehensive automated unit tests** across 18 test files with 100% pass rate.
- Added `tests/test_similarity.py`, `tests/test_intent.py`, `tests/test_phonemize.py`.
- 100% pure Python standard library for core engines (Zero mandatory external dependencies).
- Multi-dialect and MSA resilience verified under Python 3.9 through 3.14.

## [3.0.0] — 2026-11-23

### The Major Generational Release — Sovereign Arabic NLP Trinity

#### 1. Arabic Diacritization & Tashkeel Recovery Engine (`misar.diacritize` & CLI `misar diacritize`)
- Zero-dependency, pure Python standard-library engine restoring accurate Arabic short vowels (Fatha, Damma, Kasra, Sukun, Shaddah, and Tanween).
- Morphological template recognition covering major Arabic measures (تفعيل، مفعلة، استفعال، افتعال، فاعل، مفعول، فعيل، تفعل).
- Canonical grammatical particle and preposition recovery (`في` -> `فِي`، `من` -> `مِنْ`، `إلى` -> `إِلَى`، `على` -> `عَلَى`، `مع` -> `مَعَ`، `حتى` -> `حَتَّى`، `ثم` -> `ثُمَّ`، `لكن` -> `لَكِنَّ`).
- Recursive clitic prefix/suffix decomposition (`وـ`, `فـ`, `بـ`, `لـ`, `كـ`, `الـ`).
- Full preservation of pre-existing diacritics, Latin terms, technical tokens, numbers, and emojis.
- Top-level export `diacritize_text`, `diacritize_word`, `DiacritizationResult`, and `router.process(..., diacritize=True)`.
- CLI command: `misar diacritize "<text>" [--json]`.

#### 2. Arabic Text Summarization & Keyphrase Extraction Engine (`misar.summarize` & CLI `misar summarize`)
- Extractive TextRank / LexRank graph centrality algorithm specifically tuned for Arabic discourse structures.
- Morphological root and stem matching for lexical overlap without vector database or PyTorch overhead.
- Discourse connectors and structural position priors (lead sentence thesis, conclusion, and sovereign entity salience boosts).
- Salient Arabic multi-word keyphrase and concept extraction with stopword filtering.
- Compression ratio reporting, sentence extraction indices, and sub-millisecond execution.
- Top-level export `summarize_text`, `extract_keyphrases`, `SummarizationResult`, `KeyphraseSpan`.
- CLI command: `misar summarize "<text>" [--sentences 3] [--ratio 0.5] [--json]`.

#### 3. Arabic Readability & Linguistic Complexity Metric Engine (`misar.readability` & CLI `misar readability`)
- Full implementation of the academic OSMAN Readability Metric (Open Source Metric for Measuring Arabic Narratives) adapted for Arabic prose:
  $$\text{OSMAN} = 206.835 - (1.015 \times \text{ASL}) - (84.6 \times \text{ASW})$$
- Fine-grained educational stage mapping (ابتدائي، متوسط، ثانوي، جامعي، تخصصي أكاديمي).
- Syllabic mora and phonetic vowel estimator for Arabic words.
- Lexical diversity indicators: Type-Token Ratio (TTR) and Hapax Legomena frequency.
- Arabic reading time estimator calibrated for standard Arabic reading velocity (160 words/min).
- Top-level export `analyze_readability`, `ReadabilityResult`.
- CLI command: `misar readability "<text>" [--json]`.

#### 4. Web Workbench v3.0 & Sovereign AI Gateway Expansion
- New Section 17: Interactive Arabic Tashkeel & Diacritization Engine with visual vowel density and confidence metrics.
- New Section 18: Extractive Summarizer & Keyphrase Extractor with sentence count selectors and keyphrase concept chips.
- New Section 19: Arabic Readability & Osman Metric Meter with educational stage indicators and reading time estimators.
- New REST API Endpoints: POST `/api/diacritize`, POST `/api/summarize`, POST `/api/readability`.
- Unified Pipeline upgrade: `diacritize=True` support in POST `/api/process` and `MisarRouter.process()`.

#### 5. Engineering Reliability & Empirical Testing
- Expanded test suite from 138 to **154 comprehensive automated unit tests** across 15 test files with 100% pass rate.
- Added `tests/test_diacritize.py`, `tests/test_summarize.py`, `tests/test_readability.py`.
- Verified 100% compatibility across Python 3.9, 3.10, 3.11, 3.12, 3.13, and 3.14.

## [2.9.0] — 2026-11-21

### Added & Improved
- **Context-Aware Arabic Grammar & Spell Correction Engine (`misar.correct` & CLI `misar correct`):**
  - High-precision orthographic normalization and lexical error correction with zero external dependencies.
  - Precise Hamza resolution on Waw, Nabara, Line, and Yaa (`مسئول` -> `مسؤول`، `شئ` -> `شيء`، `دفئ` -> `دفء`).
  - Alef Wasla vs Qat' grammatical disambiguation for Form VII, VIII, and X infinitives (`أبتكار` -> `ابتكار`، `إستخدام` -> `استخدام`، `إقتصاد` -> `اقتصاد`).
  - Taa Marbuta vs Haa disambiguation while safeguarding attached possessive and grammatical pronoun clitics (`بيته`، `قلمه`، `مديره`، `عمله`).
  - Comprehensive composite phrase corrections (`إنشاء الله` -> `إن شاء الله`، `هاكذا` -> `هكذا`، `ذالك` -> `ذلك`).
  - Full CLI command: `misar correct "<text>"`.
- **Sovereign Arabic Named Entity Recognition & Privacy Masking (`misar.ner` & CLI `misar ner`):**
  - Gazetteer and contextual pattern extraction for sovereign entities, ministries, universities, and commercial enterprises (`سدايا`، `موهبة`، `أرامكو`، `STC`، `الراجحي`).
  - Saudi geographical regions, governorates, and major Arab cities (`عسير`، `أبها`، `الرياض`، `جدة`، `مكة`).
  - Arabic composite titles and personal names (`المهندس منير الذيب`، `الدكتور خالد`).
  - High-accuracy regex extraction for temporal expressions (Hijri and Gregorian dates) and monetary currency values (`ريال`، `SAR`، `دولار`).
  - Sovereign privacy entity masking mode (`--mask`) transforming raw prompts into anonymized representations (`[ORG_محجوب]`، `[LOC_محجوب]`) for secure LLM pipelines.
  - Full CLI command: `misar ner "<text>" [--mask]`.
- **Dialect-to-MSA Parallel Translation Engine (`misar.translate` & CLI `misar translate`):**
  - Rule-based canonical translation mapping colloquial dialect phrases into Modern Standard Arabic (الفصحى المعيارية).
  - Multi-dialect coverage: Saudi / Gulf (`وش تبغى تسوي` -> `ماذا تريد أن تفعل`، `يبيله` -> `يحتاج إلى`)، Egyptian (`عايز ايه دلوقتي` -> `ماذا تريد الآن`، `مش بطيء` -> `ليس بطيئاً`)، Levantine (`شو بدك هلأ` -> `ماذا تريد الآن`، `كتير منيح` -> `جيد جداً`)، and Maghrebi (`واش كاين دابا` -> `ماذا يوجد الآن`، `مزيان بزاف` -> `جيد جداً`).
  - Automatic dialect identification and rule telemetry reporting.
  - Full CLI command: `misar translate "<text>"`.
- **Web Workbench Major Expansion (Sections 14, 15, and 16 on `localhost:8000`):**
  - Section 14: Interactive Arabic Spell Corrector with visual diff inspection and change log tracking.
  - Section 15: Arabic Named Entity Recognition (NER) with entity tag chips and one-click sovereign privacy masking preview.
  - Section 16: Dialect-to-MSA Parallel Translation Workbench with dialect badge indicator and live comparison view.
  - New REST API Endpoints: POST `/api/correct`, POST `/api/ner`, POST `/api/translate`.
- **Expanded Automated Test Suite:**
  - Added `tests/test_correct.py`, `tests/test_ner.py`, and `tests/test_translate.py`.
  - Expanded test coverage to **131 automated unit tests**, all passing with 100% success rate in under 7 seconds.

## [2.8.0] — 2026-11-21

### Added & Improved
- **High-Performance In-Memory Arabic BM25 & Semantic Search Engine (`misar.search` & CLI `misar search`):**
  - Pure Python standard-library implementation of Okapi BM25 optimized specifically for Arabic text retrieval.
  - Native Arabic morphological root and stem matching (`use_roots=True`) and diacritic-tolerant scoring.
  - Enables sovereign, zero-dependency Arabic semantic retrieval and RAG architectures without external vector database overhead.
  - Full CLI interface: `misar search --query "..." --docs "..." "..."` or `--file corpus.txt`.
- **Dialect-Aware Arabic Sentiment & Emotion Analyzer (`misar.sentiment` & CLI `misar sentiment`):**
  - Robust rule-and-lexicon sentiment engine tuned for Modern Standard Arabic and regional dialects (Saudi, Gulf, Egyptian, Levantine, Maghrebi).
  - Nuanced handling of dialectal negation inverters (`مو حلو`, `مش بطال`, `ماكو مشاكل`).
  - Native detection of idiomatic praise phrases with grammatical negation (`ما قصرت`, `ولا غلطة`, `ما في زيه`).
  - Classifies polarity score (-1.0 to +1.0), confidence, emotion categories (`حماس ورضا`, `غضب واستياء`, `محايد وموضوعي`), and token attribution.
  - CLI command: `misar sentiment "<text>"`.
- **Cross-Model Benchmark Suite Upgrade (`misar.benchmark` & CLI `misar bench --text`):**
  - Benchmarks any custom Arabic string across all 7 supported Arabic tokenizers (`AraBERTv02`, `CAMeLBERT-Mix`, `MARBERTv2`, `ARBERTv2`, `QARiB`, `ALLaM-7B`, `Llama-3-Arabic`).
  - Computes exact token reductions, $O(L^2)$ quadratic self-attention FLOPs savings, and processing latencies.
- **Interactive Web Workbench Major Upgrade (Sections 9 through 13 on `localhost:8000`):**
  - Section 9: Live Arabizi / Franco-Arabic Decoupler & Transliteration preview.
  - Section 10: Interactive Arabic Sentiment Analyzer with emotion meter and negation particle chips.
  - Section 11: Real-time Arabic Okapi BM25 Search Engine with document scoring and root matching.
  - Section 12: Robustness & Data Augmentation tester generating synthetic perturbations live.
  - Section 13: One-click export for IEEE/ACL LaTeX academic preprint (`.tex`) and Markdown research dossier (`.md`).
- **REST API Endpoints:**
  - Added POST `/api/arabizi`, POST `/api/sentiment`, POST `/api/search`, POST `/api/augment`, POST `/api/benchmark`, and GET `/api/paper`.
- **Comprehensive Automated Test Suite:**
  - Added `tests/test_search.py` and `tests/test_sentiment.py`.
  - Expanded test suite to **117 automated unit tests**, all passing with 100% success rate in under 6 seconds.

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
