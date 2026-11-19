<div align="center">

# مِسار — Misar

**Adaptive $O(n)$ Context-Aware Arabic NLP Preprocessing Router for Transformer Language Models**  
**الموجّه اللغوي التكيفي لتعزيز كفاءة ودقة معالجة اللغة العربية في نماذج الذكاء الاصطناعي**

[![PyPI](https://img.shields.io/pypi/v/misar?color=2563eb)](https://pypi.org/project/misar/)
[![CI](https://github.com/RealMtrx/misar/actions/workflows/ci.yml/badge.svg)](https://github.com/RealMtrx/misar/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-1e40af.svg)](https://www.python.org/downloads/)
[![Dependencies](https://img.shields.io/badge/dependencies-zero%20mandatory-15803d.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-0f2942.svg)](https://github.com/RealMtrx/misar/blob/main/LICENSE)
[![Latency](https://img.shields.io/badge/CPU%20latency-0.30%20ms%2Ftext-0369a1.svg)](#empirical-benchmarks)
[![Token Reduction](https://img.shields.io/badge/tokens%20saved--45.88%25-16a34a.svg)](#empirical-benchmarks)
[![Attention FLOPs](https://img.shields.io/badge/O(L%C2%B2)%20attention%20saved--70.7%25-7c3aed.svg)](#scientific-cross-model-evaluation)
[![Throughput](https://img.shields.io/badge/ETL%20throughput-3%2C778%20texts%2Fs-0d9488.svg)](#empirical-benchmarks)

[Architecture](#architecture) • [Benchmarks](#empirical-benchmarks) • [Installation](#installation) • [Quick Start](#quick-start) • [CLI & Workbench](#cli--web-workbench) • [Citation](#citation) • [النسخة العربية](#-نظرة-عامة-بالعربية)

<br/>

<img src="https://raw.githubusercontent.com/RealMtrx/misar/main/docs/misar-architecture.png" alt="Misar Adaptive Arabic NLP Preprocessing Router Architecture" width="100%" />

</div>

---

## Why Misar?

Modern Arabic NLP deployments face a structural bottleneck: forcing heterogeneous Arabic text (**Modern Standard Arabic**, **classical diacritized prose**, **Gulf/Egyptian/Levantine/Maghrebi dialects**, and **noisy social media**) through a single static preprocessing pipeline actively degrades both accuracy and inference economics:

1. **Legacy Normalization Damages Modern Transformers:** Classical regex pipelines blindly collapse `ة → ه` and `أ/إ/آ → ا`. However, modern Transformer vocabularies such as **AraBERTv2** (`64k` WordPiece) and **CAMeLBERT** explicitly preserve Hamzas and Taa Marbuta—meaning legacy normalization shatters known whole words (e.g., `الحوسبة` as `1` token splits into `الحو` + `##سبه` as `2` tokens).
2. **Dialectal Sentiment & Negation Erasure:** Over-aggressive cleaning distorts critical colloquial negation and sentiment markers (`مو`, `مب`, `مافيه`, `ماكو`), reversing downstream classification polarity.
3. **Diacritic, Tatweel & Elongation Token Explosion:** Unstripped harakat (`تشكيل`), kashida (`ـ`), and character stretches (`بطيييئة`, `الخدمةةةة`) trigger OOV `[UNK]` blindness or inflate subword token counts—multiplying Transformer $O(L^2)$ self-attention cost and API billing.

**Misar (`مِسار`)** solves these bottlenecks **without retraining any model (`Zero Parameter Update`)** by acting as an ultra-lightweight, zero-dependency, linear-time $O(n)$ routing gateway directly in front of **AraBERTv2, MARBERTv2, CAMeLBERT-Mix, ARBERTv2, QARiB**, or generative LLMs.

### How Misar v2.1 Compares to Existing Arabic Preprocessors

| Capability | `ArabertPreprocessor` | `tnkeeh` | `CAMeL Tools` | **Misar v2.1 (`مِسار`)** |
| :--- | :---: | :---: | :---: | :---: |
| **Dynamic $O(n)$ + Clause-Aware Routing** | Static pipeline | Static pipeline | Manual utilities | **Per-text & Multi-Clause `route_document()`** |
| **Zero Mandatory Dependencies** | Requires `pyarabic` / extras | Light deps | Heavy ML / data packs | **100% Pure Python Stdlib** |
| **Unicode Presentation Forms & Farsi Lookalikes** | Partial regex | Partial regex | Partial utils | **`NFKC` + Lookalike Defense (`ﺍﻟﺒﺮﻣﺠﺔ/کود → 0 [UNK]`)** |
| **Vocab-Verified Auto-Heal & Semantic Pairs** | Collapses `ة→ه` (`0%` pair disambig) | Optional regex | Separate disambig | **`100%` Minimal-Pair Disambiguation + Auto-Heal** |
| **5 Arabic Transformer Vocabs (`do_lower_case`)** | AraBERT focus | Generic | General NLP | **AraBERTv2, MARBERTv2, ARBERTv2, CAMeLBERT, QARiB** |
| **Scientific Evaluator & $O(L^2)$ Telemetry** | — | — | CLI only | **Built-in `misar eval --all-models` & Live UI** |

---

## Architecture

Every incoming text passes once through `LinguisticProfiler` (`< 0.25 ms` on CPU) to compute five quantitative signals (`dialect_ratio` $P_{\text{dial}}$, `diacritic_density` $D_{\text{diac}}$, `elongation_noise_score` $E_{\text{noise}}$, character Shannon Entropy $H_{\text{char}}$, and `code_switching_ratio` $R_{\text{latin}}$), then dispatches dynamically to the optimal micro-pipeline:

<div align="center">
  <img src="https://raw.githubusercontent.com/RealMtrx/misar/main/docs/misar-architecture-flow.png" alt="Misar Dynamic Architecture Flow" width="100%" />
</div>

---

## Empirical Benchmarks

Measured reproducibly via `python benchmarks/run_benchmark.py` against the authentic **64,000-token `aubmindlab/bert-base-arabertv02` WordPiece vocabulary** across 15 curated multi-domain samples ([`latest_benchmark_report.json`](https://github.com/RealMtrx/misar/blob/main/benchmarks/latest_benchmark_report.json)):

| Domain Category | Samples | Raw Tokens | Fair Legacy Static (`ة→ه`, `أ→ا`) | **Misar v2.1** | Misar Gain |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MSA Engineering & Code-Switch** | 3 | 57 | 79 *(+38.6% fragmentation)* | **57** | Preserves `100%` orthography (**`-27.85%`** vs Legacy) |
| **Classical Diacritized (Heritage)** | 3 | 186 *(147 OOV `[UNK]`)* | 48 | **39** | **`-79.03%`** vs Raw · **`-18.75%`** vs Legacy (`0 [UNK]`) |
| **Saudi / Gulf / Levantine / Maghrebi Dev** | 6 | 132 | 136 | **120** | **`-9.09%`** vs Raw · **`-11.76%`** vs Legacy (+ Auto-Heal & Negations) |
| **Noisy Tech Social & Code-Switching** | 3 | 98 | 48 | **40** | **`-59.18%`** vs Raw · **`-16.67%`** vs Legacy + gemination intact |
| **Full Corpus Summary** | **15** | **473** | **311** | **256** | **`-45.88%` vs Raw · `-17.68%` vs Fair Legacy · `3,778 texts/s`** |

> **Key Scientific Proof:** Even when competing against an identical full-denoising Legacy Static baseline (`URL/Emoji/Hashtag/Stretch removal + ة→ه, أ→ا`), Misar v2.1 still saves **17.68%** fewer tokens than Legacy Static (**45.88%** vs Raw), eliminates **`159 → 0 [UNK]`** tokens, repairs broken orthography (`Auto-Heal`), preserves dialectal negations, and streams at **3,778 texts/sec** on CPU.

### Scientific Cross-Model Evaluation (`misar eval --all-models`)

Measured live across all 5 authentic Hugging Face Arabic Transformer vocabularies (`evaluation.py`):

| Model Key | Vocabulary Backend | Semantic Minimal-Pair Disambig (`Misar` vs `Legacy`) | Dialect Negation Safety | Emoji Polarity Retention | Tokens vs Raw / vs Legacy | Quadratic $O(L^2)$ Attention Saved |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`arabertv02`** | `aubmindlab/bert-base-arabertv02` | **`100%` vs `0%`** | **`100%`** | **`100%`** | **`-45.9%` / `-17.7%`** | **`-70.7%`** |
| **`arbertv2`** | `UBC-NLP/ARBERTv2` | **`100%` vs `0%`** | **`100%`** | **`100%`** | **`-47.0%` / `-15.1%`** | **`-71.9%`** |
| **`camelbert-mix`** | `CAMeL-Lab/bert-base-arabic-camelbert-mix` | **`100%` vs `0%`** | **`100%`** | **`100%`** | **`-42.3%` / `-11.0%`** | **`-66.7%`** |
| **`marbertv2`** | `UBC-NLP/MARBERTv2` | **`100%` vs `0%`** | **`100%`** | **`100%`** | **`-46.1%` / `-8.7%`** | **`-71.0%`** |
| **`qarib`** | `qarib/bert-base-qarib` | **`100%` vs `0%`** | **`100%`** | **`100%`** | **`-45.1%` / `-12.5%`** | **`-69.8%`** |

---

## Installation

Core `misar` requires **zero mandatory external packages** (Python `3.9+` Standard Library only):

```bash
# Install directly from GitHub
pip install git+https://github.com/RealMtrx/misar.git

# Or from PyPI once published
pip install misar
```

Optional extras for live Hugging Face Transformers wrapping and package development:

```bash
pip install "misar[huggingface]"   # transformers + torch
pip install "misar[dev]"           # pytest + build + twine
```

*(Note: On first use of the built-in `WordPieceTokenizer`, Misar caches the official 64k vocabulary file locally under `~/.cache/misar/`. Pass `allow_vocab_download=False` or `--offline` to run completely air-gapped using `HeuristicSubwordTokenizer`.)*

---

## Quick Start

### 1. Dynamic Text Routing & Token Telemetry

```python
from misar import MisarRouter

router = MisarRouter()  # Uses real AraBERTv2 64k WordPiece vocabulary

result = router.route("استجابة الخادم بطيييئة مرررة ومافيه أي اتصال في المنظومة أبدددداً 😡😡👎 #انقطاع_الخدمة")

print("Route:     ", result.route_name)          # NOISY_SOCIAL
print("Processed: ", result.processed_text)      # استجابة الخادم بطيئة مرة ومافيه أي اتصال في المنظومة أبدا انقطاع الخدمة
print("Tokens:    ", result.raw_token_count, "->", result.processed_token_count)  # 22 -> 13
print("Reduction: ", f"-{result.token_reduction_pct:.1f}%")                       # -40.9%
print("Latency:   ", f"{result.latency_ms:.3f} ms")
```

### 2. Clause-Aware Multi-Route Documents, Unicode PDF Defense & Batching

```python
from misar import MisarRouter, preprocess, preprocess_batch

router = MisarRouter(model_key="arabertv02", emoji_mode="sentiment", auto_heal=True)

# Recovers PDF Arabic Presentation Forms + Persian keyboard lookalikes (کود -> كود)
# AND routes formal vs. social sentences independently in multi-clause documents:
doc_res = router.route_document(
    "ﺍﻟﺒﺮﻣﺠﺔ ركيزة أساسية في بناء الأنظمة الموثوقة باستخدام کود نظيف ومعياري.\n"
    "لكن استجابة الخدمةةةة الحين تحتاج متابعة عاجلة، ومستعدين لإطلاق التحديث قريباً إن شاء الله ✅🚀"
)
print("Distribution:", doc_res.route_distribution)      # {'FAST_MSA': 1, 'DIALECT_PRESERVE': 0, 'NOISY_SOCIAL': 1}
print("Attention Saved:", f"-{doc_res.attention_savings_pct:.1f}%")

# Fast streaming batch preprocessing & Hugging Face Datasets .map(..., batched=True)
cleaned_list = preprocess_batch(["نص أول", "نص ثاااني 😡"])
# dataset = dataset.map(router.hf_dataset_map, batched=True)
```

### 3. One-Line Hugging Face Transformers Integration

```python
from transformers import pipeline
from misar import MisarRouter

sentiment = pipeline("sentiment-analysis", model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment")
smart_sentiment = MisarRouter(tokenizer=sentiment.tokenizer).wrap_hf_pipeline(sentiment)

print(smart_sentiment("بنية النظام مو معقدة أبداً ولكن استجابة الخادم بطيييئة #مشكلة_تقنية"))
```

---

## CLI & Web Workbench

Misar includes a clean command-line interface and a self-hosted HTTP server + Arabic RTL Interactive Workbench:

```bash
# Run 6 built-in cross-dialect samples with live AraBERTv2 tokenization
misar demo

# Route a sentence and compare Raw vs Legacy Static vs Misar side-by-side
misar route "توثيق البرمجه في المنصه هاذي دقيق وموثوق وماكو أي تعقيد في تكامل الـ Backend" --compare

# Select target model vocab (arabertv02, camelbert-mix, marbertv2, arbertv2, qarib)
misar route "أعلنت منصة المطورين عن ترقية واجهة البرمجة API" --model marbertv2 --json

# Run the reproducible 15-sample benchmark suite & 5-model scientific evaluation
misar bench
misar eval --all-models

# Launch the interactive RTL Web Workbench & hardened REST API on http://127.0.0.1:8050
misar serve
```

### Embedded REST API Endpoints (`misar serve`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Service status, version, and active tokenizer backend |
| `GET` | `/api/suite` | Evaluates the 6 standard cross-domain reference samples |
| `POST` | `/api/route` | Body `{"text": "..."}` → Full `RoutingResult` (supports `document_mode: true`) |
| `POST` | `/api/document` | Body `{"text": "..."}` → Clause-aware `DocumentRoutingResult` + $O(L^2)$ metrics |
| `POST` | `/api/batch` | Body `{"texts": ["...", "..."]}` → Batch routes up to 500 strings (`1 MiB` payload limit) |

---

## Repository Structure

```text
misar/
├── .github/workflows/ci.yml           # Cross-OS Python 3.9–3.13 CI pipeline
├── docs/
│   ├── misar-architecture.png        # High-resolution system architecture & benchmark banner
│   └── misar-architecture-flow.png   # Responsive mobile-optimized dynamic routing flow
├── benchmarks/
│   ├── run_benchmark.py               # Reproducible benchmark runner
│   └── latest_benchmark_report.json   # Verified telemetry snapshot
├── src/misar/
│   ├── __init__.py                    # Public API exports
│   ├── profiler.py                    # O(n) 5-signal LinguisticProfiler & Shannon entropy
│   ├── lexicons.py                    # MSA-safe multi-region dialect & negation lexicons
│   ├── pipelines.py                   # FAST_MSA, DIALECT_PRESERVE & NOISY_SOCIAL passes
│   ├── tokenizer.py                   # Pure-Python WordPieceTokenizer + HF & offline adapters
│   ├── router.py                      # MisarRouter orchestration & wrap_hf_pipeline
│   ├── evaluation.py                  # Multi-model scientific evaluation suite (disambig/polarity/O(L^2))
│   ├── benchmark.py                   # Shared benchmark suite module
│   ├── cli.py                         # `misar` CLI (`demo`, `route`, `bench`, `eval`, `serve`)
│   └── web/
│       ├── server.py                  # Hardened zero-dependency ThreadingHTTPServer REST API
│       └── index.html                 # Interactive Arabic RTL Workbench & token visualizer
├── tests/test_misar.py                # 39 unit, v2.1 feature, server & real-vocab tests
├── CITATION.cff                       # Native GitHub repository citation metadata
├── CONTRIBUTING.md                    # Developer setup & contribution standards
├── SECURITY.md                        # Built-in API & frontend hardening controls
├── CHANGELOG.md                       # Formal release history
├── LICENSE                            # MIT License
└── pyproject.toml                     # PEP 621 PyPI package specification
```

---

## Testing & Verification

```bash
python -m unittest discover -s tests -v
```

All **39 automated tests** verify profiler mathematics, Unicode Presentation Form & Farsi lookalike recovery (`ﺍﻟﺒﺮﻣﺠﺔ/کود → 0 [UNK]`), long-essay social artifact sanitization, Clause-Aware `route_document()` multi-clause dispatch, MSA homograph resistance, Saudi/Gulf/Levantine/Maghrebi negation preservation, gemination protection (`والللله → والله`), pronoun-safe Vocab Auto-Heal (`البرمجه → البرمجة` while protecting `بيته/ملفه/مديره`), clitic code-switching (`الـ Backend → الBackend`), `do_lower_case` parity for `MARBERTv2`/`ARBERTv2`, opt-in Arabizi tech safety, scientific minimal-pair evaluation (`100% vs 0%`), fast batch/HF dataset mapping, and hardened REST server HTTP security bounds.

---

## Citation

If you use **Misar** in academic research, benchmarks, or commercial NLP pipelines, please cite:

```bibtex
@software{altheeb2026misar,
  author       = {Muneer Jarab Matrak Al-Theeb},
  title        = {Misar (مِسار): Adaptive Context-Aware Arabic NLP Preprocessing Router for Transformer Language Models},
  year         = {2026},
  version      = {2.1.0},
  url          = {https://github.com/RealMtrx/misar},
  license      = {MIT}
}
```

---

## 📖 نظرة عامة بالعربية

**مِسار (`Misar v2.1.0`)** هي مكتبة وموجّه بايثون مفتوح المصدر وخفيف الوزن (بلا أي تبعيات خارجية إجبارية) صُمِّم ليعمل كطبقة بوابة ذكية في زمن خطي $O(n)$ (**~0.30 مللي ثانية** للنص وبسرعة **3,778 نص/ثانية** في المعالجة الدفعية) قبل نماذج المحولات العربية الخمسة (`AraBERTv2`, `MARBERTv2`, `CAMeLBERT`, `ARBERTv2`, `QARiB`) والنماذج التوليدية (`ALLaM` و `LLMs`) **دون الحاجة إلى إعادة تدريب النموذج (`Zero Parameter Update`)**.

* **توجيه سياقي متعدد الجمل (`route_document`) + دفاع Unicode:** يختار ديناميكياً على مستوى المستند أو كل فقرة وجملة بين `FAST_MSA` و`DIALECT_PRESERVE` و`NOISY_SOCIAL`، يستعيد أشكال العرض المعزولة لملفات PDF (`NFKC`: `ﺍﻟﺒﺮﻣﺠﺔ ← البرمجة`) ومحارف لوحات المفاتيح الفارسية/الأردية (`ک/ی ← ك/ي`)، ويرمّم أخطاء التاء المربوطة (`البرمجه ← البرمجة`) دون المساس بالضمائر (`بيته، ملفه، مديره`).
* **توفير التوكنات وتكاليف الانتباه التربيعي $O(L^2)$:** أظهرت التجارب الموثقة (`misar eval --all-models`) تفوق **100% مقابل 0%** في تمييز الثنائيات الصرفية (`عالمة/عالمه`)، وحفظ **100%** من النفي اللهجي وقطبية المشاعر، وخفض التوكنات بنسبة **45.88%** مقابل الخام و**17.68%** مقابل المناهج التقليدية (ما يوفّر **66.7%–71.9%** من عمليات الانتباه $O(L^2)$ واستعادة `159 → 0 [UNK]`).

---

<div align="center">

Authored and maintained by **Muneer Jarab Matrak Al-Theeb (منير جراب مترك الذيب)**  
Released under the [MIT License](https://github.com/RealMtrx/misar/blob/main/LICENSE) © 2026

</div>
