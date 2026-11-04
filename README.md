# AI-Based Drug Safety and Adverse Effect Analysis System

An end-to-end pharmacovigilance workbench built on the **FDA Adverse Event Reporting System (FAERS) 2025 Q1** extract:

* a reproducible **data pipeline** (ingestion → cleaning → case-version de-duplication → probable-duplicate detection → quality checks) stored as **Parquet** and queried with **DuckDB**;
* **statistical signal detection** (ROR, PRR, IC/BCPNN, chi-square, Benjamini–Hochberg q-values, Haldane–Anscombe sparse-cell correction, indication-bias flag, subgroup analysis);
* an **NLP module** built on a locally-run pretrained biomedical language model, **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`), that provisionally categorises the 12,678 reaction terms, abstains when uncertain, powers *search by meaning* and a *concept-to-signals* explorer, and is evaluated against an independent reference (NCI CTCAE v5.0) with baselines;
* an evidence-grounded **question-answering module (RAG)**: ask "Does Ozempic cause stomach problems?" and it links the drug (incl. brand names learnt from FAERS) and the reaction (exact MedDRA match or SapBERT), retrieves the stored statistics plus FDA FAERS documentation and CTCAE definitions (BM25), and answers only from that evidence with citations — or abstains;
* **external clinical evidence for the assistant**: FDA drug labels (openFDA SPL sections, label set chosen via DailyMed), FDA Drug Safety Communications and PubMed reviews are downloaded with full source metadata, chunked by label section, embedded with **BGE-small** and searched by **vector retrieval**; answers separate *what FAERS reports show* from *what labels and literature state*, with clickable citations, a label-mention check and an explicit "insufficient evidence" outcome. An optional local generative model (**Qwen2.5-0.5B-Instruct**) can write that part, shown only when every sentence is supported by the passage it cites;
* a **FastAPI** backend and a **React + TypeScript + Tailwind + shadcn/ui + Plotly.js** dashboard.

> FAERS is a spontaneous reporting system. Every count is a count of *reports*. Disproportionality measures show that a drug–reaction pair is reported together more often than expected; they do **not** establish causality, incidence or absolute risk. Model-generated categories and answers are **provisional** and are never presented as validated clinical findings.

---

## Quick start (Windows)

Requirements: **Python 3.10+** (3.11+ recommended) and **Node.js 18+** on PATH. No GPU, no paid API.

```powershell
# from the project folder
.\run.cmd                      # or: powershell -ExecutionPolicy Bypass -File scripts\run.ps1
```

The launcher creates `.venv`, installs dependencies (CPU-only PyTorch for the language model), builds the processed data if it is missing, starts the API on `http://127.0.0.1:8000` and the dashboard on `http://localhost:5173`, and opens the browser. **Press Enter in the window to stop both processes**; `scripts\run.ps1 -Stop` also stops anything left running.

| Option | Effect |
|---|---|
| `-Rebuild` | Re-run the whole pipeline from the raw zip |
| `-NoNlp` | Skip installing torch/transformers (dashboard uses precomputed predictions + lexical search) |
| `-Test` | Run pytest, Ruff, Vitest and the TypeScript check before starting |
| `-NoBrowser`, `-ApiPort`, `-WebPort` | Self-explanatory |

Logs are written to `logs\`. API documentation is at `http://127.0.0.1:8000/docs`.

### Manual commands (any OS)

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r backend/requirements-nlp.txt   # optional
python scripts/download_model.py                       # optional, pinned SapBERT revision (~440 MB)

cd backend
python -m pvsys.pipeline download                      # FAERS 2025 Q1 zip from fis.fda.gov (if not present)
python -m pvsys.pipeline all                           # ingest -> clean -> quality -> signals -> EDA -> NLP
uvicorn api.main:app --port 8000

cd ../frontend && npm ci && npm run dev                # http://localhost:5173
```

Other pipeline commands: `stats` (statistics only, no ML dependencies), `nlp` (categorisation + question-answering index + its evaluation), `evaluate`, `rag-index`, `rag-eval`, `annotate-export --n 200`, `annotate-import <csv>`.

---

## Results on FAERS 2025 Q1 (actual outputs)

| Stage | Result |
|---|---|
| Raw DEMO reports | 400,514 (DRUG 2,008,162 · REAC 1,432,926 · OUTC 304,027 rows; 0 rejected lines) |
| FDA-deleted cases removed | 1 (5,732 IDs on the quarter's delete list; 1 present in this file) |
| Older case versions removed | 0 – the quarterly file already holds one version per CASEID (rule still enforced) |
| De-duplicated cases | **400,513** |
| Probable duplicates (same clinical fingerprint, different CASEID) | 7,784 – excluded from signal detection |
| Analysis population | **392,729** cases with ≥1 suspect drug and ≥1 reaction |
| Serious / death | 224,700 (56.1%) / 32,408 (8.1%) |
| Distinct suspect drugs / reaction PTs | 5,018 / 12,678 |
| Drug–reaction pairs observed / tested (a ≥ 3) | 648,090 / 241,675 |
| Flagged by ROR / PRR / IC / all three | 192,163 / 184,795 / 173,684 / 172,074 |
| Rows needing Haldane correction | 4,201 |
| Possible indication bias (≥20% overlap) | 16,092 |

Sanity check against well-known label associations (all flagged by all three methods): metformin–lactic acidosis ROR 271.5 (95% CI 226.7–325.1), semaglutide–pancreatitis 12.5 (9.8–15.9), levofloxacin–tendon rupture 18.1 (7.4–44.1, a = 5), montelukast–nightmare 18.0 (10.1–32.1), isotretinoin–depression 6.1 (4.5–8.2), adalimumab–tuberculosis 6.3 (3.9–10.0). The indication-bias flag correctly marks dupilumab–atopic dermatitis (98% of cases list it as the indication).

Key data-quality findings: age missing in 39.2% of cases, weight 82.8%, full event date 56.9%, sex 20.7%.

### NLP module evaluation (held-out test split, NCI CTCAE v5.0 reference)

712 CTCAE terms with an unambiguous organ-system heading (99 removed because they equal an anchor phrase); 30% dev split (213 terms) used only to choose the scoring rule and abstention thresholds; numbers below on the untouched **499-term test split**.

| System | Top-1 accuracy (95% CI) | Coverage with abstention | Accuracy when answering (95% CI) | Macro-F1 (answered) |
|---|---|---|---|---|
| **SapBERT zero-shot prototypes** | **62.7%** (58.5–66.7) | 65.3% | **71.5%** (66.9–76.4) | 0.620 |
| Lexical TF-IDF char n-gram baseline | 43.1% (38.9–47.3) | 72.7% | 53.7% (48.5–59.2) | 0.458 |
| Majority class (gastrointestinal) | 15.8% | – | – | – |

On the 233 test terms that also occur in FAERS 2025 Q1: SapBERT top-1 67.8%, lexical 44.6%.
On all 12,678 FAERS terms the model assigned 7,333 and abstained on 5,345 (3,732 low similarity, 1,613 ambiguous); assigned terms cover 85.5% of case–reaction mentions. **No accuracy is claimed on FAERS terms themselves** until blinded human annotations are collected (workflow below).

### Question-answering (RAG) evaluation — independent references, `python -m pvsys.pipeline rag-eval`

| Test | Reference (not written by us) | System used | Result | Baselines |
|---|---|---|---|---|
| A. Find the answer passage | FDA FAERS FAQ: 52 questions → FDA's own answer, among 131 passages (question text removed) | BM25 | top-1 **44.2%** (30.8–57.7), top-10 80.8%, MRR 0.559 | word TF-IDF 53.8% / 82.7%; random 0.8% / 7.6% |
| B. Link words to the MedDRA term | NCI CTCAE v5.0 definitions → that term among all 12,678 FAERS PTs (299-term test split; variant chosen on 128-term dev split) | SapBERT | top-1 **47.8%** (42.1–53.5), top-10 85.6%, MRR 0.607 | char TF-IDF 11.7% / 27.1%; word BM25 12.4% / 29.1% |
| C. End to end | 150 random tested (drug, PT) pairs × 5 phrasings | full engine | **99.3%** (98.7–99.9) return exactly the stored statistics; brand aliases 55/55; 117 ms/question | – |

* All 805 evaluation answers passed the automatic grounding check (every number, citation and quotation traceable to evidence).
* Test C found three parsing bugs (reaction words treated as filler, punctuated drug names, drug look-up inside reaction terms); fixing them raised it from 78.0% → 99.3%. The 5 remaining failures are PLACEBO, deliberately excluded.
* On test A, TF-IDF beat BM25 (paired bootstrap +9.6 points top-1, 95% CI 1.9–19.2). BM25 was fixed before the test; the switch is left for a held-out question set rather than tuned on the only benchmark.
* At the 0.70 linking threshold SapBERT accepts 74.9% of (long, formal) test definitions with 58.5% precision; short lay phrases score higher (e.g. "suicidal thoughts" → Suicidal ideation 0.95).
* The optional generative summariser is **not evaluated** (no weights could be downloaded here).

---

## Architecture

```
data/raw/faers_ascii_2025q1.zip
        │  pvsys.ingest   (zip-slip safe extract, '$'-delimited -> Parquet as text, manifest + SHA-256)
        ▼
data/processed/raw/*.parquet
        │  pvsys.clean    (deletions, latest version per CASEID, unit conversion, drug-name normalisation,
        │                  probable-duplicate fingerprinting)
        ▼
data/processed/{cases,drugs,reactions,outcomes,indications,therapy,report_sources}.parquet
        │  pvsys.quality  -> quality.json         pvsys.eda -> eda_snapshot.json
        │  pvsys.signals  -> signals.parquet, signals_summary.json
        ▼
        ├────────────── statistical outputs never depend on anything below ──────────────
        │  pvsys.nlp (optional; reads processed Parquet, never writes it)
        │     encoder.py   SapBERT (CPU) | TF-IDF char n-grams (baseline / fallback)
        │     classify.py  centroid prototypes, softmax confidence, abstention, dev-split calibration
        │     evaluate.py  CTCAE reference evaluation + blinded-annotation evaluation
        │     review.py    append-only review log + blinded annotation export/import
        │     search.py    semantic retrieval over the reaction vocabulary
        │     kb.py        knowledge base (FDA README/FAQ/ASC_NTS from the zip, CTCAE definitions, method notes) + BM25
        │     rag.py       question answering: drug/reaction linking -> retrieval -> cited answer / abstention
        │     generator.py optional local generative SLM + evidence verifier
        │     rag_eval.py  FAQ retrieval, CTCAE linking and end-to-end evaluation
        ▼
data/nlp/{predictions.parquet, embeddings.npy, provenance.json, evaluation_*.json, kb_chunks.parquet, kb_manifest.json}
        │
backend/api/main.py  (FastAPI, DuckDB views over Parquet, validation, caching, graceful degradation)
        │  /api/*
frontend/            (Vite, React 18, TypeScript, Tailwind, shadcn/ui primitives, Plotly.js, TanStack Query)
```

Dashboard pages: Overview (case flow), Data quality, Demographics (subgroups + χ² test), Drug explorer, Signal detection (filters, forest plot, 2×2 detail, subgroup ROR), **Ask the data** (question box, cited answer, how the question was read, evidence with links into Signal detection, provenance, evaluation tab), Reaction categories (categories, term predictions with review, search by meaning, signals for a concept), Model & evaluation (provenance, metrics vs baselines, risk–coverage, confusion matrix, review status), Methods & limits. All model output is drawn with a dashed violet outline and a "Provisional" badge.

## The NLP module in detail

* **Model:** SapBERT-from-PubMedBERT-fulltext, BERT-base (109.5 M parameters), **Apache-2.0**, revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`, weights SHA-256 `a4696930…5614`. CLS-token embeddings, L2-normalised, max 25 tokens, CPU inference (≈2 minutes for all 12,678 terms).
* **Taxonomy:** 23 project-defined body-system categories (`pv-cat-1.0`), each defined by a description and 6–14 anchor phrases written before the evaluation reference was opened. **These are not MedDRA SOCs.**
* **Classifier:** cosine similarity to each category's centroid; the centroid rule was chosen over nearest-anchor on the dev split (66.2% vs 61.0%). Abstain if similarity < 0.46 or margin over the runner-up < 0.02 (thresholds tuned on dev to maximise accuracy at ≥70% coverage). Softmax "confidence" is a ranking aid, not a calibrated probability.
* **Provenance:** every prediction row carries `model_id`, `model_revision`, `taxonomy_version`, `run_id`, scores, margin and the lexical-baseline prediction; `provenance.json` adds weights hash, library versions, thresholds and the source data checksum.
* **Review status:** the dashboard's Accept / Correct / Reject / Unsure actions append to `data/reviews/prediction_reviews.csv` (latest entry wins). These *unblinded* reviews change a prediction's status but are **never used for accuracy**.
* **Blinded annotation workflow (ground truth):** `annotate-export` writes a stratified sample **without model output** plus `LABEL_GUIDE.md`; two annotators label independently; `annotate-import` validates and appends to `data/reviews/annotations.csv`; `evaluate` reports accuracy, macro-F1, coverage and Cohen's κ. Until that happens the dashboard states that no FAERS accuracy is claimed.
* **Graceful degradation:** without torch/transformers or weights the pipeline writes lexical-baseline predictions (`prediction_source = lexical-fallback`), the API serves precomputed predictions and search reports `mode: lexical-fallback`. Statistics are unaffected (enforced by tests).

## External clinical evidence (FDA labels, FDA safety communications, PubMed)

Build once (needs internet for the download step; uses an NVIDIA GPU if present, else the CPU):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_evidence.ps1          # -Drugs 150 -Questions 40 -SkipFetch -Cpu
```

All downloads and caches (pip, PyTorch, Hugging Face, temp) go to `.cache\` inside the project. Local models (download on Windows, Hugging Face is the source):

```powershell
.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-0.5B-Instruct', local_dir='models/generator'); d('BAAI/bge-small-en-v1.5', local_dir='models/bge-small-en-v1.5')"
```

| Stage | What happens | Output |
|---|---|---|
| Ingest (`pvsys/evidence/sources.py`) | For the 150 most-reported FAERS drugs: openFDA label sections (Boxed Warning, Contraindications, Warnings and Precautions, Adverse Reactions, Drug Interactions, Pregnancy, Overdosage, …) of the manufacturer's most-revised label set per DailyMed; up to 4 PubMed reviews/meta-analyses since 2015 (E-utilities); FDA Drug Safety Communications from fda.gov. Every raw response cached; rate-limited; per-item failures recorded | `data/evidence/raw/`, `documents.parquet`, `sources_manifest.json` |
| Chunk (`chunking.py`) | Label text split *within* SPL sections (160 words, 30 overlap), abstracts kept whole, FDA communications tagged with the drugs they mention | `chunks.parquet` |
| Embed + index (`embed.py`, `index.py`) | BGE-small-en-v1.5 (33 M params, MIT, rev `5c38ec7`), CLS pooling, query instruction; exact cosine search in NumPy; BM25 and reciprocal-rank-fusion hybrid also available | `embeddings.npy`, `index_manifest.json` (model, revision, weights hash, content hash) |
| Answer (`nlp/rag.py`) | Linked drug → label-mention check for the linked MedDRA term (+ SapBERT-related terms) → vector retrieval filtered to the drug → quotes with source details, or the optional Qwen answer if it passes the checks; abstains below the calibrated similarity threshold or when the drug is not indexed | `established` block of `/api/rag/ask` |
| Evaluate (`evidence/evaluate.py`) | See below | `evaluation_evidence.json`, `calibration.json` |

**Built on the RTX 3050 laptop (2026-10-20):** 150 drugs requested → 135 labels (13 drugs, mostly vitamins/combination products, have no openFDA label), 424 PubMed reviews, 42 FDA safety communications = 601 documents, 8,590 passages (1,180 passages of very long label sections truncated at 25 per section); embedding 110 s on the GPU (78 passages/s); 0 download failures.

**Evaluation – no hand-written labels.** Drugs split 30% dev (38) / 70% test (89), seed 42.

| Test (89 test drugs, 886 questions; gold = FDA's own SPL section coding) | Dense vectors (used) | Hybrid (RRF) | BM25 |
|---|---|---|---|
| Right label section ranked first – all questions | **81.2%** | 67.2% | 51.0% |
| … in top 5 (95% CI) | **94.6%** (93.0–95.9) | 85.6% | 70.5% |
| … lay wording only ("Who should not take Ozempic?"), top 1 / top 5 | **67.7% / 90.5%** | 39.5% / 71.6% | 18.5% / 43.3% |
| … label wording only, top 1 | 94.6% | 94.8% | 83.5% |
| MRR / nDCG@10 – all questions | **0.871 / 0.895** | 0.750 / 0.791 | 0.595 / 0.643 |

* **Retriever choice made on dev drugs only:** dev MRR dense 0.850, hybrid 0.752, BM25 0.588 → dense is the production default. BM25 fails on lay questions because they share no words with label text; fusing it in drags dense down.
* **Insufficient-evidence threshold:** 0.73 cosine, chosen on dev; on test drugs balanced accuracy **92.5%** – 93.3% of answerable questions answered, 91.6% of unanswerable ones (price, packaging colour, unrelated topics) declined.
* **Generated answers (Qwen2.5-0.5B-Instruct, GPU, 40 test drugs):** "Does X have a boxed warning?" correct **35%** (95% CI 20–50%) vs **97.5%** for the extractive baseline (does a retrieved passage come from the Boxed Warning section?); only **42.5%** of drafts passed the source checks (rejections: unsupported sentence 21, missing citation 18, misquote 2, invented id 1). Median 0.9 s per answer. **Consequence: generation is off by default**; answers are composed from the label check and verbatim quotes, and the model is an opt-in experiment. A larger model (e.g. Qwen2.5-1.5B-Instruct) can be dropped into `models/generator`.
* **Signal–label concordance (descriptive):** of the 300 most-reported strong FAERS signals (all three methods, ≥ 10 cases, no indication bias) the reaction term occurs in the label for **33.7%** (42.8% excluding product-use terms, e.g. dupilumab–pruritus/rash, relugolix–hot flush); of the 300 most *disproportionate* only 1.7% – extreme ratios are dominated by unusual report clusters. Exact text matching undercounts synonyms.

### Extractive vs generative answering, and the accuracy improvements (paper: `docs/paper/`)

**Comparison (89 test drugs, identical top-3 passages for every system; `scripts\compare_generators.ps1`, `generator_comparison.json`).** "Does X have a boxed warning?": extractive **92.1%** vs Qwen2.5-0.5B 43.8%, Qwen2.5-1.5B 69.7%, SmolLM2-1.7B 49.4% (all p < 10⁻⁴, exact McNemar). The v1 verifier (citations, numbers, quotes, word-overlap support) passed about half the drafts, but it also passed confident polarity errors (Qwen2.5-1.5B: 16 verified "no boxed warning" answers for drugs that have one), so showing verified generations *lowered* system accuracy to 74–76% for the Qwen models.

**Improvements (`scripts\overnight.ps1`, 2026-11-04; all choices made on the 38 dev drugs; results on the 89 test drugs).**

| Verifier on the *same* drafts (`generator_reverify.json`) | v1 | + structure check | + entailment check | + both |
|---|---|---|---|---|
| Qwen2.5-1.5B – system accuracy (boxed warning) | 76.4% | 92.1% | **93.3%** | 92.1% |
| Qwen2.5-1.5B – verified false "no" / false "yes" | 16 / 3 | 2 / 0 | **0** / 2 | 0 / 0 |
| SmolLM2-1.7B – system accuracy | 87.6% | 92.1% | 87.6% | 92.1% |
| SmolLM2-1.7B – verified false "no" / false "yes" | 0 / 7 | 0 / 0 | 0 / 7 | 0 / 0 |
| Extractive answer (reference) | 92.1% | 92.1% | 92.1% | 92.1% |

* **Structure check** (`nlp/checks.py`, deterministic): a draft may not deny a boxed warning when a retrieved passage is from FDA's Boxed Warning section, nor assert one when none is. **Entailment check**: `cross-encoder/nli-deberta-v3-small` (Apache-2.0, rev `fa28048`, CPU) rejects a sentence its cited passage contradicts (P(contradiction) ≥ 0.5 and > P(entailment), fixed in advance). The two are complementary: entailment removed every false "no", structure every false "yes". Both are now part of the app's verifier.
* **Retrieval** (`evaluation_evidence_v2.json`): section routing (`evidence/rerank.py` – "who should not take" → Contraindications, …; bonus weight chosen on dev) is now the default (`dense+route`). On 1,329 *re-worded* test questions top-1 rises from **54.4% to 66.9%** (MRR 0.644 → 0.733); routing covers 32% of them and misroutes 7%. On the original templates it scores 100% – by construction, because the same authors wrote both, so this is not claimed as an accuracy gain.
* **Cross-encoder re-ranking** (`BAAI/bge-reranker-base`, MIT, rev `2cfc18c`) made retrieval **worse** (top-1 65.1% → 52.0%; label wording 94.6% → 64.6%) and was rejected on dev drugs; the code keeps it as an option.
* **Larger model** – Qwen2.5-3B-Instruct (Qwen Research Licence, non-commercial; 4-bit NF4, 2.2 GB GPU, 17.5 tokens/s), with routing so the right passage is retrieved for every drug: 78.7% on the boxed-warning question vs 79.8% for Qwen2.5-1.5B and 100% for the extractive answer; v1 verifier still passed 4 false "no" and 4 false "yes" (`generator_comparison_v2.json`). Scale did not remove the polarity errors; the new verifier did (system accuracy 100% with both checks, because every disagreement with the label structure is rejected).

## The question-answering (RAG) module in detail

1. **Understanding.** Drugs: longest-first, punctuation-insensitive lookup over 5,015 normalised suspect drugs and 2,079 brand/verbatim aliases learnt from FAERS (a dose-stripped verbatim name that maps to one ingredient in ≥ 90% of ≥ 5 rows, e.g. OZEMPIC → SEMAGLUTIDE); words shared by several products ("insulin") are reported as ambiguous, typos matched at difflib ratio ≥ 0.88. Reactions: exact MedDRA PT first (a longer PT beats a drug word inside it), else SapBERT; a link is accepted only at cosine ≥ 0.70 (lexical fallback 0.45), otherwise reported as not understood. Intent: pair / drug / reaction / documentation / clarification.
2. **Retrieval.** Stored ROR/PRR/IC rows (and up to 5 SapBERT-related terms) from DuckDB, or the raw co-report count for untested pairs; BM25 (k1 1.5, b 0.75) over 942 passages; the CTCAE definition of the linked term; the causality notes.
3. **Generation.** Deterministic: each sentence is filled from retrieved values and cites evidence ids (`S1` statistics, `D1` documents); documentation answers quote FDA text verbatim; the causality caveat is always attached; abstains when nothing relevant is found.
4. **Optional generator.** Put any Hugging Face causal LM folder in `models/generator/` (or set `PV_GENERATOR_DIR`), e.g. `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, ~1 GB, CPU), add a `provenance.json` with `model_id`/`revision`, and tick the option in the dashboard. Its summary is shown only if the verifier finds every number, citation and quotation in the evidence and no unnegated causal/incidence claim; otherwise it is discarded with reasons. `PV_GENERATOR=off` disables it.
5. **Provenance & robustness.** Each answer reports the linker model and revision, threshold, KB version and content hash, NLP run id and data checksum. Answers are LRU-cached; the engine warms up in a background thread at API start (`PV_WARMUP=0` to skip); without SapBERT it falls back to lexical linking and says so; without the FAERS zip or `pypdf` the KB still builds from project notes + CTCAE.

API: `POST /api/rag/ask {"question": "...", "generate": false}`, `GET /api/rag/status`, `GET /api/rag/evaluation`.

## Licensing and data-use notes

* **FAERS/AEMS data:** public US-government data from fis.fda.gov; no licence fee. FDA's disclaimers apply (reports are unverified, causality not established).
* **SapBERT weights:** Apache-2.0 (Cambridge LTL). PubMedBERT base: MIT (Microsoft). Cite Liu et al., NAACL 2021.
* **MedDRA®:** FAERS reaction terms are MedDRA Preferred Terms, which the FDA distributes in FAERS. This project does **not** use or redistribute the MedDRA hierarchy (LLT/PT/HLT/HLGT/SOC); official SOC mapping requires an MSSO subscription. Categories here are project-defined.
* **NCI CTCAE v5.0:** published by the US National Cancer Institute (public); it embeds MedDRA v20.1 codes/SOC names under NCI's arrangement with MSSO. It is used as a local evaluation reference, and its NCI-written definitions are shown with attribution as a glossary in question-answering answers (MedDRA codes are not shown). Check with your institution before publishing the dashboard publicly.
* **FDA README / FAQ / ASC_NTS:** public U.S. government documents shipped in the FAERS zip; quoted with attribution.
* **Local models:** Qwen2.5-0.5B-Instruct Apache-2.0 (rev `7ae5576`; note some larger Qwen2.5 sizes use other licences); BGE-small-en-v1.5 MIT (rev `5c38ec7`). Weights are not redistributed with the project.
* **openFDA / DailyMed / FDA.gov:** public U.S. government content (openFDA terms: no endorsement implied; data not for clinical decisions). **PubMed:** abstracts are shown with citation and link; abstract copyright belongs to publishers – fine for local academic use, do not republish the corpus.
* Python/JS dependencies are permissive (MIT/BSD/Apache-2.0; pypdf BSD); Plotly.js is MIT.

## Validation

| Check | Command | Result |
|---|---|---|
| Backend tests | `cd backend && pytest -q` | 110 passed |
| Lint | `cd backend && ruff check .` | All checks passed |
| Frontend tests | `cd frontend && npm test` | 18 passed |
| Type check | `cd frontend && npm run typecheck` | 0 errors |
| Production build | `cd frontend && npm run build` | succeeds |

Backend tests use a synthetic FAERS quarter with hand-checkable answers (version selection, deletions, unit conversion, duplicate flags, exact 2×2 cells and Haldane-corrected ROR, indication overlap), unit tests of every formula against textbook values, NLP abstention/calibration/review validation, a test that importing the statistics modules never imports torch or the NLP package, a test that running NLP leaves `signals.parquet` byte-identical, and API validation tests, and for question answering: BM25/stemming, FAQ splitting, the verifier (8 accept/reject cases incl. invented numbers, citations, quotes and causal claims), generator mechanics with a tiny random model, linking, untested pairs, abstention, input validation and a check that answering never changes `signals.parquet`.

## Limitations

* Spontaneous-report biases (under-reporting, stimulated/notoriety reporting, Weber effect), missing fields and no exposure denominator: no incidence or risk estimates are possible.
* With a = 3 as the screening floor and a very large background, ~71% of tested pairs are flagged; many are product-use or indication terms. Use the indication-bias flag, the minimum-cases filter, and clinical review. No multiplicity control is applied to the flags themselves (q-values are reported).
* Drug names are normalised by regex/active ingredient only (no RxNorm), so salts and brands may split one substance (e.g. `CABOZANTINIB S-MALATE`).
* Probable-duplicate detection is a strict exact-match heuristic; fuzzy duplicates remain.
* One quarter only: no time-trend signal detection or cumulative analysis.
* External evidence: 150 drugs only; label sections longer than 25 passages are truncated; label-mention check is exact text (synonyms missed); small generators (0.5B–3B) make confident yes/no errors and are therefore opt-in, with a structure + entailment verifier; section routing is rule-based and was evaluated on templated and author-written paraphrased questions only; ~7% of answerable questions fall below the similarity threshold; PubMed abstracts only, no full text; sources are a snapshot of 2026-10-20.
* Question answering: evaluated on templated, not real, user questions; no comparisons, doses or time trends; BM25 is lexical (misses paraphrased documentation) and lost to TF-IDF on the FAQ test; brand names come from FAERS verbatim names, not RxNorm; the generative summariser is unevaluated.
* NLP: CTCAE is an oncology terminology and cleaner than FAERS PTs; product-use terms have no reference label; the SOC→category crosswalk is ours; 42% of distinct terms (mostly rare ones) are abstained; anchors encode the authors' judgement. Accuracy on FAERS terms awaits blinded human annotation.

## Project layout

```
backend/pvsys/        pipeline package (config, ingest, clean, quality, eda, signals, db, pipeline CLI)
backend/pvsys/nlp/    categories, encoder, classify, evaluate, review, search, run, kb, rag, generator, rag_eval
backend/pvsys/evidence/ sources, chunking, embed, index, evaluate (external clinical evidence)
backend/pvsys/device.py CPU / CUDA selection for all models (PV_DEVICE)
backend/api/main.py   FastAPI app
backend/tests/        pytest suite (synthetic FAERS fixture)
frontend/src/         React app (pages/, components/, components/ui/ shadcn primitives, lib/, test/)
data/                 raw zip, processed Parquet, nlp outputs, reviews, reference/ (CTCAE)
models/               SapBERT weights (pinned revision)
docs/                 academic_report.md, viva_guide.md
scripts/              run.ps1 launcher, download_model.py
```
