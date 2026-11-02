# AI-Based Drug Safety and Adverse Effect Analysis System

**Project report — FDA FAERS 2025 Q1**

---

## Abstract

Post-marketing drug safety depends on spontaneous adverse event reports, which are voluminous, incomplete and duplicated. This project builds an end-to-end, locally runnable pharmacovigilance workbench on the U.S. FDA Adverse Event Reporting System (FAERS) extract for January–March 2025. A reproducible pipeline ingests 400,514 reports, applies the FDA case-version rules, flags 7,784 probable duplicate reports and measures data quality. Disproportionality analysis (reporting odds ratio, proportional reporting ratio, Bayesian information component) with Haldane–Anscombe sparse-cell correction, false-discovery-rate q-values, an indication-bias flag and subgroup stratification screens 241,675 drug–reaction pairs over an analysis population of 392,729 cases. Well-known label associations are recovered (e.g. metformin–lactic acidosis ROR 271.5, 95% CI 226.7–325.1). An NLP component uses the pretrained biomedical language model SapBERT, run on CPU, to place each of the 12,678 MedDRA reaction terms into 23 project-defined body-system categories with explicit abstention, and to support semantic search over the reaction vocabulary. Against an independent reference (NCI CTCAE v5.0), on a held-out test split of 499 terms, SapBERT reaches 62.7% top-1 accuracy (95% CI 58.5–66.7) versus 43.1% for a character n-gram baseline and 15.8% for the majority class, and 71.5% accuracy on the 65.3% of terms it chooses to answer. A second NLP component answers free-text questions by retrieval-augmented generation (RAG): it links drug names (FAERS vocabulary plus 2,079 brand/verbatim aliases learnt from FAERS) and reaction phrases (exact MedDRA match, else SapBERT with an acceptance threshold) in the question, retrieves the precomputed statistics and BM25-ranked passages from FDA's FAERS documentation and NCI CTCAE definitions, and composes a cited answer only from that evidence, abstaining when nothing relevant is found. On independent references, SapBERT links NCI definitions to the correct FAERS term at rank 1 for 47.8% of 299 test terms (top-10 85.6%) versus 11.7–12.4% for lexical baselines; BM25 finds FDA's own answer to its FAQ questions at rank 1 for 44.2% of 52 questions (top-10 80.8%); and 99.3% of 750 templated questions over random real signal pairs retrieve exactly the stored statistics, with 100% of answers passing an automatic numeric-grounding check. Model outputs carry full provenance and review status and are presented as provisional. The assistant also retrieves external clinical evidence – 135 FDA labels, 42 FDA Drug Safety Communications and 424 PubMed reviews (8,590 passages) embedded with BGE-small – and keeps it separate from FAERS observations. Dense retrieval finds FDA's own gold label section first for 81.2% of 886 test questions (BM25 51.0%; lay wording 67.7% vs 18.5%), an evidence-sufficiency threshold declines 91.6% of unanswerable questions while answering 93.3% of answerable ones, and a local 0.5B generative model proved unreliable (35% vs 97.5% for extractive answering) and is therefore opt-in only. A FastAPI backend and a React/TypeScript dashboard expose all results with their limitations.

---

## 1. Introduction

### 1.1 Background

Clinical trials are too small and too short to reveal rare or delayed adverse drug reactions, so regulators rely on post-marketing surveillance. FAERS collects reports from manufacturers (mandatory) and from health professionals and consumers (voluntary). The FDA publishes the database quarterly as delimited text files. Signal detection in such databases relies on *disproportionality analysis*: comparing how often a reaction is reported with a drug against how often it is reported with all other drugs.

### 1.2 Problem statement

Raw FAERS data cannot be analysed directly: cases appear in several versions, some are deleted, the same event can be reported by several sources, units are mixed, and many fields are empty. Disproportionality statistics are easy to compute but easy to misread as risk. Finally, reaction terms are fine-grained MedDRA Preferred Terms (12,678 distinct terms in one quarter), so a reviewer interested in a clinical concept ("liver injury") must find dozens of related terms by hand. A non-specialist who simply asks "does Ozempic cause stomach problems?" cannot use the data at all: the question has to be translated into an ingredient name, a MedDRA term and the right table, and the answer needs the statistics *and* the caveats that govern their interpretation.

### 1.3 Objectives

1. Build a reproducible ingestion, cleaning, de-duplication and data-quality pipeline for FAERS 2025 Q1.
2. Implement disproportionality statistics with confidence intervals and sparse-cell handling, and present them with an explicit distinction from causality and risk.
3. Add a genuine NLP component using a locally runnable pretrained biomedical language model for provisional reaction categorisation and semantic retrieval, with provenance, abstention, a reproducible evaluation against independent ground truth and a baseline comparison.
4. Let users ask questions in natural language and receive answers grounded in, and citing, the retrieved statistics and official documentation (retrieval-augmented question answering), with each component evaluated against independent references.
5. Deliver an interactive dashboard and API that keep statistical outputs independent of model outputs.

### 1.4 Scope

One quarter (2025 Q1) of FAERS ASCII data. Drug-level analysis uses suspect drugs (primary or secondary suspect). No external exposure data, so no incidence or risk estimation.

---

## 2. Related work

* **Disproportionality methods.** The PRR with Evans criteria (PRR ≥ 2, χ² ≥ 4, a ≥ 3; Evans et al., 2001) and the ROR with lower 95% CI > 1 (van Puijenbroek et al., 2002) are standard frequentist screens. Bayesian shrinkage methods — the BCPNN information component used by the WHO Uppsala Monitoring Centre (Bate et al., 1998; Norén et al., 2013) and the FDA's MGPS/EBGM (DuMouchel, 1999) — stabilise estimates for small counts.
* **Biases.** Spontaneous reporting suffers under-reporting, stimulated reporting, notoriety bias, the Weber effect and confounding by indication (Hauben & Aronson, 2009; Bate & Evans, 2009).
* **FAERS de-duplication.** The FDA README recommends keeping the latest version of each CASEID; further duplicate detection by matching clinical fields is common practice (Hauben et al., 2021).
* **Biomedical language models.** PubMedBERT (Gu et al., 2021) is BERT pretrained from scratch on PubMed. SapBERT (Liu et al., 2021) continues training PubMedBERT with a self-alignment objective over UMLS synonyms, so that surface forms of the same concept embed close together; it is state of the art for biomedical entity linking with nearest-neighbour search and needs no task-specific fine-tuning.
* **Retrieval-augmented generation.** RAG (Lewis et al., 2020) conditions a generator on passages retrieved for the query, so answers can cite sources and stay current without retraining. Okapi BM25 (Robertson & Zaragoza, 2009) remains a strong lexical retriever. In safety-critical settings the main risk is unfaithful generation (hallucinated numbers or claims); grounding checks and extractive or template-based generation are common mitigations.
* **Terminologies.** MedDRA organises terms hierarchically into System Organ Classes (SOCs) under licence from the MSSO. NCI's Common Terminology Criteria for Adverse Events (CTCAE v5.0) is a public list of ~840 adverse-event terms grouped under MedDRA SOC headings.

---

## 3. Data

### 3.1 Source

`faers_ascii_2025q1.zip` from the FDA Quarterly Data Extract page (SHA-256 `51c1dc44…6b69`, 67.5 MB). Seven `$`-delimited tables plus a deletion list:

| Table | Content | Rows |
|---|---|---|
| DEMO | One row per case version: dates, age, sex, weight, reporter, country | 400,514 |
| DRUG | Drugs per case with role (PS, SS, C, I), names, route, dose | 2,008,162 |
| REAC | MedDRA Preferred Terms per case | 1,432,926 |
| OUTC | Outcome codes (death, hospitalisation, …) | 304,027 |
| INDI | Indications per drug | 1,205,861 |
| THER | Therapy start/end dates | 520,395 |
| RPSR | Report sources | 11,033 |
| DELETE | Case IDs deleted by FDA | 5,732 |

All tables loaded with zero rejected lines (DuckDB `read_csv` with `all_varchar`, so no value is coerced or lost on load).

### 3.2 Additional resources

* **NCI CTCAE v5.0** (`CTCAE_v5.0_MedDRA_v20.1.xlsx`, 837 rows) — evaluation reference for categorisation and linking; its NCI-written definitions are also shown, attributed, as a glossary in question-answering answers.
* **FDA FAERS documentation** shipped inside the quarterly zip — `Readme.pdf` (8 pages), `FAQs.pdf` (12 pages, 52 numbered questions), `ASCII/ASC_NTS.pdf` (19 pages) — read with `pypdf` into the question-answering knowledge base.
* **SapBERT** weights from Hugging Face, revision `090663c3…4d`, Apache-2.0.

---

## 4. Methodology

### 4.1 Pipeline

```
zip ─► ingest ─► raw Parquet ─► clean/dedup ─► processed Parquet ─┬─► quality.json
                                                                  ├─► eda_snapshot.json
                                                                  ├─► signals.parquet
                                                                  └─► (optional) NLP ─► predictions.parquet
                                                                                     └─► kb_chunks.parquet (RAG knowledge base)
```

Each stage is a Python module in the `pvsys` package and can be re-run from the CLI (`python -m pvsys.pipeline all`). A manifest records the source checksum, row counts and ingest time. Processed data are columnar Parquet files; DuckDB provides SQL over them in-process, without a database server.

### 4.2 Cleaning and normalisation

* **Dates** — `YYYYMMDD` strings parsed to dates; partial dates kept as missing (none occurred in this quarter's event dates).
* **Age** — converted to years from the `age_cod` unit (YR, MON, WK, DY, HR, DEC); values outside 0–120 set to missing (7 cases). Age groups: 0–1, 2–11, 12–17, 18–44, 45–64, 65–74, 75+.
* **Weight** — converted to kg (KG, LBS, GMS); values outside 0.2–400 kg set to missing (44 cases).
* **Sex** — F, M, otherwise "Unknown".
* **Drug names** — active ingredient (`prod_ai`) when present (98.4% of drug rows), otherwise the verbatim name upper-cased with dose strings and trailing punctuation removed.
* **Reactions** — whitespace normalised, case-insensitive key; 18,571 repeated term rows within the same report collapsed.
* **Outcomes** — a case is *serious* if it has any outcome code; flags for death, hospitalisation, life-threatening.

### 4.3 Case-version handling and de-duplication

1. Remove every CASEID on the quarter's DELETE list (1 report removed).
2. For each CASEID keep the row with the latest `FDA_DT`, ties broken by the highest `PRIMARYID` (the FDA-recommended rule). In 2025 Q1 every CASEID already appears once, so 0 older versions were removed; the rule is still enforced and unit-tested on synthetic data with multiple versions.
3. **Probable duplicates.** Reports with different CASEIDs but identical sex, age, event date, occurrence country, set of normalised suspect drugs and set of reaction terms are grouped; the most recently received is kept and the others flagged (7,784 cases). Event date and age must be present to avoid over-matching sparse reports. Flagged reports remain in descriptive statistics and are excluded from disproportionality.
4. All child tables are restricted to retained PRIMARYIDs.

Result: 400,513 de-duplicated cases; analysis population for signals 392,729.

### 4.4 Data-quality assessment

Checks across completeness, validity, consistency and uniqueness (Table 1). Missing data are disclosed, never imputed.

**Table 1. Selected data-quality results**

| Dimension | Check | Value |
|---|---|---|
| Completeness | Age missing/invalid | 39.2% of cases |
| Completeness | Weight missing | 82.8% |
| Completeness | Full event date missing | 56.9% |
| Completeness | Sex not reported | 20.7% |
| Completeness | Reporter occupation missing | 8.2% |
| Validity | Implausible age / weight | 7 / 44 cases |
| Validity | Event date after FDA receipt | 7 cases |
| Consistency | Cases without a suspect drug / without a reaction | 0 / 0 |
| Consistency | Orphan DRUG or REAC rows | 0 |
| Uniqueness | Probable duplicates | 7,784 cases |

### 4.5 Disproportionality analysis

For each suspect drug *D* and reaction term *R*, a case-level 2×2 table is built over the analysis population (*a*: cases with D and R; *b*: D without R; *c*: R without D; *d*: neither).

* **ROR** = (a/b)/(c/d); 95% CI = exp(ln ROR ± 1.96·√(1/a + 1/b + 1/c + 1/d)).
* **PRR** = [a/(a+b)]/[c/(c+d)]; 95% CI = exp(ln PRR ± 1.96·√(1/a − 1/(a+b) + 1/c − 1/(c+d))).
* **χ²** with Yates continuity correction, and its p-value; **Benjamini–Hochberg q-values** across all tested pairs.
* **IC** = log₂((a+0.5)/(E+0.5)), E = (a+b)(a+c)/N, and IC025 ≈ IC − 3.3(a+0.5)^−½ − 2(a+0.5)^−3/2 (Norén et al., 2013).
* **Sparse cells.** If any cell is zero, 0.5 is added to all four cells (Haldane–Anscombe) so that ratios and intervals are finite; flagged per row (4,201 rows).
* **Screening rules.** ROR: lower CI > 1 and a ≥ 3. PRR: PRR ≥ 2, χ² ≥ 4, a ≥ 3. IC: IC025 > 0, a ≥ 3.
* **Indication-bias flag.** The share of the *a* cases in which the reaction term is also listed as an indication; ≥ 20% flags *possible indication bias*.
* **Subgroup analysis.** The 2×2 table recomputed within sex, age-group or reporter-type strata, displayed as a forest plot; strata with a < 3 are not interpreted.

### 4.6 NLP: provisional reaction categorisation with a biomedical language model

**Why this design.** The task is to group short biomedical terms by organ system without any labelled FAERS training data, on a CPU, without inventing labels. A zero-shot, embedding-based classifier over a domain language model meets these constraints: no fine-tuning data are needed, the decision rule is transparent, and uncertainty can be expressed by abstaining. Generative LLMs were not used for categorisation because they need GPUs or paid APIs and their free-text outputs are harder to audit. The same embeddings also provide *semantic retrieval* over the reaction vocabulary, which the question-answering module (§4.8) reuses for linking.

**Model.** `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` — 109.5 M parameters, Apache-2.0. The [CLS] vector of each term (max 25 tokens) is L2-normalised. Embedding all 12,678 terms takes about two minutes on a laptop-class CPU.

**Taxonomy.** 23 project-defined categories (cardiovascular, respiratory, gastrointestinal, hepatobiliary, renal & urinary, nervous system, psychiatric, skin, musculoskeletal, blood, immune & hypersensitivity, infections, metabolism & nutrition, endocrine, eye, ear, reproductive, pregnancy & perinatal, neoplasms, general & administration site, investigations, injury/poisoning/procedural, product use/errors/efficacy). Each has a description and 6–14 anchor phrases (251 in total) written by the project before the evaluation reference was inspected. The *product use* category is included because FAERS is dominated by non-clinical terms such as "Off label use" (32,441 cases). These categories are deliberately **not** MedDRA SOCs: official MedDRA mappings require an MSSO licence and are not claimed.

**Classifier.** Each category's anchor embeddings are averaged into a normalised centroid; a term's score for a category is its cosine similarity to that centroid. Prediction = highest-scoring category, unless the top score is below *s_min* (low similarity) or the margin over the runner-up is below *m_min* (ambiguous), in which case the model **abstains**. A softmax over scores (temperature 0.05) is reported as a relative confidence for ranking only.

**Baselines.** (i) The same prototype classifier on TF-IDF character 3–5-gram vectors (lexical baseline, also the fallback when the neural model is unavailable). (ii) Majority class.

**Provenance and review.** Each prediction stores model id, revision, taxonomy version, run id, score, margin, confidence, abstention reason, nearest anchor and the baseline prediction. `provenance.json` records the weights' SHA-256, library versions, thresholds and source-data checksum. Dashboard reviewers can accept, correct, reject or mark unsure; decisions are appended to an audit log with reviewer and time.

### 4.7 Evaluation protocol for the NLP component

Labels were never produced by the authors or by a model. Two sources of ground truth are supported:

1. **External reference — NCI CTCAE v5.0.** Each CTCAE term sits under a MedDRA SOC heading chosen by NCI. A project-defined crosswalk maps the 23 SOC headings present to our categories. "Other, specify" entries (26) were removed, then terms identical to any anchor phrase (99) to prevent leakage, leaving 712 terms. A stratified split (seed 42) put 30% (213) in a **development** set, used to choose the scoring rule (centroid vs nearest-anchor) and tune abstention thresholds (grid search maximising accuracy at ≥ 70% coverage), and 70% (499) in a **test** set used once for reporting. 233 test terms also occur in FAERS 2025 Q1. Metrics: top-1 accuracy without abstention, coverage, accuracy on answered items, macro-F1, per-category precision/recall, confusion matrix, accuracy–coverage curve; 95% CIs by bootstrap (2,000 resamples).
2. **Blinded human annotation of FAERS terms.** `annotate-export` samples terms stratified by provisional group but writes the sheet *without* model output; two annotators label independently with a written guide; `annotate-import` validates labels; `evaluate` reports accuracy and Cohen's κ. Unblinded dashboard reviews are kept separate and never used for metrics because seeing the model's answer biases the reviewer (anchoring).

### 4.8 NLP: evidence-grounded question answering (RAG)

FAERS has no free-text narratives, but the project has two retrievable bodies of evidence: the computed statistics, and the documentation that explains how to read them. The question-answering module (`pvsys/nlp/rag.py`) retrieves from both and generates an answer that is constrained to what it retrieved.

**Knowledge base** (`kb.py`, `kb_chunks.parquet`, 942 chunks). FDA's README and ASC_NTS are split into 140-word windows with a 100-word stride (30 and 40 chunks); the FAQ is split into its 52 numbered question–answer pairs; 811 NCI CTCAE definitions form a glossary; and 9 short *project methods notes* (ROR, PRR, IC, Haldane, indication bias, de-duplication, minimum case count, "disproportionality is not causality", "categories are provisional") are attributed to the project, never to FDA. Each chunk keeps its source, page and the SHA-256 of its source file; the knowledge base has its own content hash. PDFs are read directly from the FAERS zip.

**Question understanding.**
1. *Drugs.* Longest-first n-gram lookup (punctuation-insensitive, so combination products with `/` or `\` match) against the 5,015 normalised suspect-drug names, plus 2,079 brand/verbatim aliases learnt from FAERS itself (a dose-stripped verbatim `DRUGNAME` maps to one normalised name in ≥ 90% of ≥ 5 rows, e.g. OZEMPIC → SEMAGLUTIDE). A first word shared by several products (e.g. "insulin") is reported as ambiguous instead of guessed; a spelling-tolerant match (difflib ratio ≥ 0.88, names with ≥ 3 cases) catches typos ("metformn").
2. *Reactions.* Exact MedDRA PT matches first (longest-first, punctuation-insensitive; a longer exact PT overrides any drug match inside it, so "Blood **magnesium** decreased" is not read as the drug magnesium). An adjacent modifier is absorbed when SapBERT finds the combined phrase with similarity ≥ 0.90 ("acute pancreatitis" → "Pancreatitis acute"). Remaining phrases are linked with SapBERT and accepted only if the top cosine similarity is ≥ 0.70 (lexical fallback: ≥ 0.45); otherwise the phrase is reported as not understood, with the rejected best guess. The threshold separates correct lay links (0.72–0.96, e.g. "stomach problems" → Dyspepsia 0.73, "suicidal thoughts" → Suicidal ideation 0.95) from noise (≤ 0.66).
3. *Intent.* Pair (drug + reaction), drug only, reaction only, documentation (a question word plus a methods/data term, or a weak reaction link with a strong documentation hit), or clarification.

**Retrieval.** Pair: the stored ROR/PRR/IC row for the linked PT and up to five SapBERT neighbours (within 0.05 of the top similarity, floor 0.75); for untested pairs (a < 3) the raw co-report count from DuckDB. Drug: the most frequently reported pairs flagged by all three methods, excluding indication bias. Reaction: the drugs most often co-reported and flagged. Documentation: BM25 (k1 = 1.5, b = 0.75) over FDA and project passages; the module abstains if the best score is < 3. The CTCAE definition of the linked term, the project causality note and the FDA README clinical-caveats passage are attached as supporting evidence.

**Generation.** The default generator is deterministic: each sentence is filled from retrieved values with one formatting function and carries the ids of the evidence it uses (S = statistics, D = documents). Documentation answers are extractive (the best-matching sentences quoted verbatim). The causality caveat is attached to every statistical answer. An **optional** local generative small language model (any Hugging Face causal LM in `models/generator/`, e.g. the Apache-2.0 Qwen2.5-0.5B-Instruct; greedy decoding on CPU) can additionally write a summary; it is shown only if a verifier finds that (i) every citation exists and at least one is given, (ii) every number appears in the evidence, (iii) every quotation is verbatim evidence, and (iv) there is no unnegated causal or incidence wording ("causes", "proves", "risk of 5%"). Otherwise it is discarded with the reasons. No generator weights could be downloaded for this project (network policy), so this path is tested only mechanically with a randomly initialised tiny model — whose output the verifier correctly rejects — and is not evaluated.

### 4.8b External clinical evidence (labels, regulatory communications, literature)

FAERS shows what was *reported*; whether a reaction is an *established* effect is stated in drug labelling and the literature. For the 150 most-reported FAERS suspect drugs, `pvsys/evidence/sources.py` downloads (i) the openFDA structured-product-labeling record of the manufacturer's most-revised label set (selected through DailyMed, repackagers de-prioritised), keeping 11 safety-relevant SPL sections; (ii) up to four PubMed reviews, systematic reviews or meta-analyses on adverse effects published since 2015 (NCBI E-utilities, query recorded); (iii) the FDA Drug Safety Communications linked from fda.gov. Every response is cached with retrieval time, URL, set id, SPL version, effective date, PMID/DOI or publication date, so the index can be rebuilt offline and every citation resolved. Documents are chunked *within* SPL sections (160 words, 30-word overlap) so a citation always names its section; FDA communications are tagged with the drugs (and label brand names) they mention.

Passages are embedded with **BAAI/bge-small-en-v1.5** (33 M parameters, MIT; CLS pooling, query instruction) and searched by exact cosine similarity (8,590 × 384 matrix – no approximate index needed); BM25 and reciprocal-rank-fusion hybrid search are implemented as alternatives. The production retriever was chosen on development drugs (§6.5). For a question, the engine (1) checks whether the drug's label text mentions the linked MedDRA term or a SapBERT-related term and in which sections; (2) retrieves passages restricted to the drug; (3) declines if the best similarity is below a threshold calibrated on development drugs, or if the drug is not indexed; (4) otherwise answers with verbatim quotes and full source details. An optional generator (**Qwen2.5-0.5B-Instruct**, Apache-2.0, fp16 on GPU / fp32 on CPU, greedy decoding, one-shot prompt) may write this part; its text is shown only if every citation exists, every number and quotation is in the cited passages, and each sentence shares at least half its content words with the passage it cites. All models run locally on an NVIDIA GPU when available (`pvsys/device.py`), else on the CPU.

### 4.9 Evaluation protocol for question answering (`rag_eval.py`)

No label was written by the authors or by a model.

* **A. Passage retrieval — FDA FAQ.** Queries: the 52 FAQ question texts. Index: the 52 FDA answers *with the question text removed* (otherwise the task is string matching), plus every README/ASC_NTS window and project note as distractors (131 passages). Gold: FDA's own answer. BM25 parameters are textbook values, so no dev split is needed; a word TF-IDF retriever and the random expectation are baselines; the TF-IDF − BM25 difference is tested with a paired bootstrap.
* **B. Reaction linking — CTCAE definitions.** The 427 CTCAE terms whose name is also a FAERS 2025 Q1 PT; query = NCI's definition, gold = that PT among all 12,678 FAERS PTs. Seeded 30/70 dev/test split (128/299). The input variant (full definition vs. with the boilerplate "A disorder characterized by" removed; 25 vs. 64 tokens) is chosen per system on dev MRR. Baselines: character TF-IDF and word BM25. Only 2 of 299 test definitions contain the term itself (leakage check).
* **C. End to end.** 150 tested (drug, PT) pairs drawn at random (seed 42), each phrased with five templates ("Is X associated with Y?", "X Y", "does x cause y", "Is there a signal for Y with X?", "Any reports of y in patients taking x?"), plus 55 brand-alias questions. Success = the engine links exactly that drug and PT and returns exactly the stored *a* and ROR. Every answer is also checked by the generator's verifier (numeric/citation/quotation grounding).

* **D. External evidence (`evidence/evaluate.py`).** Gold relevance is FDA's own SPL section coding. For each of the 127 drugs with an indexed label and each topic present (boxed warning, contraindications, interactions, pregnancy, overdosage, adverse reactions) a question is asked in label wording and in lay wording; a hit is correct if it comes from that drug's passages of the gold section. Drugs are split 30% dev (38) / 70% test (89). The retriever (dense / BM25 / hybrid) and the sufficiency threshold (answerable questions vs templated unanswerable ones: price, packaging colour, unrelated topics) are chosen on dev; everything is reported on test. Generated answers are scored on "Does X have a boxed warning?" (gold: presence of the Boxed Warning section) against an extractive baseline, plus verifier pass rate and latency. Signal–label concordance is descriptive.

---

## 5. System implementation

| Layer | Technology | Notes |
|---|---|---|
| Storage | Parquet (zstd) | ~90 MB for all processed tables |
| Query engine | DuckDB 1.x, in-process | Views over Parquet; thread-safe wrapper |
| Statistics | NumPy, SciPy, pandas | Vectorised over all pairs in ~10 s |
| NLP | PyTorch (CPU), Hugging Face Transformers, scikit-learn, pypdf | Lazy-loaded; never imported by statistics |
| Question answering | SapBERT linking, BM25 (own implementation on scikit-learn), template/extractive generation, optional local SLM + verifier | ~120 ms per question after a background warm-up at API start |
| API | FastAPI, Pydantic | Typed query validation, gzip, LRU caching, 503/409/422 errors with actionable messages |
| UI | React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui (Radix), Plotly.js, TanStack Query | 9 pages; light/dark themes; keyboard accessible |
| Launcher | PowerShell | One command: setup, data build, start, open browser, clean stop |

**Independence of statistics from NLP.** The statistics modules do not import the NLP package; a test asserts that importing them loads neither `torch` nor `pvsys.nlp`, and another that running the NLP stage leaves `signals.parquet` byte-identical. NLP predictions are joined only at query time in the API, for optional filtering and display.

**Graceful degradation.** Without weights or torch, the NLP stage produces lexical predictions labelled `lexical-fallback`, search reports its mode, and the dashboard explains the situation; all statistical pages are unaffected.

**Dashboard pages.** Overview (case-flow funnel, outcomes, daily receipts), Data quality, Demographics (age–sex pyramid, serious share by subgroup, χ² test), Drug explorer, Signal detection (filters, forest plot, 2×2 detail with all measures, subgroup ROR), Ask the data (question box, cited answer, how the question was read, statistical and documentary evidence with deep links into Signal detection, provenance, and a tab with the question-answering evaluation), Reaction categories (category distribution, term predictions with review, search by meaning, signals for a concept), Model & evaluation, Methods & limits. Model output is always drawn with a dashed violet outline and a "Provisional" badge.

---

## 6. Results

### 6.1 Descriptive analysis

* 400,513 de-duplicated cases received 1 Jan – 31 Mar 2025 (three stray earlier receipt dates). 224,700 (56.1%) serious; 32,408 (8.1%) with death reported; 83,294 hospitalisations.
* Sex: female 186,230 (46.5%), male 131,469 (32.8%), not reported 82,814 (20.7%). Median age 59 years.
* Serious share rises with age (18–44: 60.2%; 75+: 68.3%) and death share most steeply (75+: 14.6%). Infants 0–1 have a high serious share (71.4%).
* Male cases were more often serious (60.4% vs 55.7%; χ² = 712.6, p < 10⁻¹⁵⁶) but Cramér's V = 0.047: statistically significant only because *n* is huge, practically negligible, and a property of reports rather than patients.
* Reporters: consumers 165,641 (42.3% serious) vs physicians 87,265 (72.6% serious).
* Countries: US 246,922; Canada 30,004; Japan 16,260; UK 14,119; France 14,094.
* Most reported suspect drugs: dupilumab (34,416), tirzepatide (11,367), lenalidomide (8,074), adalimumab (6,853), acetaminophen (6,351).
* Most reported terms are non-clinical: "Off label use" (32,441), "Drug ineffective" (23,304), followed by fatigue, "Product dose omission issue", nausea, diarrhoea, death.

### 6.2 Disproportionality

Of 648,090 observed suspect-drug–reaction pairs, 241,675 had a ≥ 3. ROR flagged 192,163, PRR 184,795, IC 173,684; 172,074 (71.2%) were flagged by all three. The high proportion is expected: with a background of ~390,000 cases, almost any co-reporting of a specific drug and term exceeds expectation, and many top pairs are product-use terms (e.g. somatropin–"Drug dose omission by device", ROR 79.1) or indication artefacts. 16,092 pairs were flagged for possible indication bias, including dupilumab–atopic dermatitis (98.4% overlap).

**Table 2. Recovery of well-known label associations (positive controls)**

| Drug | Reaction | a | ROR (95% CI) | PRR | IC025 |
|---|---|---|---|---|---|
| Metformin hydrochloride | Lactic acidosis | 280 | 271.5 (226.7–325.1) | 233.9 | 6.29 |
| Semaglutide | Pancreatitis | 74 | 12.5 (9.8–15.9) | 12.2 | 3.00 |
| Tirzepatide | Pancreatitis | 88 | 4.2 (3.3–5.2) | 4.1 | 1.55 |
| Levofloxacin | Tendon rupture | 5 | 18.1 (7.4–44.1) | 17.9 | 1.24 |
| Montelukast sodium | Nightmare | 12 | 18.0 (10.1–32.1) | 17.6 | 2.41 |
| Isotretinoin | Depression | 48 | 6.1 (4.5–8.2) | 5.8 | 1.98 |
| Adalimumab | Tuberculosis | 19 | 6.3 (3.9–10.0) | 6.2 | 1.58 |
| Dupilumab | Conjunctivitis | 256 | 8.4 (7.2–9.9) | 8.4 | 2.14 |

All were flagged by all three methods. This is a plausibility check of the implementation, not a validation of the method's sensitivity or specificity, which would require a reference set of positive and negative controls.

### 6.3 NLP evaluation

**Table 3. Held-out CTCAE test split (499 terms)**

| System | Top-1 accuracy (95% CI) | Coverage | Accuracy on answered (95% CI) | Macro-F1 (answered) |
|---|---|---|---|---|
| SapBERT prototypes | **62.7%** (58.5–66.7) | 65.3% | **71.5%** (66.9–76.4) | 0.620 |
| TF-IDF character n-grams | 43.1% (38.9–47.3) | 72.7% | 53.7% (48.5–59.2) | 0.458 |
| Majority class | 15.8% | 100% | 15.8% | – |

Tuned thresholds: SapBERT abstains if similarity < 0.46 or margin < 0.02; lexical if similarity < 0.06 or margin < 0.01. On the 233 test terms that also occur in FAERS, SapBERT top-1 accuracy was 67.8% (lexical 44.6%). The confidence intervals of the language model and the lexical baseline do not overlap.

The scoring rule was chosen on the dev split: centroid prototypes 66.2% vs nearest anchor 61.0% (also mean-of-anchors 65.3%, top-2 mean 63.4%, category description only 52.6%).

Per-category recall on test was high for eye (85%), reproductive (77%), skin (77%), hepatobiliary and renal (73%), and low for injury/procedural (11%), general (8%) and neoplasms (0%, 4 terms). Low recall for gastrointestinal (33%) and nervous system (32%) is mostly abstention (42 of 79 GI terms abstained) rather than error. The confusion matrix shows that the weakest groups are those defined by mechanism or circumstance rather than anatomy (injury/poisoning/procedural, general), where most terms name an affected organ and the model abstains or follows that organ instead.

**FAERS run.** 7,333 of 12,678 terms were assigned and 5,345 abstained (3,732 low similarity, 1,613 ambiguous). Because frequent terms are clearer, assigned terms account for 85.5% of all case–reaction mentions. Top-1 agreement with the lexical baseline was 41.9%. Illustrative behaviour: "Off label use" → product use (confidence 0.98); "Hepatotoxicity" → hepatobiliary (0.97); "Pancreatitis" → abstained as ambiguous between hepatobiliary and gastrointestinal (margin 0.009) — abstention prevented an error, since MedDRA files pancreatitis under gastrointestinal disorders; "Type IIa hyperlipidaemia" → hepatobiliary, an error that a reviewer can correct in the dashboard.

**Semantic retrieval.** The query "liver damage" returns "Liver injury" (similarity 0.82) and "Hepatocellular injury", and "pancreas inflammation" gathers "Pancreatitis", whose signals include semaglutide and tirzepatide — terms that a substring search would not connect to the query.

No accuracy is claimed for predictions on FAERS terms; the blinded annotation workflow is implemented and tested but no human annotations have been collected yet.

### 6.4 Question-answering evaluation

**Table 4. A — FDA FAQ passage retrieval (52 questions, 131 passages)**

| Retriever | Top-1 (95% CI) | Top-3 | Top-5 | Top-10 | MRR (95% CI) |
|---|---|---|---|---|---|
| BM25 (used) | 44.2% (30.8–57.7) | 65.4% | 71.2% | 80.8% | 0.559 (0.445–0.676) |
| Word TF-IDF | 53.8% (38.5–67.3) | 69.2% | 76.9% | 82.7% | 0.635 (0.518–0.749) |
| Random | 0.8% | 2.3% | 3.8% | 7.6% | 0.042 |

Both retrievers are far above chance. TF-IDF beat BM25 on this benchmark: paired bootstrap difference in top-1 +9.6 points (95% CI 1.9–19.2), MRR +0.076 (0.024–0.140). BM25 had been fixed before the test; switching on the evidence of the only benchmark would be selection on test data, so the result is reported and the switch is left to future work with a second, held-out question set. No FAQ question fell below the abstention score.

**Table 5. B — Linking NCI definitions to FAERS terms (299 test terms, 12,678 candidates)**

| System (chosen variant) | Top-1 (95% CI) | Top-3 | Top-5 | Top-10 | MRR |
|---|---|---|---|---|---|
| SapBERT (boilerplate stripped, 64 tokens) | **47.8%** (42.1–53.5) | 67.2% | 77.3% | 85.6% | 0.607 |
| Character TF-IDF (stripped) | 11.7% (8.4–15.7) | 19.1% | 22.4% | 27.1% | 0.173 |
| Word BM25 (stripped) | 12.4% (8.7–16.4) | 19.7% | 23.1% | 29.1% | 0.177 |

Dev MRR for the SapBERT variants: full definition, 25 tokens 0.557; stripped, 25 tokens 0.570; stripped, 64 tokens 0.602. At the production threshold (0.70) SapBERT accepts 74.9% of test definitions and 58.5% of accepted links are the gold term. Definitions are long, formal paraphrases (e.g. "a dysrhythmia without discernible P waves…" for atrial fibrillation, gold rank 463), so this is a conservative proxy; typical misses are near-synonyms ("Eosinophil count abnormal" for *Eosinophilia*, gold at rank 2).

**C — End to end.** 745 of 750 templated questions (99.3%, 95% CI 98.7–99.9) returned exactly the stored statistics; every template scored the same, and 55 of 55 brand-alias questions succeeded. All 5 failures are PLACEBO, which the linker deliberately excludes. The first version of the engine scored 78.0%: the evaluation exposed that reaction words were being discarded as filler ("Troponin **increased**", "**Drug** hypersensitivity"), that drug names with `/` or `\` never matched, and that drug look-up captured words inside reaction terms; fixing these raised the score to 94.7% and then 99.3%. 100% of the 805 answers passed the grounding check. Mean latency was 117 ms per question on CPU.

Examples (real output): "Does Ozempic cause stomach problems?" → OZEMPIC → SEMAGLUTIDE (alias), "stomach problems" → Dyspepsia (0.73): 177 cases, ROR 10.6 (9.10–12.5), flagged by all three methods, plus the CTCAE definition and the causality caveat. "Is dupilumab associated with atopic dermatitis?" → ROR 151.3 with the warning that 98% of the cases list it as the indication. "Why are some ages very large?" → FDA FAQ 6, quoted: some manufacturers submit age in days. "What is the weather today?" → abstains.

### 6.5 External evidence evaluation

Built on an RTX 3050 laptop GPU: 601 documents (135 labels, 424 reviews, 42 FDA communications), 8,590 passages, embedded in 110 s; no download failures (13 drugs have no openFDA label).

**Table 6. Finding FDA's gold label section (89 test drugs, 886 questions, search within the drug)**

| Test (89 test drugs, 886 questions; gold = FDA's own SPL section coding) | Dense vectors (used) | Hybrid (RRF) | BM25 |
|---|---|---|---|
| Right label section ranked first – all questions | **81.2%** | 67.2% | 51.0% |
| … in top 5 (95% CI) | **94.6%** (93.0–95.9) | 85.6% | 70.5% |
| … lay wording only ("Who should not take Ozempic?"), top 1 / top 5 | **67.7% / 90.5%** | 39.5% / 71.6% | 18.5% / 43.3% |
| … label wording only, top 1 | 94.6% | 94.8% | 83.5% |
| MRR / nDCG@10 – all questions | **0.871 / 0.895** | 0.750 / 0.791 | 0.595 / 0.643 |

On development drugs the MRR was 0.850 (dense), 0.752 (hybrid) and 0.588 (BM25), so dense retrieval was selected; the test results confirm the choice. Label-worded questions are easy for every method (≥ 83.5% top-1); lay questions separate them, because they share almost no words with labelling ("who should not take" vs "contraindications"). Searching the whole index without the drug filter lowers dense top-1 only from 81.2% to 76.1%.

**Evidence sufficiency.** The 0.73 cosine threshold chosen on dev gives a balanced accuracy of 92.5% on test drugs (answerable kept 93.3%, unanswerable declined 91.6%; median similarity 0.80 vs 0.53).

**Generation.** Qwen2.5-0.5B-Instruct answered "Does X have a boxed warning?" correctly for 35% of 40 test drugs (95% CI 20–50%), against 97.5% for the extractive rule; 21 of its answers could not even be classified. Only 42.5% of drafts passed the source checks; the commonest failures were sentences not supported by the cited passage (21) and missing citations (18) – including a fluent but unsupported claim that metformin should be avoided in pregnancy, caught by the support rule during development. Because a fluent wrong answer is worse than a quoted correct one, generation is disabled by default.

**Concordance.** Among the 300 most-reported strong signals of labelled drugs, the reaction term appears in the label text for 33.7% (42.8% excluding product-use terms); among the 300 most disproportionate pairs, for 1.7%, which mostly reflects unusual report clusters and terms that labels do not use (e.g. desoximetasone–pemphigus, 482 reports). Exact matching undercounts synonyms, so these are lower bounds.

---

## 7. Discussion

**Statistics.** The pipeline reproduces established associations and makes the main biases visible in the interface: the indication-bias flag removes the most misleading top results, the Haldane flag marks unstable estimates, and subgroup forest plots show when apparent sex or age differences are within noise. The very large number of flagged pairs illustrates that frequentist screens with a ≥ 3 are hypothesis generators; the q-values, effect sizes and the minimum-case filter are needed to prioritise.

**NLP.** A frozen biomedical encoder with a few hand-written anchors per category cuts the error rate of a strong lexical baseline by about a third (from 56.9% to 37.3%) without any training data. Abstention trades coverage for accuracy in a controllable way, and the accuracy–coverage curve lets a reviewer pick an operating point. The model is most useful for navigation (search by meaning, concept-level signal lookup, filtering out product-use terms) rather than for coding. Its weaknesses mirror the taxonomy: categories defined by mechanism ("injury", "general") are poorly captured by an encoder trained on synonymy.

**Question answering.** Grounding by construction (templates and verbatim quotes) rather than by hoping a generator stays faithful made the answers checkable: every number and citation can be traced, and the automatic check passed for all evaluated answers. The evaluation was useful as an engineering tool as well as a measurement — it found three parsing bugs that a handful of manual examples had not. The weak point is the language side rather than the statistics: SapBERT is trained on short names, so long descriptions link less reliably than lay phrases, and BM25 misses documentation phrased differently from the question ("standardised drug names" vs FDA's "drug dictionary").

**External evidence.** Dense embeddings were decisive for lay questions, where keyword search fails; rank fusion did not help because one weak ranker drags down a strong one. The negative generation result is itself a finding: at this size a generative model is a liability for factual safety questions, and the evidence-support checks are what made that visible and kept unsupported text off the screen.

**Independence and honesty.** Keeping the language model out of the statistical path means a model error can never change an ROR. Model outputs are visually distinct, carry provenance and are labelled provisional everywhere.

---

## 8. Limitations

1. **Spontaneous reporting.** Under-reporting, stimulated and notoriety reporting, the Weber effect, duplicate reporting beyond what is detected, and missing fields. No exposure denominator: no incidence, prevalence or absolute/relative risk.
2. **Disproportionality ≠ causality.** Signals need clinical case review, literature, and pharmacological plausibility.
3. **Single quarter.** No temporal or cumulative analysis; three receipt dates fall before 2025.
4. **Drug normalisation.** Regex and active ingredient only; salts and brand variants can split a substance. RxNorm mapping was not available offline.
5. **Duplicate detection.** Exact-match heuristic; fuzzy duplicates remain, and a few true distinct cases with identical fields could be flagged.
6. **Multiplicity.** Screening flags are not FDR-controlled; q-values are reported alongside.
7. **NLP reference.** CTCAE is oncology-oriented and cleaner than FAERS terms; product-use terms have no reference label; the SOC→category crosswalk is ours; the test set is small for rare categories (wide CIs, some n < 5). Results measure agreement with NCI's grouping, not clinical correctness.
8. **NLP coverage.** 42% of distinct terms are abstained (mostly rare terms). Anchors encode the authors' judgement and a different anchor set would give different results.
9. **No FAERS ground truth yet.** Accuracy on FAERS terms awaits blinded annotation by qualified reviewers.
10. **Question answering.** The templated end-to-end test uses vocabulary terms, so real questions (misspelt reactions, several drugs, comparisons, time) will do worse; no set of real user questions with independent answers exists yet. Only one question type per drug–reaction pair is supported; comparisons ("A vs B"), doses and time trends are not. Brand aliases are learnt from FAERS verbatim names, not RxNorm. The FAQ benchmark is small (52 questions) and BM25 was beaten by TF-IDF on it. The optional generative model is unevaluated because no weights were available.
11. **External evidence.** 150 drugs; long label sections truncated at 25 passages; exact-text label check; abstracts only; snapshot dated 2026-10-20; gold relevance is section-level, so a useful passage outside the gold section counts as a miss; the generative model is small and evaluated on one task.

---

## 9. Licensing, ethics and data governance

* FAERS data are public, de-identified U.S. government data; FDA disclaimers apply.
* SapBERT: Apache-2.0; PubMedBERT: MIT. All software dependencies are permissively licensed.
* MedDRA® (ICH trademark): FAERS PTs are used as distributed by the FDA; no MedDRA hierarchy is used or redistributed; the categories are project-defined. CTCAE is used locally as an evaluation reference only.
* FDA's README/FAQ/ASC_NTS are public U.S. government documents and are quoted with attribution. CTCAE v5.0 definitions are NCI text shown with attribution as a glossary; MedDRA codes from the CTCAE file are not displayed. Confirm with your institution that this use of CTCAE is acceptable before publishing the dashboard.
* Local models: Qwen2.5-0.5B-Instruct (Apache-2.0) and BGE-small-en-v1.5 (MIT); weights not redistributed. openFDA, DailyMed and fda.gov content is public U.S. government information; PubMed abstracts are shown with citation and link for academic use and the corpus should not be republished.
* No patient-level re-identification is attempted; outputs are aggregated.
* Model outputs are not clinical advice and are presented as provisional.

---

## 10. Conclusion and future work

The system turns a raw FAERS quarter into a reproducible, audited analysis with honest statistical screening and a genuinely useful, measured NLP layer. Future work: multi-quarter ingestion and time-to-signal analysis; Bayesian shrinkage (EBGM/MGPS); RxNorm drug normalisation; fuzzy duplicate detection; positive/negative control sets (e.g. OMOP/EU-ADR) to estimate signal-detection performance; collecting blinded annotations from pharmacists to evaluate on FAERS terms; with licensed MedDRA access, evaluating against official SOC assignments; a held-out set of real user questions with pharmacist-written answers to evaluate question answering end to end; hybrid lexical+dense passage retrieval; and evaluating a local generative SLM behind the existing verifier.

---

## References

* Bate A, Evans SJW. Quantitative signal detection using spontaneous ADR reporting. *Pharmacoepidemiol Drug Saf.* 2009;18:427–436.
* Bate A, Lindquist M, Edwards IR, et al. A Bayesian neural network method for adverse drug reaction signal generation. *Eur J Clin Pharmacol.* 1998;54:315–321.
* DuMouchel W. Bayesian data mining in large frequency tables, with an application to the FDA spontaneous reporting system. *Am Stat.* 1999;53:177–190.
* Evans SJW, Waller PC, Davis S. Use of proportional reporting ratios (PRRs) for signal generation from spontaneous adverse drug reaction reports. *Pharmacoepidemiol Drug Saf.* 2001;10:483–486.
* Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS* 2020;33:9459–9474.
* Xiao S, Liu Z, Zhang P, Muennighoff N. C-Pack: packaged resources to advance general Chinese embedding (BGE). arXiv:2309.07597, 2023.
* Qwen Team. Qwen2.5 technical report. arXiv:2412.15115, 2024.
* Kass-Hout TA, Xu Z, Mohebbi M, et al. OpenFDA: an innovative platform providing access to a wealth of FDA's publicly available data. *J Am Med Inform Assoc.* 2016;23:596–600.
* Robertson S, Zaragoza H. The probabilistic relevance framework: BM25 and beyond. *Found Trends Inf Retr.* 2009;3(4):333–389.
* Gu Y, Tinn R, Cheng H, et al. Domain-specific language model pretraining for biomedical natural language processing. *ACM Trans Comput Healthc.* 2021;3(1):1–23.
* Hauben M, Aronson JK. Defining 'signal' and its subtypes in pharmacovigilance based on a systematic review of previous definitions. *Drug Saf.* 2009;32:99–110.
* Liu F, Shareghi E, Meng Z, Basaldella M, Collier N. Self-alignment pretraining for biomedical entity representations. *NAACL-HLT* 2021:4228–4238.
* Norén GN, Hopstadius J, Bate A. Shrinkage observed-to-expected ratios for robust and transparent large-scale pattern discovery. *Stat Methods Med Res.* 2013;22:57–69.
* van Puijenbroek EP, Bate A, Leufkens HGM, et al. A comparison of measures of disproportionality for signal detection in spontaneous reporting systems for adverse drug reactions. *Pharmacoepidemiol Drug Saf.* 2002;11:3–10.
* U.S. FDA. FAERS Quarterly Data Extract Files — README / ASC_NTS. 2025.
* National Cancer Institute. Common Terminology Criteria for Adverse Events (CTCAE) v5.0. 2017.

---

## Appendix A. Reproducing every number in this report

```bash
cd backend
python -m pvsys.pipeline all        # writes data/processed/* and data/nlp/*
python -m pvsys.pipeline evaluate   # re-prints Table 3
python -m pvsys.pipeline rag-eval   # Tables 4-5 and the end-to-end result (~3 min on CPU)
pytest -q                           # 101 tests
# external evidence (from the project root, Windows): scripts\build_evidence.ps1  -> data/evidence/evaluation_evidence.json
```
Numbers are read from `data/processed/{manifest,dedup_stats,quality,signals_summary,eda_snapshot}.json` and `data/nlp/{provenance,evaluation_reference,evaluation_rag,kb_manifest}.json`.
