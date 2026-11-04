# Viva guide — AI-Based Drug Safety and Adverse Effect Analysis System

Short, defensible answers grounded in what the system actually does and the numbers it actually produced. Figures refer to FAERS 2025 Q1.

## 1. Two-minute pitch

"We built a pharmacovigilance workbench on the FDA's FAERS 2025 Q1 data. A reproducible pipeline takes the 400,514 raw reports, applies the FDA's case-version rules, flags 7,784 probable duplicates and audits data quality. We then screen 241,675 drug–reaction pairs with ROR, PRR and the information component, with sparse-cell correction, confidence intervals, q-values and an indication-bias flag, and it recovers known associations such as metformin–lactic acidosis. On top of that, a biomedical language model, SapBERT, running on a normal CPU, groups the 12,678 reaction terms into 23 body-system categories and abstains when unsure; it also lets you search reactions by meaning. On an independent NCI reference it scores 62.7% against 43.1% for a lexical baseline. Finally, you can simply ask a question — 'does Ozempic cause stomach problems?' — and a retrieval-augmented question-answering module finds the drug and the MedDRA term you mean, retrieves the stored statistics and FDA's own documentation, and answers only from that evidence, with citations and the causality caveat. 99.3% of templated questions retrieve exactly the right statistics. The model never touches the statistics, and everything it produces is labelled provisional."

## 2. Live demo script (≈6 minutes)

1. `run.cmd` → dashboard opens. **Overview**: case-flow funnel (400,514 → 392,729), serious 56.1%, deaths 8.1%, daily receipts with weekend dips. Read the disclaimer aloud.
2. **Data quality**: age missing 39%, weight 83% — "we disclose, we never impute".
3. **Demographics**: serious share rises with age; sex χ² is significant but Cramér's V = 0.047 — "significance is not importance at n = 400k".
4. **Signal detection**: filter drug `SEMAGLUTIDE`, reaction `pancrea` → ROR 12.5 (9.8–15.9). Open the row: 2×2 table, PRR, IC025, q-value, subgroup forest plot by sex.
5. Untick "Hide likely indication bias", drug `DUPILUMAB` → atopic dermatitis appears with "Indication?" (98% overlap). Tick "Hide product-use and error terms" to show the NLP-assisted clean-up.
6. **Reaction categories → Search by meaning**: "liver damage" → Liver injury, Hepatocellular injury. **Signals for a concept**: "pancreas inflammation" → Pancreatitis → semaglutide, tirzepatide.
7. **Term predictions**: show an abstention (Pancreatitis, ambiguous), then review a wrong one (Type IIa hyperlipidaemia → correct to Metabolism). Explain that this review is logged but not scored.
8. **Model & evaluation**: provenance (revision, weights hash), accuracy vs baselines, accuracy–coverage curve, confusion matrix, "no annotations yet → no FAERS accuracy claimed".
9. **Ask the data**: click "Does Ozempic cause stomach problems?" → point to *How the question was read* (OZEMPIC → SEMAGLUTIDE via a FAERS-learnt alias; "stomach problems" → Dyspepsia, similarity 0.73), the citation chips, the evidence table (click *details* → opens Signal detection), the CTCAE definition and the caveat. Then "Is dupilumab associated with atopic dermatitis?" (indication-bias warning), "Why are some ages very large?" (quotes FDA FAQ 6), "insulin and low blood sugar" (ambiguity reported, not guessed), and "what is the weather today" (abstains). Finish on the *How well it works* tab.

## 3. Likely questions and answers

### Data and pipeline

**Why Parquet + DuckDB instead of MySQL/PostgreSQL?**
The workload is analytical (scans, group-bys over millions of rows) and single-user. Parquet is columnar and compressed (~90 MB for everything), DuckDB queries it in-process with no server to install, and the whole pipeline runs in about a minute on a laptop.

**How did you handle duplicates and case versions?**
FDA rule: drop CASEIDs on the delete list (1 removed), then keep the latest FDA_DT per CASEID, ties by highest PRIMARYID. In this quarter every CASEID already appeared once, so 0 versions were removed — the rule is still applied and unit-tested with multi-version synthetic data. Separately, reports with different CASEIDs but identical sex, age, event date, country, suspect-drug set and reaction set are flagged as probable duplicates (7,784) and excluded from signal detection.

**Why not impute missing age or weight?**
Missingness in FAERS is not random (consumer reports are less complete), so imputation would inject assumptions into a hypothesis-generating analysis. We report completeness instead.

**Why are 56% of cases "serious"?**
Manufacturers must expedite serious reports and many non-serious ones go unreported, so FAERS over-represents serious events. It is a property of the reporting system.

### Statistics

**Explain ROR and PRR.**
2×2 table of cases: a = drug & reaction, b = drug & other reactions, c = other drugs & reaction, d = neither. ROR = (a/b)/(c/d), an odds ratio of reporting. PRR = [a/(a+b)]/[c/(c+d)], the proportion of the drug's reports with the reaction relative to everyone else. Both get log-normal 95% CIs.

**What is the Haldane–Anscombe correction and when do you apply it?**
If any cell is zero the ratio or its variance is undefined; adding 0.5 to all four cells makes them finite. We apply it only when a cell is zero (4,201 rows) and flag those rows as unstable.

**What thresholds did you use?**
ROR lower 95% CI > 1 with a ≥ 3; Evans criteria for PRR (PRR ≥ 2, χ² ≥ 4, a ≥ 3); IC025 > 0. They are conventional screening rules, configurable in `SignalConfig`.

**71% of tested pairs are flagged — isn't that useless?**
It shows why screens are only hypothesis generators. With ~390,000 cases, a specific drug–term pairing quickly exceeds expectation. We prioritise with the minimum-case filter, confidence-interval width, IC025, q-values, the indication-bias flag and (optionally) by hiding product-use terms. Final assessment is clinical.

**Does a high ROR mean the drug is dangerous?**
No. It means the pair is reported together disproportionately. Without the number of exposed patients we cannot compute incidence or risk; reporting can be driven by media, litigation, the Weber effect or the indication itself.

**What is confounding by indication and how do you detect it?**
The reaction is part of the disease being treated (dupilumab for atopic dermatitis → "Dermatitis atopic"). We compute the share of the a cases that list the reaction as an indication; ≥ 20% flags it (16,092 pairs).

**Why both frequentist and Bayesian measures?**
IC (BCPNN) shrinks estimates for small counts; it is the WHO-UMC standard and is more conservative (173,684 vs 192,163 ROR flags). Agreement of all three methods (172,074) is a simple robustness criterion.

**What about multiple testing?**
We report Benjamini–Hochberg q-values across all 241,675 tested pairs. The screening flags themselves follow the published conventions and are not FDR-controlled; we state this as a limitation.

**How do you know the implementation is correct?**
Unit tests compare every formula with hand-computed textbook values; an end-to-end test on a synthetic quarter checks exact 2×2 cells and the Haldane-corrected ROR; and on real data known label associations are recovered (metformin–lactic acidosis, levofloxacin–tendon rupture, montelukast–nightmares, isotretinoin–depression).

### NLP / language model

**What NLP is in the project, concretely?**
A pretrained biomedical transformer, SapBERT (BERT-base, 110 M parameters), turns every reaction term into a 768-dimensional embedding. We use those embeddings for (1) zero-shot categorisation of terms into 23 body-system categories with abstention, (2) semantic search over the reaction vocabulary and (3) concept-level signal lookup, plus an optional filter to hide product-use terms.

**Why SapBERT and not ChatGPT/an LLM?**
Constraints: local, CPU-only, no paid API, auditable. SapBERT was trained precisely so that synonyms of the same biomedical concept embed together, which is our task. Generative models need GPUs or APIs and produce free text that is harder to audit — which is why our RAG module generates from templates and quotes, and treats any generative model as optional and verified.

**Is it an "SLM"?**
It is a small language model in the encoder sense — 110 M parameters, 438 MB, runs on CPU in about two minutes for all terms. It is not generative. The question-answering module can additionally use a small *generative* model (e.g. Qwen2.5-0.5B-Instruct, Apache-2.0) if its weights are placed in `models/generator/`; we could not download one in this project, so we say plainly that path is unevaluated.

### Question answering (RAG)

**Is it really RAG if there is no large language model?**
RAG = retrieve evidence for the query, then generate the answer conditioned on it. We retrieve from two sources — the statistics table (structured) and FDA documentation, project methods notes and CTCAE definitions (942 passages, BM25) — and generate the answer from what was retrieved. Our generator is deterministic (templates for statistics, verbatim extracts for documents) because in drug safety a fluent but wrong number is worse than a plain sentence. A neural generator can be plugged in; its text is shown only if a verifier confirms every number, citation and quotation is in the evidence and there is no causal claim.

**Where is the language model in it?**
SapBERT links free phrases to MedDRA terms ("suicidal thoughts" → Suicidal ideation, 0.95; "low blood sugar" → Blood glucose decreased, 0.91) and finds related terms. Without it, the lexical baselines find the right term only ~12% of the time on our linking test versus 48% (top-10: 27–29% vs 86%).

**How do you stop it from making things up?**
(1) Every sentence is filled from retrieved values and carries evidence ids; (2) the threshold (0.70) makes it say "not understood" rather than guess; (3) ambiguous drug words ("insulin") are reported, not resolved; (4) it abstains if no drug, term or passage is found; (5) an automatic check confirmed every number/citation/quotation in all 805 evaluation answers is traceable; (6) the causality caveat is attached to every statistical answer.

**Why 0.70?**
Correct lay links scored 0.72–0.96 and noise words ≤ 0.66 in a probe; on the CTCAE dev/test split, 0.70 accepts ~75% of formal definitions with ~58% precision — definitions are much harder than short phrases, so that is a lower bound.

**How did you evaluate it?**
Three independent tests, none labelled by us: FDA's 52 FAQ questions vs FDA's own answers (BM25 top-1 44.2%, top-10 80.8%, random 0.8%); NCI definitions → correct FAERS term among 12,678 (SapBERT top-1 47.8%, top-10 85.6% vs ~12% / ~28% lexical); 750 templated questions over 150 random real signal pairs (99.3% retrieve exactly the stored statistics). The last test found three real parsing bugs; fixing them raised it from 78% to 99.3%.

**TF-IDF beat BM25 on the FAQ test — why didn't you switch?**
Because that was the only documentation benchmark: switching after seeing it would be tuning on the test set. We report the difference (+9.6 points, paired-bootstrap CI 1.9–19.2) and list a held-out question set + switch/hybrid as future work.

**What can't it answer?**
Comparisons between drugs, doses, time trends, multi-step questions, and documentation phrased very differently from FDA's wording (BM25 is lexical). Brand names come from FAERS verbatim names, not RxNorm.

**How does the classifier work?**
Each category has a description and 6–14 anchor phrases we wrote (251 total). We average each category's anchor embeddings into a centroid. A term goes to the most similar centroid unless the best similarity is below 0.46 or the gap to the second best is below 0.02 — then it abstains.

**Where do the thresholds come from?**
A development split (30% of the CTCAE reference, 213 terms) — grid search for the best accuracy while answering at least 70%. The test split (499 terms) was used only for the final numbers. The centroid rule itself was also chosen on dev (66.2% vs 61.0% for nearest-anchor).

**What are your results?**
Test top-1 accuracy 62.7% (95% CI 58.5–66.7) vs 43.1% for character n-gram TF-IDF and 15.8% for majority class; when abstention is allowed, 71.5% accuracy on 65.3% of terms. On test terms that also occur in FAERS: 67.8%.

**Why CTCAE as ground truth? Isn't that MedDRA?**
We cannot label data ourselves without bias, and official MedDRA hierarchies need a licence. CTCAE is a public NCI terminology in which experts placed each term under an organ-system heading. We map those headings to our categories with a documented crosswalk and remove any term identical to an anchor to avoid leakage. It is a proxy — it measures agreement with NCI, not correctness on FAERS.

**So what is the model's accuracy on FAERS?**
Unknown, and we say so. We built a blinded annotation workflow (export without model output, two annotators, import, Cohen's κ, scoring). Until annotations exist, the dashboard claims no FAERS accuracy.

**Why aren't dashboard reviews used for accuracy?**
Reviewers see the model's answer first, which anchors them — that would inflate accuracy. Reviews only change a prediction's status (accepted, corrected, rejected, unsure) and are logged with reviewer and time.

**What does "confidence" mean?**
A softmax over category similarities. It ranks predictions but is not a calibrated probability of being right; the accuracy–coverage curve is the honest view of how confidence relates to correctness.

**Give an example of abstention helping.**
"Pancreatitis" scored 0.67 for hepatobiliary and 0.66 for gastrointestinal — margin 0.009, so it abstained. Choosing hepatobiliary would have been wrong (MedDRA files it under gastrointestinal).

**And an error?**
"Type IIa hyperlipidaemia" → hepatobiliary (should be metabolism). It is shown as provisional and a reviewer can correct it.

**Could the NLP corrupt the statistics?**
No. The statistics modules don't import it (tested), running NLP leaves `signals.parquet` byte-identical (tested), and categories are only joined at display time.

**What if the model weights are missing?**
The NLP stage writes lexical-baseline predictions marked `lexical-fallback`, the API serves precomputed predictions, search reports its mode, and statistics are unaffected.

**Licensing?**
SapBERT Apache-2.0, PubMedBERT MIT, FAERS and FDA's README/FAQ public (quoted with attribution), CTCAE definitions shown with attribution as a glossary (MedDRA codes not shown), no MedDRA hierarchy used or redistributed; categories are ours; pypdf BSD. An optional generator must allow local use (Qwen2.5-0.5B-Instruct: Apache-2.0).

### External clinical evidence

**Where does "established evidence" come from?** FDA labels (openFDA SPL sections; DailyMed picks the manufacturer's most-revised label set), FDA Drug Safety Communications and PubMed reviews/meta-analyses since 2015, for the 150 most-reported FAERS drugs – 601 documents, 8,590 passages, each with section, version/date, identifier and link.

**What does "vector retrieval" mean here?** Each passage is turned into a 384-number vector by BGE-small; the question is embedded the same way and the closest passages by cosine similarity are returned (exact search – 8,590 vectors need no special index).

**How do you know retrieval works?** FDA codes every label passage by section. We ask 886 test questions like "Who should not take Ozempic?" whose correct answer must come from that drug's Contraindications section: the right section is ranked first 81% of the time, in the top 5 95% (keyword search 51% / 71%). Lay questions are where vectors win: 68% vs 19%.

**Why not hybrid search?** We tried it; on the development drugs dense MRR 0.85 beat hybrid 0.75 and BM25 0.59, so dense was chosen before looking at the test drugs, and the test drugs confirmed it.

**When does it refuse?** Below a similarity of 0.73 (chosen on development drugs): it answers 93% of answerable questions and declines 92% of questions no source can answer; also when the drug isn't indexed.

**Did the generative model work?** Not well. In the first pilot Qwen2.5-0.5B got the boxed-warning question right 35% of the time vs 97.5% for simply checking which section the retrieved passage came from, and only 43% of its drafts passed our source checks. So it is off by default – an honest negative result. The checks (citations exist, numbers/quotes in the source, each sentence supported by its cited passage) stop unsupported text reaching the user.

**Did you train or fine-tune any model?** No. All models are pretrained and used as downloaded (zero-shot); only decision settings (thresholds, retrieval mode, routing weight) were chosen on the 38 development drugs, and every result is reported on the 89 held-out test drugs. FAERS has no labelled training data, and the research question is whether off-the-shelf local models are safe to use.

**Extractive or generative – what did the comparison show?** Same three passages for every system, 89 test drugs: quoting the evidence answered "Does X have a boxed warning?" correctly 92.1% of the time; Qwen 0.5B 43.8%, Qwen 1.5B 69.7%, SmolLM2 1.7B 49.4%, Qwen 3B (4-bit) no better than 1.5B. The errors were confident yes/no flips ("No, there is no boxed warning [L1]" citing the boxed warning itself) that a word-overlap verifier passes.

**How did you fix that?** Two extra checks: a structure check (a draft can't contradict which label section the passage came from) and an entailment model (nli-deberta-v3-small) that rejects sentences their cited passage contradicts. Qwen 1.5B's verified false "no" answers fell from 16 to 0 and system accuracy rose from 76.4% to 93.3%, level with extractive answering.

**Did the re-ranker and routing help?** Routing (mapping "who should not take" to Contraindications) raised top-1 on re-worded questions from 54% to 67%; on our own templates it is 100%, but that is by construction, so we don't claim it. The cross-encoder re-ranker made retrieval worse (65% → 52%) and was rejected on development drugs – a negative result we report.

**How do you separate reports from evidence?** Two panels: "Observed in FAERS reports" (counts and disproportionality – not causal) and "Established clinical evidence" (what the label/literature states, with a direct check whether the label mentions the reaction).

**Interesting finding?** Of the 300 most-reported strong FAERS signals, about a third (43% excluding product-use terms) are already in the label text; the most extreme ratios are mostly not – they tend to be report clusters, not new drug effects.

### Engineering

**How is the API protected against bad input?**
Typed FastAPI/Pydantic parameters with ranges and length limits, whitelisted sort columns (no SQL built from user text), parameterised DuckDB queries, validated review payloads and reviewer names; errors return 404/409/422/503 with actionable messages.

**Caching and performance?**
Descriptive aggregates are LRU-cached in the API; the frontend caches with TanStack Query; semantic-search query embeddings are cached; question answers are LRU-cached; the QA engine warms up in a background thread at API start (first question ~0.6 s instead of ~30 s); Plotly is split into its own bundle chunk.

**Testing?**
110 backend tests (synthetic FAERS quarter, formula checks, NLP abstention/calibration/review/annotation, retrieval, linking, verifier, generator mechanics with a tiny random model, abstention, independence, API validation), Ruff lint, 18 frontend tests (Vitest + Testing Library), TypeScript check and production build.

## 4. Numbers to remember

| | |
|---|---|
| Raw reports / de-duplicated cases | 400,514 / 400,513 |
| Probable duplicates | 7,784 |
| Analysis population | 392,729 |
| Serious / deaths | 56.1% / 32,408 |
| Suspect drugs / reaction terms | 5,018 / 12,678 |
| Pairs tested / flagged by all three | 241,675 / 172,074 |
| Indication-bias flags | 16,092 |
| SapBERT vs lexical vs majority (test top-1) | 62.7% / 43.1% / 15.8% |
| SapBERT with abstention | 71.5% accuracy at 65.3% coverage |
| FAERS terms assigned / abstained | 7,333 / 5,345 (85.5% of mentions covered) |
| QA knowledge base | 942 passages (52 FDA FAQ, 30 README, 40 ASC_NTS, 9 project notes, 811 CTCAE definitions) |
| FAQ retrieval, BM25 top-1 / top-10 | 44.2% / 80.8% (TF-IDF 53.8% / 82.7%, random 0.8%) |
| Linking NCI definitions, SapBERT vs lexical top-1 | 47.8% vs 11.7–12.4% (top-10 85.6% vs 27–29%) |
| End-to-end exact retrieval | 99.3% of 750 (95% CI 98.7–99.9); brand aliases 55/55 |
| Grounding check | 100% of 805 answers |
| External evidence | 601 docs (135 labels, 424 reviews, 42 FDA DSCs), 8,590 passages |
| Label-section retrieval, dense vs BM25 (top-1 / top-5) | 81.2% / 94.6% vs 51.0% / 70.5%; lay wording 67.7% vs 18.5% top-1 |
| Insufficient-evidence threshold | 0.73; balanced accuracy 92.5% |
| Boxed-warning test (89 drugs), extractive vs generators | 92.1% vs 43.8% (Qwen 0.5B) / 69.7% (Qwen 1.5B) / 49.4% (SmolLM2) / 78.7% (Qwen 3B, 4-bit, routed retrieval) |
| Verifier v2 (structure + entailment), Qwen 1.5B | verified false "no" 16 → 0; system 76.4% → 93.3% (extractive 92.1%) |
| Section routing, re-worded questions | top-1 54.4% → 66.9%; re-ranker 65.1% → 52.0% (rejected) |
| Strong signals already in label | 33.7% (most-reported 300) |

## 5. Limitations to volunteer

No denominator (no risk); disproportionality ≠ causality; one quarter; regex drug normalisation (no RxNorm); exact-match duplicate detection; screening flags not FDR-controlled; NLP reference is a proxy with small per-category samples; 42% of distinct terms abstained; no FAERS ground truth yet; QA evaluated on templated not real user questions, no comparisons/doses/trends, BM25 is lexical and lost to TF-IDF on the FAQ test, small generators make confident polarity errors (kept opt-in, behind the structure + entailment verifier); routing evaluated on author-written questions only.
