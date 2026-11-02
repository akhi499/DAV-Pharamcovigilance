// Typed client for the FastAPI backend. All requests go through /api (proxied by Vite).

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function apiGet<T>(path: string, params?: Record<string, string | number | boolean | undefined | null>): Promise<T> {
  const qs = new URLSearchParams();
  Object.entries(params ?? {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  });
  const url = `/api${path}${qs.toString() ? `?${qs}` : ""}`;
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError(0, "The analysis server is not reachable. Start the backend (scripts/run.ps1) and reload.");
  }
  if (!res.ok) throw new ApiError(res.status, await errorText(res));
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await errorText(res));
  return res.json() as Promise<T>;
}

async function errorText(res: Response): Promise<string> {
  try {
    const j = await res.json();
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return j.detail.map((d: { msg: string }) => d.msg).join("; ");
  } catch {
    /* fall through */
  }
  return `Request failed (${res.status})`;
}

// ---------- types ----------
export interface Kpis {
  cases: number; serious: number; deaths: number; hospitalizations: number; probable_duplicates: number;
  expedited: number; literature: number; first_fda_date: string; last_fda_date: string; median_age: number;
  drug_rows: number; distinct_suspect_drugs: number; reaction_rows: number; distinct_reaction_terms: number;
}
export interface DedupStats {
  demo_rows_raw: number; rows_removed_deleted: number; rows_removed_older_versions: number;
  cases_after_dedup: number; probable_duplicates_flagged: number; reaction_rows_raw: number; reactions_rows: number;
}
export interface SignalsSummary {
  analysis_population_cases: number; drug_event_pairs_observed: number; pairs_tested_min_cases: number;
  min_cases: number; ror_signals: number; prr_signals: number; ic_signals: number; all_three_methods: number;
  haldane_corrected_rows: number; q_below_0_05: number; possible_indication_bias_rows: number;
  distinct_suspect_drugs: number; distinct_pts: number; criteria?: Record<string, string>; disclaimer?: string;
}
export interface Overview {
  kpis: Kpis; dedup: DedupStats | null; signals_summary: SignalsSummary | null;
  manifest: { quarter: string; source_zip: string; source_sha256: string; ingested_at: string; deleted_case_ids: number };
  disclaimer: string;
}
export interface QualityCheck { check: string; dimension: string; value: number | null; unit: string; status: "pass" | "warn" | "info"; detail: string }
export interface Quality { n_cases: number; field_missing_pct: Record<string, number>; checks: QualityCheck[]; summary: Record<string, number> }
export interface DistRow { level: string; label?: string; cases: number; serious: number; deaths: number; serious_pct: number; death_pct: number }
export interface Demographics {
  sex: DistRow[]; age_group: DistRow[]; reporter_type: DistRow[];
  countries: { country: string; cases: number }[]; outcomes: { code: string; label: string; cases: number }[];
  pyramid: { age_group: string; sex: string; cases: number }[]; age_histogram: { bin_start: number; cases: number }[];
  serious_by_sex: { chi2: number; p_value: number; cramers_v: number; note: string; table: { sex: string; s: number; ns: number }[] };
  note: string;
}
export interface TrendRow { date: string; cases: number; serious: number }
export interface DrugCount { drug: string; cases: number; serious: number; deaths: number }
export interface ReactionCount { pt: string; pt_key: string; cases: number; serious: number; deaths: number }
export interface DrugProfile {
  drug: string; cases: number; serious: number; deaths: number; hospitalizations: number; median_age: number | null;
  demographics: { sex: string; age_group: string; cases: number }[];
  top_reactions: { pt: string; pt_key: string; cases: number }[]; top_indications: { indi_pt: string; cases: number }[];
}
export interface SignalRow {
  drug: string; pt: string; pt_key: string; a: number; b: number; c: number; d: number; n_drug: number; n_pt: number;
  expected: number; ror: number; ror_lower: number; ror_upper: number; prr: number; prr_lower: number; prr_upper: number;
  chi2: number; p_value: number; q_value: number; ic: number; ic025: number; ror_signal: boolean; prr_signal: boolean;
  ic_signal: boolean; methods_flagging: number; haldane_applied: boolean; indication_overlap: number;
  possible_indication_bias: boolean; nlp_category?: string | null; nlp_confidence?: number | null;
}
export interface SignalsPage { total: number; limit: number; offset: number; rows: SignalRow[]; disclaimer: string }
export interface Stratum { stratum: string; a: number; b: number; c: number; d: number; ror: number; ror_lower: number; ror_upper: number; prr: number; estimable: boolean; haldane_applied: boolean }

export interface Category { id: string; label: string; description: string; anchors: string[]; terms: number; case_mentions: number }
export interface CategoriesResp { categories: Category[]; abstained: { terms: number; case_mentions: number }; disclaimer: string }
export type ReviewStatus = "unreviewed" | "accepted" | "corrected" | "rejected" | "uncertain";
export interface Prediction {
  pt: string; pt_key: string; cases: number; prediction: string; top_category: string; second_category: string;
  score: number; second_score: number; margin: number; confidence: number; abstain_reason: string; nearest_anchor: string;
  baseline_prediction: string; baseline_top_category: string; baseline_score: number; agrees_with_baseline: boolean;
  prediction_source: string; model_id: string; model_revision: string; taxonomy_version: string; run_id: string;
  review_status: ReviewStatus; reviewed_category?: string | null; reviewer?: string; reviewed_at?: string;
}
export interface PredictionsPage { total: number; rows: Prediction[]; disclaimer: string }
export interface SearchHit { pt: string; pt_key: string; cases: number; similarity: number; prediction: string; top_category: string; review_status?: ReviewStatus }
export interface SearchResp { query: string; mode: "neural" | "lexical-fallback"; results: SearchHit[] }
export interface ConceptSignals {
  query: string; mode: string; terms: SearchHit[];
  rows: Pick<SignalRow, "drug" | "pt" | "pt_key" | "a" | "ror" | "ror_lower" | "ror_upper" | "prr" | "ic025" | "methods_flagging" | "possible_indication_bias">[];
  by_drug: { drug: string; flagged_terms: number; total_cases: number; max_ror_lower: number }[]; disclaimer: string;
}
export interface Metrics {
  n: number; coverage: number; answered: number; abstained: number; selective_accuracy: number | null;
  selective_accuracy_95ci: [number, number] | null; macro_f1_answered: number | null;
  top1_accuracy_no_abstention: number; top1_accuracy_95ci: [number, number] | null;
  per_category: Record<string, { support: number; precision: number | null; recall: number | null; abstained: number }>;
}
export interface SystemEval {
  status: string; encoder?: string; thresholds?: { min_score: number; min_margin: number; source: string };
  test?: Metrics; test_in_faers?: Metrics; risk_coverage_test?: { coverage: number; accuracy: number }[];
  confusion_test?: { labels_gold: string[]; labels_pred: string[]; matrix: number[][] };
  predicts?: string; test_accuracy?: number; reason?: string;
}
export interface ReferenceEval {
  status?: string; reason?: string; reference: string; taxonomy_version: string;
  protocol: { seed: number; dev_fraction: number; threshold_objective: string };
  dataset: Record<string, unknown> & { evaluation_terms: number; dev_terms: number; test_terms: number; test_terms_in_faers_2025q1: number; excluded_equal_to_anchor: number };
  systems: Record<string, SystemEval>; caveats: string[]; created_at: string;
}
export interface Provenance {
  run_id: string; prediction_source: string; created_at: string; n_terms: number; n_predicted: number; n_abstained: number;
  abstain_reasons: Record<string, number>; case_weighted_coverage: number; agreement_with_lexical_top1: number;
  model: Record<string, string | number> | null; baseline: Record<string, string>;
  thresholds: { min_score: number; min_margin: number; temperature: number; source: string };
  taxonomy: { version: string; hash: string; n_categories: number; n_anchors: number; note: string };
  data_manifest: { quarter: string; source_zip: string; source_sha256: string } | null;
}
export interface NlpStatus {
  available: boolean; neural_encoder: { available: boolean; weights_present: boolean; deps_installed: boolean; model_id: string };
  provenance: Provenance | null; evaluation_reference: ReferenceEval | null;
  evaluation_annotations: { status: string; message?: string; n_gold_terms?: number; cohens_kappa_double_annotated?: number | null; neural?: Metrics; lexical?: Metrics } | null;
  disclaimer: string;
}

// ---------- question answering (RAG) ----------
export interface AnswerSentence { text: string; cites: string[] }
export interface StatEvidence {
  id: string; kind: "statistic" | "aggregate"; drug?: string; pt?: string; pt_key?: string; a?: number; b?: number; c?: number; d?: number;
  expected?: number; ror?: number; ror_lower?: number; ror_upper?: number; prr?: number; ic025?: number; q_value?: number;
  methods_flagging?: number; possible_indication_bias?: boolean; indication_overlap_pct?: number; haldane_applied?: boolean;
  tested?: boolean; min_cases?: number; suspect_cases?: number; pairs_tested?: number; pairs_flagged_all_three?: number;
  reaction_cases?: number; similarity?: number; link?: string; source: string;
}
export interface DocEvidence { id: string; kind: "document"; chunk_id: string; source: string; title: string; page: number; text: string; score: number | null }
export type Evidence = StatEvidence | DocEvidence;
export interface LinkedDrug { drug: string; text: string; match: "exact" | "alias" | "prefix" | "fuzzy"; cases: number; score: number }
export interface LinkedReaction {
  phrase: string; pt: string; pt_key: string; similarity: number; match: "exact" | "semantic"; cases: number;
  related: { pt: string; pt_key: string; similarity: number; cases: number }[];
}
export interface GeneratedSummary {
  status: "unavailable" | "not_requested" | "verified" | "rejected" | "error" | "skipped"; reason?: string | null; text?: string | null;
  rejected_text?: string | null; problems?: string[]; model_id?: string; seconds?: number;
}
export interface AskResponse {
  question: string; status: "answered" | "abstained" | "needs_clarification"; intent: "pair" | "drug" | "reaction" | "documentation" | "clarify";
  answer: AnswerSentence[];
  understanding: {
    drugs: LinkedDrug[]; reactions: LinkedReaction[]; link_mode: string;
    unlinked_phrases: { phrase: string; best_candidate: string | null; best_similarity: number | null; threshold: number }[];
    ambiguous_drugs: { text: string; candidates: { drug: string; cases: number }[] }[];
  };
  evidence: Evidence[]; generated_summary: GeneratedSummary; established?: Established; provenance: Record<string, string | number | null>;
  elapsed_ms: number; disclaimer: string;
}
export interface RagStatus {
  available: boolean; rag_version: string; examples: string[]; link_thresholds: Record<string, number>; evaluation_available: boolean;
  knowledge_base: { kb_version: string; n_chunks: number; by_source: Record<string, number>; content_sha256: string; sources: Record<string, { title: string; pages?: number; sha256?: string; n?: number }> } | null;
  generator: { available: boolean; weights_present: boolean; deps_installed: boolean; enabled: boolean; model_dir: string; model_id: string | null };
  disclaimer: string;
}
export interface RankMetrics {
  n: number; recall_at_1: number; recall_at_3: number; recall_at_5: number; recall_at_10: number; mrr: number;
  recall_at_1_95ci: [number, number] | null; mrr_95ci: [number, number] | null;
}
export interface RagEvaluation {
  created_at: string; caveats: string[];
  faq_retrieval: { status: string; reason?: string; queries?: number; corpus_size?: number; corpus?: string; systems?: Record<string, Partial<RankMetrics> & { note?: string }>;
    paired_difference_tfidf_minus_bm25?: Record<string, { mean: number; "95ci": [number, number] }> };
  reaction_linking: { status: string; reason?: string; queries?: number; dev?: number; test?: number; candidates?: number;
    systems?: Record<string, { status: string; chosen_variant?: string; test?: RankMetrics; dev_mrr_by_variant?: Record<string, number>;
      at_production_threshold?: { threshold: number; test_coverage: number; test_precision_at_1_when_accepted: number | null; note: string } }>;
    examples_test?: { definition: string; gold: string; sapbert_top1: string; rank: number }[] };
  end_to_end: { status: string; questions?: number; pairs_sampled?: number; exact_retrieval_generic_templates?: number; exact_retrieval_95ci?: [number, number] | null;
    per_template?: Record<string, { n: number; exact_retrieval: number }>; failure_reasons?: Record<string, number>;
    deterministic_answers_passing_grounding_check?: number | null; mean_latency_ms?: number; note?: string; link_mode?: string };
  generator: { status: string; reason: string };
}

// ---------- external clinical evidence ----------
export type SourceType = "label" | "literature" | "regulatory";
export interface ExternalEvidence {
  id: string; kind: "external"; chunk_id: string; doc_id: string; source_type: SourceType; drugs: string; section_key: string; section: string;
  title: string; publisher: string; date: string; version: string; url: string; identifier: string; text: string;
  score: number | null; dense_similarity: number | null; bm25: number | null;
}
export interface ExternalGenerated {
  status: "verified" | "rejected" | "insufficient" | "unavailable" | "skipped" | "not_requested" | "error";
  text?: string | null; rejected_text?: string | null; problems?: string[]; reason?: string; model_id?: string; seconds?: number;
  citation_support?: number | null;
}
export interface Established {
  status: "answered" | "partial" | "insufficient" | "unavailable" | "not_applicable" | "not_requested";
  reason?: string; retrieval_mode?: string; threshold?: number; top_similarity?: number | null; drugs_searched?: string[]; not_indexed?: string[];
  label_mentions?: { drug: string; pt: string; found: boolean; sections: string[]; label_indexed: boolean; cite?: string }[];
  sentences?: { text: string; cites: string[]; generated?: boolean }[]; evidence?: ExternalEvidence[]; generated?: ExternalGenerated;
  index?: { index_version: string; created_at: string; documents: number; chunks: number; content_sha256: string };
}
export interface EvidenceStatus {
  available: boolean; failures: number | null; sufficiency_threshold: number; evaluation_available: boolean;
  index: { created_at: string; documents: number; chunks: number; documents_by_type: Record<string, number>; chunks_by_type: Record<string, number>;
    drugs_with_label: number; content_sha256: string; embedding: { model_id?: string; revision?: string; status?: string; reason?: string; seconds?: number }; retrieval_modes: string[] } | null;
  sources: { started_at: string; finished_at: string; drugs_requested: number; stats: Record<string, Record<string, number>> } | null;
  embedder: { available: boolean; model_id: string | null }; generator: { available: boolean; model_id: string | null };
}
export interface EvidenceDoc { doc_id: string; source_type: SourceType; title: string; publisher: string; date: string; version: string; url: string; identifier: string; drugs: string; chunks: number }
export interface EvidenceDocs { total: number; by_type: Record<string, number>; rows: EvidenceDoc[] }
export interface IrMetrics { n: number; recall_at_1: number; recall_at_3: number; recall_at_5: number; recall_at_10: number; mrr: number; ndcg_at_10: number; recall_at_5_95ci: [number, number] | null; mrr_95ci: [number, number] | null }
export interface EvidenceEvaluation {
  created_at: string; drugs_with_label: number; dev_drugs: number; test_drugs: number; caveats: string[];
  retrieval: { modes: string[]; test_queries: number; within_drug: Record<string, Record<string, IrMetrics>>; whole_index: Record<string, Record<string, IrMetrics>>;
    per_topic_hybrid_within_drug_recall_at_5?: Record<string, number> };
  sufficiency: { status: string; reason?: string; threshold?: number; test?: { balanced_accuracy: number; answerable_kept: number; unanswerable_rejected: number; n: number } };
  generation: { status: string; reason?: string; n?: number; boxed_warning_accuracy?: number; boxed_warning_accuracy_95ci?: [number, number] | null;
    baseline_retrieval_only_accuracy?: number; verifier_pass_rate?: number; explicit_insufficient_rate?: number; citation_support_mean?: number | null;
    seconds_median?: number; rejection_reasons?: Record<string, number>; generator?: { model_id: string | null } };
  concordance: { status: string; caveat?: string; most_reported?: Concordance; most_disproportionate?: Concordance };
  mode_selection?: { selected_mode: string; criterion: string; dev: Record<string, { recall_at_1: number; mrr: number; n: number }> };
}
export interface Concordance { signals_checked: number; share_reaction_text_in_label: number | null; share_in_label_excluding_product_use_terms: number | null; product_use_terms: number }
