import { vi } from "vitest";

export const categories = {
  categories: [
    { id: "hepatobiliary", label: "Hepatobiliary", description: "Liver.", anchors: ["hepatitis"], terms: 300, case_mentions: 9000 },
    { id: "skin", label: "Skin & subcutaneous", description: "Skin.", anchors: ["rash"], terms: 360, case_mentions: 88000 },
  ],
  abstained: { terms: 5000, case_mentions: 200000 },
  disclaimer: "provisional",
};

export const signalRow = {
  drug: "DRUGX", pt: "Hepatitis", pt_key: "hepatitis", a: 12, b: 100, c: 50, d: 10000, n_drug: 112, n_pt: 62, expected: 0.7,
  ror: 23.8, ror_lower: 12.1, ror_upper: 46.6, prr: 21.5, prr_lower: 11, prr_upper: 40, chi2: 180, p_value: 1e-30, q_value: 1e-28,
  ic: 3.9, ic025: 3.1, ror_signal: true, prr_signal: true, ic_signal: true, methods_flagging: 3, haldane_applied: false,
  indication_overlap: 0.3, possible_indication_bias: true, nlp_category: "hepatobiliary", nlp_confidence: 0.91,
};

export const prediction = {
  pt: "Hepatitis", pt_key: "hepatitis", cases: 1200, prediction: "hepatobiliary", top_category: "hepatobiliary", second_category: "infections",
  score: 0.8, second_score: 0.6, margin: 0.2, confidence: 0.95, abstain_reason: "", nearest_anchor: "hepatitis",
  baseline_prediction: "hepatobiliary", baseline_top_category: "hepatobiliary", baseline_score: 0.7, agrees_with_baseline: true,
  prediction_source: "neural", model_id: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext", model_revision: "090663c", taxonomy_version: "pv-cat-1.0",
  run_id: "abc", review_status: "unreviewed",
};

export const routes: Record<string, unknown> = {
  "/api/nlp/categories": categories,
  "/api/signals/summary": {
    analysis_population_cases: 392729, drug_event_pairs_observed: 648090, pairs_tested_min_cases: 241675, min_cases: 3, ror_signals: 1,
    prr_signals: 1, ic_signals: 1, all_three_methods: 172074, haldane_corrected_rows: 1, q_below_0_05: 1, possible_indication_bias_rows: 16092,
    distinct_suspect_drugs: 5018, distinct_pts: 12678,
  },
  "/api/signals": { total: 1, limit: 50, offset: 0, rows: [signalRow], disclaimer: "not causality" },
  "/api/nlp/predictions": { total: 1, rows: [prediction], disclaimer: "provisional" },
  "/api/nlp/status": { available: true, neural_encoder: { available: true, weights_present: true, deps_installed: true, model_id: "m" }, provenance: null, evaluation_reference: null, evaluation_annotations: null, disclaimer: "" },
};

export const askResponse = {
  question: "Is DRUGX associated with hepatitis?", status: "answered", intent: "pair",
  answer: [
    { text: "In FAERS 2025 Q1, DRUGX and “Hepatitis” were reported together in 12 cases (about 0.70 expected if they were unrelated).", cites: ["S1"] },
    { text: "This is a reporting association in a spontaneous-reporting database. It does not show that the drug causes the reaction.", cites: ["D1"] },
  ],
  understanding: {
    drugs: [{ drug: "DRUGX", text: "DRUGX", match: "exact", cases: 112, score: 1 }],
    reactions: [{ phrase: "hepatitis", pt: "Hepatitis", pt_key: "hepatitis", similarity: 1, match: "exact", cases: 62, related: [{ pt: "Hepatitis acute", pt_key: "hepatitis acute", similarity: 0.9, cases: 5 }] }],
    unlinked_phrases: [], ambiguous_drugs: [], link_mode: "neural",
  },
  evidence: [
    { id: "S1", kind: "statistic", ...signalRow, tested: true, indication_overlap_pct: 30, source: "data/processed/signals.parquet" },
    { id: "D1", kind: "document", chunk_id: "PROJECT-1", source: "PROJECT", title: "Project methods note: Disproportionality is not causality", page: 0, text: "Disproportionality statistics compare reporting.", score: null },
  ],
  generated_summary: { status: "unavailable", reason: "no local generator model installed" },
  established: {
    status: "answered", retrieval_mode: "hybrid", threshold: 0.69, top_similarity: 0.81, drugs_searched: ["DRUGX"], not_indexed: [],
    label_mentions: [{ drug: "DRUGX", pt: "Hepatitis", found: true, sections: ["Warnings and Precautions"], label_indexed: true, cite: "L1" }],
    sentences: [
      { text: "The FDA label for DRUGX mentions “hepatitis” (sections: Warnings and Precautions).", cites: ["L1"] },
      { text: "The DRUGX label warns about hepatitis [L1].", cites: [], generated: true },
    ],
    evidence: [{ id: "L1", kind: "external", chunk_id: "c1", doc_id: "label:abc", source_type: "label", drugs: "DRUGX", section_key: "warnings_and_cautions",
      section: "Warnings and Precautions", title: "Drugxa (Drugx) - FDA prescribing information", publisher: "Acme Pharma", date: "20260130", version: "14",
      url: "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc", identifier: "abc", text: "5.1 Hepatitis: monitor liver tests.", score: 0.03,
      dense_similarity: 0.81, bm25: 7.2 }],
    generated: { status: "verified", text: "The DRUGX label warns about hepatitis [L1].", model_id: "Qwen/Qwen2.5-0.5B-Instruct", seconds: 4.2 },
  },
  provenance: { rag_version: "pv-rag-1.0", reaction_linker: "SapBERT", link_threshold: 0.7 },
  elapsed_ms: 120, disclaimer: "Answers are provisional.",
};

export const ragStatus = {
  available: true, rag_version: "pv-rag-1.0", examples: ["Is semaglutide associated with pancreatitis?"], link_thresholds: { neural: 0.7 },
  evaluation_available: true, knowledge_base: null,
  generator: { available: false, weights_present: false, deps_installed: true, enabled: true, model_dir: "models/generator", model_id: null },
  disclaimer: "provisional",
};

const rank = { n: 299, recall_at_1: 0.48, recall_at_3: 0.67, recall_at_5: 0.77, recall_at_10: 0.86, mrr: 0.61, recall_at_1_95ci: [0.42, 0.54], mrr_95ci: [0.56, 0.65] };
export const ragEvaluation = {
  created_at: "2026-10-19T17:00:00+00:00", caveats: ["The FAQ benchmark has 52 questions."],
  faq_retrieval: { status: "ok", queries: 52, corpus_size: 131, corpus: "answers", systems: { bm25: rank, tfidf: rank, random: { recall_at_1: 0.01, mrr: 0.04 } } },
  reaction_linking: { status: "ok", queries: 427, dev: 128, test: 299, candidates: 12678,
    systems: { sapbert: { status: "ok", chosen_variant: "sapbert_stripped_64tok", test: rank, at_production_threshold: { threshold: 0.7, test_coverage: 0.75, test_precision_at_1_when_accepted: 0.58, note: "n" } },
      tfidf_char: { status: "ok", chosen_variant: "tfidf_char_stripped", test: { ...rank, recall_at_1: 0.12 } } } },
  end_to_end: { status: "ok", pairs_sampled: 150, exact_retrieval_generic_templates: 0.97, exact_retrieval_95ci: [0.95, 0.98],
    per_template: { is_associated: { n: 150, exact_retrieval: 0.97 } }, failure_reasons: { "wrong reaction term": 3 },
    deterministic_answers_passing_grounding_check: 1, mean_latency_ms: 140, note: "Templated questions." },
  generator: { status: "not_evaluated", reason: "No generative model weights were available." },
};

routes["/api/rag/status"] = ragStatus;
routes["/api/evidence/status"] = {
  available: true, failures: 0, sufficiency_threshold: 0.69, evaluation_available: false,
  index: { created_at: "2026-10-20", documents: 3, chunks: 30, documents_by_type: { label: 1, literature: 1, regulatory: 1 }, chunks_by_type: {},
    drugs_with_label: 1, content_sha256: "abc", embedding: { model_id: "BAAI/bge-small-en-v1.5", revision: "5c38ec7" }, retrieval_modes: ["hybrid", "dense", "bm25"] },
  sources: { started_at: "", finished_at: "", drugs_requested: 1, stats: {} },
  embedder: { available: true, model_id: "BAAI/bge-small-en-v1.5" }, generator: { available: true, model_id: "Qwen/Qwen2.5-0.5B-Instruct" },
};
routes["/api/evidence/documents"] = { total: 1, by_type: { label: 1 }, rows: [{ doc_id: "label:abc", source_type: "label", title: "Drugxa (Drugx) - FDA prescribing information",
  publisher: "Acme Pharma", date: "20260130", version: "14", url: "https://dailymed.nlm.nih.gov/x", identifier: "abc", drugs: "DRUGX", chunks: 10 }] };
routes["/api/rag/evaluation"] = ragEvaluation;
routes["POST /api/rag/ask"] = askResponse;

/** Install a fetch mock answering from `routes` by path; returns the mock for call assertions. */
export function mockFetch(extra: Record<string, unknown> = {}) {
  const table = { ...routes, ...extra };
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), "http://localhost");
    if (init?.method === "POST") {
      const posted = table[`POST ${url.pathname}`];
      return new Response(JSON.stringify(posted ?? { ok: true }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    const body = table[url.pathname];
    if (body === undefined) return new Response(JSON.stringify({ detail: `no mock for ${url.pathname}` }), { status: 404 });
    return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fn);
  return fn;
}
