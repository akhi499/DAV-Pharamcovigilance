"""Evaluation of the question-answering (RAG) module against independent references.

No label is written by the project authors or by a language model. Three tasks:

A. **Passage retrieval - FDA FAQ.** FDA's FAERS FAQ is a list of real user
   questions with FDA-written answers. Queries = the 52 question texts;
   the index holds the *answers only* (question text removed, so the task is
   not string matching) plus every README / ASC_NTS window and project note as
   distractors. Gold = the answer FDA wrote for that question. BM25 vs a word
   TF-IDF baseline vs the random expectation. BM25 uses textbook parameters
   (k1 = 1.5, b = 0.75); nothing is tuned, so there is no dev split.

B. **Reaction linking - NCI CTCAE definitions.** For every CTCAE v5.0 term
   whose name is also a FAERS 2025 Q1 MedDRA PT, the NCI-written definition is
   the query and the gold answer is that PT, among all 12,678 FAERS PTs.
   SapBERT (two input variants) vs character TF-IDF vs word BM25. The input
   variant is chosen on a 30% dev split; numbers are reported on the 70% test
   split, overall and for the harder subset whose definition does not contain
   the term itself. Coverage/precision at the production acceptance threshold
   are reported too. Definitions are longer and more formal than user phrases,
   so this is a hard, conservative proxy for lay-language linking.

C. **End-to-end retrieval on templated questions.** Random (seeded) tested
   drug-reaction pairs are inserted into five question templates (and a
   brand-name template using FAERS-derived aliases). Gold = the pair the
   question was built from; success = the engine links that drug and that PT
   and returns exactly the stored statistics. This checks parsing and
   retrieval, not clinical correctness. Every answered question is also run
   through the generator verifier as a numeric-grounding self-check.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import Paths, get_paths
from .kb import BM25, TfidfRetriever, load_ctcae_definitions, load_kb
from .rag import LINK_THRESHOLD

log = logging.getLogger(__name__)
SEED = 42
DEV_FRACTION = 0.3
DEF_PREFIX = re.compile(
    r"^(a|an)\s+(disorder|finding|injury|condition|sensation|syndrome|procedure|event)\s+"
    r"(characterized by|based on|of|in which|resulting from|that)\s+", re.I)


def _bootstrap(x: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> list[float] | None:
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def ranking_metrics(ranks: np.ndarray, ks=(1, 3, 5, 10)) -> dict:
    """ranks: 1-based rank of the gold item per query (np.inf if not retrieved)."""
    out = {"n": int(len(ranks))}
    for k in ks:
        hit = (ranks <= k).astype(float)
        out[f"recall_at_{k}"] = round(float(hit.mean()), 4) if len(ranks) else None
    out["recall_at_1_95ci"] = _bootstrap((ranks <= 1).astype(float))
    rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
    out["mrr"] = round(float(rr.mean()), 4) if len(ranks) else None
    out["mrr_95ci"] = _bootstrap(rr)
    return out


def _rank_of(scores: np.ndarray, gold: int) -> float:
    g = scores[gold]
    if not np.isfinite(g) or g <= 0 and scores.max() <= 0:
        return float("inf")
    return float(1 + (scores > g).sum())


# --------------------------------------------------------------------------
# A. FAQ passage retrieval
# --------------------------------------------------------------------------

def eval_faq_retrieval(paths: Paths) -> dict:
    kb = load_kb(str(paths.nlp))
    faq = kb[kb["source"] == "FDA-FAQ"].reset_index(drop=True)
    if faq.empty:
        return {"status": "unavailable", "reason": "FDA FAQ not in knowledge base (FAERS zip or pypdf missing)"}
    other = kb[kb["source"].isin(["FDA-README", "FDA-ASC_NTS", "PROJECT"])]
    docs = faq["answer"].tolist() + other["text"].tolist()
    queries = faq["question"].tolist()
    res = {"status": "ok", "queries": len(queries), "corpus_size": len(docs),
           "corpus": "52 FDA FAQ answers (questions removed) + README/ASC_NTS windows + project notes",
           "systems": {}}
    per_query = []
    for name, cls in (("bm25", BM25), ("tfidf", TfidfRetriever)):
        r = cls(docs)
        ranks = np.array([_rank_of(r.scores(q), i) for i, q in enumerate(queries)])
        res["systems"][name] = ranking_metrics(ranks)
        if name == "bm25":
            tops = [float(r.scores(q).max()) for q in queries]
            res["systems"][name]["top_score_median"] = round(float(np.median(tops)), 2)
            res["systems"][name]["share_top_score_below_abstain_threshold_3"] = round(float(np.mean(np.array(tops) < 3.0)), 4)
            for i, q in enumerate(queries):
                per_query.append({"faq": int(faq.loc[i, "faq_number"]), "question": q[:160],
                                  "bm25_rank": None if not np.isfinite(ranks[i]) else int(ranks[i])})
    # paired bootstrap of the difference (tfidf - bm25); the production retriever was fixed before this test,
    # so a difference here is reported, not used to re-select the retriever on the same 52 questions
    rk = {name: np.array([_rank_of(cls(docs).scores(q), i) for i, q in enumerate(queries)])
          for name, cls in (("bm25", BM25), ("tfidf", TfidfRetriever))}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(queries), size=(2000, len(queries)))
    for metric, f in (("recall_at_1", lambda r: (r <= 1).astype(float)),
                      ("mrr", lambda r: np.where(np.isfinite(r), 1 / r, 0.0))):
        d = f(rk["tfidf"])[idx].mean(axis=1) - f(rk["bm25"])[idx].mean(axis=1)
        res.setdefault("paired_difference_tfidf_minus_bm25", {})[metric] = {
            "mean": round(float(f(rk["tfidf"]).mean() - f(rk["bm25"]).mean()), 4),
            "95ci": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
    n = len(docs)
    res["systems"]["random"] = {f"recall_at_{k}": round(k / n, 4) for k in (1, 3, 5, 10)} | {"mrr": round(
        float(np.mean([1 / i for i in range(1, n + 1)])), 4), "note": "expected value of a random ranking"}
    res["per_query_bm25"] = per_query
    return res


# --------------------------------------------------------------------------
# B. CTCAE definition -> FAERS PT linking
# --------------------------------------------------------------------------

def eval_reaction_linking(paths: Paths) -> dict:
    from .encoder import EncoderUnavailable, LexicalEncoder, get_neural_encoder, similarity

    pred_file = paths.nlp / "predictions.parquet"
    if not pred_file.exists():
        return {"status": "unavailable", "reason": "run the NLP stage first"}
    terms = pd.read_parquet(pred_file, columns=["pt", "pt_key", "cases"])
    key_to_idx = {k: i for i, k in enumerate(terms["pt_key"])}
    ctc = load_ctcae_definitions(paths)
    if ctc.empty:
        return {"status": "unavailable", "reason": "CTCAE v5.0 file not found in data/reference/"}
    ctc = ctc[ctc["term_key"].isin(key_to_idx)].reset_index(drop=True)
    ctc["gold"] = ctc["term_key"].map(key_to_idx)
    ctc["stripped"] = ctc["definition"].str.replace(DEF_PREFIX, "", regex=True)
    ctc["contains_term"] = [t in d.lower() for t, d in zip(ctc["term_key"], ctc["definition"], strict=True)]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ctc))
    is_dev = np.zeros(len(ctc), dtype=bool)
    is_dev[perm[: int(round(DEV_FRACTION * len(ctc)))]] = True
    gold = ctc["gold"].to_numpy()
    pts = terms["pt"].tolist()

    def ranks_from(sim: np.ndarray) -> np.ndarray:
        g = sim[np.arange(len(gold)), gold]
        return 1 + (sim > g[:, None]).sum(axis=1).astype(float)

    systems: dict[str, dict] = {}
    raw_ranks: dict[str, np.ndarray] = {}
    top_sims: dict[str, np.ndarray] = {}
    top_idx: dict[str, np.ndarray] = {}
    # neural variants
    try:
        enc = get_neural_encoder()
        emb_file = paths.nlp / "embeddings.npy"
        cand = np.load(emb_file).astype(np.float32) if emb_file.exists() else enc.encode(pts)
        for variant, col, ml in (("sapbert_full_25tok", "definition", 25), ("sapbert_stripped_25tok", "stripped", 25),
                                 ("sapbert_stripped_64tok", "stripped", 64)):
            q = enc.encode(ctc[col].tolist(), max_length=ml)
            sim = similarity(q, cand)
            raw_ranks[variant] = ranks_from(sim)
            top_sims[variant] = sim.max(axis=1)
            top_idx[variant] = sim.argmax(axis=1)
    except EncoderUnavailable as exc:
        systems["sapbert"] = {"status": "unavailable", "reason": str(exc)}
    # lexical baselines (fitted on the candidate vocabulary)
    lex = LexicalEncoder(pts)
    for variant, col in (("tfidf_char_full", "definition"), ("tfidf_char_stripped", "stripped")):
        sim = similarity(lex.encode(ctc[col].tolist()), lex.encode(pts))
        raw_ranks[variant] = ranks_from(sim)
        top_sims[variant] = sim.max(axis=1)
        top_idx[variant] = sim.argmax(axis=1)
    bm = BM25(pts)
    for variant, col in (("bm25_word_stripped", "stripped"),):
        sim = np.vstack([bm.scores(q) for q in ctc[col]])
        rk = ranks_from(sim)
        rk[sim[np.arange(len(gold)), gold] <= 0] = np.inf
        raw_ranks[variant] = rk

    # choose the input variant per family on dev (MRR), report test
    families = {"sapbert": [v for v in raw_ranks if v.startswith("sapbert")],
                "tfidf_char": [v for v in raw_ranks if v.startswith("tfidf")],
                "bm25_word": [v for v in raw_ranks if v.startswith("bm25")]}
    chosen = {}
    for fam, variants in families.items():
        if not variants:
            continue
        dev_mrr = {v: float(np.where(np.isfinite(raw_ranks[v][is_dev]), 1 / raw_ranks[v][is_dev], 0).mean())
                   for v in variants}
        best = max(dev_mrr, key=dev_mrr.get)
        chosen[fam] = best
        test = ~is_dev
        hard = test & ~ctc["contains_term"].to_numpy()
        systems[fam] = {"status": "ok", "chosen_variant": best, "dev_mrr_by_variant": {k: round(v, 4) for k, v in dev_mrr.items()},
                        "test": ranking_metrics(raw_ranks[best][test]),
                        "test_definition_without_term": ranking_metrics(raw_ranks[best][hard])}
        if best in top_sims and fam == "sapbert":
            thr = LINK_THRESHOLD["neural"]
            acc = top_sims[best] >= thr
            correct = top_idx[best] == gold
            systems[fam]["at_production_threshold"] = {
                "threshold": thr,
                "dev_coverage": round(float(acc[is_dev].mean()), 4),
                "dev_precision_at_1_when_accepted": round(float(correct[is_dev & acc].mean()), 4) if (is_dev & acc).any() else None,
                "test_coverage": round(float(acc[~is_dev].mean()), 4),
                "test_precision_at_1_when_accepted": round(float(correct[~is_dev & acc].mean()), 4) if (~is_dev & acc).any() else None,
                "note": "Definitions are long and formal; production queries are short phrases, whose similarities are higher.",
            }
    examples = []
    if "sapbert" in chosen:
        v = chosen["sapbert"]
        for i in np.where(~is_dev)[0][:12]:
            examples.append({"definition": ctc.loc[i, "definition"][:160], "gold": ctc.loc[i, "term"],
                             "sapbert_top1": pts[int(top_idx[v][i])], "rank": float(raw_ranks[v][i])})
    return {"status": "ok", "queries": int(len(ctc)), "dev": int(is_dev.sum()), "test": int((~is_dev).sum()),
            "test_definition_without_term": int((~is_dev & ~ctc["contains_term"].to_numpy()).sum()),
            "candidates": int(len(pts)), "systems": systems, "examples_test": examples,
            "protocol": {"seed": SEED, "dev_fraction": DEV_FRACTION, "selection": "input variant with best dev MRR per family"}}


# --------------------------------------------------------------------------
# C. end-to-end templated questions
# --------------------------------------------------------------------------

TEMPLATES = {
    "is_associated": "Is {drug} associated with {pt}?",
    "bare_pair": "{drug} {pt}",
    "does_cause_lower": "does {drug_l} cause {pt_l}",
    "signal_for": "Is there a signal for {pt} with {drug}?",
    "reports_of": "Any reports of {pt_l} in patients taking {drug_l}?",
}


def eval_end_to_end(paths: Paths, n_pairs: int = 150, n_alias: int = 60) -> dict:
    from ..db import Database
    from .generator import evidence_block, verify
    from .rag import RAGEngine

    db = Database(paths)
    eng = RAGEngine(paths, db)
    rng = np.random.default_rng(SEED)
    all_pairs = db.query("SELECT drug, pt, pt_key, a, ror FROM signals ORDER BY drug, pt_key")
    pick = rng.choice(len(all_pairs), size=min(n_pairs, len(all_pairs)), replace=False)
    pairs = [all_pairs[i] for i in pick]
    cases = []
    for p in pairs:
        for tname, tpl in TEMPLATES.items():
            cases.append((tname, tpl.format(drug=p["drug"], pt=p["pt"], drug_l=p["drug"].lower(), pt_l=p["pt"].lower()), p))
    # brand-name aliases learnt from FAERS: a consistency check of alias resolution
    aliases = sorted(eng.drugs.alias.items())
    if aliases:
        apick = rng.choice(len(aliases), size=min(n_alias, len(aliases)), replace=False)
        for i in apick:
            alias, drug = aliases[i]
            row = db.query("SELECT drug, pt, pt_key, a, ror FROM signals WHERE drug = ? ORDER BY a DESC LIMIT 1", [drug])
            if row:
                cases.append(("brand_alias", f"Is {alias.lower()} associated with {row[0]['pt'].lower()}?", row[0]))
    by_t: dict[str, list[int]] = {}
    failures: dict[str, int] = {}
    fail_examples = []
    grounded, answered = 0, 0
    t0 = time.perf_counter()
    for tname, q, p in cases:
        r = eng.answer(q)
        u = r["understanding"]
        drugs = [d["drug"] for d in u["drugs"]]
        rx = [x["pt_key"] for x in u["reactions"]]
        s1 = next((e for e in r["evidence"] if e["kind"] == "statistic"), None)
        ok = (r["intent"] == "pair" and drugs[:1] == [p["drug"]] and rx[:1] == [p["pt_key"]] and s1 is not None
              and s1.get("a") == p["a"] and s1.get("pt_key") == p["pt_key"])
        by_t.setdefault(tname, []).append(int(ok))
        if not ok:
            reason = ("drug not linked" if p["drug"] not in drugs else "reaction not linked" if not rx else
                      "wrong reaction term" if rx[0] != p["pt_key"] else f"intent {r['intent']}")
            failures[reason] = failures.get(reason, 0) + 1
            if len(fail_examples) < 15:
                fail_examples.append({"template": tname, "question": q, "reason": reason, "linked_drugs": drugs,
                                      "linked_terms": rx[:2]})
        if r["status"] == "answered":
            answered += 1
            text = " ".join(f"{s['text']} [{', '.join(s['cites'])}]" if s["cites"] else s["text"] for s in r["answer"])
            if not verify(text, evidence_block(r["evidence"]), {e["id"] for e in r["evidence"]},
                          {"2025", "95"} | set(re.findall(r"\d+", q))):  # template constants + user's numbers
                grounded += 1
    per_template = {k: {"n": len(v), "exact_retrieval": round(float(np.mean(v)), 4), "95ci": _bootstrap(np.array(v, float))}
                    for k, v in by_t.items()}
    allv = np.array([x for k, v in by_t.items() if k != "brand_alias" for x in v], float)
    return {"status": "ok", "pairs_sampled": len(pairs), "questions": len(cases),
            "exact_retrieval_generic_templates": round(float(allv.mean()), 4), "exact_retrieval_95ci": _bootstrap(allv),
            "per_template": per_template, "failure_reasons": failures, "failure_examples": fail_examples,
            "answered": answered,
            "deterministic_answers_passing_grounding_check": round(grounded / answered, 4) if answered else None,
            "mean_latency_ms": round(1000 * (time.perf_counter() - t0) / max(1, len(cases)), 1),
            "link_mode": eng.mode,
            "note": "Templated questions built from real (drug, PT) pairs; tests parsing+retrieval, not clinical truth. "
                    "Brand aliases are learnt from the same FAERS data, so that row is a consistency check."}


def evaluate_rag(paths: Paths | None = None, n_pairs: int = 150) -> dict:
    paths = paths or get_paths()
    started = datetime.now(timezone.utc)
    out = {"created_at": started.isoformat(timespec="seconds"),
           "faq_retrieval": eval_faq_retrieval(paths),
           "reaction_linking": eval_reaction_linking(paths),
           "end_to_end": eval_end_to_end(paths, n_pairs=n_pairs),
           "generator": {"status": "not_evaluated",
                         "reason": "No generative model weights were available when this evaluation was run; only the "
                                   "verifier is unit-tested. Generated summaries, if enabled, are unevaluated."},
           "caveats": [
               "The FAQ benchmark has 52 questions; confidence intervals are wide.",
               "CTCAE definitions are formal NCI text, not patient language; they are a conservative proxy for linking.",
               "Templated end-to-end questions use exact or lower-cased vocabulary terms; real questions are messier.",
               "No metric here measures clinical correctness; the statistics themselves are validated separately.",
           ]}
    out["duration_seconds"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    paths.nlp.mkdir(parents=True, exist_ok=True)
    (paths.nlp / "evaluation_rag.json").write_text(json.dumps(out, indent=2, default=float))
    return out
