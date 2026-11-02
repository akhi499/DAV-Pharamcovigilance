"""Evaluation of external-evidence retrieval and answers, without hand-written labels.

Ground truth comes from structure that FDA, not this project, assigned:

A. **Label-section retrieval.** Every label chunk carries the SPL section FDA coded it under (Boxed
   Warning, Contraindications, Drug Interactions, ...). For each indexed label and each topic present,
   a templated question is asked twice - in label wording ("What are the contraindications of X?") and
   in lay wording ("Who should not take X?") - and a hit is correct if it comes from that drug's chunks
   of the gold section(s). Measured within the drug (the production setting, drug filter on) and across
   the whole index (drug must also be found). Systems: dense (BGE), BM25, hybrid (RRF).
B. **Insufficient-evidence threshold.** Answerable questions (A, dev drugs) vs questions whose answer is
   not in any indexed source (price, colour, manufacturer address, unrelated topics). The threshold on
   the best dense similarity is chosen on dev drugs (max balanced accuracy) and reported on test drugs.
C. **Generated answers** (only if the local generator is installed), on test drugs:
   * "Does X have a boxed warning?" - gold: whether FDA's label has a Boxed Warning section;
   * verifier pass rate, explicit-insufficiency rate, lexical citation support and latency.
D. **Signal-label concordance** (descriptive, not an accuracy): for strong FAERS signals of indexed drugs,
   how often the reaction term appears in the drug's label text.

Drugs are split 30% dev / 70% test with a fixed seed; nothing is tuned on test drugs.
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
from .sources import evidence_dir

log = logging.getLogger(__name__)
SEED = 42
DEV_FRACTION = 0.3
DEFAULT_THRESHOLD = 0.60  # used only until evaluate() has calibrated one on dev drugs

TOPICS = {
    "boxed_warning": (("boxed_warning",), "What is the boxed warning for {d}?", "What is the most serious warning about {b}?"),
    "contraindications": (("contraindications",), "What are the contraindications of {d}?", "Who should not take {b}?"),
    "interactions": (("drug_interactions",), "What are the drug interactions of {d}?",
                     "Which other medicines should not be combined with {b}?"),
    "pregnancy": (("pregnancy", "use_in_specific_populations"), "What does the label say about {d} use in pregnancy?",
                  "Is it safe to take {b} while pregnant?"),
    "overdosage": (("overdosage",), "What are the signs of {d} overdosage?", "What happens if someone takes too much {b}?"),
    "adverse_reactions": (("adverse_reactions",), "What are the most common adverse reactions of {d}?",
                          "What side effects do people get from {b}?"),
}
UNANSWERABLE = ["How much does {b} cost in pharmacies?", "What colour is the {b} packaging?",
                "What is the street address of the company that sells {b}?", "Who won the football world cup?",
                "What is a good recipe for chocolate cake?", "How tall is the Eiffel Tower?"]


def sufficiency_threshold(paths: Paths | None = None) -> float:
    f = evidence_dir(paths or get_paths()) / "calibration.json"
    if f.exists():
        try:
            return float(json.loads(f.read_text())["sufficiency_threshold"])
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return DEFAULT_THRESHOLD


def _boot(x: np.ndarray, n: int = 2000) -> list[float] | None:
    if len(x) == 0:
        return None
    rng = np.random.default_rng(SEED)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def _metrics(ranks: list[float]) -> dict:
    r = np.array(ranks, dtype=float)
    if len(r) == 0:
        return {"n": 0}
    rr = np.where(np.isfinite(r), 1 / r, 0.0)
    ndcg = np.where(r <= 10, 1 / np.log2(r + 1), 0.0)  # binary relevance, first relevant hit
    return {"n": int(len(r)), "recall_at_1": round(float((r <= 1).mean()), 4), "recall_at_3": round(float((r <= 3).mean()), 4),
            "recall_at_5": round(float((r <= 5).mean()), 4), "recall_at_10": round(float((r <= 10).mean()), 4),
            "mrr": round(float(rr.mean()), 4), "ndcg_at_10": round(float(ndcg.mean()), 4),
            "recall_at_5_95ci": _boot((r <= 5).astype(float)), "mrr_95ci": _boot(rr)}


def split_drugs(drugs: list[str]) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(SEED)
    d = sorted(drugs)
    rng.shuffle(d)
    k = int(round(DEV_FRACTION * len(d)))
    return set(d[:k]), set(d[k:])


def _brand(title: str, drug: str) -> str:
    b = title.split(" (")[0].split(",")[0].strip()
    return b if b and b.upper() != drug.upper() else drug.lower()


def build_queries(idx) -> list[dict]:
    labels = idx.chunks[idx.chunks["source_type"] == "label"]
    out = []
    for drug, g in labels.groupby("drugs"):
        present = set(g["section_key"])
        brand = _brand(g["title"].iloc[0], drug)
        for topic, (secs, direct, lay) in TOPICS.items():
            if not present & set(secs):
                continue
            gold = set(g[g["section_key"].isin(secs)]["chunk_id"])
            for style, tpl in (("label_wording", direct), ("lay_wording", lay)):
                out.append({"drug": drug, "topic": topic, "style": style, "gold": gold,
                            "query": tpl.format(d=drug.lower(), b=brand)})
    return out


def _rank(hits: list[dict], gold: set[str]) -> float:
    for i, h in enumerate(hits, 1):
        if h["chunk_id"] in gold:
            return float(i)
    return float("inf")


def eval_retrieval(idx, queries: list[dict], test: set[str]) -> dict:
    modes = ["bm25"] + (["dense", "hybrid"] if idx.vectors is not None else [])
    res: dict = {"queries_total": len(queries), "modes": modes, "within_drug": {}, "whole_index": {}}
    rows = []
    for q in queries:
        for mode in modes:
            r1 = _rank(idx.search(q["query"], k=10, drugs=[q["drug"]], mode=mode)["hits"], q["gold"])
            r2 = _rank(idx.search(q["query"], k=10, mode=mode)["hits"], q["gold"])
            rows.append({"drug": q["drug"], "topic": q["topic"], "style": q["style"], "mode": mode,
                         "rank_within": r1, "rank_whole": r2, "split": "test" if q["drug"] in test else "dev"})
    df = pd.DataFrame(rows)
    t = df[df["split"] == "test"]
    for mode in modes:
        m = t[t["mode"] == mode]
        res["within_drug"][mode] = {"all": _metrics(m["rank_within"].tolist()),
                                    **{s: _metrics(m[m["style"] == s]["rank_within"].tolist())
                                       for s in ("label_wording", "lay_wording")}}
        res["whole_index"][mode] = {"all": _metrics(m["rank_whole"].tolist()),
                                    **{s: _metrics(m[m["style"] == s]["rank_whole"].tolist())
                                       for s in ("label_wording", "lay_wording")}}
    if "hybrid" in modes:
        res["per_topic_hybrid_within_drug_recall_at_5"] = {
            tp: round(float((g["rank_within"] <= 5).mean()), 4)
            for tp, g in t[t["mode"] == "hybrid"].groupby("topic")}
    res["test_queries"] = int(len(t) // max(1, len(modes)))
    return res


def select_mode_on_dev(idx, queries: list[dict], dev: set[str]) -> dict:
    """Choose the production retrieval mode on DEV drugs only (MRR, within-drug search)."""
    modes = ["bm25"] + (["dense", "hybrid"] if idx.vectors is not None else [])
    res = {}
    for mode in modes:
        ranks = [_rank(idx.search(q["query"], k=10, drugs=[q["drug"]], mode=mode)["hits"], q["gold"])
                 for q in queries if q["drug"] in dev]
        res[mode] = {k: v for k, v in _metrics(ranks).items() if not k.endswith("95ci")}
    best = max(res, key=lambda m: res[m].get("mrr", 0))
    return {"selected_mode": best, "criterion": "highest MRR on dev drugs", "dev": res}


def calibrate_threshold(idx, queries: list[dict], dev: set[str], test: set[str]) -> dict:
    if idx.vectors is None:
        return {"status": "unavailable", "reason": "no dense embeddings; insufficiency falls back to 'no hits'"}
    drugs = sorted({q["drug"] for q in queries})
    titles = idx.chunks[idx.chunks["source_type"] == "label"].groupby("drugs")["title"].first()

    def top(q, d):
        hits = idx.search(q, k=5, drugs=[d], mode="dense")["hits"]
        return max((h["dense_similarity"] for h in hits), default=0.0)

    rows = [{"drug": q["drug"], "y": 1, "s": top(q["query"], q["drug"])} for q in queries]
    for d in drugs:
        b = _brand(titles.get(d, d), d)
        rows += [{"drug": d, "y": 0, "s": top(t.format(b=b), d)} for t in UNANSWERABLE]
    df = pd.DataFrame(rows)
    dv, ts = df[df["drug"].isin(dev)], df[df["drug"].isin(test)]

    def bal(frame, th):
        pos, neg = frame[frame.y == 1], frame[frame.y == 0]
        tpr = float((pos.s >= th).mean()) if len(pos) else 0.0
        tnr = float((neg.s < th).mean()) if len(neg) else 0.0
        return (tpr + tnr) / 2, tpr, tnr

    grid = np.round(np.arange(0.40, 0.95, 0.005), 3)
    best = max(grid, key=lambda th: (bal(dv, th)[0], -th))
    b_dev, b_test = bal(dv, best), bal(ts, best)
    return {"status": "ok", "threshold": float(best), "objective": "max balanced accuracy on dev drugs",
            "dev": {"balanced_accuracy": round(b_dev[0], 4), "answerable_kept": round(b_dev[1], 4),
                    "unanswerable_rejected": round(b_dev[2], 4), "n": int(len(dv))},
            "test": {"balanced_accuracy": round(b_test[0], 4), "answerable_kept": round(b_test[1], 4),
                     "unanswerable_rejected": round(b_test[2], 4), "n": int(len(ts))},
            "unanswerable_templates": UNANSWERABLE,
            "similarity_quartiles": {"answerable": np.percentile(df[df.y == 1].s, [25, 50, 75]).round(3).tolist(),
                                     "unanswerable": np.percentile(df[df.y == 0].s, [25, 50, 75]).round(3).tolist()}}


def classify_boxed(text: str | None, status: str) -> str:
    if status == "insufficient":
        return "no_answer"
    if not text:
        return "no_answer"
    low = text.lower()
    if re.search(r"\b(no|not|without|does not (have|carry|include|contain))\b[^.]{0,40}boxed warning", low) or \
            re.search(r"boxed warning[^.]{0,30}\b(not|none)\b", low):
        return "no"
    if "boxed warning" in low or "black box" in low or "warning: " in low:
        return "yes"
    return "undetermined"


def eval_generation(idx, test: set[str], n: int = 40) -> dict:
    from ..nlp.generator import status as gstatus
    from ..nlp.generator import summarise_external

    st = gstatus()
    if not st["available"]:
        return {"status": "unavailable", "reason": "local generator not installed", "generator": st}
    labels = idx.chunks[idx.chunks["source_type"] == "label"]
    has_bw = labels.groupby("drugs")["section_key"].apply(lambda s: "boxed_warning" in set(s))
    drugs = sorted(d for d in has_bw.index if d in test)
    rng = np.random.default_rng(SEED)
    pos = [d for d in drugs if has_bw[d]]
    neg = [d for d in drugs if not has_bw[d]]
    k = n // 2
    pick = list(rng.choice(pos, min(k, len(pos)), replace=False)) + list(rng.choice(neg, min(k, len(neg)), replace=False))
    rows = []
    for d in pick:
        q = f"Does {d.lower()} have a boxed warning?"
        hits = idx.search(q, k=4, drugs=[d])["hits"]
        items = [{"id": f"L{i + 1}", "kind": "external", **h} for i, h in enumerate(hits)]
        t0 = time.perf_counter()
        g = summarise_external(q, items)
        pred = classify_boxed(g.get("text") or g.get("rejected_text"), g["status"])
        rows.append({"drug": d, "gold": "yes" if has_bw[d] else "no", "pred": pred, "status": g["status"],
                     "verified": g["status"] == "verified", "support": g.get("citation_support"),
                     "seconds": round(time.perf_counter() - t0, 2), "retrieval_has_boxed": any(
                         h["section_key"] == "boxed_warning" for h in hits),
                     "text": (g.get("text") or g.get("rejected_text") or "")[:400], "problems": g.get("problems", [])})
    df = pd.DataFrame(rows)
    correct = (df.pred == df.gold)
    retr = df.retrieval_has_boxed.map({True: "yes", False: "no"}) == df.gold
    sup = df.support.dropna()
    return {"status": "ok", "generator": {k: st.get(k) for k in ("model_id", "model_dir")}, "n": int(len(df)),
            "boxed_warning_accuracy": round(float(correct.mean()), 4), "boxed_warning_accuracy_95ci": _boot(correct.to_numpy(float)),
            "prediction_counts": df.pred.value_counts().to_dict(),
            "baseline_retrieval_only_accuracy": round(float(retr.mean()), 4),
            "verifier_pass_rate": round(float(df.verified.mean()), 4),
            "explicit_insufficient_rate": round(float((df.status == "insufficient").mean()), 4),
            "rejection_reasons": pd.Series([p.split(":")[0] for ps in df.problems for p in ps]).value_counts().to_dict(),
            "citation_support_mean": round(float(sup.mean()), 4) if len(sup) else None,
            "seconds_median": round(float(df.seconds.median()), 2), "seconds_p90": round(float(df.seconds.quantile(0.9)), 2),
            "examples": df.head(8).to_dict(orient="records")}


def eval_concordance(idx, paths: Paths, top_n: int = 300) -> dict:
    """Two samples of strong signals: the most-reported ones and the most disproportionate ones."""
    out = {"status": "ok"}
    for name, order in (("most_reported", "a DESC, ror_lower DESC"), ("most_disproportionate", "ror_lower DESC")):
        out[name] = _concordance(idx, paths, top_n, order)
    out["caveat"] = out["most_reported"].get("caveat")
    return out


def _concordance(idx, paths: Paths, top_n: int, order: str) -> dict:
    import duckdb

    f = str(paths.processed / "signals.parquet").replace("\\", "/")
    drugs = [d for d in idx.indexed_drugs if idx.mask([d], ("label",)).any()]
    if not drugs:
        return {"status": "unavailable"}
    marks = ",".join("?" * len(drugs))
    rows = duckdb.execute(
        f"""SELECT drug, pt, a, ror_lower FROM read_parquet('{f}') WHERE methods_flagging = 3 AND NOT possible_indication_bias
            AND a >= 10 AND drug IN ({marks}) ORDER BY {order} LIMIT {int(top_n)}""", drugs).fetchall()
    if not rows:
        return {"status": "ok", "signals_checked": 0, "share_reaction_text_in_label": None,
                "share_in_label_excluding_product_use_terms": None, "product_use_terms": 0,
                "caveat": "No strong signals met the criteria for drugs with an indexed label."}
    found = [bool(idx.label_mentions(d, [pt.lower()])) for d, pt, _, _ in rows]
    product = [bool(re.search(r"off label|product|dose|device|medication error|drug ineffective|wrong|use issue|"
                              r"administration|prescribed|underdose|overdose|omission", pt, re.I)) for _, pt, _, _ in rows]
    df = pd.DataFrame({"drug": [r[0] for r in rows], "pt": [r[1] for r in rows], "a": [r[2] for r in rows],
                       "in_label": found, "product_use_term": product})
    clin = df[~df.product_use_term]
    return {"status": "ok", "signals_checked": int(len(df)), "criteria": "all three methods, no indication-bias flag, a >= 10, "
            f"top {top_n} by {order.split(',')[0]} among drugs with an indexed label",
            "share_reaction_text_in_label": round(float(df.in_label.mean()), 4) if len(df) else None,
            "share_in_label_excluding_product_use_terms": round(float(clin.in_label.mean()), 4) if len(clin) else None,
            "product_use_terms": int(df.product_use_term.sum()),
            "examples_not_in_label": clin[~clin.in_label].head(10).to_dict(orient="records"),
            "examples_in_label": clin[clin.in_label].head(10).to_dict(orient="records"),
            "caveat": "Exact-term text match: a reaction worded differently in the label counts as 'not found'."}


def evaluate(paths: Paths | None = None, n_generation: int = 40, skip_generation: bool = False) -> dict:
    from .index import EvidenceIndex

    paths = paths or get_paths()
    t0 = time.time()
    idx = EvidenceIndex(paths)
    queries = build_queries(idx)
    drugs = sorted({q["drug"] for q in queries})
    dev, test = split_drugs(drugs)
    out = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "index": idx.manifest,
           "retrieval_mode_available": idx.mode, "drugs_with_label": len(drugs), "dev_drugs": len(dev), "test_drugs": len(test),
           "protocol": {"seed": SEED, "dev_fraction": DEV_FRACTION, "topics": {k: list(v[0]) for k, v in TOPICS.items()},
                        "templates": {k: [v[1], v[2]] for k, v in TOPICS.items()}}}
    log.info("evidence eval: retrieval (%d queries)", len(queries))
    out["retrieval"] = eval_retrieval(idx, queries, test)
    out["mode_selection"] = select_mode_on_dev(idx, queries, dev)
    log.info("evidence eval: threshold calibration")
    cal = calibrate_threshold(idx, [q for q in queries if q["drug"] in dev or q["drug"] in test], dev, test)
    out["sufficiency"] = cal
    if cal.get("status") == "ok":
        (evidence_dir(paths) / "calibration.json").write_text(json.dumps(
            {"sufficiency_threshold": cal["threshold"], "created_at": out["created_at"],
             "source": "evidence.evaluate.calibrate_threshold (dev drugs)"}, indent=2))
    log.info("evidence eval: generation")
    out["generation"] = {"status": "skipped"} if skip_generation else eval_generation(idx, test, n_generation)
    log.info("evidence eval: signal-label concordance")
    out["concordance"] = eval_concordance(idx, paths)
    out["seconds"] = round(time.time() - t0, 1)
    out["caveats"] = [
        "Gold relevance is FDA's own SPL section coding; a passage outside the gold section can still be useful.",
        "Questions are templated (two wordings per topic); real questions are more varied.",
        "Unanswerable questions are templated by the project; their answers are absent from all indexed sources by construction.",
        "The generator is evaluated on one yes/no task plus automatic faithfulness checks, not on free-form clinical accuracy.",
        "Concordance uses exact text matching and is descriptive, not a measure of signal validity.",
    ]
    (evidence_dir(paths) / "evaluation_evidence.json").write_text(json.dumps(out, indent=2, default=str))
    return out
