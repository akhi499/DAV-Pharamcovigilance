"""FastAPI backend for the pharmacovigilance dashboard.

Run (from the backend folder):  uvicorn api.main:app --port 8000
Statistical endpoints work without any NLP dependency. NLP endpoints degrade
gracefully: if the SapBERT weights or torch are missing, precomputed
predictions are still served and semantic search falls back to the lexical
encoder, with the active mode reported in every response.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import threading
import time
from functools import lru_cache
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from pvsys import __version__, eda
from pvsys import signals as sigmod
from pvsys.config import SignalConfig, get_paths
from pvsys.db import Database

log = logging.getLogger("pvsys.api")

app = FastAPI(title="AI-Based Drug Safety & Adverse Effect Analysis API", version=__version__)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DISCLAIMER = (
    "FAERS is a spontaneous reporting system. Counts reflect reports, not patients or incidence. "
    "Disproportionality measures (ROR/PRR/IC) indicate reporting associations that warrant review; "
    "they do not establish causality or absolute risk."
)

RAG_DISCLAIMER = (
    "Answers are assembled automatically from retrieved FAERS statistics and documents and are provisional. "
    "Drug and reaction links are made by a language model and may be wrong; check the evidence. Not medical advice; "
    "reporting associations do not establish causality, incidence or risk."
)

NLP_DISCLAIMER = (
    "Reaction categories are provisional, model-generated groupings into project-defined (non-MedDRA) "
    "categories. They are not validated clinical codes and must not be used for clinical decisions."
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _clean(obj):
    """Make DuckDB/numpy results JSON-safe (NaN/inf -> None, numpy scalars -> python)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


_db: Database | None = None


def db() -> Database:
    global _db
    if _db is None:
        _db = Database(get_paths())
    tables = _db.tables()
    if "cases" not in tables:
        raise HTTPException(503, "Processed data not found. Run `python -m pvsys.pipeline stats` first.")
    return _db


def _read_json(name: str, folder: str = "processed"):
    p = getattr(get_paths(), folder) / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - t) * 1000:.1f}"
    return response


# --------------------------------------------------------------------------
# health & meta
# --------------------------------------------------------------------------

@app.get("/api/health")
def health():
    from pvsys.nlp.encoder import neural_status

    d = Database(get_paths()) if _db is None else _db
    paths = get_paths()
    return {
        "status": "ok",
        "version": __version__,
        "tables": sorted(d.tables()),
        "nlp": {
            "predictions_available": (paths.nlp / "predictions.parquet").exists(),
            "neural_encoder": neural_status(),
            "qa_knowledge_base_available": (paths.nlp / "kb_chunks.parquet").exists(),
        },
    }


@app.post("/api/admin/reload")
def reload_data():
    """Re-open Parquet views and clear caches after re-running the pipeline."""
    db().reload()
    _cached.cache_clear()
    from pvsys.evidence.index import reset_index as reset_evidence
    from pvsys.nlp.rag import reset_engine
    from pvsys.nlp.search import reset_index

    reset_index()
    reset_engine()
    reset_evidence()
    _ask_cached.cache_clear()
    return {"reloaded": True}


@lru_cache(maxsize=64)
def _cached(key: str):
    d = db()
    if key == "overview":
        return eda.overview(d)
    if key.startswith("dist:"):
        return eda.distribution(d, key[5:])
    if key == "outcomes":
        return eda.outcomes(d)
    if key == "pyramid":
        return eda.age_sex_pyramid(d)
    if key == "age_hist":
        return eda.age_histogram(d)
    if key == "trend":
        return eda.receipt_trend(d)
    if key == "countries":
        return eda.countries(d, 20)
    if key == "sex_test":
        return eda.serious_by_sex_test(d)
    raise KeyError(key)


@app.get("/api/overview")
def overview():
    return _clean({
        "kpis": _cached("overview"),
        "dedup": _read_json("dedup_stats.json"),
        "manifest": {k: v for k, v in (_read_json("manifest.json") or {}).items() if k != "tables"},
        "signals_summary": _read_json("signals_summary.json"),
        "disclaimer": DISCLAIMER,
    })


@app.get("/api/quality")
def quality():
    q = _read_json("quality.json")
    if q is None:
        raise HTTPException(503, "quality.json not found - run the pipeline")
    return q


@app.get("/api/demographics")
def demographics():
    return _clean({
        "sex": _cached("dist:sex"),
        "age_group": _cached("dist:age_group"),
        "reporter_type": _cached("dist:reporter_type"),
        "countries": _cached("countries"),
        "outcomes": _cached("outcomes"),
        "pyramid": _cached("pyramid"),
        "age_histogram": _cached("age_hist"),
        "serious_by_sex": _cached("sex_test"),
        "note": "Distributions of reported cases; they describe who reports, not who is at risk.",
    })


@app.get("/api/trend")
def trend():
    return _clean(_cached("trend"))


@app.get("/api/drugs/top")
def drugs_top(limit: int = Query(20, ge=1, le=200), role: Literal["suspect", "primary", "any"] = "suspect"):
    return _clean(eda.top_drugs(db(), limit, role))


@app.get("/api/reactions/top")
def reactions_top(limit: int = Query(20, ge=1, le=200)):
    return _clean(eda.top_reactions(db(), limit))


@app.get("/api/drugs/search")
def drug_search(q: str = Query(..., min_length=2, max_length=80), limit: int = Query(15, ge=1, le=50)):
    return db().query(
        """
        SELECT drug, count(DISTINCT primaryid) AS cases FROM drugs
        WHERE role_cod IN ('PS','SS') AND drug ILIKE ? GROUP BY 1 ORDER BY 2 DESC LIMIT ?
        """,
        [f"%{q.strip()}%", limit],
    )


@app.get("/api/reactions/search")
def reaction_search(q: str = Query(..., min_length=2, max_length=80), limit: int = Query(15, ge=1, le=50)):
    return db().query(
        "SELECT pt, pt_key, count(*) AS cases FROM reactions WHERE pt_key ILIKE ? GROUP BY 1, 2 ORDER BY 3 DESC LIMIT ?",
        [f"%{q.strip().lower()}%", limit],
    )


@app.get("/api/drugs/{drug}/profile")
def drug_profile(drug: str):
    if not drug.strip() or len(drug) > 200:
        raise HTTPException(422, "invalid drug name")
    prof = eda.drug_profile(db(), drug)
    if not prof.get("cases"):
        raise HTTPException(404, f"No suspect-drug reports for {drug!r}")
    return _clean(prof)


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------

SIGNAL_SORTS = {
    "ror_lower": "ror_lower DESC", "ror": "ror DESC", "prr": "prr DESC", "a": "a DESC",
    "chi2": "chi2 DESC", "ic025": "ic025 DESC", "q_value": "q_value ASC",
}


@app.get("/api/signals/summary")
def signals_summary():
    s = _read_json("signals_summary.json")
    if s is None:
        raise HTTPException(503, "signals not computed")
    cfg = SignalConfig()
    return {**s, "criteria": {
        "ROR": f"lower 95% CI > {cfg.ror_lower_ci_threshold} and a >= {cfg.min_cases}",
        "PRR": f"PRR >= {cfg.prr_threshold}, chi2(Yates) >= {cfg.chi2_threshold}, a >= {cfg.min_cases}",
        "IC": f"IC025 > 0 and a >= {cfg.min_cases}",
        "sparse_cells": "Haldane-Anscombe +0.5 when any cell is zero",
    }, "disclaimer": DISCLAIMER}


@app.get("/api/signals")
def signals(
    drug: str | None = Query(None, max_length=200),
    pt: str | None = Query(None, max_length=200),
    min_cases: int = Query(3, ge=3, le=100000),
    only_signals: bool = True,
    method: Literal["any", "ror", "prr", "ic", "all"] = "all",
    exclude_indication_bias: bool = False,
    category: str | None = Query(None, max_length=40),
    exclude_category: str | None = Query(None, max_length=40),
    sort: str = Query("ror_lower"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if sort not in SIGNAL_SORTS:
        raise HTTPException(422, f"sort must be one of {sorted(SIGNAL_SORTS)}")
    d = db()
    where, params = ["s.a >= ?"], [min_cases]
    if drug:
        where.append("s.drug = ?")
        params.append(drug.strip().upper())
    if pt:
        where.append("s.pt_key ILIKE ?")
        params.append(f"%{pt.strip().lower()}%")
    if only_signals:
        where.append({"any": "s.methods_flagging >= 1", "ror": "s.ror_signal", "prr": "s.prr_signal",
                      "ic": "s.ic_signal", "all": "s.methods_flagging = 3"}[method])
    if exclude_indication_bias:
        where.append("NOT s.possible_indication_bias")
    has_nlp = "nlp_predictions" in d.tables()
    join = "LEFT JOIN nlp_predictions n ON n.pt_key = s.pt_key" if has_nlp else ""
    cat_cols = ", n.prediction AS nlp_category, n.confidence AS nlp_confidence" if has_nlp else ""
    if category or exclude_category:
        if not has_nlp:
            raise HTTPException(409, "NLP predictions unavailable; category filter cannot be applied")
    if category:
        where.append("n.prediction = ?")
        params.append(category)
    if exclude_category:
        # only removes terms the model confidently assigned; abstained terms are kept
        where.append("coalesce(n.prediction, '') <> ?")
        params.append(exclude_category)
    w = " AND ".join(where)
    total = d.scalar(f"SELECT count(*) FROM signals s {join} WHERE {w}", params)
    rows = d.query(
        f"""
        SELECT s.drug, s.pt, s.pt_key, s.a, s.b, s.c, s.d, s.n_drug, s.n_pt, s.expected,
               s.ror, s.ror_lower, s.ror_upper, s.prr, s.prr_lower, s.prr_upper, s.chi2, s.p_value, s.q_value,
               s.ic, s.ic025, s.ror_signal, s.prr_signal, s.ic_signal, s.methods_flagging, s.haldane_applied,
               s.indication_overlap, s.possible_indication_bias {cat_cols}
        FROM signals s {join} WHERE {w}
        ORDER BY {SIGNAL_SORTS[sort]}, s.a DESC LIMIT ? OFFSET ?
        """,
        params + [limit, offset],
    )
    return _clean({"total": total, "limit": limit, "offset": offset, "rows": rows, "disclaimer": DISCLAIMER})


@app.get("/api/signals/stratified")
def signals_stratified(drug: str = Query(..., max_length=200), pt_key: str = Query(..., max_length=200),
                       by: Literal["sex", "age_group", "reporter_type"] = "sex"):
    from pvsys.db import connect

    con = connect(get_paths(), tables=("cases", "drugs", "reactions"))
    try:
        rows = sigmod.stratified(con, drug.strip().upper(), pt_key.strip().lower(), by)
    finally:
        con.close()
    return _clean({"drug": drug.upper(), "pt_key": pt_key.lower(), "by": by, "strata": rows,
                   "note": "Stratum-specific ROR with a < 3 is shown but not interpretable."})


# --------------------------------------------------------------------------
# NLP
# --------------------------------------------------------------------------

def _nlp_required(d: Database):
    if "nlp_predictions" not in d.tables():
        raise HTTPException(503, "NLP predictions not found. Run `python -m pvsys.pipeline nlp`.")


@app.get("/api/nlp/status")
def nlp_status():
    from pvsys.nlp.encoder import neural_status

    prov = _read_json("provenance.json", "nlp")
    ev = _read_json("evaluation_reference.json", "nlp")
    ann = _read_json("evaluation_annotations.json", "nlp")
    return _clean({
        "available": prov is not None,
        "neural_encoder": neural_status(),
        "provenance": prov,
        "evaluation_reference": ev,
        "evaluation_annotations": ann,
        "disclaimer": NLP_DISCLAIMER,
    })


@app.get("/api/nlp/categories")
def nlp_categories():
    from pvsys.nlp.categories import as_dicts

    cats = as_dicts()
    d = db()
    counts = {}
    if "nlp_predictions" in d.tables():
        for r in d.query("SELECT prediction, count(*) AS terms, sum(cases) AS case_mentions FROM nlp_predictions GROUP BY 1"):
            counts[r["prediction"]] = r
    for c in cats:
        c["terms"] = counts.get(c["id"], {}).get("terms", 0)
        c["case_mentions"] = counts.get(c["id"], {}).get("case_mentions", 0)
    abstain = counts.get("abstain", {})
    return _clean({"categories": cats, "abstained": {"terms": abstain.get("terms", 0),
                                                     "case_mentions": abstain.get("case_mentions", 0)},
                   "disclaimer": NLP_DISCLAIMER})


def _review_overlay(rows: list[dict]) -> list[dict]:
    from pvsys.nlp.review import load_reviews

    rev = load_reviews(get_paths())
    rmap = {r["pt_key"]: r for r in rev.to_dict(orient="records")} if not rev.empty else {}
    for r in rows:
        rv = rmap.get(r["pt_key"])
        if rv:
            r["review_status"] = rv["decision"]
            r["reviewed_category"] = rv["corrected_category"] or None
            r["reviewer"] = rv["reviewer"]
            r["reviewed_at"] = rv["reviewed_at"]
        else:
            r["review_status"] = "unreviewed"
    return rows


@app.get("/api/nlp/predictions")
def nlp_predictions(
    category: str | None = Query(None, max_length=40),
    q: str | None = Query(None, max_length=100),
    min_confidence: float = Query(0.0, ge=0, le=1),
    review_status: Literal["any", "unreviewed", "accepted", "corrected", "rejected", "uncertain"] = "any",
    disagree_only: bool = False,
    sort: Literal["cases", "confidence", "score", "margin"] = "cases",
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    d = db()
    _nlp_required(d)
    where, params = ["confidence >= ?"], [min_confidence]
    if category:
        where.append("prediction = ?")
        params.append(category)
    if q:
        where.append("pt_key ILIKE ?")
        params.append(f"%{q.strip().lower()}%")
    if disagree_only:
        where.append("NOT agrees_with_baseline")
    w = " AND ".join(where)
    cols = ("pt, pt_key, cases, prediction, top_category, second_category, score, second_score, margin, "
            "confidence, abstain_reason, nearest_anchor, baseline_prediction, baseline_top_category, baseline_score, "
            "agrees_with_baseline, prediction_source, model_id, model_revision, taxonomy_version, run_id")
    if review_status == "any":
        total = d.scalar(f"SELECT count(*) FROM nlp_predictions WHERE {w}", params)
        rows = d.query(f"SELECT {cols} FROM nlp_predictions WHERE {w} ORDER BY {sort} DESC, pt_key LIMIT ? OFFSET ?",
                       params + [limit, offset])
        rows = _review_overlay(rows)
    else:
        rows = _review_overlay(d.query(f"SELECT {cols} FROM nlp_predictions WHERE {w} ORDER BY {sort} DESC, pt_key", params))
        rows = [r for r in rows if r["review_status"] == review_status]
        total = len(rows)
        rows = rows[offset: offset + limit]
    return _clean({"total": total, "rows": rows, "disclaimer": NLP_DISCLAIMER})


@app.get("/api/nlp/evaluation")
def nlp_evaluation():
    ev = _read_json("evaluation_reference.json", "nlp")
    ann = _read_json("evaluation_annotations.json", "nlp")
    if ev is None and ann is None:
        raise HTTPException(503, "No evaluation has been run")
    return _clean({"reference": ev, "annotations": ann})


@app.get("/api/nlp/search")
def nlp_search(q: str = Query(..., min_length=2, max_length=200), k: int = Query(20, ge=1, le=100)):
    from pvsys.nlp.search import get_index

    try:
        res = get_index().search(q, k)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    res["results"] = _review_overlay(res["results"])
    return _clean(res)


@app.get("/api/nlp/concept-signals")
def concept_signals(
    q: str = Query(..., min_length=2, max_length=200),
    drug: str | None = Query(None, max_length=200),
    k: int = Query(15, ge=1, le=50),
    min_similarity: float = Query(0.7, ge=0, le=1),
    limit: int = Query(100, ge=1, le=500),
):
    """Semantic expansion of a free-text concept to FAERS terms, then look up their statistical signals.

    The statistics themselves are unchanged pre-computed ROR/PRR/IC values; NLP is only used to choose
    which reaction terms are relevant to the user's concept.
    """
    from pvsys.nlp.search import get_index

    res = get_index().search(q, k, min_similarity=min_similarity)
    keys = [r["pt_key"] for r in res["results"]]
    if not keys:
        return {"query": q, "mode": res["mode"], "terms": [], "rows": [], "disclaimer": DISCLAIMER}
    marks = ",".join("?" * len(keys))
    params: list = list(keys)
    w = f"pt_key IN ({marks}) AND methods_flagging >= 1"
    if drug:
        w += " AND drug = ?"
        params.append(drug.strip().upper())
    rows = db().query(
        f"""SELECT drug, pt, pt_key, a, ror, ror_lower, ror_upper, prr, ic025, methods_flagging,
                   possible_indication_bias FROM signals WHERE {w} ORDER BY ror_lower DESC LIMIT ?""",
        params + [limit],
    )
    by_drug = db().query(
        f"""SELECT drug, count(*) AS flagged_terms, sum(a) AS total_cases, max(ror_lower) AS max_ror_lower
            FROM signals WHERE {w} GROUP BY drug ORDER BY flagged_terms DESC, total_cases DESC LIMIT 25""",
        params,
    )
    return _clean({"query": q, "mode": res["mode"], "terms": res["results"], "rows": rows,
                   "by_drug": by_drug, "disclaimer": DISCLAIMER})


@app.get("/api/nlp/drug-profile")
def nlp_drug_profile(drug: str = Query(..., max_length=200)):
    """Share of a drug's reaction mentions per provisional category vs. the whole database."""
    d = db()
    _nlp_required(d)
    drug = drug.strip().upper()
    rows = d.query(
        """
        WITH dr AS (SELECT DISTINCT primaryid FROM drugs WHERE drug = ? AND role_cod IN ('PS','SS')),
        m AS (SELECT n.prediction, count(*) AS n_drug FROM reactions r JOIN dr USING (primaryid)
              JOIN nlp_predictions n USING (pt_key) GROUP BY 1),
        allm AS (SELECT n.prediction, count(*) AS n_all FROM reactions r JOIN nlp_predictions n USING (pt_key) GROUP BY 1)
        SELECT allm.prediction AS category, coalesce(m.n_drug, 0) AS drug_mentions, allm.n_all AS all_mentions
        FROM allm LEFT JOIN m USING (prediction) ORDER BY drug_mentions DESC
        """,
        [drug],
    )
    tot_d = sum(r["drug_mentions"] for r in rows) or 1
    tot_a = sum(r["all_mentions"] for r in rows) or 1
    for r in rows:
        r["drug_share"] = r["drug_mentions"] / tot_d
        r["all_share"] = r["all_mentions"] / tot_a
    return _clean({"drug": drug, "rows": rows, "note": "Exploratory: shares of reaction mentions by provisional "
                   "category. Not a disproportionality statistic.", "disclaimer": NLP_DISCLAIMER})


class ReviewIn(BaseModel):
    pt_key: str = Field(..., min_length=1, max_length=200)
    decision: Literal["accepted", "corrected", "rejected", "uncertain"]
    reviewer: str = Field(..., min_length=1, max_length=64)
    corrected_category: str | None = Field(None, max_length=40)
    note: str = Field("", max_length=500)

    @field_validator("pt_key")
    @classmethod
    def _norm(cls, v: str) -> str:
        return v.strip().lower()


@app.post("/api/nlp/review")
def nlp_review(body: ReviewIn):
    from pvsys.nlp.review import ReviewError, record_review

    d = db()
    _nlp_required(d)
    rows = d.query("SELECT pt, prediction, run_id FROM nlp_predictions WHERE pt_key = ?", [body.pt_key])
    if not rows:
        raise HTTPException(404, "unknown reaction term")
    try:
        rec = record_review(body.pt_key, rows[0]["pt"], rows[0]["prediction"], body.decision, body.reviewer,
                            body.corrected_category, body.note, rows[0]["run_id"], get_paths())
    except ReviewError as exc:
        raise HTTPException(422, str(exc)) from exc
    return rec


@app.get("/api/nlp/reviews")
def nlp_reviews(limit: int = Query(100, ge=1, le=1000)):
    from pvsys.nlp.review import load_reviews

    df = load_reviews(get_paths(), latest_only=False)
    counts = load_reviews(get_paths())["decision"].value_counts().to_dict() if not df.empty else {}
    rows = df.sort_values("reviewed_at", ascending=False).head(limit).to_dict(orient="records") if not df.empty else []
    return {"counts": counts, "rows": rows,
            "note": "Unblinded verification log. Not used for accuracy metrics (see blinded annotation workflow)."}


# --------------------------------------------------------------------------
# Question answering (retrieval-augmented, evidence-grounded)
# --------------------------------------------------------------------------

EXAMPLE_QUESTIONS = [
    "Is semaglutide associated with pancreatitis?",
    "Does Ozempic cause stomach problems?",
    "Which drugs are linked to suicidal thoughts?",
    "What reactions are reported with montelukast?",
    "Is dupilumab associated with atopic dermatitis?",
    "What does a high ROR mean? Does it prove causation?",
    "Why are some ages in FAERS very large?",
]
_rag_lock = threading.Lock()


@app.on_event("startup")
def _warm_question_answering() -> None:
    """Load the QA engine (SapBERT + indexes, ~10 s on CPU) in the background so the first question is fast.
    Failures are only logged: the dashboard's statistical pages never depend on it."""
    import os

    if os.environ.get("PV_WARMUP", "1") == "0":
        return

    def _load():
        try:
            if (get_paths().nlp / "kb_chunks.parquet").exists():
                with _rag_lock:
                    _rag_engine().answer("Is semaglutide associated with pancreatitis?", generate=False)
                log.info("question-answering engine warmed up")
        except Exception as exc:  # pragma: no cover - best effort
            log.warning("question-answering warm-up skipped: %s", exc)

    threading.Thread(target=_load, name="qa-warmup", daemon=True).start()


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=300)
    generate: bool = False  # opt-in: Qwen2.5-0.5B scored 35% vs 97.5% for the extractive answer (evaluation_evidence.json)
    external: bool = True

    @field_validator("question")
    @classmethod
    def _q(cls, v: str) -> str:
        v = " ".join(v.split())
        if not re.search(r"[A-Za-z]{2,}", v):
            raise ValueError("question must contain words")
        return v


def _rag_engine():
    from pvsys.nlp.rag import get_engine

    d = db()
    try:
        return get_engine(d)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"{exc}. Run `python -m pvsys.pipeline nlp` (or rag-index).") from exc


@lru_cache(maxsize=256)
def _ask_cached(question: str, generate: bool, external: bool = True) -> dict:
    with _rag_lock:
        return _clean(_rag_engine().answer(question, generate=generate, external=external))


@app.post("/api/rag/ask")
def rag_ask(body: AskIn):
    """Answer a free-text question: FAERS statistics (observed reports) plus FDA label / FDA safety communication /
    literature evidence (established clinical evidence), each with citations and source metadata."""
    res = copy.deepcopy(_ask_cached(body.question, body.generate, body.external))
    res["disclaimer"] = RAG_DISCLAIMER
    return res


@app.get("/api/rag/status")
def rag_status():
    from pvsys.nlp.generator import status as gen_status
    from pvsys.nlp.rag import LINK_THRESHOLD, RAG_VERSION

    kb = _read_json("kb_manifest.json", "nlp")
    ev = _read_json("evaluation_rag.json", "nlp")
    return _clean({
        "available": kb is not None and (get_paths().nlp / "predictions.parquet").exists(),
        "rag_version": RAG_VERSION,
        "knowledge_base": kb,
        "link_thresholds": LINK_THRESHOLD,
        "generator": gen_status(),
        "evaluation_available": ev is not None,
        "examples": EXAMPLE_QUESTIONS,
        "disclaimer": RAG_DISCLAIMER,
    })


@app.get("/api/rag/evaluation")
def rag_evaluation():
    ev = _read_json("evaluation_rag.json", "nlp")
    if ev is None:
        raise HTTPException(503, "Question-answering evaluation not run. Run `python -m pvsys.pipeline rag-eval`.")
    return _clean(ev)


# --------------------------------------------------------------------------
# External clinical evidence (FDA labels, FDA Drug Safety Communications, PubMed)
# --------------------------------------------------------------------------

def _evidence_json(name: str):
    from pvsys.evidence.sources import evidence_dir

    p = evidence_dir(get_paths()) / name
    return json.loads(p.read_text()) if p.exists() else None


@app.get("/api/evidence/status")
def evidence_status():
    from pvsys.evidence.embed import status as embed_status
    from pvsys.evidence.evaluate import sufficiency_threshold
    from pvsys.nlp.generator import status as gen_status

    src = _evidence_json("sources_manifest.json")
    return _clean({
        "available": _evidence_json("index_manifest.json") is not None,
        "index": _evidence_json("index_manifest.json"),
        "sources": {k: src.get(k) for k in ("started_at", "finished_at", "drugs_requested", "stats", "sources")} if src else None,
        "failures": len(src.get("failures", [])) if src else None,
        "embedder": embed_status(),
        "generator": gen_status(),
        "sufficiency_threshold": sufficiency_threshold(get_paths()),
        "evaluation_available": _evidence_json("evaluation_evidence.json") is not None,
    })


@app.get("/api/evidence/documents")
def evidence_documents(drug: str | None = Query(None, max_length=200),
                       source_type: Literal["label", "literature", "regulatory"] | None = None,
                       q: str | None = Query(None, max_length=100),
                       limit: int = Query(100, ge=1, le=1000)):
    import pandas as pd

    from pvsys.evidence.sources import evidence_dir

    f = evidence_dir(get_paths()) / "chunks.parquet"
    if not f.exists():
        raise HTTPException(503, "Evidence index not built. Run scripts\\build_evidence.ps1.")
    ch = pd.read_parquet(f, columns=["doc_id", "source_type", "drugs", "title", "publisher", "date", "version", "url",
                                     "identifier"])
    docs = ch.groupby("doc_id", as_index=False).agg(
        source_type=("source_type", "first"), title=("title", "first"), publisher=("publisher", "first"),
        date=("date", "first"), version=("version", "first"), url=("url", "first"), identifier=("identifier", "first"),
        drugs=("drugs", lambda s: "|".join(sorted({d for x in s for d in (x or "").split("|") if d}))),
        chunks=("doc_id", "size"))
    if drug:
        docs = docs[docs["drugs"].str.split("|").apply(lambda ds: drug.strip().upper() in ds)]
    if source_type:
        docs = docs[docs["source_type"] == source_type]
    if q:
        docs = docs[docs["title"].str.contains(q, case=False, regex=False)]
    counts = docs["source_type"].value_counts().to_dict()
    docs = docs.sort_values(["source_type", "title"]).head(limit)
    return _clean({"total": int(sum(counts.values())), "by_type": counts, "rows": docs.to_dict(orient="records")})


@app.get("/api/evidence/search")
def evidence_search(q: str = Query(..., min_length=3, max_length=300), drug: str | None = Query(None, max_length=200),
                    mode: Literal["hybrid", "dense", "bm25"] = "hybrid", k: int = Query(8, ge=1, le=30)):
    from pvsys.evidence.index import get_evidence_index

    try:
        idx = get_evidence_index()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    res = idx.search(q, k=k, drugs=[drug.strip().upper()] if drug else None, mode=mode)
    return _clean({"query": q, **res, "index_mode": idx.mode})


@app.get("/api/evidence/evaluation")
def evidence_evaluation():
    ev = _evidence_json("evaluation_evidence.json")
    if ev is None:
        raise HTTPException(503, "Evidence evaluation not run. Run scripts\\build_evidence.ps1.")
    ev.pop("index", None)
    return _clean(ev)
