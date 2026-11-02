"""NLP stage - provisional categorisation of every FAERS reaction term.

Outputs (data/nlp/):
  terms.parquet          unique reaction terms with case counts
  embeddings.npy         SapBERT vectors (float16) aligned with terms.parquet
  predictions.parquet    per-term provisional category, scores, abstention,
                         lexical-baseline prediction and provenance columns
  provenance.json        model id/revision/weights hash, library versions,
                         taxonomy version/hash, thresholds and run id
  evaluation_*.json      see evaluate.py

This stage only reads processed Parquet; the statistical outputs never depend
on it. Re-running it does not modify any statistical file.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import Paths, get_paths
from .categories import CATEGORIES, TAXONOMY_VERSION, anchor_table, as_dicts, taxonomy_hash
from .classify import DEFAULT_THRESHOLDS, PrototypeClassifier, Thresholds
from .encoder import EncoderUnavailable, LexicalEncoder, get_neural_encoder
from .evaluate import evaluate_annotations, evaluate_reference

log = logging.getLogger(__name__)


def load_terms(paths: Paths) -> pd.DataFrame:
    import duckdb

    f = str(paths.processed / "reactions.parquet").replace("\\", "/")
    return duckdb.sql(
        f"SELECT pt_key, any_value(pt) AS pt, count(DISTINCT primaryid) AS cases FROM read_parquet('{f}') "
        "GROUP BY pt_key ORDER BY cases DESC, pt_key"
    ).df()


def _thresholds(paths: Paths, kind: str) -> Thresholds:
    f = paths.nlp / "calibration.json"
    if f.exists():
        d = json.loads(f.read_text()).get(kind)
        if d:
            return Thresholds(**d)
    return DEFAULT_THRESHOLDS[kind]


def run(paths: Paths | None = None, skip_evaluation: bool = False) -> dict:
    paths = (paths or get_paths()).ensure()
    terms = load_terms(paths)
    texts = terms["pt"].tolist()
    started = datetime.now(timezone.utc)

    evaluation = None
    if not skip_evaluation:
        evaluation = evaluate_reference(paths, faers_terms=texts)

    corpus = [a for _, a in anchor_table()] + texts
    lexical = LexicalEncoder(corpus)
    lex_clf = PrototypeClassifier(lexical, _thresholds(paths, "lexical"))
    lex = lex_clf.predict(texts)

    neural_prov = None
    try:
        enc = get_neural_encoder()
        vec = enc.encode(texts)
        np.save(paths.nlp / "embeddings.npy", vec.astype(np.float16))
        clf = PrototypeClassifier(enc, _thresholds(paths, "neural"))
        main = clf.predict(texts, vectors=vec)
        neural_prov = enc.provenance
        source = "neural"
    except EncoderUnavailable as exc:
        log.warning("Neural encoder unavailable (%s); using lexical baseline as primary output", exc)
        main = lex.copy()
        clf = lex_clf
        source = "lexical-fallback"

    run_id = hashlib.sha256(
        json.dumps([source, neural_prov and neural_prov.get("weights_sha256"), taxonomy_hash(),
                    clf.thresholds.to_dict(), len(texts)], sort_keys=True).encode()
    ).hexdigest()[:12]

    out = terms.copy()
    for col in ("top_category", "second_category", "score", "second_score", "margin", "confidence",
                "prediction", "abstain_reason", "nearest_anchor"):
        out[col] = main[col].to_numpy()
    out["baseline_top_category"] = lex["top_category"].to_numpy()
    out["baseline_prediction"] = lex["prediction"].to_numpy()
    out["baseline_score"] = lex["score"].to_numpy()
    out["agrees_with_baseline"] = out["top_category"] == out["baseline_top_category"]
    out["prediction_source"] = source
    out["model_id"] = (neural_prov or lexical.provenance)["model_id"]
    out["model_revision"] = (neural_prov or lexical.provenance)["model_revision"]
    out["taxonomy_version"] = TAXONOMY_VERSION
    out["run_id"] = run_id
    out["status"] = "provisional"  # review status is overlaid at query time from the review log
    out.to_parquet(paths.nlp / "predictions.parquet", index=False)
    terms[["pt_key", "pt", "cases"]].to_parquet(paths.nlp / "terms.parquet", index=False)

    answered = out["prediction"] != "abstain"
    summary = {
        "run_id": run_id,
        "prediction_source": source,
        "created_at": started.isoformat(timespec="seconds"),
        "duration_seconds": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "n_terms": int(len(out)),
        "n_predicted": int(answered.sum()),
        "n_abstained": int((~answered).sum()),
        "abstain_reasons": out.loc[~answered, "abstain_reason"].value_counts().to_dict(),
        "case_weighted_coverage": float(out.loc[answered, "cases"].sum() / out["cases"].sum()),
        "agreement_with_lexical_top1": float(out["agrees_with_baseline"].mean()),
        "category_counts": out["prediction"].value_counts().to_dict(),
    }
    provenance = {
        **summary,
        "model": neural_prov,
        "baseline": lexical.provenance,
        "thresholds": clf.thresholds.to_dict(),
        "taxonomy": {"version": TAXONOMY_VERSION, "hash": taxonomy_hash(), "n_categories": len(CATEGORIES),
                     "n_anchors": len(anchor_table()),
                     "note": "Project-defined categories; not MedDRA SOCs."},
        "data_manifest": json.loads((paths.processed / "manifest.json").read_text())
        if (paths.processed / "manifest.json").exists() else None,
    }
    if provenance["data_manifest"]:
        provenance["data_manifest"] = {k: provenance["data_manifest"][k] for k in ("quarter", "source_zip", "source_sha256")}
    (paths.nlp / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    (paths.nlp / "categories.json").write_text(json.dumps(as_dicts(), indent=2))
    evaluate_annotations(paths)
    log.info("NLP run %s: %s", run_id, summary)
    return {"summary": summary, "evaluation": evaluation}
