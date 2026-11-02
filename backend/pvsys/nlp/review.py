"""Human-in-the-loop review and blinded annotation workflow.

Two separate records are kept, because they answer different questions:

1. **Prediction reviews** (``data/reviews/prediction_reviews.csv``) - a reviewer
   looks at a model prediction in the dashboard and accepts, corrects or
   rejects it. This is *unblinded verification*: the reviewer sees the model
   output, so it is subject to anchoring bias and is NOT used to compute
   accuracy. It only changes a prediction's review status.

2. **Blinded annotations** (``data/reviews/annotations.csv``) - reviewers
   label a random, stratified batch of terms exported WITHOUT model output.
   These are the only labels used for accuracy metrics (evaluate.py).

Both files are append-only CSV logs (latest entry wins) so every decision is
auditable and the workflow is reproducible from the files alone.
"""

from __future__ import annotations

import csv
import math
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import Paths, get_paths
from .categories import ABSTAIN, CATEGORY_BY_ID

_lock = threading.Lock()

DECISIONS = ("accepted", "corrected", "rejected", "uncertain")
REVIEW_COLUMNS = ["pt_key", "pt", "model_prediction", "decision", "corrected_category", "reviewer", "note",
                  "model_run_id", "reviewed_at"]
ANNOTATION_COLUMNS = ["pt_key", "term", "annotator_id", "label", "notes", "batch_id", "imported_at"]
VALID_LABELS = set(CATEGORY_BY_ID) | {"uncertain"}
_NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-@]{1,64}$")


class ReviewError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


# --------------------------------------------------------------------------
# 1. Prediction reviews (unblinded verification from the dashboard)
# --------------------------------------------------------------------------

def reviews_file(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).reviews / "prediction_reviews.csv"


def record_review(pt_key: str, pt: str, model_prediction: str, decision: str, reviewer: str,
                  corrected_category: str | None = None, note: str = "", model_run_id: str = "",
                  paths: Paths | None = None) -> dict:
    if decision not in DECISIONS:
        raise ReviewError(f"decision must be one of {DECISIONS}")
    if not reviewer or not _NAME_RE.match(reviewer):
        raise ReviewError("reviewer must be 1-64 characters (letters, digits, space, _.-@)")
    if decision == "corrected":
        if corrected_category not in CATEGORY_BY_ID:
            raise ReviewError("corrected_category must be a valid category id when decision is 'corrected'")
    else:
        corrected_category = None
    if model_prediction not in CATEGORY_BY_ID and model_prediction != ABSTAIN:
        raise ReviewError("unknown model_prediction")
    row = {
        "pt_key": pt_key.strip().lower(), "pt": pt.strip(), "model_prediction": model_prediction,
        "decision": decision, "corrected_category": corrected_category or "", "reviewer": reviewer.strip(),
        "note": (note or "")[:500].replace("\n", " "), "model_run_id": model_run_id, "reviewed_at": _now(),
    }
    with _lock:
        _append(reviews_file(paths), REVIEW_COLUMNS, [row])
    return row


def load_reviews(paths: Paths | None = None, latest_only: bool = True) -> pd.DataFrame:
    f = reviews_file(paths)
    if not f.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    if latest_only and not df.empty:
        df = df.sort_values("reviewed_at").groupby("pt_key", as_index=False).tail(1)
    return df


# --------------------------------------------------------------------------
# 2. Blinded annotation batches (ground truth for evaluation)
# --------------------------------------------------------------------------

def annotations_file(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).reviews / "annotations.csv"


def load_annotations(paths: Paths | None = None) -> pd.DataFrame:
    f = annotations_file(paths)
    if not f.exists():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)
    return pd.read_csv(f, dtype=str, keep_default_na=False)


def export_annotation_batch(n: int = 200, seed: int = 7, paths: Paths | None = None) -> Path:
    """Write a blinded CSV of terms to label (no model output included).

    Terms are sampled stratified by the model's *provisional* group (incl.
    abstentions) so rare groups are represented, but the sheet itself never
    shows that group, preventing anchoring.
    """
    paths = paths or get_paths()
    preds = pd.read_parquet(paths.nlp / "predictions.parquet")
    done = set(load_annotations(paths)["pt_key"])
    pool = preds[~preds["pt_key"].isin(done)]
    strata = pool.groupby("prediction")
    per = max(1, math.ceil(n / max(1, strata.ngroups)))
    sample = strata.apply(lambda g: g.sample(min(len(g), per), random_state=seed), include_groups=False)
    sample = sample.reset_index(drop=True).sample(frac=1, random_state=seed).head(n)
    batch_id = datetime.now(timezone.utc).strftime("batch_%Y%m%dT%H%M%S")
    out_dir = paths.reviews / "annotation_batches"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{batch_id}.csv"
    sheet = pd.DataFrame({"pt_key": sample["pt_key"], "term": sample["pt"], "annotator_id": "", "label": "",
                          "notes": "", "batch_id": batch_id})
    sheet.to_csv(out, index=False)
    guide = out_dir / "LABEL_GUIDE.md"
    lines = ["# Labelling guide (blinded)", "",
             "Fill `annotator_id` and `label` for every row. Use one category id below, or `uncertain`.",
             "Do not look at dashboard predictions while labelling. Two independent annotators per batch are",
             "recommended so agreement (Cohen's kappa) can be reported.", "",
             "| id | category | meaning |", "|---|---|---|"]
    lines += [f"| `{c.id}` | {c.label} | {c.description} |" for c in CATEGORY_BY_ID.values()]
    lines += ["| `uncertain` | - | Cannot decide / not a single organ system |"]
    guide.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def import_annotations(csv_path: Path, paths: Paths | None = None) -> dict:
    """Validate a completed batch and append it to the canonical annotations file."""
    paths = paths or get_paths()
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = {"pt_key", "term", "annotator_id", "label"} - set(df.columns)
    if missing:
        raise ReviewError(f"missing columns: {sorted(missing)}")
    df = df[df["label"].str.strip() != ""].copy()
    df["label"] = df["label"].str.strip().str.lower()
    bad = df[~df["label"].isin(VALID_LABELS)]
    if not bad.empty:
        raise ReviewError(f"invalid labels: {sorted(bad['label'].unique())[:10]}")
    if (df["annotator_id"].str.strip() == "").any():
        raise ReviewError("annotator_id is required for every labelled row")
    if not df["annotator_id"].map(lambda s: bool(_NAME_RE.match(s.strip()))).all():
        raise ReviewError("annotator_id contains invalid characters")
    df["imported_at"] = _now()
    if "batch_id" not in df.columns:
        df["batch_id"] = Path(csv_path).stem
    if "notes" not in df.columns:
        df["notes"] = ""
    with _lock:
        _append(annotations_file(paths), ANNOTATION_COLUMNS, df.to_dict(orient="records"))
    return {"imported_rows": int(len(df)), "annotators": sorted(df["annotator_id"].unique().tolist())}
