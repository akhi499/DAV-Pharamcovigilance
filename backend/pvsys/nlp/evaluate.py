"""Evaluation of the provisional categoriser - only against independent references.

Two kinds of ground truth are supported; nothing is ever labelled by the model
authors or by an LLM:

1. **NCI CTCAE v5.0 reference** (``data/reference/CTCAE_v5.0*.xlsx``): a public
   US-government terminology in which NCI experts group adverse-event terms
   under organ-system headings. A project-defined crosswalk
   (categories.CTCAE_SOC_TO_CATEGORY) maps those headings to our categories.
   This is an *external proxy* reference: it measures agreement with NCI's
   grouping, not clinical correctness on FAERS data.
2. **Blinded human annotations** (``data/reviews/annotations.csv`` produced by
   review.export_annotation_batch and filled in by reviewers who do not see
   model output). Metrics are computed only once such labels exist.

Protocol for (1): reference terms are de-duplicated, terms equal to any anchor
phrase are removed (leakage control), then split 30% dev / 70% test,
stratified by category with a fixed seed. Abstention thresholds are tuned on
dev only; all reported numbers are on the untouched test split.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, get_paths
from .categories import ABSTAIN, CATEGORIES, CTCAE_SOC_TO_CATEGORY, TAXONOMY_VERSION, anchor_table
from .classify import PrototypeClassifier, calibrate
from .encoder import EncoderUnavailable, LexicalEncoder, get_neural_encoder

log = logging.getLogger(__name__)
SEED = 42
DEV_FRACTION = 0.3


def find_ctcae(paths: Paths) -> Path | None:
    ref = paths.root / "reference"
    if not ref.exists():
        return None
    files = sorted(ref.glob("CTCAE*v5*.xlsx")) + sorted(ref.glob("ctcae*v5*.xlsx"))
    return files[0] if files else None


def load_ctcae(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(path, sheet_name=0)
    soc_col = next(c for c in raw.columns if "SOC" in str(c))
    term_col = next(c for c in raw.columns if "Term" in str(c))
    df = pd.DataFrame({"soc": raw[soc_col].astype(str).str.strip(), "term": raw[term_col].astype(str).str.strip()})
    info = {"source_file": path.name, "rows": int(len(df))}
    df = df[~df["term"].str.contains(r"other, specify", case=False, regex=True)]
    info["after_removing_other_specify"] = int(len(df))
    df["soc_key"] = df["soc"].str.lower()
    df["category"] = df["soc_key"].map(CTCAE_SOC_TO_CATEGORY)
    unmapped = df[df["category"].isna()]["soc"].value_counts().to_dict()
    info["excluded_unmapped_groups"] = {k: int(v) for k, v in unmapped.items()}
    df = df.dropna(subset=["category"])
    df["term_key"] = df["term"].str.lower().str.replace(r"\s+", " ", regex=True)
    # a term listed under two groups would be ambiguous ground truth -> drop
    multi = df.groupby("term_key")["category"].nunique()
    df = df[df["term_key"].isin(multi[multi == 1].index)].drop_duplicates("term_key")
    info["unique_mapped_terms"] = int(len(df))
    anchors = {a.lower() for _, a in anchor_table()}
    leak = df["term_key"].isin(anchors)
    info["excluded_equal_to_anchor"] = int(leak.sum())
    df = df[~leak].reset_index(drop=True)
    info["evaluation_terms"] = int(len(df))
    return df, info


def stratified_split(labels: pd.Series, dev_fraction: float = DEV_FRACTION, seed: int = SEED) -> np.ndarray:
    """Boolean mask of dev items; every class with >=2 items contributes >=1 to each side."""
    rng = np.random.default_rng(seed)
    is_dev = np.zeros(len(labels), dtype=bool)
    for _, idx in labels.groupby(labels).groups.items():
        idx = np.array(sorted(idx))
        rng.shuffle(idx)
        if len(idx) < 2:
            continue
        k = max(1, int(round(len(idx) * dev_fraction)))
        k = min(k, len(idx) - 1)
        is_dev[idx[:k]] = True
    return is_dev


def _bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> list[float] | None:
    if len(correct) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def metrics(pred: np.ndarray, top1: np.ndarray, gold: np.ndarray) -> dict:
    from sklearn.metrics import f1_score

    answered = pred != ABSTAIN
    correct_ans = (pred == gold)[answered]
    top1_correct = top1 == gold
    labels = sorted(set(gold))
    per_cat = {}
    for c in labels:
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn_ = int(((pred != c) & (gold == c)).sum())
        per_cat[c] = {
            "support": int((gold == c).sum()),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn_) if tp + fn_ else None,
            "abstained": int(((gold == c) & ~answered).sum()),
        }
    return {
        "n": int(len(gold)),
        "coverage": float(answered.mean()) if len(gold) else None,
        "answered": int(answered.sum()),
        "abstained": int((~answered).sum()),
        "selective_accuracy": float(correct_ans.mean()) if answered.any() else None,
        "selective_accuracy_95ci": _bootstrap_ci(correct_ans.astype(float)),
        "macro_f1_answered": float(f1_score(gold[answered], pred[answered], labels=labels, average="macro",
                                            zero_division=0)) if answered.any() else None,
        "top1_accuracy_no_abstention": float(top1_correct.mean()) if len(gold) else None,
        "top1_accuracy_95ci": _bootstrap_ci(top1_correct.astype(float)),
        "per_category": per_cat,
    }


def risk_coverage(scores: np.ndarray, top1: np.ndarray, gold: np.ndarray) -> list[dict]:
    """Accuracy of the k most-confident predictions (by top score) for k = 10%..100%."""
    order = np.argsort(-scores)
    correct = (top1 == gold)[order]
    out = []
    for frac in np.arange(0.1, 1.01, 0.1):
        k = max(1, int(round(frac * len(order))))
        out.append({"coverage": round(float(frac), 2), "accuracy": float(correct[:k].mean())})
    return out


def evaluate_reference(paths: Paths | None = None, faers_terms: list[str] | None = None) -> dict:
    paths = paths or get_paths()
    ref_path = find_ctcae(paths)
    if ref_path is None:
        return {"status": "unavailable", "reason": "CTCAE v5.0 reference file not found in data/reference/"}
    ref, info = load_ctcae(ref_path)
    faers_keys = {t.lower() for t in (faers_terms or [])}
    ref["in_faers"] = ref["term_key"].isin(faers_keys)
    is_dev = stratified_split(ref["category"])
    ref["split"] = np.where(is_dev, "dev", "test")
    info["dev_terms"] = int(is_dev.sum())
    info["test_terms"] = int((~is_dev).sum())
    info["test_terms_in_faers_2025q1"] = int((~is_dev & ref["in_faers"]).sum())

    gold = ref["category"].to_numpy()
    cat_ids = [c.id for c in CATEGORIES]
    systems = {}
    calib = {}
    encoders = {}
    corpus = [a for _, a in anchor_table()] + list(faers_terms or [])
    encoders["lexical"] = LexicalEncoder(corpus)
    try:
        encoders["neural"] = get_neural_encoder()
    except EncoderUnavailable as exc:
        systems["neural"] = {"status": "unavailable", "reason": str(exc)}

    preds_out = []
    for name, enc in encoders.items():
        clf = PrototypeClassifier(enc)
        vec = enc.encode(ref["term"].tolist())
        scores = clf.score_matrix(vec)
        th = calibrate(scores[is_dev], gold[is_dev], cat_ids)
        calib[name] = th.to_dict()
        dec = clf.decide(scores, th)
        test = ~is_dev
        res = {
            "status": "ok",
            "encoder": enc.provenance.get("encoder"),
            "thresholds": th.to_dict(),
            "test": metrics(dec["prediction"].to_numpy()[test], dec["top_category"].to_numpy()[test], gold[test]),
            "test_in_faers": metrics(
                dec["prediction"].to_numpy()[test & ref["in_faers"].to_numpy()],
                dec["top_category"].to_numpy()[test & ref["in_faers"].to_numpy()],
                gold[test & ref["in_faers"].to_numpy()],
            ),
            "risk_coverage_test": risk_coverage(dec["score"].to_numpy()[test], dec["top_category"].to_numpy()[test], gold[test]),
        }
        cm = pd.crosstab(pd.Series(gold[test], name="gold"), pd.Series(dec["prediction"].to_numpy()[test], name="pred"))
        res["confusion_test"] = {"labels_gold": cm.index.tolist(), "labels_pred": cm.columns.tolist(),
                                 "matrix": cm.to_numpy().tolist()}
        systems[name] = res
        part = ref[["term", "category", "split", "in_faers"]].copy()
        part["system"] = name
        part = pd.concat([part.reset_index(drop=True), dec.reset_index(drop=True)], axis=1)
        preds_out.append(part)

    majority = pd.Series(gold[is_dev]).value_counts().idxmax()
    systems["majority_class"] = {
        "status": "ok",
        "predicts": majority,
        "test_accuracy": float((gold[~is_dev] == majority).mean()),
    }
    result = {
        "reference": "NCI CTCAE v5.0 (MedDRA v20.1 SOC groupings) via project crosswalk",
        "taxonomy_version": TAXONOMY_VERSION,
        "protocol": {"seed": SEED, "dev_fraction": DEV_FRACTION, "threshold_objective":
                     "max selective accuracy on dev s.t. coverage >= 0.70"},
        "dataset": info,
        "systems": systems,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveats": [
            "CTCAE is an oncology-oriented terminology; its terms are cleaner and more 'disorder-like' than many FAERS PTs.",
            "Product-use/medication-error terms common in FAERS have no CTCAE counterpart and are not evaluated here.",
            "Agreement with NCI's organ-system grouping is not proof of clinical correctness on FAERS reports.",
            "The SOC-to-category crosswalk is defined by this project, not by MedDRA/MSSO.",
        ],
    }
    paths.nlp.mkdir(parents=True, exist_ok=True)
    (paths.nlp / "evaluation_reference.json").write_text(json.dumps(result, indent=2, default=float))
    (paths.nlp / "calibration.json").write_text(json.dumps(calib, indent=2))
    pd.concat(preds_out).to_parquet(paths.nlp / "evaluation_reference_predictions.parquet", index=False)
    return result


def evaluate_annotations(paths: Paths | None = None) -> dict:
    """Score current predictions against blinded human annotations, if any exist."""
    from .review import load_annotations

    paths = paths or get_paths()
    ann = load_annotations(paths)
    pred_file = paths.nlp / "predictions.parquet"
    if ann.empty or not pred_file.exists():
        result = {"status": "no_annotations",
                  "message": "No blinded human annotations have been imported yet; no accuracy is claimed."}
        (paths.nlp / "evaluation_annotations.json").write_text(json.dumps(result, indent=2))
        return result
    # inter-annotator agreement on doubly-annotated terms
    kappa = None
    multi = ann.groupby("pt_key")["annotator_id"].nunique()
    double = ann[ann["pt_key"].isin(multi[multi >= 2].index)]
    if not double.empty:
        from sklearn.metrics import cohen_kappa_score

        pairs = double.sort_values("annotator_id").groupby("pt_key")["label"].apply(list)
        a1 = [p[0] for p in pairs]
        a2 = [p[1] for p in pairs]
        kappa = float(cohen_kappa_score(a1, a2)) if len(set(a1 + a2)) > 1 else None
    # gold = majority label per term (ties -> excluded); 'uncertain' labels excluded
    votes = ann[ann["label"] != "uncertain"].groupby(["pt_key", "label"]).size().reset_index(name="v")
    top = votes.sort_values("v", ascending=False).groupby("pt_key")
    gold = top.head(1).set_index("pt_key")["label"]
    tied = top["v"].apply(lambda s: len(s) > 1 and s.iloc[0] == s.iloc[1])
    gold = gold[~gold.index.isin(tied[tied].index)]
    preds = pd.read_parquet(pred_file).set_index("pt_key")
    common = gold.index.intersection(preds.index)
    if len(common) == 0:
        return {"status": "no_overlap"}
    res = {"status": "ok", "n_annotated_terms": int(ann["pt_key"].nunique()),
           "n_gold_terms": int(len(common)), "annotators": sorted(ann["annotator_id"].unique().tolist()),
           "cohens_kappa_double_annotated": kappa}
    for system, pcol, tcol in (("neural", "prediction", "top_category"),
                               ("lexical", "baseline_prediction", "baseline_top_category")):
        if pcol in preds.columns:
            res[system] = metrics(preds.loc[common, pcol].to_numpy(), preds.loc[common, tcol].to_numpy(),
                                  gold.loc[common].to_numpy())
    (paths.nlp / "evaluation_annotations.json").write_text(json.dumps(res, indent=2, default=float))
    return res
