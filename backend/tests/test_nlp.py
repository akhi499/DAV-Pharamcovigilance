"""Tests of the NLP module: classifier/abstention, calibration, review workflow,
graceful degradation and independence from the statistical pipeline."""

from __future__ import annotations

import hashlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pvsys.config import PROJECT_ROOT
from pvsys.nlp.categories import ABSTAIN, CATEGORIES, CTCAE_SOC_TO_CATEGORY, anchor_table, taxonomy_hash
from pvsys.nlp.classify import PrototypeClassifier, Thresholds, calibrate
from pvsys.nlp.encoder import EncoderUnavailable, LexicalEncoder, SapBERTEncoder
from pvsys.nlp.evaluate import metrics, stratified_split

CAT_IDS = [c.id for c in CATEGORIES]


@pytest.fixture(scope="module")
def lexical():
    return LexicalEncoder([a for _, a in anchor_table()] + ["drug induced liver injury", "zzqx wvvk"])


def test_taxonomy_integrity():
    ids = [c.id for c in CATEGORIES]
    assert len(ids) == len(set(ids)) and ABSTAIN not in ids
    assert all(len(c.anchors) >= 5 for c in CATEGORIES)
    assert set(CTCAE_SOC_TO_CATEGORY.values()) <= set(ids)
    assert len(taxonomy_hash()) == 16


def test_lexical_classifier_obvious_and_abstain(lexical):
    clf = PrototypeClassifier(lexical, Thresholds(min_score=0.2, min_margin=0.0))
    out = clf.predict(["drug induced liver injury", "zzqx wvvk"])
    assert out.loc[0, "prediction"] == "hepatobiliary"
    assert out.loc[1, "prediction"] == ABSTAIN and out.loc[1, "abstain_reason"] == "low_similarity"
    assert 0 <= out.loc[0, "confidence"] <= 1


def test_decide_margin_abstention(lexical):
    clf = PrototypeClassifier(lexical)
    scores = np.zeros((2, len(CAT_IDS)), dtype=np.float32)
    scores[0, 0], scores[0, 1] = 0.80, 0.79  # ambiguous
    scores[1, 2], scores[1, 3] = 0.80, 0.50  # clear
    dec = clf.decide(scores, Thresholds(min_score=0.5, min_margin=0.05))
    assert dec.loc[0, "prediction"] == ABSTAIN and dec.loc[0, "abstain_reason"] == "ambiguous"
    assert dec.loc[1, "prediction"] == CAT_IDS[2]
    assert dec.loc[1, "margin"] == pytest.approx(0.3, abs=1e-6)


def test_scoring_option_validated(lexical):
    with pytest.raises(ValueError):
        PrototypeClassifier(lexical, scoring="median")


def test_calibrate_respects_coverage():
    rng = np.random.default_rng(0)
    n = 200
    gold = np.array([CAT_IDS[i % 4] for i in range(n)])
    scores = rng.uniform(0, 0.5, size=(n, len(CAT_IDS))).astype(np.float32)
    for i in range(n):
        scores[i, CAT_IDS.index(gold[i])] += 0.3 if i % 5 else -0.3  # 80% easy, 20% wrong
    th = calibrate(scores, gold, CAT_IDS, min_coverage=0.7)
    s1 = scores.max(1)
    margin = s1 - np.sort(scores, 1)[:, -2]
    assert ((s1 >= th.min_score) & (margin >= th.min_margin)).mean() >= 0.7
    assert "calibrated" in th.source


def test_metrics_counts_abstentions():
    gold = np.array(["a", "a", "b", "b"])
    pred = np.array(["a", ABSTAIN, "b", "a"])
    top1 = np.array(["a", "b", "b", "a"])
    m = metrics(pred, top1, gold)
    assert m["coverage"] == 0.75 and m["answered"] == 3
    assert m["selective_accuracy"] == pytest.approx(2 / 3)
    assert m["top1_accuracy_no_abstention"] == pytest.approx(0.5)


def test_stratified_split_is_deterministic_and_balanced():
    labels = pd.Series(["x"] * 10 + ["y"] * 4 + ["z"])
    a = stratified_split(labels)
    b = stratified_split(labels)
    assert (a == b).all()
    assert a[:10].sum() == 3 and a[10:14].sum() == 1 and a[14] == np.False_  # singleton stays in test


def test_missing_weights_raise_unavailable(tmp_path):
    with pytest.raises(EncoderUnavailable):
        SapBERTEncoder(model_dir=tmp_path)


SAPBERT_DIR = PROJECT_ROOT / "models" / "SapBERT-from-PubMedBERT-fulltext"


@pytest.mark.skipif(not (SAPBERT_DIR / "model.safetensors").exists(), reason="SapBERT weights not present")
def test_sapbert_synonymy():
    try:
        enc = SapBERTEncoder(model_dir=SAPBERT_DIR)
    except EncoderUnavailable as exc:
        pytest.skip(str(exc))
    v = enc.encode(["heart attack", "myocardial infarction", "skin rash"])
    assert v.shape == (3, 768)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4)
    assert v[0] @ v[1] > v[0] @ v[2]
    prov = enc.provenance
    assert prov["model_id"] == "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" and len(prov["weights_sha256"]) == 64


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def nlp_run(pipeline_paths):
    from pvsys.nlp import run as nlp_run

    before = _sha(pipeline_paths.processed / "signals.parquet")
    res = nlp_run.run(pipeline_paths)
    return pipeline_paths, res, before


def test_nlp_run_falls_back_gracefully(nlp_run):
    paths, res, _ = nlp_run
    assert res["summary"]["prediction_source"] == "lexical-fallback"
    assert res["evaluation"]["status"] == "unavailable"  # no CTCAE file in the test data dir
    preds = pd.read_parquet(paths.nlp / "predictions.parquet")
    assert {"pt_key", "prediction", "score", "margin", "confidence", "model_id", "run_id", "taxonomy_version",
            "baseline_prediction", "status"} <= set(preds.columns)
    assert (preds["status"] == "provisional").all()
    assert set(preds["prediction"]) <= set(CAT_IDS) | {ABSTAIN}
    assert (paths.nlp / "provenance.json").exists()


def test_nlp_does_not_touch_statistics(nlp_run):
    paths, _, before = nlp_run
    assert _sha(paths.processed / "signals.parquet") == before


def test_statistics_do_not_import_nlp_stack():
    code = ("import sys; import pvsys.signals, pvsys.clean, pvsys.ingest, pvsys.quality, pvsys.eda; "
            "bad=[m for m in ('torch','transformers','pvsys.nlp') if m in sys.modules]; print(bad); sys.exit(bool(bad))")
    r = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT / "backend", capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_review_log_validation_and_latest_wins(nlp_run):
    from pvsys.nlp.review import ReviewError, load_reviews, record_review

    paths, _, _ = nlp_run
    record_review("nausea", "Nausea", "gastrointestinal", "accepted", "rev1", paths=paths)
    record_review("nausea", "Nausea", "gastrointestinal", "corrected", "rev2", corrected_category="general", paths=paths)
    latest = load_reviews(paths)
    row = latest[latest.pt_key == "nausea"].iloc[0]
    assert row.decision == "corrected" and row.corrected_category == "general"
    with pytest.raises(ReviewError):
        record_review("nausea", "Nausea", "gastrointestinal", "corrected", "rev1", corrected_category="nope", paths=paths)
    with pytest.raises(ReviewError):
        record_review("nausea", "Nausea", "gastrointestinal", "maybe", "rev1", paths=paths)
    with pytest.raises(ReviewError):
        record_review("nausea", "Nausea", "gastrointestinal", "accepted", "<script>", paths=paths)


def test_blinded_annotation_roundtrip(nlp_run):
    from pvsys.nlp.evaluate import evaluate_annotations
    from pvsys.nlp.review import ReviewError, export_annotation_batch, import_annotations

    paths, _, _ = nlp_run
    assert evaluate_annotations(paths)["status"] == "no_annotations"
    batch = export_annotation_batch(n=3, paths=paths)
    sheet = pd.read_csv(batch, dtype=str, keep_default_na=False)
    assert "prediction" not in sheet.columns and "score" not in sheet.columns  # blinded
    assert (batch.parent / "LABEL_GUIDE.md").exists()
    sheet["annotator_id"] = "ann1"
    sheet["label"] = "general"
    sheet.to_csv(batch, index=False)
    assert import_annotations(batch, paths)["imported_rows"] == len(sheet)
    res = evaluate_annotations(paths)
    assert res["status"] == "ok" and res["n_gold_terms"] == len(sheet)
    sheet["label"] = "not-a-category"
    sheet.to_csv(batch, index=False)
    with pytest.raises(ReviewError):
        import_annotations(batch, paths)


def test_semantic_search_lexical_fallback(nlp_run):
    from pvsys.nlp.search import SemanticIndex

    paths, _, _ = nlp_run
    idx = SemanticIndex(paths)
    res = idx.search("nausea", k=3)
    assert res["mode"] == "lexical-fallback"
    assert res["results"][0]["pt_key"] == "nausea"
    with pytest.raises(ValueError):
        idx.search("   ")
