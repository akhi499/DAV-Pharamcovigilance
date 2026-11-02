"""Zero-shot prototype classifier with explicit abstention.

For each category the anchors are embedded and averaged into one L2-normalised
prototype (centroid); a term's score for a category is its cosine similarity
to that prototype. (The nearest-anchor "max" rule is kept as an option; the
centroid rule was selected because it was more accurate on the CTCAE *dev*
split - the test split was not used for this choice.) The nearest single
anchor is still reported as a human-readable explanation.
The prediction is the top-scoring category,
unless the classifier abstains because

* the top score is below ``min_score`` (term is not close to any category), or
* the margin between the best and second-best category is below
  ``min_margin`` (term is ambiguous between categories).

A softmax over category scores (temperature ``temperature``) is reported as a
relative confidence. It is NOT a calibrated probability of correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .categories import ABSTAIN, CATEGORIES, anchor_table
from .encoder import similarity


@dataclass
class Thresholds:
    min_score: float
    min_margin: float
    temperature: float = 0.05
    source: str = "a-priori default"

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_THRESHOLDS = {
    # a-priori values used only when no calibration file exists
    "neural": Thresholds(min_score=0.55, min_margin=0.03),
    "lexical": Thresholds(min_score=0.30, min_margin=0.02),
}


class PrototypeClassifier:
    def __init__(self, encoder, thresholds: Thresholds | None = None, scoring: str = "centroid"):
        if scoring not in ("centroid", "max"):
            raise ValueError("scoring must be 'centroid' or 'max'")
        self.scoring = scoring
        self.encoder = encoder
        self.thresholds = thresholds or DEFAULT_THRESHOLDS[encoder.kind]
        table = anchor_table()
        self.category_ids = [c.id for c in CATEGORIES]
        self._anchor_cat = np.array([self.category_ids.index(cid) for cid, _ in table])
        self.anchor_vectors = encoder.encode([a for _, a in table])
        dense = self.anchor_vectors.toarray() if hasattr(self.anchor_vectors, "toarray") else self.anchor_vectors
        cent = np.stack([dense[self._anchor_cat == k].mean(axis=0) for k in range(len(self.category_ids))])
        norms = np.linalg.norm(cent, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.centroids = (cent / norms).astype(np.float32)

    def score_matrix(self, vectors: np.ndarray) -> np.ndarray:
        if self.scoring == "centroid":
            return similarity(vectors, self.centroids)
        sims = similarity(vectors, self.anchor_vectors)  # (n, n_anchors)
        scores = np.full((vectors.shape[0], len(self.category_ids)), -1.0, dtype=np.float32)
        for k in range(len(self.category_ids)):
            cols = self._anchor_cat == k
            scores[:, k] = sims[:, cols].max(axis=1)
        return scores

    def decide(self, scores: np.ndarray, thresholds: Thresholds | None = None) -> pd.DataFrame:
        t = thresholds or self.thresholds
        order = np.argsort(-scores, axis=1)
        top = order[:, 0]
        second = order[:, 1]
        rows = np.arange(scores.shape[0])
        s1 = scores[rows, top]
        s2 = scores[rows, second]
        margin = s1 - s2
        z = (scores - s1[:, None]) / t.temperature
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        conf = probs[rows, top]
        low = s1 < t.min_score
        amb = (~low) & (margin < t.min_margin)
        reason = np.where(low, "low_similarity", np.where(amb, "ambiguous", ""))
        cats = np.array(self.category_ids)
        pred = np.where(low | amb, ABSTAIN, cats[top])
        return pd.DataFrame(
            {
                "top_category": cats[top],
                "second_category": cats[second],
                "score": s1.astype(float),
                "second_score": s2.astype(float),
                "margin": margin.astype(float),
                "confidence": conf.astype(float),
                "prediction": pred,
                "abstain_reason": reason,
            }
        )

    def nearest_anchor(self, vectors: np.ndarray) -> list[str]:
        table = anchor_table()
        idx = similarity(vectors, self.anchor_vectors).argmax(axis=1)
        return [table[i][1] for i in idx]

    def predict(self, texts: list[str], vectors: np.ndarray | None = None) -> pd.DataFrame:
        if vectors is None:
            vectors = self.encoder.encode(texts)
        df = self.decide(self.score_matrix(vectors))
        df.insert(0, "term", texts)
        df["nearest_anchor"] = self.nearest_anchor(vectors)
        return df


def calibrate(
    scores: np.ndarray,
    gold: np.ndarray,
    category_ids: list[str],
    min_coverage: float = 0.7,
    temperature: float = 0.05,
) -> Thresholds:
    """Grid-search abstention thresholds on a *development* split.

    Picks the (min_score, min_margin) pair that maximises accuracy on answered
    items subject to answering at least ``min_coverage`` of the items.
    """
    cats = np.array(category_ids)
    order = np.argsort(-scores, axis=1)
    rows = np.arange(scores.shape[0])
    s1 = scores[rows, order[:, 0]]
    margin = s1 - scores[rows, order[:, 1]]
    correct = cats[order[:, 0]] == gold
    best = (-1.0, 0.0, 0.0, 0.0)
    for ms in np.round(np.arange(0.0, 1.0, 0.01), 2):
        for mm in np.round(np.arange(0.0, 0.2, 0.005), 3):
            answered = (s1 >= ms) & (margin >= mm)
            cov = answered.mean()
            if cov < min_coverage or answered.sum() == 0:
                continue
            acc = correct[answered].mean()
            if acc > best[0] + 1e-12 or (abs(acc - best[0]) < 1e-12 and cov > best[3]):
                best = (acc, ms, mm, cov)
    if best[0] < 0:
        return Thresholds(0.0, 0.0, temperature, source="calibration failed - no abstention")
    return Thresholds(float(best[1]), float(best[2]), temperature,
                      source=f"calibrated on dev split (coverage>={min_coverage})")
