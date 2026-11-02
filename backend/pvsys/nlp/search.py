"""Semantic retrieval over the FAERS reaction vocabulary.

A free-text query ("liver damage", "irregular heartbeat") is embedded with the
same encoder used for categorisation and compared to every cached term
embedding. This lets users find all related MedDRA-coded terms that a plain
substring search would miss, and aggregate disproportionality results over a
concept. If the neural model is unavailable the index transparently falls back
to the lexical (character n-gram) encoder and says so in its response.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import Paths, get_paths
from .categories import anchor_table
from .encoder import EncoderUnavailable, LexicalEncoder, get_neural_encoder, similarity

log = logging.getLogger(__name__)
MAX_QUERY_LEN = 200


class SemanticIndex:
    def __init__(self, paths: Paths | None = None):
        self.paths = paths or get_paths()
        pred_file = self.paths.nlp / "predictions.parquet"
        if not pred_file.exists():
            raise FileNotFoundError("NLP predictions not found - run `python -m pvsys.pipeline nlp` first")
        self.terms = pd.read_parquet(pred_file)
        self._lock = threading.Lock()
        self.mode = "lexical-fallback"
        self.encoder = None
        emb_file = self.paths.nlp / "embeddings.npy"
        try:
            if not emb_file.exists():
                raise EncoderUnavailable("cached embeddings.npy missing")
            vec = np.load(emb_file).astype(np.float32)
            if vec.shape[0] != len(self.terms):
                raise EncoderUnavailable("embeddings do not match predictions (re-run the NLP stage)")
            self.encoder = get_neural_encoder()
            self.vectors = vec
            self.mode = "neural"
        except EncoderUnavailable as exc:
            log.warning("Semantic search using lexical fallback: %s", exc)
            corpus = [a for _, a in anchor_table()] + self.terms["pt"].tolist()
            self.encoder = LexicalEncoder(corpus)
            self.vectors = self.encoder.encode(self.terms["pt"].tolist())

    @lru_cache(maxsize=512)  # noqa: B019 - index is a long-lived singleton
    def _embed(self, text: str):
        with self._lock:
            return self.encoder.encode([text])

    def search(self, query: str, k: int = 20, min_similarity: float = 0.0) -> dict:
        q = " ".join(query.split())[:MAX_QUERY_LEN]
        if not q:
            raise ValueError("query must not be empty")
        sims = similarity(self.vectors, self._embed(q)).ravel()
        k = max(1, min(int(k), 200))
        idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        rows = []
        for i in idx:
            if sims[i] < min_similarity:
                continue
            t = self.terms.iloc[i]
            rows.append({"pt": t["pt"], "pt_key": t["pt_key"], "cases": int(t["cases"]),
                         "similarity": round(float(sims[i]), 4), "prediction": t["prediction"],
                         "top_category": t["top_category"]})
        return {"query": q, "mode": self.mode, "results": rows}


_index: SemanticIndex | None = None
_index_lock = threading.Lock()


def get_index() -> SemanticIndex:
    global _index
    with _index_lock:
        if _index is None:
            _index = SemanticIndex()
        return _index


def reset_index() -> None:
    global _index
    with _index_lock:
        _index = None
