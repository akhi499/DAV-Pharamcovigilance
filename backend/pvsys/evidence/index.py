"""Reproducible evidence index: chunks + dense vectors + BM25, with hybrid retrieval.

Build (offline, from cached raw files)::

    data/evidence/chunks.parquet        chunk text + full source metadata
    data/evidence/embeddings.npy        float16 BGE vectors aligned with chunks.parquet (absent if no model)
    data/evidence/index_manifest.json   counts, embedding model id/revision/hash, content hash, timings

Search modes: ``dense`` (cosine over BGE vectors, exact search with NumPy - ~10^4 vectors, so no ANN
library is needed), ``bm25`` (lexical), ``hybrid`` (reciprocal-rank fusion of both, k = 60), and the
improvements of ``rerank.py``: ``dense+route`` (bonus for the label section the question asks about),
``rerank`` (cross-encoder re-ranking of the top dense candidates) and ``rerank+route``. The default
mode is the one chosen on development drugs by ``evaluate.select_mode_on_dev`` (dense on FAERS 2025 Q1).
Results can be restricted to drugs and source types. Without the embedding model, ``hybrid`` and
``dense`` degrade to ``bm25`` and the response says so.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import Paths, get_paths
from .chunking import chunk_documents
from .embed import EmbedderUnavailable, get_embedder, text_hash
from .rerank import RERANK_CANDIDATES, get_reranker, minmax, reranker_available, route_sections
from .sources import build_documents, evidence_dir

log = logging.getLogger(__name__)
RRF_K = 60
CANDIDATES = 100
MODES = ("dense", "bm25", "hybrid", "dense+route", "rerank", "rerank+route")
DEFAULT_ROUTE_WEIGHT = 0.3  # used only until evaluate_v2 has chosen one on dev drugs
INDEX_VERSION = "pv-evidence-index-1.0"


def _passage(r) -> str:
    return f"{r['title']} | {r['section']}: {r['text']}"


def build_index(paths: Paths | None = None, rebuild_documents: bool = True) -> dict:
    paths = paths or get_paths()
    root = evidence_dir(paths)
    t0 = time.time()
    docs = build_documents(paths) if rebuild_documents or not (root / "documents.parquet").exists() \
        else pd.read_parquet(root / "documents.parquet")
    chunks, cinfo = chunk_documents(docs)
    chunks.to_parquet(root / "chunks.parquet", index=False)
    passages = [_passage(r) for r in chunks.to_dict(orient="records")]
    manifest = {"index_version": INDEX_VERSION, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "documents": int(len(docs)), "documents_by_type": docs["source_type"].value_counts().to_dict(),
                "chunks": int(len(chunks)), "chunks_by_type": chunks["source_type"].value_counts().to_dict(),
                "drugs_with_label": int((docs["source_type"] == "label").sum()),
                "content_sha256": text_hash(passages), **cinfo}
    emb_file = root / "embeddings.npy"
    try:
        emb = get_embedder()
        t1 = time.time()
        last = [0.0]

        def progress(i, n):
            if time.time() - last[0] > 15:
                last[0] = time.time()
                log.info("embedding passages: %d/%d", i, n)

        vec = emb.encode(passages, progress=progress)
        np.save(emb_file, vec.astype(np.float16))
        manifest["embedding"] = {**emb.provenance, "dim": emb.dim, "seconds": round(time.time() - t1, 1),
                                 "passages_per_second": round(len(passages) / max(1e-6, time.time() - t1), 1)}
        manifest["retrieval_modes"] = ["hybrid", "dense", "bm25"]
    except EmbedderUnavailable as exc:
        if emb_file.exists():
            emb_file.unlink()
        manifest["embedding"] = {"status": "unavailable", "reason": str(exc)}
        manifest["retrieval_modes"] = ["bm25"]
        log.warning("Embedding model unavailable (%s): index is BM25-only", exc)
    manifest["seconds_total"] = round(time.time() - t0, 1)
    (root / "index_manifest.json").write_text(json.dumps(manifest, indent=2))
    reset_index()
    return manifest


class EvidenceIndex:
    def __init__(self, paths: Paths | None = None):
        from ..nlp.kb import BM25

        self.paths = paths or get_paths()
        root = evidence_dir(self.paths)
        f = root / "chunks.parquet"
        if not f.exists():
            raise FileNotFoundError("Evidence index not built - run scripts\\build_evidence.ps1 "
                                    "(or `python -m pvsys.pipeline evidence-index`)")
        self.chunks = pd.read_parquet(f).reset_index(drop=True)
        man = root / "index_manifest.json"
        self.manifest = json.loads(man.read_text()) if man.exists() else {}
        self.drug_sets = [set(filter(None, (d or "").split("|"))) for d in self.chunks["drugs"]]
        self.indexed_drugs = sorted(set().union(*self.drug_sets)) if self.drug_sets else []
        self.bm25 = BM25([_passage(r) for r in self.chunks.to_dict(orient="records")])
        self.vectors = None
        self.mode = "bm25-only"
        emb_file = root / "embeddings.npy"
        if emb_file.exists():
            vec = np.load(emb_file).astype(np.float32)
            if vec.shape[0] == len(self.chunks):
                try:
                    self.embedder = get_embedder()
                    self.vectors = vec
                    self.mode = "dense"
                except EmbedderUnavailable as exc:
                    log.warning("Query embedder unavailable (%s); evidence search is BM25-only", exc)
            else:
                log.warning("embeddings.npy does not match chunks.parquet; rebuild the index")
        # production mode chosen on dev drugs by evaluate.select_mode_on_dev (stored in evaluation_evidence.json)
        self.route_weight: dict[str, float] = {}
        self.default_mode = self._selected_mode() if self.vectors is not None else "bm25"
        self._qcache: dict[str, np.ndarray] = {}
        self._rcache: dict[tuple[str, int], float] = {}
        self._reranker_ok: bool | None = None
        self._lock = threading.Lock()

    def _selected_mode(self) -> str:
        """Mode chosen on dev drugs: evaluation_evidence_v2.json (improved retrieval) if present, else v1."""
        root = evidence_dir(self.paths)
        for name in ("evaluation_evidence_v2.json", "evaluation_evidence.json"):
            try:
                sel = json.loads((root / name).read_text())["mode_selection"]
            except (OSError, KeyError, ValueError):
                continue
            self.route_weight = {k: float(v) for k, v in sel.get("route_weight", {}).items()}
            if sel.get("selected_mode") in MODES:
                return sel["selected_mode"]
        return "dense"

    def has_reranker(self) -> bool:
        if self._reranker_ok is None:
            self._reranker_ok = reranker_available()
        return self._reranker_ok

    def _rerank(self, q: str, rows: np.ndarray) -> np.ndarray:
        todo = [int(r) for r in rows if (q, int(r)) not in self._rcache]
        if todo:
            texts = [f"{self.chunks.at[r, 'section']}: {self.chunks.at[r, 'text']}" for r in todo]
            for r, v in zip(todo, get_reranker().score(q, texts), strict=True):
                self._rcache[(q, r)] = float(v)
            if len(self._rcache) > 200_000:
                self._rcache.clear()
        return np.array([self._rcache.get((q, int(r)), 0.0) for r in rows])

    def _qvec(self, q: str) -> np.ndarray:
        with self._lock:
            if q not in self._qcache:
                if len(self._qcache) > 512:
                    self._qcache.clear()
                self._qcache[q] = self.embedder.encode([q], is_query=True)[0]
            return self._qcache[q]

    def mask(self, drugs: list[str] | None = None, source_types: tuple[str, ...] | None = None,
             include_untagged_regulatory: bool = False) -> np.ndarray:
        m = np.ones(len(self.chunks), dtype=bool)
        if source_types:
            m &= self.chunks["source_type"].isin(source_types).to_numpy()
        if drugs:
            want = set(drugs)
            m &= np.array([bool(s & want) or (include_untagged_regulatory and not s) for s in self.drug_sets])
        return m

    def search(self, query: str, k: int = 6, drugs: list[str] | None = None, source_types: tuple[str, ...] | None = None,
               mode: str | None = None, section_keys: tuple[str, ...] | None = None) -> dict:
        mode = mode or self.default_mode
        q = " ".join(query.split())
        if not q:
            raise ValueError("query must not be empty")
        m = self.mask(drugs, source_types)
        if section_keys:
            m &= self.chunks["section_key"].isin(section_keys).to_numpy()
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")
        used = mode if (mode == "bm25" or self.vectors is not None) else "bm25"
        note = None
        if used.startswith("rerank") and not self.has_reranker():
            used, note = used.replace("rerank", "dense"), "re-ranker model not installed; dense retrieval used"
        if not m.any():
            return {"mode": used, "hits": [], "note": note}
        idx = np.where(m)[0]
        dense = bm = None
        if used != "bm25" and used != "bm25-only":
            dense = self.vectors[idx] @ self._qvec(q)
        if used in ("bm25", "hybrid"):
            bm = self.bm25.scores(q)[idx]
        routed = route_sections(q) if used.endswith("+route") else ()
        boost = (np.isin(self.chunks["section_key"].to_numpy()[idx], routed).astype(float) if routed
                 else np.zeros(len(idx)))
        beta = self.route_weight.get(used, DEFAULT_ROUTE_WEIGHT) if routed else 0.0
        if used == "dense":
            order = np.argsort(-dense)[:k]
            fused = dense
        elif used == "dense+route":
            fused = minmax(dense) + beta * boost
            order = np.argsort(-fused, kind="stable")[:k]
        elif used.startswith("rerank"):
            pre = minmax(dense) + beta * boost
            cand = np.argsort(-pre, kind="stable")[:RERANK_CANDIDATES]
            final = minmax(self._rerank(q, idx[cand])) + beta * boost[cand]
            order = cand[np.argsort(-final, kind="stable")[:k]]
            fused = np.full(len(idx), -1.0)
            fused[cand] = final
        elif used == "bm25":
            order = [i for i in np.argsort(-bm)[:k] if bm[i] > 0]
            fused = bm
        else:
            rd = np.argsort(-dense)[:CANDIDATES]
            rb = [i for i in np.argsort(-bm)[:CANDIDATES] if bm[i] > 0]
            score: dict[int, float] = {}
            for rank, i in enumerate(rd):
                score[i] = score.get(i, 0) + 1 / (RRF_K + rank + 1)
            for rank, i in enumerate(rb):
                score[i] = score.get(i, 0) + 1 / (RRF_K + rank + 1)
            order = sorted(score, key=lambda i: -score[i])[:k]
            fused = np.zeros(len(idx))
            for i, v in score.items():
                fused[i] = v
        hits = []
        for i in order:
            r = self.chunks.iloc[int(idx[i])]
            hits.append({**{c: (r[c] if not isinstance(r[c], np.generic) else r[c].item()) for c in self.chunks.columns},
                         "score": round(float(fused[i]), 5),
                         "dense_similarity": round(float(dense[i]), 4) if dense is not None else None,
                         "bm25": round(float(bm[i]), 3) if bm is not None else None})
        return {"mode": used, "hits": hits, "note": note, "routed_sections": list(routed),
                "top_dense_similarity": round(float(dense.max()), 4) if dense is not None and len(dense) else None}

    def label_mentions(self, drug: str, terms: list[str]) -> list[dict]:
        """Label chunks of `drug` whose text mentions any of `terms` (whole-word, case-insensitive)."""
        import re

        pats = [re.compile(r"\b" + re.escape(t.lower()) + r"\b") for t in terms if len(t) >= 4]
        if not pats:
            return []
        m = self.mask([drug], ("label",))
        out = []
        for i in np.where(m)[0]:
            r = self.chunks.iloc[int(i)]
            low = r["text"].lower()
            found = [t for t, p in zip(terms, pats, strict=False) if p.search(low)]
            if found:
                out.append({**{c: r[c] for c in self.chunks.columns}, "matched_terms": found})
        order = {k: i for i, k in enumerate(["boxed_warning", "contraindications", "warnings_and_cautions", "warnings",
                                             "precautions", "adverse_reactions"])}
        out.sort(key=lambda r: (order.get(r["section_key"], 9), r["ordinal"]))
        return out


_index: EvidenceIndex | None = None
_index_lock = threading.Lock()


def get_evidence_index() -> EvidenceIndex:
    global _index
    with _index_lock:
        if _index is None:
            _index = EvidenceIndex()
        return _index


def reset_index() -> None:
    global _index
    with _index_lock:
        _index = None
