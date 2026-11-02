"""Reproducible evidence index: chunks + dense vectors + BM25, with hybrid retrieval.

Build (offline, from cached raw files)::

    data/evidence/chunks.parquet        chunk text + full source metadata
    data/evidence/embeddings.npy        float16 BGE vectors aligned with chunks.parquet (absent if no model)
    data/evidence/index_manifest.json   counts, embedding model id/revision/hash, content hash, timings

Search modes: ``dense`` (cosine over BGE vectors, exact search with NumPy - ~10^4 vectors, so no ANN
library is needed), ``bm25`` (lexical) and ``hybrid`` (reciprocal-rank fusion of both, k = 60). The default
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
from .sources import build_documents, evidence_dir

log = logging.getLogger(__name__)
RRF_K = 60
CANDIDATES = 100
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
        self.default_mode = self._selected_mode() if self.vectors is not None else "bm25"
        self._qcache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _selected_mode(self) -> str:
        f = evidence_dir(self.paths) / "evaluation_evidence.json"
        try:
            return json.loads(f.read_text())["mode_selection"]["selected_mode"]
        except (OSError, KeyError, ValueError):
            return "dense"

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
        used = mode if (mode == "bm25" or self.vectors is not None) else "bm25"
        if not m.any():
            return {"mode": used, "hits": []}
        idx = np.where(m)[0]
        dense = bm = None
        if used in ("dense", "hybrid"):
            dense = self.vectors[idx] @ self._qvec(q)
        if used in ("bm25", "hybrid"):
            bm = self.bm25.scores(q)[idx]
        if used == "dense":
            order = np.argsort(-dense)[:k]
            fused = dense
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
        return {"mode": used, "hits": hits}

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
