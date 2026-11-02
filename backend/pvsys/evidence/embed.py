"""Sentence embeddings for passage retrieval: BAAI/bge-small-en-v1.5 (33M parameters, MIT, CPU).

BGE uses the [CLS] vector, L2-normalised; queries (not passages) get BGE's retrieval instruction.
Weights are loaded only from a local folder (``models/bge-small-en-v1.5`` or ``PV_EMBED_DIR``).
If torch/transformers or the weights are missing, ``EmbedderUnavailable`` is raised and the index
falls back to BM25 only (and says so).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import get_paths

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
MAX_LENGTH = 512


class EmbedderUnavailable(RuntimeError):
    pass


def default_embed_dir() -> Path:
    return Path(os.environ.get("PV_EMBED_DIR", get_paths().models / "bge-small-en-v1.5"))


def status() -> dict:
    d = default_embed_dir()
    st = {"model_dir": str(d), "weights_present": (d / "config.json").exists() and
          ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists())}
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        st["deps_installed"] = True
    except ImportError:
        st["deps_installed"] = False
    meta = d / "provenance.json"
    st["model_id"] = json.loads(meta.read_text()).get("model_id") if meta.exists() else None
    st["available"] = st["weights_present"] and st["deps_installed"]
    return st


class Embedder:
    def __init__(self, model_dir: Path | None = None, batch_size: int = 32):
        self.model_dir = Path(model_dir or default_embed_dir())
        weights = [p for p in ("model.safetensors", "pytorch_model.bin") if (self.model_dir / p).exists()]
        if not (self.model_dir / "config.json").exists() or not weights:
            raise EmbedderUnavailable(f"embedding model not found in {self.model_dir}")
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise EmbedderUnavailable(f"torch/transformers not installed: {exc}") from exc
        from ..device import device_info, pick_device

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self._torch = torch
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModel.from_pretrained(str(self.model_dir)).eval().to(self.device)
        self.batch_size = batch_size * (4 if self.device == "cuda" else 1)
        self.dim = int(self.model.config.hidden_size)
        meta = self.model_dir / "provenance.json"
        m = json.loads(meta.read_text()) if meta.exists() else {}
        self.provenance = {"model_id": m.get("model_id", self.model_dir.name), "revision": m.get("revision"),
                           "license": m.get("license"), "weights_sha256": m.get("weights_sha256"),
                           "pooling": "CLS, L2-normalised", "max_length": MAX_LENGTH,
                           "query_instruction": QUERY_INSTRUCTION, **device_info()}
        self._lock = threading.Lock()

    def encode(self, texts: list[str], is_query: bool = False, progress=None) -> np.ndarray:
        torch = self._torch
        if is_query:
            texts = [QUERY_INSTRUCTION + t for t in texts]
        out = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                b = self.tok(texts[i : i + self.batch_size], padding=True, truncation=True, max_length=MAX_LENGTH,
                             return_tensors="pt").to(self.device)
                v = self.model(**b).last_hidden_state[:, 0].float().cpu().numpy()
                out.append(v)
                if progress:
                    progress(min(i + self.batch_size, len(texts)), len(texts))
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = np.vstack(out)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1
        return (v / n).astype(np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()


def text_hash(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]
