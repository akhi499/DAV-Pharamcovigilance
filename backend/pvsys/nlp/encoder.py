"""Text encoders for reaction terms.

Two interchangeable encoders produce L2-normalised vectors:

* ``SapBERTEncoder`` - the pretrained biomedical language model
  ``cambridgeltl/SapBERT-from-PubMedBERT-fulltext`` (BERT-base, 110M params,
  Apache-2.0). SapBERT is PubMedBERT further trained with self-alignment
  pre-training on UMLS synonyms, so that different surface forms of the same
  biomedical concept get similar [CLS] embeddings. Runs on CPU; no GPU or
  paid API required. Weights are loaded from a local folder only.
* ``LexicalEncoder`` - TF-IDF over character n-grams: the non-neural baseline
  and the graceful fallback when torch/transformers or the weights are absent.

The statistical pipeline never imports this module.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import get_paths

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DEFAULT_MODEL_REVISION = "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"  # HF commit the weights were downloaded from
MODEL_LICENSE = "Apache-2.0"
MAX_LENGTH = 25  # SapBERT was trained with max_length 25 for short entity names


class EncoderUnavailable(RuntimeError):
    """Raised when the neural encoder cannot be loaded (missing deps or weights)."""


def default_model_dir() -> Path:
    return Path(os.environ.get("PV_SAPBERT_DIR", get_paths().models / "SapBERT-from-PubMedBERT-fulltext"))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def similarity(a, b) -> np.ndarray:
    """Dense cosine-similarity matrix a @ b.T for dense or sparse L2-normalised inputs."""
    out = a @ b.T
    if hasattr(out, "toarray"):
        out = out.toarray()
    return np.asarray(out, dtype=np.float32)


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class SapBERTEncoder:
    kind = "neural"

    def __init__(self, model_dir: Path | None = None, batch_size: int = 128):
        self.model_dir = Path(model_dir or default_model_dir())
        required = ["config.json", "vocab.txt"]
        weights = [p for p in ("model.safetensors", "pytorch_model.bin") if (self.model_dir / p).exists()]
        missing = [f for f in required if not (self.model_dir / f).exists()]
        if missing or not weights:
            raise EncoderUnavailable(
                f"SapBERT weights not found in {self.model_dir} (missing: {missing or 'weights file'}). "
                "Run scripts/download_model.py or see README 'NLP model'."
            )
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EncoderUnavailable(f"torch/transformers not installed: {exc}") from exc

        from ..device import pick_device

        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self._torch = torch
        self.device = pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModel.from_pretrained(str(self.model_dir)).eval().to(self.device)
        self.batch_size = batch_size
        self.weights_file = self.model_dir / weights[0]
        self._provenance: dict | None = None

    @property
    def provenance(self) -> dict:
        if self._provenance is None:
            import transformers

            meta_file = self.model_dir / "provenance.json"
            meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
            self._provenance = {
                "encoder": "SapBERT",
                "model_id": meta.get("model_id", DEFAULT_MODEL_ID),
                "model_revision": meta.get("revision", DEFAULT_MODEL_REVISION),
                "license": MODEL_LICENSE,
                "weights_file": self.weights_file.name,
                "weights_sha256": _file_sha256(self.weights_file),
                "pooling": "CLS token, L2-normalised",
                "max_length": MAX_LENGTH,
                "torch": self._torch.__version__,
                "transformers": transformers.__version__,
                "device": self.device,
                "python": platform.python_version(),
            }
        return self._provenance

    def encode(self, texts: list[str], max_length: int = MAX_LENGTH) -> np.ndarray:
        torch = self._torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                toks = self.tokenizer(batch, padding=True, truncation=True, max_length=max_length,
                                      return_tensors="pt").to(self.device)
                cls = self.model(**toks).last_hidden_state[:, 0, :]
                out.append(cls.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return _l2(np.vstack(out))


class LexicalEncoder:
    """Character n-gram TF-IDF encoder (baseline / fallback)."""

    kind = "lexical"

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True)
        self.vectorizer.fit(corpus)
        self.provenance = {
            "encoder": "TF-IDF char n-grams (3-5, char_wb)",
            "model_id": "sklearn-tfidf-char",
            "model_revision": f"vocab={len(self.vectorizer.vocabulary_)}",
            "license": "BSD-3-Clause (scikit-learn)",
        }

    def encode(self, texts: list[str]):
        """Return a sparse CSR matrix; TfidfVectorizer rows are already L2-normalised."""
        return self.vectorizer.transform(texts).astype(np.float32).tocsr()


@lru_cache(maxsize=1)
def get_neural_encoder() -> SapBERTEncoder:
    """Load (once per process) the neural encoder; raises EncoderUnavailable."""
    return SapBERTEncoder()


def neural_status() -> dict:
    """Report availability without raising (used by the API health endpoint)."""
    d = default_model_dir()
    status = {"model_id": DEFAULT_MODEL_ID, "model_dir": str(d), "weights_present": False, "deps_installed": False}
    status["weights_present"] = (d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists()
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        status["deps_installed"] = True
    except ImportError:
        pass
    status["available"] = status["weights_present"] and status["deps_installed"]
    return status
