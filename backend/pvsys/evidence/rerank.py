"""Retrieval improvements: question -> label-section routing and an optional cross-encoder re-ranker.

* **Section routing** maps everyday wording to the SPL label sections that answer it ("who should not take"
  -> Contraindications, "too much" -> Overdosage, ...). Matching passages get a score bonus ``beta`` (chosen on
  development drugs); nothing is filtered out, so a wrongly routed question still sees every passage.
  The rules use general vocabulary, but they were written by the same people who wrote the evaluation
  templates - results with routing are therefore an optimistic estimate for real user questions.
* **Re-ranking** re-scores the top candidates of dense retrieval with a cross-encoder that reads the question
  and passage together (BAAI/bge-reranker-base, MIT licence, ~280M parameters, local; optional). Without
  the model, re-ranking modes fall back to dense retrieval and say so.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import get_paths

log = logging.getLogger(__name__)

RERANK_CANDIDATES = 20
MAX_PASSAGE_TOKENS = 384

SECTION_ROUTES: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (re.compile(r"boxed warning|black[- ]?box|most (serious|severe|important) (warning|risk|danger)", re.I),
     ("boxed_warning",)),
    (re.compile(r"contraindicat|should(n't| not) (take|use|be given|receive)|must not (take|use)|"
                r"(can't|cannot|can not) (take|use)|who (should|must) avoid|not (be )?suitable for", re.I),
     ("contraindications",)),
    (re.compile(r"interact|combin|together with|taken with|take with|mix(ed|ing)? with|alongside|"
                r"other (medicines|medications|drugs)", re.I),
     ("drug_interactions",)),
    (re.compile(r"pregnan|breast[- ]?feed|lactat|nursing|unborn|f[o]?etus|expecting a baby", re.I),
     ("pregnancy", "use_in_specific_populations")),
    (re.compile(r"overdos|too much|too many|extra dose|double dose|poison", re.I),
     ("overdosage",)),
    (re.compile(r"side[- ]?effects?|adverse (reactions?|events?|effects?)|common reactions?|"
                r"(problems|reactions) (people|patients) (get|have)", re.I),
     ("adverse_reactions",)),
    (re.compile(r"\bwarnings?\b|precautions?|cautions?|dangers?|safe(ty)? (concerns?|risks?)", re.I),
     ("warnings_and_cautions", "warnings", "precautions")),
]


def route_sections(question: str) -> tuple[str, ...]:
    """Label sections that a question asks about (may be empty). Specific routes win over the generic one."""
    found: list[str] = []
    for i, (pat, secs) in enumerate(SECTION_ROUTES):
        if pat.search(question):
            if i == len(SECTION_ROUTES) - 1 and found:
                break  # generic "warning" wording adds nothing once a specific section matched
            found += [s for s in secs if s not in found]
    return tuple(found)


def minmax(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x, dtype=float) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


class RerankerUnavailable(RuntimeError):
    pass


def default_reranker_dir() -> Path:
    return Path(os.environ.get("PV_RERANKER_DIR", get_paths().models / "reranker"))


class Reranker:
    def __init__(self, model_dir: Path | None = None):
        d = Path(model_dir or default_reranker_dir())
        if not (d / "config.json").exists():
            raise RerankerUnavailable(f"no re-ranker model in {d}")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RerankerUnavailable(f"torch/transformers not installed: {exc}") from exc
        from ..device import pick_device

        self._torch = torch
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(str(d))
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(str(d), dtype=dtype).eval().to(self.device)
        meta = d / "provenance.json"
        self.provenance = json.loads(meta.read_text()) if meta.exists() else {"model_id": d.name}
        self._lock = threading.Lock()

    def score(self, query: str, passages: list[str], batch_size: int = 16) -> np.ndarray:
        torch = self._torch
        out: list[float] = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(passages), batch_size):
                chunk = passages[i:i + batch_size]
                enc = self.tok([query] * len(chunk), chunk, padding=True, truncation="only_second",
                               max_length=MAX_PASSAGE_TOKENS, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                out += (logits[:, 0] if logits.shape[1] == 1 else logits[:, -1]).cpu().tolist()
        return np.array(out, dtype=float)


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()


def reranker_available() -> bool:
    try:
        get_reranker()
        return True
    except RerankerUnavailable:
        return False
    except Exception as exc:  # broken model folder, CUDA problems, ...: never break retrieval
        log.warning("re-ranker could not be loaded: %s", exc)
        return False
