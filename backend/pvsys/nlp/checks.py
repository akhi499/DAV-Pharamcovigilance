"""Polarity-aware checks for generated answers (verifier v2), added after the extractive-vs-generative study.

The v1 verifier (generator.verify + sentence support) checks citations, numbers, quotations and word overlap;
it cannot see a flipped polarity ("No, there is no boxed warning [L1]" shares every word with a boxed-warning
passage). Two further checks:

* **Structure check** (deterministic): a draft may not deny a boxed warning when a retrieved passage comes from
  FDA's Boxed Warning section, nor assert one when none does.
* **Entailment check** (optional model): each cited sentence is tested against each passage it cites with a
  natural-language-inference cross-encoder (cross-encoder/nli-deberta-v3-small, Apache-2.0, ~140M parameters,
  local). A sentence is rejected when the model judges it contradicted by its cited passage(s):
  P(contradiction) >= 0.5 and higher than P(entailment). The threshold is fixed a priori, not tuned.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from functools import lru_cache
from pathlib import Path

from ..config import get_paths

log = logging.getLogger(__name__)

CONTRADICTION_THRESHOLD = 0.5
_CITE_RE = re.compile(r"\[([A-Z]\d+(?:\s*[,;]\s*[A-Z]\d+)*)\]")
_DENY_BOXED = re.compile(r"\b(no|not|without|does not (have|carry|include|contain)|doesn't (have|carry))\b[^.]{0,40}"
                         r"(boxed|black[- ]?box) warning|(boxed|black[- ]?box) warning[^.]{0,30}\b(not|none)\b", re.I)
_ASSERT_BOXED = re.compile(r"(boxed|black[- ]?box) warning|\bwarning: ", re.I)


def boxed_polarity(text: str | None) -> str | None:
    """'no' if the text denies a boxed warning, 'yes' if it asserts one, None if it does not mention one."""
    if not text:
        return None
    if _DENY_BOXED.search(text):
        return "no"
    return "yes" if _ASSERT_BOXED.search(text) else None


def structure_problems(text: str | None, items: list[dict]) -> list[str]:
    pol = boxed_polarity(text)
    if pol is None:
        return []
    has_boxed = any(i.get("section_key") == "boxed_warning" for i in items)
    if pol == "no" and has_boxed:
        return ["contradicts label structure: denies a boxed warning, but a retrieved passage is from the "
                "label's Boxed Warning section"]
    if pol == "yes" and not has_boxed:
        return ["not supported by label structure: asserts a boxed warning, but no retrieved passage is from a "
                "Boxed Warning section"]
    return []


class NLIUnavailable(RuntimeError):
    pass


def default_nli_dir() -> Path:
    return Path(os.environ.get("PV_NLI_DIR", get_paths().models / "nli"))


class NLI:
    def __init__(self, model_dir: Path | None = None):
        d = Path(model_dir or default_nli_dir())
        if not (d / "config.json").exists():
            raise NLIUnavailable(f"no entailment model in {d}")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise NLIUnavailable(f"torch/transformers not installed: {exc}") from exc
        self._torch = torch
        # CPU by default: the model is small and the GPU is kept for the generator (4 GB laptop GPUs)
        self.device = os.environ.get("PV_NLI_DEVICE", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.tok = AutoTokenizer.from_pretrained(str(d))
        # DeBERTa-v3 is numerically fragile in fp16; it is small enough to run in fp32
        self.model = AutoModelForSequenceClassification.from_pretrained(str(d)).eval().to(self.device)
        labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        try:
            self.i_con = next(i for i, v in labels.items() if v.startswith("contra"))
            self.i_ent = next(i for i, v in labels.items() if v.startswith("entail"))
        except StopIteration as exc:
            raise NLIUnavailable(f"model labels {labels} are not NLI labels") from exc
        meta = d / "provenance.json"
        self.provenance = json.loads(meta.read_text()) if meta.exists() else {"model_id": d.name}
        self._lock = threading.Lock()

    def probs(self, premises: list[str], hypotheses: list[str]) -> list[tuple[float, float]]:
        """(P(contradiction), P(entailment)) for each premise/hypothesis pair."""
        torch = self._torch
        out: list[tuple[float, float]] = []
        with self._lock, torch.inference_mode():
            for i in range(0, len(premises), 16):
                enc = self.tok(premises[i:i + 16], hypotheses[i:i + 16], padding=True, truncation="only_first",
                               max_length=512, return_tensors="pt").to(self.device)
                p = self.model(**enc).logits.float().softmax(-1).cpu()
                out += [(float(r[self.i_con]), float(r[self.i_ent])) for r in p]
        return out


@lru_cache(maxsize=1)
def get_nli() -> NLI:
    return NLI()


def nli_available() -> bool:
    try:
        get_nli()
        return True
    except NLIUnavailable:
        return False
    except Exception as exc:
        log.warning("entailment model could not be loaded: %s", exc)
        return False


def _sentences(text: str) -> list[tuple[str, set[str]]]:
    out = []
    for snt in [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]:
        ids = {c.strip() for grp in _CITE_RE.findall(snt) for c in re.split(r"[,;]", grp)}
        clean = " ".join(_CITE_RE.sub(" ", snt).split())
        if ids and len(clean) > 3:
            out.append((clean, ids))
    return out


def entailment_check(text: str | None, items: list[dict], nli: NLI | None = None) -> dict:
    """Per cited sentence: max P(contradiction) and max P(entailment) over the passages it cites.

    Returns {"status": "ok"|"unavailable", "problems": [...], "sentences": [...]}."""
    if not text:
        return {"status": "ok", "problems": [], "sentences": []}
    try:
        nli = nli or get_nli()
    except NLIUnavailable as exc:
        return {"status": "unavailable", "reason": str(exc), "problems": [], "sentences": []}
    by_id = {i["id"]: f"{i.get('section', '')}: {i.get('text', '')}" for i in items}
    pairs, owners = [], []
    sents = _sentences(text)
    for k, (snt, ids) in enumerate(sents):
        for c in sorted(ids):
            if c in by_id:
                pairs.append((by_id[c], snt))
                owners.append(k)
    scores = nli.probs([p for p, _ in pairs], [h for _, h in pairs]) if pairs else []
    res, problems = [], []
    for k, (snt, ids) in enumerate(sents):
        mine = [s for s, o in zip(scores, owners, strict=True) if o == k]
        con = max((c for c, _ in mine), default=0.0)
        ent = max((e for _, e in mine), default=0.0)
        res.append({"sentence": snt, "cites": sorted(ids), "contradiction": round(con, 3), "entailment": round(ent, 3)})
        if con >= CONTRADICTION_THRESHOLD and con > ent:
            problems.append(f"contradicted by its cited passage (entailment model): '{snt[:80]}'")
    return {"status": "ok", "problems": problems, "sentences": res}
