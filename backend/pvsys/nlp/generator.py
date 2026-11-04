"""Optional local generative small language model (SLM) for paraphrasing a grounded answer.

The question-answering module works without this file's model: its answer is
composed deterministically from retrieved evidence (``rag.py``). If a small
instruction-tuned causal language model is placed in ``models/generator/``
(any Hugging Face ``AutoModelForCausalLM`` folder, e.g. the Apache-2.0
``Qwen/Qwen2.5-0.5B-Instruct``; override with ``PV_GENERATOR_DIR``), the
engine can additionally ask it to write a short summary of the evidence.

Generated text is never trusted. ``verify`` accepts it only if

* every citation id it uses exists in the evidence and it cites at least one,
* every number it contains appears in the evidence text,
* it makes no unnegated causal or incidence claim ("causes", "proves", "risk of 5%" ...),

otherwise the summary is discarded and the reason is reported. The model runs
on CPU with greedy decoding (deterministic); no paid API or GPU is used.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path

from ..config import get_paths

log = logging.getLogger(__name__)

MAX_NEW_TOKENS = 160
SYSTEM_PROMPT = (
    "You are a pharmacovigilance assistant. Summarise the EVIDENCE in at most three sentences for the QUESTION. "
    "Use only facts and numbers that appear in the EVIDENCE and cite them with their ids in square brackets, "
    "e.g. [S1]. These are spontaneous-report statistics: never say that a drug causes a reaction, never give an "
    "incidence or risk percentage, and say so if the evidence is insufficient."
)
EXTERNAL_PROMPT = (
    "You answer drug-safety questions using ONLY the numbered passages (FDA labels [L#], FDA Drug Safety "
    "Communications [R#], published reviews [P#]). Write two or three short sentences. End every sentence with the id "
    "of the passage it comes from in square brackets, like [L1]. Report what the passage says (e.g. 'The label lists "
    "...', 'The label warns ...'); do not add facts, numbers or advice. If the passages do not answer the question, "
    "reply exactly: The retrieved sources do not answer this question."
)
# one worked example (fictional drug) teaches the small model the citation format
EXTERNAL_SHOTS = [
    ("QUESTION: Does drugx cause headache?\n\nPASSAGES:\n[L1] DRUGX label - Adverse Reactions: The most common adverse "
     "reactions (>=5%) were headache and nausea.\n[L2] DRUGX label - Warnings and Precautions: Hypotension: monitor "
     "blood pressure.\n\nANSWER:",
     "The DRUGX label lists headache among the most common adverse reactions [L1]."),
]
MIN_SENTENCE_SUPPORT = 0.5
_REPORTING_WORDS = {"label", "fda", "prescrib", "information", "stat", "list", "warn", "report", "passag", "evidenc",
                    "say", "mention", "describ", "note", "indicat", "includ", "drug", "also"}
INSUFFICIENT_REPLY = "The retrieved sources do not answer this question."

# unsigned: "9.80-15.9" is a range, not a negative number (signs are therefore not checked)
_NUM_RE = re.compile(r"(?<![A-Za-z0-9.])\d[\d,]*(?:\.\d+)?")
_QUOTE_RE = re.compile(r"[\u201c\"]([^\u201d\"]{3,})[\u201d\"]")
_CITE_RE = re.compile(r"\[([A-Z]\d+(?:\s*[,;]\s*[A-Z]\d+)*)\]")
_CAUSAL_RE = re.compile(
    r"\b(causes?|caused|causing|proves?|proven|confirms?|is responsible for|leads? to|results? in|"
    r"incidence of|risk of \d|\d+(?:\.\d+)?\s*% (?:risk|chance|of patients))\b", re.I)
_NEGATION_RE = re.compile(r"\b(not|no|cannot|can't|does not|doesn't|do not|don't|never|nor|without)\b[^.]{0,40}$", re.I)


def default_generator_dir() -> Path:
    return Path(os.environ.get("PV_GENERATOR_DIR", get_paths().models / "generator"))


def _norm_num(s: str) -> str:
    s = s.replace(",", "").lstrip("+")
    try:
        f = float(s)
    except ValueError:
        return s
    return f"{f:.6g}"


def evidence_numbers(evidence_text: str) -> set[str]:
    return {_norm_num(m.group(0)) for m in _NUM_RE.finditer(evidence_text)}


def verify(text: str, evidence_text: str, valid_ids: set[str], allowed_numbers: set[str] | None = None,
           check_causal: bool = True) -> list[str]:
    """Return a list of problems; an empty list means the text passed every check.

    ``check_causal`` applies to statements about FAERS reports; summaries of FDA labels may legitimately
    report what the label states, so for them only citations, numbers and quotations are checked."""
    problems: list[str] = []
    if not text.strip():
        return ["empty output"]
    # Quoted passages must be verbatim evidence; once verified they are exempt from the number/causal checks
    # (e.g. a CTCAE definition "loss of consciousness caused by insufficient blood supply" is physiology).
    flat_ev = " ".join(evidence_text.split())
    for q in _QUOTE_RE.findall(text):
        if " ".join(q.split()) not in flat_ev:
            problems.append(f"quotation not found verbatim in evidence: '{q[:60]}'")
    body = _CITE_RE.sub(" ", _QUOTE_RE.sub(" ", text))
    cited = {c.strip() for grp in _CITE_RE.findall(text) for c in re.split(r"[,;]", grp)}
    if not cited:
        problems.append("no citation")
    bad_ids = sorted(cited - valid_ids)
    if bad_ids:
        problems.append(f"unknown citation ids: {', '.join(bad_ids)}")
    ok_nums = evidence_numbers(evidence_text) | (allowed_numbers or set())
    unsupported = sorted({m.group(0) for m in _NUM_RE.finditer(body) if _norm_num(m.group(0)) not in ok_nums})
    if unsupported:
        problems.append(f"numbers not in evidence: {', '.join(unsupported[:8])}")
    for m in (_CAUSAL_RE.finditer(body) if check_causal else ()):
        before = body[max(0, m.start() - 60): m.start()]
        sentence_start = max(before.rfind("."), before.rfind("!"), before.rfind("?"))
        if not _NEGATION_RE.search(before[sentence_start + 1:]):
            problems.append(f"unsupported causal/incidence claim: '{m.group(0)}'")
            break
    return problems


def evidence_block(evidence: list[dict], max_chars: int | None = None) -> str:
    """Model-readable rendering of the evidence; the full version is the verifier's reference text."""
    lines = []
    for e in evidence:
        if e["kind"] == "document":
            lines.append(f"[{e['id']}] {e['title']}: {e['text'][:max_chars] if max_chars else e['text']}")
        elif e["kind"] == "external":
            txt = e["text"][:max_chars] if max_chars else e["text"]
            lines.append(f"[{e['id']}] {e['title']} - {e['section']} ({e.get('date') or 'undated'}): {txt}")
        else:
            items = {k: v for k, v in e.items() if k not in ("id", "kind", "source") and v is not None}
            parts = []
            for k, v in items.items():
                if isinstance(v, float):
                    from .rag import fmt

                    v = fmt(v)
                parts.append(f"{k}={v}")
            lines.append(f"[{e['id']}] " + ", ".join(parts))
    return "\n".join(lines)


def status() -> dict:
    d = default_generator_dir()
    st = {"model_dir": str(d), "weights_present": (d / "config.json").exists() and any(
        (d / f).exists() for f in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")),
          "enabled": os.environ.get("PV_GENERATOR", "auto").lower() != "off"}
    meta = d / "provenance.json"
    st["model_id"] = json.loads(meta.read_text()).get("model_id") if meta.exists() else (d.name if st["weights_present"] else None)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        st["deps_installed"] = True
    except ImportError:
        st["deps_installed"] = False
    st["available"] = st["weights_present"] and st["deps_installed"] and st["enabled"]
    return st


def _wants_4bit(model_dir: Path, torch) -> bool:
    """4-bit loading for models whose fp16 weights would not fit the GPU (e.g. a 3B model on a 4 GB laptop GPU).

    PV_GEN_QUANT=4bit forces it, PV_GEN_QUANT=none disables it; needs the optional bitsandbytes package."""
    mode = os.environ.get("PV_GEN_QUANT", "auto").lower()
    if mode in ("none", "off", "fp16"):
        return False
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        if mode == "4bit":
            log.warning("PV_GEN_QUANT=4bit but bitsandbytes is not installed; loading in fp16")
        return False
    if mode == "4bit":
        return True
    weights = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
    total = torch.cuda.get_device_properties(0).total_memory
    return weights > 0.9 * total


class Generator:
    def __init__(self, model_dir: Path | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..device import device_info, pick_device

        self.model_dir = Path(model_dir or default_generator_dir())
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self._torch = torch
        self.device = pick_device()
        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.quantization = None
        try:
            if self.device == "cuda" and _wants_4bit(self.model_dir, torch):
                from transformers import BitsAndBytesConfig

                q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
                self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir), quantization_config=q,
                                                                  device_map={"": 0}, dtype=torch.float16).eval()
                self.quantization = "4-bit NF4 (bitsandbytes), fp16 compute"
            else:
                dtype = torch.float16 if self.device == "cuda" else torch.float32  # fp16 fits a 4 GB laptop GPU
                self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir), dtype=dtype).eval().to(self.device)
        except torch.cuda.OutOfMemoryError:  # model too large for the GPU: fall back to the CPU
            log.warning("%s does not fit in GPU memory; running on the CPU", self.model_dir.name)
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir), dtype=torch.float32).eval()
        meta = self.model_dir / "provenance.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {"model_id": self.model_dir.name}
        self.meta.update(device_info())
        self.meta["quantization"] = self.quantization
        self._lock = threading.Lock()

    def _prompt(self, question: str, block: str, system: str = SYSTEM_PROMPT,
                shots: list[tuple[str, str]] | None = None) -> str:
        user = (f"QUESTION: {question}\n\nPASSAGES:\n{block}\n\nANSWER:" if shots else
                f"QUESTION: {question}\n\nEVIDENCE:\n{block}\n\nAnswer with citations:")
        msgs = [{"role": "system", "content": system}]
        for u, a in shots or []:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        msgs.append({"role": "user", "content": user})
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(m["content"] for m in msgs) + "\n"

    def generate(self, question: str, evidence: list[dict], system: str = SYSTEM_PROMPT, max_chars: int = 600,
                 shots: list[tuple[str, str]] | None = None) -> str:
        block = evidence_block(evidence, max_chars=max_chars)
        prompt = self._prompt(question, block, system, shots)
        torch = self._torch
        with self._lock, torch.inference_mode():
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
            ids = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1536).to(self.device)
            try:
                out = self.model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=pad)
            except torch.cuda.OutOfMemoryError:  # 4 GB laptop GPUs: move to the CPU once and retry
                if self.quantization:  # 4-bit weights cannot be moved to the CPU
                    raise
                log.warning("GPU out of memory while generating with %s; switching to the CPU", self.model_dir.name)
                self.model = self.model.float().to("cpu")
                torch.cuda.empty_cache()
                self.device = "cpu"
                self.meta["device"] = "cpu (fell back after GPU out-of-memory)"
                ids = ids.to("cpu")
                out = self.model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=pad)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    return Generator()


def summarise(question: str, evidence: list[dict], grounded_text: str = "") -> dict:
    """Run the optional generator and verify its output. Never raises."""
    st = status()
    if not st["available"]:
        reason = ("disabled (PV_GENERATOR=off)" if not st["enabled"] else
                  "torch/transformers not installed" if not st["deps_installed"] else
                  f"no generator model in {st['model_dir']}")
        return {"status": "unavailable", "reason": reason}
    if not evidence:
        return {"status": "skipped", "reason": "no evidence to summarise"}
    t = time.perf_counter()
    try:
        gen = get_generator()
        text = gen.generate(question, evidence)
    except Exception as exc:  # model errors must never break the grounded answer
        log.exception("generator failed")
        return {"status": "error", "reason": str(exc)[:300]}
    # numbers in the deterministic answer are derived from the evidence; numbers in the question are the user's own
    allowed = evidence_numbers(grounded_text) | evidence_numbers(question)
    problems = verify(text, evidence_block(evidence), {e["id"] for e in evidence}, allowed)
    return {"status": "verified" if not problems else "rejected", "text": text if not problems else None,
            "rejected_text": text if problems else None, "problems": problems,
            "model_id": gen.meta.get("model_id"), "model_revision": gen.meta.get("revision"), "device": gen.device,
            "seconds": round(time.perf_counter() - t, 2), "decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS}


def sentence_support(text: str, evidence: list[dict]) -> list[dict]:
    """Per generated sentence: share of its content words that occur in the passages it cites
    (title + section + text; reporting words such as 'label', 'states', 'lists' are ignored)."""
    from .kb import tokenize

    by_id = {e["id"]: set(tokenize(f"{e.get('title', '')} {e.get('section', '')} {e.get('text', '')}")) for e in evidence}
    out = []
    for snt in [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]:
        ids = {c.strip() for grp in _CITE_RE.findall(snt) for c in re.split(r"[,;]", grp)}
        words = set(tokenize(_CITE_RE.sub(" ", snt))) - _REPORTING_WORDS
        pool = set().union(*(by_id.get(i, set()) for i in ids)) if ids else set()
        out.append({"sentence": snt, "cites": sorted(ids),
                    "support": round(len(words & pool) / len(words), 3) if words and ids else (1.0 if ids else 0.0)})
    return out


def citation_support(text: str, evidence: list[dict]) -> float | None:
    """Share of sentences whose content is mostly (>= MIN_SENTENCE_SUPPORT) found in the passages they cite."""
    ss = sentence_support(text, evidence)
    if not ss:
        return None
    return round(sum(s["support"] >= MIN_SENTENCE_SUPPORT for s in ss) / len(ss), 3)


def summarise_external(question: str, items: list[dict], gen: Generator | None = None) -> dict:
    """Generate a cited answer from external passages (labels, FDA communications, literature) and verify it.

    ``gen`` lets an evaluation pass a specific model; by default the configured local generator is used."""
    if gen is None:
        st = status()
        if not st["available"]:
            return {"status": "unavailable", "reason": "no local generator model" if st["deps_installed"] else
                    "torch/transformers not installed"}
    if not items:
        return {"status": "skipped", "reason": "no external evidence retrieved"}
    t = time.perf_counter()
    try:
        gen = gen or get_generator()
        text = gen.generate(question, items, system=EXTERNAL_PROMPT, max_chars=500, shots=EXTERNAL_SHOTS)
    except Exception as exc:
        log.exception("generator failed")
        return {"status": "error", "reason": str(exc)[:300]}
    base = {"model_id": gen.meta.get("model_id"), "model_revision": gen.meta.get("revision"), "device": gen.device,
            "seconds": round(time.perf_counter() - t, 2), "decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS}
    if text.strip().rstrip(".").lower() == INSUFFICIENT_REPLY.rstrip(".").lower():
        return {**base, "status": "insufficient", "text": None, "problems": [], "raw_text": text}
    problems = verify(text, evidence_block(items), {e["id"] for e in items}, evidence_numbers(question),
                      check_causal=False)
    support = sentence_support(text, items)
    weak = [s for s in support if s["support"] < MIN_SENTENCE_SUPPORT]
    if weak:  # a cited sentence whose content is not in the cited passage is an unsupported claim
        problems.append(f"sentence not supported by its cited passage: '{weak[0]['sentence'][:80]}'")
    # verifier v2: polarity-aware checks (label structure, and entailment if the NLI model is installed)
    from .checks import entailment_check, structure_problems

    v1 = list(problems)
    struct = structure_problems(text, items)
    nli = entailment_check(text, items) if os.environ.get("PV_NLI", "on").lower() != "off" else \
        {"status": "disabled", "problems": [], "sentences": []}
    problems = v1 + struct + nli["problems"]
    return {**base, "status": "verified" if not problems else "rejected", "text": text if not problems else None,
            "rejected_text": text if problems else None, "problems": problems,
            "checks": {"v1": v1, "structure": struct, "entailment": nli["problems"], "entailment_status": nli["status"],
                       "entailment_sentences": nli["sentences"]},
            "citation_support": citation_support(text, items), "sentence_support": support}
