"""Extractive vs generative answering across local language models of different sizes.

Design (all on the 89 *test* drugs of evaluate.split_drugs, so nothing here was tuned on them):

* Retrieval is run once with the production retriever and cached; every model sees identical passages
  (top 3, as in the app), so differences come from the generator alone.
* Task A - "Does <drug> have a boxed warning?"  Gold: whether FDA's label has a Boxed Warning section.
  Extractive answer: "yes" iff one of the retrieved passages comes from the Boxed Warning section.
* Task B - one lay-worded question per drug about a label topic (contraindications, interactions, pregnancy,
  overdosage, adverse reactions; topic drawn with a fixed seed). Gold: FDA's section coding. Extractive answer
  is correct iff the passage it quotes first (top-1) is from a gold section; a generated answer is correct iff it
  cites at least one passage and every cited passage is from a gold section.
* For each model: raw accuracy, verifier pass rate, accuracy of verified answers, and *system* accuracy (what the
  app would show: the generated answer if it passes the checks, else the extractive answer). Paired exact McNemar
  tests compare each with the extractive answer on the same items. Latency, tokens/s and peak GPU memory recorded.

Output: data/evidence/generator_comparison.json (summary + every item for every model).
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from datetime import datetime, timezone
from math import comb, sqrt
from pathlib import Path

import numpy as np

from ..config import Paths, get_paths
from .evaluate import SEED, TOPICS, _brand, build_queries, classify_boxed, split_drugs
from .sources import evidence_dir

log = logging.getLogger(__name__)
TOP_K = 3


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Two-sided exact McNemar test on paired correctness vectors."""
    n01 = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    n10 = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n)
    return {"only_first_correct": n01, "only_second_correct": n10, "p_value": p}


def build_tasks(idx, paths: Paths) -> dict:
    """Questions + retrieved passages for both tasks (cached so every model gets identical inputs)."""
    queries = build_queries(idx)
    drugs = sorted({q["drug"] for q in queries})
    _, test = split_drugs(drugs)
    labels = idx.chunks[idx.chunks["source_type"] == "label"]
    sections = labels.groupby("drugs")["section_key"].apply(set)
    titles = labels.groupby("drugs")["title"].first()
    rng = np.random.default_rng(SEED)

    def items_for(q: str, d: str) -> list[dict]:
        hits = idx.search(q, k=5, drugs=[d])["hits"][:TOP_K]
        return [{"id": f"L{i + 1}", "kind": "external", **h} for i, h in enumerate(hits)]

    task_a, task_b = [], []
    for d in sorted(test):
        q = f"Does {d.lower()} have a boxed warning?"
        items = items_for(q, d)
        gold = "yes" if "boxed_warning" in sections[d] else "no"
        task_a.append({"drug": d, "question": q, "gold": gold, "items": items,
                       "extractive_pred": "yes" if any(i["section_key"] == "boxed_warning" for i in items) else "no"})
        topics = [t for t in TOPICS if t != "boxed_warning" and sections[d] & set(TOPICS[t][0])]
        topic = topics[int(rng.integers(len(topics)))]
        secs, _, lay = TOPICS[topic]
        qb = lay.format(d=d.lower(), b=_brand(titles[d], d))
        items = items_for(qb, d)
        task_b.append({"drug": d, "topic": topic, "question": qb, "gold_sections": list(secs), "items": items,
                       "extractive_correct": bool(items) and items[0]["section_key"] in secs})
    return {"retriever": idx.default_mode, "top_k": TOP_K, "test_drugs": len(test), "task_a": task_a, "task_b": task_b}


def _cited(text: str | None) -> set[str]:
    if not text:
        return set()
    return {c.strip() for grp in re.findall(r"\[([A-Z]\d+(?:\s*[,;]\s*[A-Z]\d+)*)\]", text) for c in re.split(r"[,;]", grp)}


def run_model(model_dir: Path, tasks: dict) -> dict:
    import torch

    from ..nlp.generator import Generator, summarise_external

    t0 = time.perf_counter()
    gen = Generator(model_dir)
    load_s = round(time.perf_counter() - t0, 1)
    if gen.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    summarise_external(tasks["task_a"][0]["question"], tasks["task_a"][0]["items"], gen)  # warm-up
    rows_a, rows_b, n_tokens, gen_seconds = [], [], 0, 0.0
    for t in tasks["task_a"]:
        g = summarise_external(t["question"], t["items"], gen)
        text = g.get("text") or g.get("rejected_text")
        pred = classify_boxed(text, g["status"])
        n_tokens += len(gen.tok(text or "")["input_ids"])
        gen_seconds += g.get("seconds", 0)
        verified = g["status"] == "verified"
        rows_a.append({"drug": t["drug"], "gold": t["gold"], "pred": pred, "status": g["status"], "verified": verified,
                       "correct": pred == t["gold"], "extractive_correct": t["extractive_pred"] == t["gold"],
                       "system_correct": (pred == t["gold"]) if verified else (t["extractive_pred"] == t["gold"]),
                       "problems": g.get("problems", []), "seconds": g.get("seconds"), "text": text})
    for t in tasks["task_b"]:
        g = summarise_external(t["question"], t["items"], gen)
        text = g.get("text") or g.get("rejected_text")
        by_id = {i["id"]: i["section_key"] for i in t["items"]}
        cited = _cited(text)
        correct = bool(cited) and all(by_id.get(c) in t["gold_sections"] for c in cited)
        n_tokens += len(gen.tok(text or "")["input_ids"])
        gen_seconds += g.get("seconds", 0)
        verified = g["status"] == "verified"
        rows_b.append({"drug": t["drug"], "topic": t["topic"], "status": g["status"], "verified": verified,
                       "correct": correct, "extractive_correct": t["extractive_correct"],
                       "system_correct": correct if verified else t["extractive_correct"],
                       "problems": g.get("problems", []), "seconds": g.get("seconds"), "text": text})
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if gen.device == "cuda" else None
    meta = {"model_id": gen.meta.get("model_id", model_dir.name), "revision": gen.meta.get("revision"),
            "parameters_millions": round(sum(p.numel() for p in gen.model.parameters()) / 1e6),
            "device": gen.device, "dtype": str(next(gen.model.parameters()).dtype).replace("torch.", ""),
            "load_seconds": load_s, "peak_gpu_gib": peak,
            "tokens_per_second": round(n_tokens / gen_seconds, 1) if gen_seconds else None}
    del gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"meta": meta, "task_a": rows_a, "task_b": rows_b}


def summarise_task(rows: list[dict]) -> dict:
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    ke = sum(r["extractive_correct"] for r in rows)
    ks = sum(r["system_correct"] for r in rows)
    ver = [r for r in rows if r["verified"]]
    kv = sum(r["correct"] for r in ver)
    reasons: dict[str, int] = {}
    for r in rows:
        for p in r["problems"]:
            key = p.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    secs = [r["seconds"] for r in rows if r["seconds"] is not None]
    return {"n": n, "generative_accuracy": round(k / n, 4), "generative_95ci": wilson(k, n),
            "extractive_accuracy": round(ke / n, 4), "extractive_95ci": wilson(ke, n),
            "verifier_pass_rate": round(len(ver) / n, 4), "verifier_pass_95ci": wilson(len(ver), n),
            "accuracy_of_verified": round(kv / len(ver), 4) if ver else None, "verified_correct": kv, "verified_n": len(ver),
            "system_accuracy": round(ks / n, 4), "system_95ci": wilson(ks, n),
            "mcnemar_extractive_vs_generative": mcnemar_exact([r["extractive_correct"] for r in rows], [r["correct"] for r in rows]),
            "mcnemar_extractive_vs_system": mcnemar_exact([r["extractive_correct"] for r in rows], [r["system_correct"] for r in rows]),
            "undetermined": sum(r.get("pred") in ("undetermined", "no_answer") for r in rows) if "pred" in rows[0] else None,
            "rejection_reasons": reasons, "seconds_median": round(float(np.median(secs)), 2) if secs else None}


def compare(model_dirs: list[Path], paths: Paths | None = None, limit: int | None = None) -> dict:
    from .index import EvidenceIndex

    paths = paths or get_paths()
    idx = EvidenceIndex(paths)
    tasks = build_tasks(idx, paths)
    if limit:  # quick smoke test on the first `limit` drugs
        tasks["task_a"], tasks["task_b"] = tasks["task_a"][:limit], tasks["task_b"][:limit]
    out = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "retriever": tasks["retriever"],
           "top_k_passages": TOP_K, "test_drugs": tasks["test_drugs"],
           "task_a_gold_counts": {g: sum(t["gold"] == g for t in tasks["task_a"]) for g in ("yes", "no")},
           "task_b_topics": {tp: sum(t["topic"] == tp for t in tasks["task_b"]) for tp in TOPICS},
           "protocol": __doc__, "models": []}
    for d in model_dirs:
        log.info("generator comparison: %s", d)
        try:
            res = run_model(Path(d), tasks)
        except Exception as exc:  # one model failing must not lose the others
            log.exception("model %s failed", d)
            out["models"].append({"model_dir": str(d), "status": "error", "error": str(exc)[:300]})
            continue
        out["models"].append({"model_dir": str(d), "status": "ok", **res["meta"],
                              "task_a": summarise_task(res["task_a"]), "task_b": summarise_task(res["task_b"]),
                              "items_task_a": res["task_a"], "items_task_b": res["task_b"]})
        (evidence_dir(paths) / "generator_comparison.json").write_text(json.dumps(out, indent=2, default=str))
    out["inputs"] = {"task_a": [{k: v for k, v in t.items() if k != "items"} | {"chunk_ids": [i["chunk_id"] for i in t["items"]]}
                                for t in tasks["task_a"]],
                     "task_b": [{k: v for k, v in t.items() if k != "items"} | {"chunk_ids": [i["chunk_id"] for i in t["items"]]}
                                for t in tasks["task_b"]]}
    (evidence_dir(paths) / "generator_comparison.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def default_model_dirs(paths: Paths | None = None) -> list[Path]:
    """Every folder under models/ whose name starts with 'generator' and contains weights, smallest first."""
    root = (paths or get_paths()).models
    dirs = [d for d in root.glob("generator*") if (d / "config.json").exists() and
            (any(d.glob("*.safetensors")) or any(d.glob("*.bin")))]
    return sorted(dirs, key=lambda d: sum(f.stat().st_size for f in d.glob("*.safetensors")))
