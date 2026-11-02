"""Command-line entry point for the full pipeline.

    python -m pvsys.pipeline all            # ingest -> clean -> quality -> signals -> eda -> nlp
    python -m pvsys.pipeline stats          # statistical stages only (no NLP dependencies needed)
    python -m pvsys.pipeline nlp            # NLP stage only (needs processed data)
    python -m pvsys.pipeline annotate-export --n 200
    python -m pvsys.pipeline annotate-import path/to/batch.csv
    python -m pvsys.pipeline evaluate       # re-score against CTCAE + any blinded annotations
    python -m pvsys.pipeline rag-index      # (re)build the question-answering knowledge base
    python -m pvsys.pipeline rag-eval       # evaluate the question-answering module
    python -m pvsys.pipeline evidence-fetch [--n 150] [--drug "X" ...] [--refresh]   # FDA labels, FDA DSCs, PubMed
    python -m pvsys.pipeline evidence-index # chunk + embed (offline, from the cached downloads)
    python -m pvsys.pipeline evidence-eval  # retrieval / threshold / generation / concordance evaluation

Run from the ``backend`` folder (or with ``backend`` on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from . import clean, eda, ingest, quality, signals
from .config import QUARTER, get_paths

log = logging.getLogger("pvsys")


def _timed(name, fn, *a, **kw):
    t = time.time()
    log.info("== %s ...", name)
    out = fn(*a, **kw)
    log.info("== %s done in %.1fs", name, time.time() - t)
    return out


def run_stats(quarter: str = QUARTER) -> dict:
    paths = get_paths().ensure()
    res = {}
    res["ingest"] = _timed("ingest", ingest.ingest, paths, quarter)
    res["clean"] = _timed("clean", clean.clean, paths)
    res["quality"] = _timed("quality", quality.run, paths)["summary"]
    res["signals"] = _timed("signals", signals.run, paths)
    _timed("eda snapshot", eda.snapshot, paths)
    return res


def run_nlp() -> dict:
    from .nlp import run as nlp_run  # imported lazily: statistics never depend on NLP

    summary = _timed("nlp", nlp_run.run, get_paths())["summary"]
    try:
        summary["rag"] = run_rag(evaluate=True)
    except Exception as exc:  # question answering is optional; categorisation outputs are already written
        log.error("Question-answering stage failed: %s", exc)
        summary["rag"] = {"error": str(exc)}
    return summary


def run_rag(evaluate: bool = True, n_pairs: int = 150) -> dict:
    from .nlp import kb, rag, rag_eval, search

    paths = get_paths()
    out = {"knowledge_base": _timed("rag knowledge base", kb.build, paths)}
    search.reset_index()
    rag.reset_engine()
    if evaluate:
        ev = _timed("rag evaluation", rag_eval.evaluate_rag, paths, n_pairs)
        out["evaluation"] = {
            "faq_bm25_recall_at_1": ev["faq_retrieval"].get("systems", {}).get("bm25", {}).get("recall_at_1"),
            "linking_sapbert_test_recall_at_1": ev["reaction_linking"].get("systems", {}).get("sapbert", {})
            .get("test", {}).get("recall_at_1"),
            "end_to_end_exact_retrieval": ev["end_to_end"].get("exact_retrieval_generic_templates"),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pvsys.pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "all", "stats", "nlp", "evaluate",
                                            "annotate-export", "annotate-import", "rag-index", "rag-eval",
                                            "evidence-fetch", "evidence-index", "evidence-eval"])
    parser.add_argument("path", nargs="?", help="CSV path for annotate-import")
    parser.add_argument("--quarter", default=QUARTER)
    parser.add_argument("--n", type=int, default=200, help="batch size for annotate-export / pairs for rag-eval")
    parser.add_argument("--drug", action="append", help="evidence-fetch: drug name(s) to fetch (repeatable)")
    parser.add_argument("--refresh", action="store_true", help="evidence-fetch: ignore cached downloads")
    parser.add_argument("--skip-generation", action="store_true", help="evidence-eval: skip the generator part")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "download":
        print(ingest.download(get_paths(), args.quarter))
    elif args.command == "stats":
        print(json.dumps(run_stats(args.quarter)["signals"], indent=2))
    elif args.command == "all":
        res = run_stats(args.quarter)
        try:
            res["nlp"] = run_nlp()
        except Exception as exc:  # NLP failure must not invalidate statistical outputs
            log.error("NLP stage failed (statistical outputs are intact): %s", exc)
            res["nlp"] = {"error": str(exc)}
        print(json.dumps({"signals": res["signals"], "nlp": res["nlp"]}, indent=2, default=str))
    elif args.command == "nlp":
        print(json.dumps(run_nlp(), indent=2, default=str))
    elif args.command == "evaluate":
        from .nlp import evaluate
        from .nlp import run as nlp_run

        terms = nlp_run.load_terms(get_paths())["pt"].tolist()
        ref = evaluate.evaluate_reference(get_paths(), faers_terms=terms)
        ann = evaluate.evaluate_annotations(get_paths())
        print(json.dumps({"reference": {k: v.get("test") if isinstance(v, dict) else v
                                        for k, v in ref.get("systems", {}).items()},
                          "annotations": ann}, indent=2, default=str))
    elif args.command == "rag-index":
        print(json.dumps(run_rag(evaluate=False), indent=2, default=str))
    elif args.command == "rag-eval":
        from .nlp import rag_eval

        res = rag_eval.evaluate_rag(get_paths(), n_pairs=args.n if args.n != 200 else 150)
        print(json.dumps({k: res[k] for k in ("faq_retrieval", "reaction_linking", "end_to_end")}, indent=2,
                         default=str)[:6000])
    elif args.command == "evidence-fetch":
        from .evidence import sources

        drugs = [d.strip().upper() for d in args.drug] if args.drug else None
        n = args.n if args.n != 200 else 150
        if drugs:  # add to the existing set: keep previously fetched drugs in the manifest
            m = sources.fetch_all(get_paths(), drugs=drugs, refresh=args.refresh, max_dsc=0)
        else:
            m = sources.fetch_all(get_paths(), n_drugs=n, refresh=args.refresh)
        print(json.dumps({k: m[k] for k in ("drugs_requested", "requests", "stats")}, indent=2))
        if m["failures"]:
            print(f"{len(m['failures'])} item(s) failed - see data/evidence/sources_manifest.json")
    elif args.command == "evidence-index":
        from .evidence import index

        m = index.build_index(get_paths())
        print(json.dumps({k: m.get(k) for k in ("documents", "documents_by_type", "chunks", "chunks_by_type",
                                                 "retrieval_modes", "seconds_total")}, indent=2))
        print("embedding:", json.dumps({k: m["embedding"].get(k) for k in ("model_id", "status", "reason", "seconds")}))
    elif args.command == "evidence-eval":
        from .evidence import evaluate as ev

        n = args.n if args.n != 200 else 40
        r = ev.evaluate(get_paths(), n_generation=n, skip_generation=args.skip_generation)
        summary = {"retrieval_within_drug_test": {m: v["all"] for m, v in r["retrieval"]["within_drug"].items()},
                   "sufficiency": {k: r["sufficiency"].get(k) for k in ("status", "threshold", "test")},
                   "generation": {k: r["generation"].get(k) for k in ("status", "n", "boxed_warning_accuracy",
                                                                      "baseline_retrieval_only_accuracy",
                                                                      "verifier_pass_rate", "seconds_median")},
                   "concordance": {k: r["concordance"].get(k) for k in ("signals_checked", "share_reaction_text_in_label")},
                   "seconds": r["seconds"]}
        print(json.dumps(summary, indent=2, default=str))
    elif args.command == "annotate-export":
        from .nlp import review

        print(review.export_annotation_batch(args.n))
    elif args.command == "annotate-import":
        from .nlp import review

        if not args.path:
            parser.error("annotate-import requires a CSV path")
        print(json.dumps(review.import_annotations(args.path), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
