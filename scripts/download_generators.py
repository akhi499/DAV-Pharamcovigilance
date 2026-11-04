"""Download local generator models for the extractive-vs-generative comparison, recording exact provenance.

    python scripts/download_generators.py            # Qwen2.5-1.5B-Instruct + SmolLM2-1.7B-Instruct
    python scripts/download_generators.py --with-3b  # also Qwen2.5-3B-Instruct (Qwen research licence; 4-bit on 4 GB GPUs)
    python scripts/download_generators.py --only-improvements  # just the re-ranker and the entailment model

Each model goes to models/<folder>/ with a provenance.json (model id, commit, licence, weights SHA-256).
Already-downloaded models are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "generator-1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", "Apache-2.0"),
    "generator-smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "Apache-2.0"),
}
OPTIONAL_3B = {"generator-3b": ("Qwen/Qwen2.5-3B-Instruct", "Qwen Research License (non-commercial)")}
IMPROVEMENTS = {  # retrieval re-ranker and entailment checker for the verifier
    "reranker": ("BAAI/bge-reranker-base", "MIT"),
    "nli": ("cross-encoder/nli-deberta-v3-small", "Apache-2.0"),
}
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab.json", "LICENSE*"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-3b", action="store_true")
    ap.add_argument("--with-improvements", action="store_true", help="also the re-ranker and entailment model")
    ap.add_argument("--only-improvements", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    todo = {} if args.only_improvements else dict(MODELS, **(OPTIONAL_3B if args.with_3b else {}))
    if args.with_improvements or args.only_improvements:
        todo = dict(IMPROVEMENTS, **todo)  # small models first
    failed = []
    api = HfApi()
    for folder, (repo, licence) in todo.items():
        dest = ROOT / "models" / folder
        if (dest / "provenance.json").exists() and (any(dest.glob("*.safetensors")) or any(dest.glob("*.bin"))):
            print(f"[skip] {repo} already in models/{folder}")
            continue
        try:
            info = api.model_info(repo)
            rev = info.sha
            print(f"[download] {repo} @ {rev[:7]} -> models/{folder}")
            files = {s.rfilename for s in (info.siblings or [])}
            patterns = PATTERNS + (["pytorch_model.bin"] if not any(f.endswith(".safetensors") for f in files) else [])
            snapshot_download(repo, revision=rev, local_dir=str(dest), allow_patterns=patterns)
        except Exception as exc:  # keep going: the other models may still download
            print(f"[failed] {repo}: {exc}")
            failed.append(repo)
            continue
        weights = sorted(dest.glob("*.safetensors")) or sorted(dest.glob("pytorch_model.bin"))
        meta = {"model_id": repo, "revision": rev, "license": licence,
                "weights_files": [w.name for w in weights], "weights_sha256": {w.name: sha256(w) for w in weights},
                "source": f"https://huggingface.co/{repo}",
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (dest / "provenance.json").write_text(json.dumps(meta, indent=2))
        print(f"[ok] {repo}: {sum(w.stat().st_size for w in weights) / 2**30:.2f} GiB")
    if failed:
        print(f"{len(failed)} model(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
