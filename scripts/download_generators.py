"""Download local generator models for the extractive-vs-generative comparison, recording exact provenance.

    python scripts/download_generators.py            # Qwen2.5-1.5B-Instruct + SmolLM2-1.7B-Instruct
    python scripts/download_generators.py --with-3b  # also Qwen2.5-3B-Instruct (Qwen research licence; CPU on 4 GB GPUs)

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
    args = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    todo = dict(MODELS, **(OPTIONAL_3B if args.with_3b else {}))
    api = HfApi()
    for folder, (repo, licence) in todo.items():
        dest = ROOT / "models" / folder
        if (dest / "provenance.json").exists() and any(dest.glob("*.safetensors")):
            print(f"[skip] {repo} already in models/{folder}")
            continue
        rev = api.model_info(repo).sha
        print(f"[download] {repo} @ {rev[:7]} -> models/{folder}")
        snapshot_download(repo, revision=rev, local_dir=str(dest), allow_patterns=PATTERNS)
        weights = sorted(dest.glob("*.safetensors"))
        meta = {"model_id": repo, "revision": rev, "license": licence,
                "weights_files": [w.name for w in weights], "weights_sha256": {w.name: sha256(w) for w in weights},
                "source": f"https://huggingface.co/{repo}",
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        (dest / "provenance.json").write_text(json.dumps(meta, indent=2))
        print(f"[ok] {repo}: {sum(w.stat().st_size for w in weights) / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
