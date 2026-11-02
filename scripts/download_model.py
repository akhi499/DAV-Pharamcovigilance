"""Download the pinned SapBERT revision from Hugging Face into models/ (one-time, ~440 MB).

    python scripts/download_model.py

Only the files needed for CPU inference are fetched, at a fixed commit so the
weights are reproducible; SHA-256 of the weights is printed and also recorded
in every prediction run's provenance.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MODEL_ID = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
REVISION = "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"
FILES = ["config.json", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json", "README.md", "model.safetensors"]
EXPECTED_SHA256 = {"model.safetensors": "a4696930afef9aab296196d3d2142216c44cba24f21b4f285ceca7af21025614"}

dest = Path(__file__).resolve().parents[1] / "models" / "SapBERT-from-PubMedBERT-fulltext"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        target = dest / f
        if target.exists() and (f not in EXPECTED_SHA256 or sha256(target) == EXPECTED_SHA256[f]):
            print(f"ok      {f}")
            continue
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{f}"
        print(f"fetch   {url}")
        tmp = target.with_suffix(target.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
        tmp.replace(target)
        if f in EXPECTED_SHA256 and sha256(target) != EXPECTED_SHA256[f]:
            print(f"ERROR: checksum mismatch for {f}", file=sys.stderr)
            return 1
    (dest / "provenance.json").write_text(json.dumps({"model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0"}, indent=2))
    print(f"SapBERT ready in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
