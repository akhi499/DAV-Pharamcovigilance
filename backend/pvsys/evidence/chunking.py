"""Section-aware chunking of external evidence documents.

* Labels: one or more chunks per SPL section (Boxed Warning, Warnings and Precautions, ...), so a
  citation always names the section it came from.
* Literature: title + abstract (split only if long).
* Regulatory (FDA Drug Safety Communications): word windows; each chunk is tagged with the FAERS
  drug names (and label brand/generic aliases) it mentions.

Chunks are ~CHUNK_WORDS words with OVERLAP_WORDS overlap, cut at sentence boundaries where possible.
"""

from __future__ import annotations

import hashlib
import json
import re

import pandas as pd

from .sources import LABEL_SECTIONS

CHUNK_WORDS = 160
OVERLAP_WORDS = 30
MAX_CHUNKS_PER_SECTION = 25  # very long Adverse Reactions tables are truncated (recorded in the manifest)


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.;:!?])\s+(?=[A-Z0-9•(\[])|\s+(?=•)", text)
    return [p.strip() for p in parts if p.strip()]


def split_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Pack sentences into ~size-word chunks; sentences longer than size are hard-split."""
    sents = []
    for s in _sentences(text):
        w = s.split()
        while len(w) > size:
            sents.append(" ".join(w[:size]))
            w = w[size - overlap:]
        if w:
            sents.append(" ".join(w))
    chunks, cur = [], []
    for s in sents:
        n = len(s.split())
        if cur and sum(len(x.split()) for x in cur) + n > size:
            chunks.append(" ".join(cur))
            tail = " ".join(cur).split()[-overlap:] if overlap else []
            cur = [" ".join(tail)] if tail else []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c.split()) >= 5]


def _cid(doc_id: str, section: str, i: int, text: str) -> str:
    return hashlib.sha256(f"{doc_id}|{section}|{i}|{text[:80]}".encode()).hexdigest()[:12]


def tag_drugs(text: str, vocab: dict[str, str]) -> list[str]:
    """FAERS drug names whose name or alias (vocab: ALIAS -> DRUG) appears as a whole word in text."""
    up = " " + re.sub(r"[^A-Z0-9]+", " ", text.upper()) + " "
    return sorted({drug for alias, drug in vocab.items() if f" {alias} " in up})


def drug_vocab(docs: pd.DataFrame) -> dict[str, str]:
    vocab: dict[str, str] = {}
    for r in docs[docs["source_type"] == "label"].itertuples():
        norm = re.sub(r"[^A-Z0-9]+", " ", r.drug.upper()).strip()
        vocab[norm] = r.drug
        first = norm.split()[0]
        if len(first) >= 6:  # "METFORMIN" for METFORMIN HYDROCHLORIDE
            vocab.setdefault(first, r.drug)
        for a in (r.aliases or "").split("|"):
            a = re.sub(r"[^A-Z0-9]+", " ", a).strip()
            if len(a) >= 4:
                vocab.setdefault(a, r.drug)
    return vocab


def chunk_documents(docs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    vocab = drug_vocab(docs)
    rows, truncated = [], 0
    meta_cols = ["doc_id", "source_type", "title", "publisher", "date", "version", "url", "identifier"]
    for d in docs.itertuples():
        meta = {c: getattr(d, c) for c in meta_cols}
        if d.source_type == "label":
            secs = json.loads(d.sections_json)
            for key, name in LABEL_SECTIONS.items():
                if key not in secs:
                    continue
                parts = split_words(secs[key])
                if len(parts) > MAX_CHUNKS_PER_SECTION:
                    truncated += len(parts) - MAX_CHUNKS_PER_SECTION
                    parts = parts[:MAX_CHUNKS_PER_SECTION]
                for i, t in enumerate(parts):
                    rows.append({**meta, "chunk_id": _cid(d.doc_id, key, i, t), "section_key": key, "section": name,
                                 "ordinal": i, "drugs": d.drug, "text": t})
        elif d.source_type == "literature":
            for i, t in enumerate(split_words(f"{d.title} {d.text}", size=220)):
                rows.append({**meta, "chunk_id": _cid(d.doc_id, "abstract", i, t), "section_key": "abstract",
                             "section": "Abstract", "ordinal": i, "drugs": d.drug, "text": t})
        else:
            for i, t in enumerate(split_words(d.text)):
                tags = tag_drugs(f"{d.title} {t}", vocab) or tag_drugs(d.title, vocab)
                rows.append({**meta, "chunk_id": _cid(d.doc_id, "body", i, t), "section_key": "body",
                             "section": "Drug Safety Communication", "ordinal": i, "drugs": "|".join(tags), "text": t})
    cols = ["chunk_id", "doc_id", "source_type", "drugs", "section_key", "section", "ordinal", "title", "publisher",
            "date", "version", "url", "identifier", "text"]
    df = pd.DataFrame(rows, columns=cols).drop_duplicates("chunk_id").reset_index(drop=True)
    return df, {"chunk_words": CHUNK_WORDS, "overlap_words": OVERLAP_WORDS, "max_chunks_per_section": MAX_CHUNKS_PER_SECTION,
                "chunks_truncated": truncated, "drug_vocabulary": len(vocab)}
