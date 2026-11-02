"""Document knowledge base and passage retrieval for the question-answering module.

Sources (all public, all shipped with or referenced by the project):

* ``FDA-README``  - "README.DOC" for the FAERS Quarterly Data Extract (Readme.pdf)
* ``FDA-FAQ``     - FDA's FAQs for the FAERS Quarterly Data Extract (FAQs.pdf)
* ``FDA-ASC_NTS`` - FDA's ASCII file/field descriptions (ASCII/ASC_NTS.pdf)
* ``CTCAE``       - NCI CTCAE v5.0 term definitions (data/reference), used as a
  glossary for reaction terms that appear in it
* ``PROJECT``     - this project's own methods notes (``METHOD_NOTES`` below),
  clearly attributed to the project, never to FDA

The PDFs are read directly from the FAERS zip (``data/raw/faers_ascii_*.zip``)
so the knowledge base can be rebuilt without the extracted folder. Each chunk
keeps its source, page and the SHA-256 of the file it came from.

Retrieval is lexical (Okapi BM25, implemented here on top of scikit-learn's
CountVectorizer) with a TF-IDF cosine baseline. SapBERT is deliberately *not*
used for passages: it is trained on short biomedical entity names (max 25
tokens) and its [CLS] vector is not a passage representation; it is used for
entity linking in ``rag.py`` instead.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths, get_paths

log = logging.getLogger(__name__)

KB_VERSION = "pv-kb-1.0"
PDF_SOURCES = {  # zip member (case-insensitive suffix) -> (source id, title)
    "readme.pdf": ("FDA-README", "FDA FAERS Quarterly Data Extract README"),
    "faqs.pdf": ("FDA-FAQ", "FDA FAERS Quarterly Data Extract FAQs"),
    "asc_nts.pdf": ("FDA-ASC_NTS", "FDA FAERS ASCII file descriptions (ASC_NTS)"),
}
WINDOW_WORDS = 140
STRIDE_WORDS = 100

# Short, factual notes describing *this project's* methods. They are attributed to the project
# (source "PROJECT") so the answer never presents them as FDA statements.
METHOD_NOTES: list[tuple[str, str]] = [
    ("Disproportionality is not causality",
     "Disproportionality statistics (ROR, PRR, IC) compare how often a reaction is reported with a drug versus with all "
     "other drugs in FAERS. A high value means the drug-reaction pair is reported together more often than expected. It "
     "does not show that the drug causes the reaction, it is not an incidence rate and it is not an absolute risk, "
     "because FAERS has no denominator of exposed patients and reporting is voluntary and selective."),
    ("Reporting odds ratio (ROR)",
     "The reporting odds ratio is (a/b)/(c/d) from the 2x2 table of reports: a = reports with the drug and the reaction, "
     "b = drug without the reaction, c = reaction with other drugs, d = neither. The 95% confidence interval is "
     "exp(ln ROR +/- 1.96 sqrt(1/a+1/b+1/c+1/d)). The project flags a ROR signal when the lower 95% limit exceeds 1 and a >= 3."),
    ("Proportional reporting ratio (PRR)",
     "The proportional reporting ratio is [a/(a+b)] / [c/(c+d)]. Following Evans et al. (2001) the project flags a PRR "
     "signal when PRR >= 2, the Yates-corrected chi-square >= 4 and a >= 3."),
    ("Information component (IC)",
     "The information component is the Bayesian log2 ratio of observed to expected co-reporting (BCPNN). The project "
     "flags an IC signal when the lower credibility bound IC025 is above 0 and a >= 3."),
    ("Sparse cells and Haldane correction",
     "When any cell of the 2x2 table is zero, 0.5 is added to every cell (Haldane-Anscombe correction) so that the ROR "
     "and its confidence interval remain defined. Such rows are marked haldane_applied and should be interpreted with care."),
    ("Indication bias flag",
     "A pair is marked as possible indication bias when at least 20% of the drug-reaction cases also list the reaction "
     "term as an indication for a drug, for example a skin disease reported for a drug used to treat that skin disease."),
    ("Case de-duplication",
     "The pipeline removes cases on FDA's quarterly delete list, keeps only the latest version of each CASEID (latest "
     "FDA_DT, then highest PRIMARYID) and flags probable duplicates with an identical clinical fingerprint (sex, age, "
     "event date, country, suspect drugs and reactions) under different CASEIDs; probable duplicates are excluded from "
     "signal detection."),
    ("Minimum case count",
     "Pairs with fewer than 3 co-reports are not tested for disproportionality, because ratios estimated from one or two "
     "reports are unstable."),
    ("Reaction categories are provisional",
     "Reaction categories in this project are project-defined body-system groupings assigned by the SapBERT language "
     "model. They are not MedDRA System Organ Classes and are not validated clinical codes."),
]

_STOP = None


def _stopwords() -> frozenset[str]:
    global _STOP
    if _STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        _STOP = frozenset(ENGLISH_STOP_WORDS) - {"not", "no"}
    return _STOP


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _stem(w: str) -> str:
    """Tiny, deterministic suffix stripper (keeps the module dependency-free).

    duplicates/duplicate/duplicated -> duplicat, ages/age -> ag, handled/handle -> handl.
    """
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    for suf in ("ing", "ed"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    stop = _stopwords()
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in stop and len(t) > 1]


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------
# building the knowledge base
# --------------------------------------------------------------------------

def find_faers_zip(paths: Paths) -> Path | None:
    zips = sorted(paths.raw.glob("faers_ascii_*.zip"))
    return zips[0] if zips else None


def _pdf_pages(data: bytes) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [(p.extract_text() or "") for p in reader.pages]


def split_faq(pages: list[str]) -> list[dict]:
    """Split the FDA FAQ into one record per numbered question (question + answer kept separately)."""
    marked = []
    for i, txt in enumerate(pages, start=1):
        marked.append(f"\n<<PAGE {i}>>\n{txt}")
    full = "".join(marked)
    # numbered questions start a line: "12. Why ..." ; numbers must increase monotonically
    starts = []
    expected = 1
    for m in re.finditer(r"(?m)^\s*(\d{1,2})\.\s+(?=[A-Z])", full):
        if int(m.group(1)) == expected:
            starts.append((m.start(), expected))
            expected += 1
    out = []
    for j, (pos, num) in enumerate(starts):
        end = starts[j + 1][0] if j + 1 < len(starts) else len(full)
        block = full[pos:end]
        page_marks = re.findall(r"<<PAGE (\d+)>>", full[:pos])
        page = int(page_marks[-1]) if page_marks else 1
        block = re.sub(r"<<PAGE \d+>>", " ", block)
        block = re.sub(r"(?m)^\s*\d+\s*$", " ", block)  # page numbers
        q, _, a = block.partition("ANSWER:")
        q = _clean_ws(re.sub(r"^\s*\d{1,2}\.\s+", "", q))
        a = _clean_ws(a)
        if q and a:
            out.append({"number": num, "question": q, "answer": a, "page": page})
    return out


def window_chunks(pages: list[str], window: int = WINDOW_WORDS, stride: int = STRIDE_WORDS) -> list[dict]:
    """Overlapping word windows over a document, each tagged with the page it starts on."""
    words, page_of = [], []
    for i, txt in enumerate(pages, start=1):
        w = _clean_ws(txt).split(" ")
        w = [x for x in w if x]
        words += w
        page_of += [i] * len(w)
    out = []
    for s in range(0, max(1, len(words)), stride):
        seg = words[s : s + window]
        if len(seg) < 25 and out:
            break
        out.append({"page": page_of[s] if s < len(page_of) else 1, "text": " ".join(seg)})
        if s + window >= len(words):
            break
    return out


def load_ctcae_definitions(paths: Paths) -> pd.DataFrame:
    """CTCAE term definitions (term, term_key, soc, definition) - empty frame if the file is absent."""
    from .evaluate import find_ctcae

    f = find_ctcae(paths)
    if f is None:
        return pd.DataFrame(columns=["term", "term_key", "soc", "definition"])
    raw = pd.read_excel(f, sheet_name=0)
    term_col = next(c for c in raw.columns if "Term" in str(c))
    soc_col = next(c for c in raw.columns if "SOC" in str(c))
    def_col = next(c for c in raw.columns if "Definition" in str(c))
    df = pd.DataFrame({"term": raw[term_col].astype(str).str.strip(), "soc": raw[soc_col].astype(str).str.strip(),
                       "definition": raw[def_col].astype(str).str.strip()})
    df = df[(df["definition"].str.len() > 10) & (df["definition"].str.lower() != "nan")]
    df = df[~df["term"].str.contains("other, specify", case=False)]
    df["term_key"] = df["term"].str.lower().str.replace(r"\s+", " ", regex=True)
    return df.drop_duplicates("term_key").reset_index(drop=True)


def build(paths: Paths | None = None) -> dict:
    """(Re)build ``data/nlp/kb_chunks.parquet`` and ``kb_manifest.json``."""
    paths = (paths or get_paths()).ensure()
    rows: list[dict] = []
    sources: dict[str, dict] = {}
    z = find_faers_zip(paths)
    if z is None:
        log.warning("FAERS zip not found; knowledge base will contain project notes and CTCAE only")
    else:
        try:
            import pypdf  # noqa: F401
        except ImportError:
            log.warning("pypdf not installed; FDA documents skipped (pip install pypdf)")
            z = None
    if z is not None:
        with zipfile.ZipFile(z) as zf:
            for info in zf.infolist():
                key = info.filename.replace("\\", "/").split("/")[-1].lower()
                if key not in PDF_SOURCES:
                    continue
                src, title = PDF_SOURCES[key]
                data = zf.read(info)
                pages = _pdf_pages(data)
                sources[src] = {"title": title, "file": info.filename, "sha256": _sha256(data), "pages": len(pages),
                                "zip": z.name}
                if src == "FDA-FAQ":
                    for q in split_faq(pages):
                        rows.append({"source": src, "title": f"FAQ {q['number']}: {q['question']}", "page": q["page"],
                                     "text": f"{q['question']} {q['answer']}", "question": q["question"],
                                     "answer": q["answer"], "faq_number": q["number"]})
                else:
                    for k, ch in enumerate(window_chunks(pages)):
                        rows.append({"source": src, "title": f"{title}, p. {ch['page']}", "page": ch["page"],
                                     "text": ch["text"], "question": "", "answer": "", "faq_number": 0,
                                     "chunk_in_doc": k})
    for title, text in METHOD_NOTES:
        rows.append({"source": "PROJECT", "title": f"Project methods note: {title}", "page": 0, "text": text,
                     "question": "", "answer": "", "faq_number": 0})
    sources["PROJECT"] = {"title": "Project methods notes (pvsys.nlp.kb.METHOD_NOTES)", "n": len(METHOD_NOTES)}
    ctcae = load_ctcae_definitions(paths)
    for r in ctcae.itertuples():
        rows.append({"source": "CTCAE", "title": f"CTCAE v5.0 definition: {r.term}", "page": 0,
                     "text": f"{r.term}: {r.definition}", "question": "", "answer": "", "faq_number": 0,
                     "term_key": r.term_key})
    if len(ctcae):
        sources["CTCAE"] = {"title": "NCI CTCAE v5.0 term definitions", "n": int(len(ctcae))}
    df = pd.DataFrame(rows)
    for col in ("chunk_in_doc", "term_key"):
        if col not in df.columns:
            df[col] = None
    df["chunk_in_doc"] = pd.to_numeric(df["chunk_in_doc"], errors="coerce").fillna(-1).astype(int)
    df["term_key"] = df["term_key"].fillna("").astype(str)
    df.insert(0, "chunk_id", [f"{s}-{i}" for i, s in enumerate(df["source"])])
    df.to_parquet(paths.nlp / "kb_chunks.parquet", index=False)
    manifest = {"kb_version": KB_VERSION, "n_chunks": int(len(df)),
                "by_source": df["source"].value_counts().to_dict(), "sources": sources,
                "window_words": WINDOW_WORDS, "stride_words": STRIDE_WORDS,
                "content_sha256": _sha256(pd.util.hash_pandas_object(df[["chunk_id", "text"]], index=False)
                                          .to_numpy().tobytes())[:16]}
    (paths.nlp / "kb_manifest.json").write_text(json.dumps(manifest, indent=2))
    load_kb.cache_clear()
    return manifest


@lru_cache(maxsize=4)
def load_kb(root: str) -> pd.DataFrame:
    f = Path(root) / "kb_chunks.parquet"
    if not f.exists():
        raise FileNotFoundError("Knowledge base not built - run `python -m pvsys.pipeline rag-index`")
    return pd.read_parquet(f)


# --------------------------------------------------------------------------
# retrievers
# --------------------------------------------------------------------------

@dataclass
class Hit:
    index: int
    score: float


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75) over pre-tokenised documents."""

    name = "bm25"

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        from sklearn.feature_extraction.text import CountVectorizer

        self.k1, self.b = k1, b
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(docs).tocsc().astype(np.float32)
        n = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avg = dl.mean() if n else 1.0
        tf = tf.tocsr()
        # pre-compute the BM25 term weight for every (doc, term) entry
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        denom = tf.data + k1 * (1 - b + b * dl[rows] / (avg or 1.0))
        tf.data = (tf.data * (k1 + 1) / denom).astype(np.float32)
        self.weights = tf.tocsc()
        self.n = n

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        idx = q.indices
        if len(idx) == 0:
            return np.zeros(self.n, dtype=np.float32)
        w = self.weights[:, idx] @ (self.idf[idx] * q.data)
        return np.asarray(w).ravel().astype(np.float32)


class TfidfRetriever:
    """Word TF-IDF cosine baseline (same tokeniser as BM25)."""

    name = "tfidf"

    def __init__(self, docs: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
        self.m = self.vec.fit_transform(docs)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray((self.m @ self.vec.transform([query]).T).toarray()).ravel().astype(np.float32)


def top_k(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[Hit]:
    s = scores.copy()
    if mask is not None:
        s[~mask] = -np.inf
    k = max(0, min(k, int(np.isfinite(s).sum())))
    if k == 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx], kind="stable")]
    return [Hit(int(i), float(s[i])) for i in idx if s[i] > 0]


class PassageIndex:
    """BM25 index over the knowledge base used at question time."""

    def __init__(self, paths: Paths | None = None):
        self.paths = paths or get_paths()
        self.kb = load_kb(str(self.paths.nlp)).reset_index(drop=True)
        self.bm25 = BM25(self.kb["text"].tolist())
        man = self.paths.nlp / "kb_manifest.json"
        self.manifest = json.loads(man.read_text()) if man.exists() else {}

    def search(self, query: str, k: int = 4, sources: tuple[str, ...] | None = None) -> list[dict]:
        mask = self.kb["source"].isin(sources).to_numpy() if sources else None
        hits = top_k(self.bm25.scores(query), k, mask)
        out = []
        for h in hits:
            r = self.kb.iloc[h.index]
            out.append({"chunk_id": r["chunk_id"], "source": r["source"], "title": r["title"], "page": int(r["page"]),
                        "text": r["text"], "answer": r.get("answer") or "", "score": round(h.score, 3)})
        return out

    def by_title(self, title_prefix: str) -> dict | None:
        m = self.kb[self.kb["title"].str.startswith(title_prefix)]
        if m.empty:
            return None
        r = m.iloc[0]
        return {"chunk_id": r["chunk_id"], "source": r["source"], "title": r["title"], "page": int(r["page"]),
                "text": r["text"], "score": None}

    def ctcae_definition(self, pt_key: str) -> dict | None:
        m = self.kb[(self.kb["source"] == "CTCAE") & (self.kb["term_key"] == pt_key)]
        if m.empty:
            return None
        r = m.iloc[0]
        return {"chunk_id": r["chunk_id"], "source": "CTCAE", "title": r["title"], "page": 0, "text": r["text"],
                "score": None}
