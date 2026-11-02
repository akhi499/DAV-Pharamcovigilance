"""Evidence-grounded question answering over FAERS 2025 Q1 (retrieval-augmented generation).

Pipeline for one question::

    question
      -> 1. understanding   drug mentions (FAERS vocabulary, FAERS-derived brand aliases, typo-tolerant),
                            reaction mentions (exact MedDRA PT match, else SapBERT semantic linking with
                            an acceptance threshold), question type (pair / drug / reaction / documentation)
      -> 2. retrieval       structured: precomputed ROR/PRR/IC rows from DuckDB (or raw co-report counts
                            for untested pairs); unstructured: BM25 over FDA FAERS documentation and project
                            method notes; CTCAE definition of the linked reaction term
      -> 3. generation      every answer sentence is composed from retrieved values and carries citation
                            ids of the evidence it uses; the causality caveat is always attached
      -> 4. (optional)      a local generative model may paraphrase the evidence; its text is accepted only
                            if a verifier finds no number, citation or causal claim absent from the evidence
                            (``generator.py``)

The engine abstains - says it could not answer - when no drug, reaction or
relevant passage is found, and asks for clarification when a drug name is
ambiguous. It only *reads* processed statistics; it never computes or
alters them, so the statistical pipeline stays independent of it.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Paths, SignalConfig, get_paths
from .kb import KB_VERSION, PassageIndex, tokenize

log = logging.getLogger(__name__)

RAG_VERSION = "pv-rag-1.0"
MAX_QUESTION_LEN = 300
# Reaction-linking acceptance thresholds (cosine similarity of the top match). Chosen from the observed
# similarity of correct lay-term links (0.72-0.96) versus noise words (<= 0.66) and checked on the CTCAE
# dev split by evaluate_rag(); a phrase whose best match is below the threshold is reported as not understood.
LINK_THRESHOLD = {"neural": 0.70, "lexical-fallback": 0.45}
EXPAND_WITHIN = 0.05       # related terms: within this similarity of the top match ...
EXPAND_MIN = {"neural": 0.75, "lexical-fallback": 0.55}  # ... and above this floor
MAX_RELATED = 5
DOC_SOURCES = ("FDA-README", "FDA-FAQ", "FDA-ASC_NTS", "PROJECT")
NOT_DRUGS = {"WATER", "PLACEBO", "UNKNOWN", "UNSPECIFIED", "SOAP"}

_FILLER = set("""
is are was were be been there any a an the of for in on to with without and or between about from by into than
does do did can could may might will would should shall has have had cause causes caused causing causal link linked
links association associated associate relationship related relation connected connection signal signals
signalling disproportionality disproportionate risk risks increase increased reports reported report reporting
case cases adverse event events reaction reactions effect effects side drug drugs medicine medicines medication
medications what which who how many much show me tell list give find top most common frequent frequently
please faers 2025 q1 quarter data database patients patient people taking take takes use using used user users
i my we our you it its this that these those like such as also more less vs versus statistically statistical
ror prr ic score scores value values strength strong evidence
""".split())

_DOC_TRIGGERS = re.compile(
    r"\b(what is|what are|what does|what do|how is|how are|how do|how does|why|explain|define|definition|meaning|"
    r"mean by|difference between|interpret)\b", re.I)
_METHOD_TERMS = re.compile(
    r"\b(ror|prr|ic025|ic|bcpnn|information component|odds ratio|proportional reporting|disproportionality|"
    r"chi[- ]?square|confidence interval|haldane|sparse|indication bias|signal detection|duplicate|duplicates|"
    r"de-?duplicat\w*|case ?version|primaryid|caseid|isr|field|fields|file|files|ascii|xml|extract|qde|meddra|"
    r"faers|causality|causation|incidence|absolute risk|age_cod|age|weight|reporter|occp_cod|rept_cod|role_cod|"
    r"delete|deleted|quarterly|legacy|laers|dictionary|threshold|thresholds|minimum|ages|weights|dates?|dose|doses|route|"
    r"version|versions|reporters?|countr(?:y|ies)|outcomes?|indications?|serious|expedited|periodic)\b", re.I)


_TOK_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def tokens(text: str) -> list[tuple[str, int, int]]:
    """Word tokens with character offsets; drug and reaction vocabularies are normalised with the same rule."""
    return [(m.group(0), m.start(), m.end()) for m in _TOK_RE.finditer(text)]


def norm_key(text: str) -> str:
    return " ".join(t[0] for t in tokens(text)).upper()


def fmt(x: float | None) -> str:
    """Deterministic number formatting shared by the answer text and the evidence (verifier relies on it)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    ax = abs(x)
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


@dataclass
class DrugMention:
    drug: str
    text: str
    match: str  # exact | alias | prefix | fuzzy
    cases: int
    score: float = 1.0
    span: tuple[int, int] = (0, 0)


@dataclass
class ReactionMention:
    phrase: str
    pt: str
    pt_key: str
    similarity: float
    match: str  # exact | semantic
    cases: int
    related: list[dict] = field(default_factory=list)


@dataclass
class Parsed:
    question: str
    intent: str
    drugs: list[DrugMention]
    reactions: list[ReactionMention]
    unlinked_phrases: list[dict]
    ambiguous_drugs: list[dict]
    link_mode: str


class DrugLinker:
    """Find suspect-drug names in free text using the FAERS vocabulary and FAERS-derived brand aliases."""

    def __init__(self, db):
        rows = db.query(
            """SELECT drug, count(DISTINCT primaryid) AS cases FROM drugs
               WHERE role_cod IN ('PS','SS') AND drug IS NOT NULL GROUP BY 1""")
        self.cases = {r["drug"]: int(r["cases"]) for r in rows if r["drug"] not in NOT_DRUGS}
        # Brand / verbatim names -> normalised ingredient, learnt from FAERS itself: the verbatim DRUGNAME (dose
        # stripped) maps to the normalised name in >= 90% of >= 5 rows, and is not itself a drug in the vocabulary.
        alias_rows = db.query(
            r"""WITH v AS (
                  SELECT upper(trim(regexp_replace(drugname, '\s+\d+(\.\d+)?\s*(MG|MCG|G|ML|IU|UNITS?|%).*$', '', 'i')))
                         AS alias, drug FROM drugs WHERE role_cod IN ('PS','SS') AND drug IS NOT NULL AND drugname IS NOT NULL),
                c AS (SELECT alias, drug, count(*) AS n FROM v GROUP BY 1, 2),
                t AS (SELECT alias, drug, n, sum(n) OVER (PARTITION BY alias) AS tot,
                             row_number() OVER (PARTITION BY alias ORDER BY n DESC, drug) AS rk FROM c)
                SELECT alias, drug, n, tot FROM t WHERE rk = 1 AND n >= 5 AND n >= 0.9 * tot AND alias <> drug""")
        self.alias = {r["alias"]: r["drug"] for r in alias_rows
                      if r["drug"] in self.cases and r["alias"] not in self.cases and len(r["alias"]) >= 4
                      and re.fullmatch(r"[A-Z0-9][A-Z0-9 \-]*", r["alias"])}
        # normalised lookup (punctuation-insensitive: "A/CALIFORNIA/7/2009 (H1N1)" == "A CALIFORNIA 7 2009 H1N1")
        self.lookup: dict[str, tuple[str, str]] = {}
        for a, d in self.alias.items():
            self.lookup.setdefault(norm_key(a), (d, "alias"))
        for d in self.cases:
            self.lookup[norm_key(d)] = (d, "exact")
        self.max_words = min(max([len(k.split()) for k in self.lookup] + [1]), 60)
        # typo-tolerant matching only against names with >= 3 suspect cases (limits spurious fuzzy hits)
        self.single = sorted({k for k in list(self.cases) + list(self.alias) if " " not in k and len(k) >= 5
                              and self.cases.get(self.alias.get(k, k), 0) >= 3})
        self.first_word: dict[str, list[str]] = {}
        for d in self.cases:
            if " " in d:
                self.first_word.setdefault(d.split()[0], []).append(d)

    def link(self, text: str) -> tuple[list[DrugMention], list[tuple[int, int]], list[dict]]:
        """Return (mentions, consumed character spans, ambiguous candidates)."""
        toks = [(t.upper(), s, e) for t, s, e in tokens(text)]
        used = [False] * len(toks)
        found: list[DrugMention] = []
        spans: list[tuple[int, int]] = []
        ambiguous: list[dict] = []
        for n in range(min(self.max_words, len(toks)), 0, -1):
            for i in range(0, len(toks) - n + 1):
                if any(used[i : i + n]):
                    continue
                cand = " ".join(t[0] for t in toks[i : i + n])
                if n == 1 and (len(cand) < 3 or cand.lower() in _FILLER):
                    continue
                drug, how = self.lookup.get(cand, (None, None))
                if drug:
                    for j in range(i, i + n):
                        used[j] = True
                    spans.append((toks[i][1], toks[i + n - 1][2]))
                    if drug not in {d.drug for d in found}:
                        found.append(DrugMention(drug, text[toks[i][1]: toks[i + n - 1][2]], how, self.cases[drug],
                                                 span=(toks[i][1], toks[i + n - 1][2])))
        for i, (tok, s, e) in enumerate(toks):
            if used[i] or len(tok) < 5 or tok.lower() in _FILLER:
                continue
            if tok in self.first_word:  # e.g. "insulin" -> INSULIN GLARGINE, INSULIN LISPRO ...
                cands = sorted(self.first_word[tok], key=lambda d: -self.cases[d])
                if len(cands) == 1:
                    used[i] = True
                    spans.append((s, e))
                    found.append(DrugMention(cands[0], text[s:e], "prefix", self.cases[cands[0]], span=(s, e)))
                else:
                    used[i] = True
                    spans.append((s, e))
                    ambiguous.append({"text": text[s:e], "span": [s, e],
                                      "candidates": [{"drug": d, "cases": self.cases[d]} for d in cands[:6]]})
                continue
            if len(tok) >= 6:
                m = difflib.get_close_matches(tok, self.single, n=1, cutoff=0.88)
                if m:
                    drug = self.alias.get(m[0], m[0])
                    used[i] = True
                    spans.append((s, e))
                    if drug not in {d.drug for d in found}:
                        ratio = difflib.SequenceMatcher(None, tok, m[0]).ratio()
                        found.append(DrugMention(drug, text[s:e], "fuzzy", self.cases[drug], round(ratio, 3), (s, e)))
        return found[:3], spans, ambiguous


class RAGEngine:
    def __init__(self, paths: Paths | None = None, db=None):
        from ..db import Database
        from .search import get_index

        self.paths = paths or get_paths()
        self.db = db or Database(self.paths)
        self.cfg = SignalConfig()
        self.drugs = DrugLinker(self.db)
        self.index = get_index()
        self.mode = self.index.mode
        self.terms_by_key = {r.pt_key: (r.pt, int(r.cases)) for r in self.index.terms[["pt_key", "pt", "cases"]].itertuples()}
        self.pt_lookup = {norm_key(k): k for k in self.terms_by_key}
        self.pt_max_words = min(max([len(k.split()) for k in self.pt_lookup] + [1]), 12)
        self.passages = PassageIndex(self.paths)
        self.caveat_project = self.passages.by_title("Project methods note: Disproportionality is not causality")
        cav = self.passages.search("clinical caveats causal relationship not established reporting", 1, ("FDA-README",))
        self.caveat_fda = cav[0] if cav else None
        self._lock = threading.Lock()
        prov = self.paths.nlp / "provenance.json"
        p = json.loads(prov.read_text()) if prov.exists() else {}
        man = self.paths.processed / "manifest.json"
        m = json.loads(man.read_text()) if man.exists() else {}
        self.provenance = {
            "rag_version": RAG_VERSION,
            "kb_version": KB_VERSION,
            "kb_content_sha256": self.passages.manifest.get("content_sha256"),
            "kb_chunks": self.passages.manifest.get("n_chunks"),
            "reaction_linker": "SapBERT" if self.mode == "neural" else "TF-IDF char n-gram (fallback)",
            "reaction_linker_model": (p.get("model") or {}).get("model_id") if self.mode == "neural" else "sklearn-tfidf-char",
            "reaction_linker_revision": (p.get("model") or {}).get("model_revision") if self.mode == "neural" else None,
            "link_threshold": LINK_THRESHOLD[self.mode],
            "passage_retriever": "Okapi BM25 (k1=1.5, b=0.75)",
            "nlp_run_id": p.get("run_id"),
            "data_source_sha256": m.get("source_sha256"),
            "quarter": m.get("quarter"),
            "drug_vocabulary": len(self.drugs.cases),
            "brand_aliases": len(self.drugs.alias),
        }

    # ------------------------------------------------------------------ understanding
    def _exact_terms(self, text: str, spans: list[tuple[int, int]]) -> tuple[list[str], list[tuple[int, int]]]:
        """Longest-first exact MedDRA PT matches (punctuation-insensitive) outside the given spans."""
        toks = [t for t in tokens(text) if not any(s <= t[1] < e for s, e in spans)]
        used = [False] * len(toks)
        keys, out_spans = [], []
        for n in range(min(self.pt_max_words, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(used[i : i + n]):
                    # only contiguous runs: tokens must be adjacent in the original text
                    continue
                if any(text[toks[j][2]: toks[j + 1][1]].strip(" ,;:") for j in range(i, i + n - 1)):
                    continue
                cand = " ".join(t[0] for t in toks[i : i + n]).upper()
                if n == 1 and (len(cand) < 4 or cand.lower() in _FILLER):
                    continue
                k = self.pt_lookup.get(cand)
                if k and k not in keys:
                    keys.append(k)
                    out_spans.append((toks[i][1], toks[i + n - 1][2]))
                    for j in range(i, i + n):
                        used[j] = True
        return keys, out_spans

    def _resolve_overlaps(self, text: str, drugs: list[DrugMention], ambiguous: list[dict]):
        """A word inside an exact MedDRA term is part of the reaction ('Blood magnesium decreased', 'Influenza'):
        a longer exact PT beats any drug match it contains; an equally long PT beats prefix/fuzzy/ambiguous matches."""
        _, pt_spans = self._exact_terms(text, [])

        def beaten(span, strong: bool) -> bool:
            for s, e in pt_spans:
                if s <= span[0] and span[1] <= e and ((e - s) > (span[1] - span[0]) or not strong):
                    return True
            return False

        drugs = [d for d in drugs if not beaten(d.span, d.match in ("exact", "alias"))]
        ambiguous = [a for a in ambiguous if not beaten(tuple(a["span"]), False)]
        spans = [d.span for d in drugs] + [tuple(a["span"]) for a in ambiguous]
        return drugs, ambiguous, spans

    def _absorb_modifier(self, text: str, key: str, span: tuple[int, int], taken: list[tuple[int, int]]) -> str:
        """'acute pancreatitis': try the exact term together with an adjacent free word (e.g. -> 'Pancreatitis acute')."""
        free = [t for t in tokens(text) if t[0].lower() not in _FILLER and not any(s <= t[1] < e for s, e in taken)]
        near = [t for t in free if abs(t[2] - span[0]) <= 1 or abs(t[1] - span[1]) <= 1]
        if not near:
            return key
        lo = min([span[0]] + [t[1] for t in near])
        hi = max([span[1]] + [t[2] for t in near])
        res = self.index.search(text[lo:hi].lower(), k=1)["results"]
        if res and res[0]["similarity"] >= 0.90 and res[0]["pt_key"] != key:
            return res[0]["pt_key"]
        return key

    def _segments(self, text: str, spans: list[tuple[int, int]],
                  exact_spans: list[tuple[int, int]] | None = None) -> list[str]:
        exact_spans = exact_spans or []
        spans = list(spans) + list(exact_spans)
        chars = list(text)
        for s, e in spans:
            for k in range(s, e):
                chars[k] = " "
        rest = "".join(chars).lower()
        rest = re.sub(r"[^a-z0-9\-' ]+", " | ", rest)
        segs, cur = [], []
        for w in rest.split():
            if w == "|" or w in _FILLER or w in {"and", "or", "vs"}:
                if cur:
                    segs.append(" ".join(cur))
                cur = []
            else:
                cur.append(w)
        if cur:
            segs.append(" ".join(cur))
        segs = [s for s in segs if len(s) >= 3]
        if exact_spans:  # "acute" left over from "acute pancreatitis" is a modifier, not a new reaction
            low = text.lower()

            def adjacent(seg: str) -> bool:
                for m in re.finditer(re.escape(seg), low):
                    if any(abs(m.end() - s) <= 1 or abs(e - m.start()) <= 1 for s, e in exact_spans):
                        return True
                return False

            segs = [s for s in segs if " " in s or not adjacent(s)]
        return segs[:4]

    def _link_reaction(self, phrase: str) -> tuple[ReactionMention | None, dict]:
        key = " ".join(phrase.split())
        if key in self.terms_by_key:
            pt, cases = self.terms_by_key[key]
            res = self.index.search(key, k=MAX_RELATED + 1)["results"]
            related = [r for r in res if r["pt_key"] != key and r["similarity"] >= EXPAND_MIN[self.mode]][:MAX_RELATED]
            return ReactionMention(phrase, pt, key, 1.0, "exact", cases, related), {}
        res = self.index.search(key, k=MAX_RELATED + 1)["results"]
        best = res[0] if res else None
        if best is None or best["similarity"] < LINK_THRESHOLD[self.mode]:
            return None, {"phrase": phrase, "best_candidate": best and best["pt"],
                          "best_similarity": best and best["similarity"], "threshold": LINK_THRESHOLD[self.mode]}
        floor = max(EXPAND_MIN[self.mode], best["similarity"] - EXPAND_WITHIN)
        related = [r for r in res[1:] if r["similarity"] >= floor][:MAX_RELATED]
        return ReactionMention(phrase, best["pt"], best["pt_key"], best["similarity"], "semantic", best["cases"],
                               related), {}

    def parse(self, question: str) -> Parsed:
        q = " ".join(question.split())[:MAX_QUESTION_LEN]
        drugs, spans, ambiguous = self.drugs.link(q)
        drugs, ambiguous, spans = self._resolve_overlaps(q, drugs, ambiguous)
        is_doc = bool(_DOC_TRIGGERS.search(q)) and bool(_METHOD_TERMS.search(q))
        reactions, unlinked = [], []
        if not is_doc or drugs:
            seen = set()
            exact_keys, exact_spans = self._exact_terms(q, spans)
            exact_keys = [self._absorb_modifier(q, k, sp, spans + exact_spans) for k, sp in zip(exact_keys, exact_spans,
                                                                                                strict=True)]
            for k in exact_keys:
                m, _ = self._link_reaction(k)
                if m and m.pt_key not in seen:
                    m.phrase = k
                    reactions.append(m)
                    seen.add(m.pt_key)
            for seg in self._segments(q, spans, exact_spans):
                m, miss = self._link_reaction(seg)
                if m and m.pt_key not in seen:
                    reactions.append(m)
                    seen.add(m.pt_key)
                elif miss:
                    unlinked.append(miss)
        if (not is_doc and not drugs and _DOC_TRIGGERS.search(q) and reactions
                and all(r.match == "semantic" and r.similarity < 0.85 for r in reactions)):
            top = self.passages.search(q, 1, DOC_SOURCES)
            if top and top[0]["score"] >= 6.0:  # a strong documentation hit beats a weak reaction link
                is_doc, reactions = True, []
        if is_doc and not drugs:
            intent = "documentation"
        elif drugs and reactions:
            intent = "pair"
        elif drugs:
            intent = "drug"
        elif reactions:
            intent = "reaction"
        elif ambiguous:
            intent = "clarify"
        else:
            intent = "documentation"  # last resort: look for a relevant passage, abstain if none
        return Parsed(q, intent, drugs, reactions[:3], unlinked, ambiguous, self.mode)

    # ------------------------------------------------------------------ retrieval helpers
    def _signal_rows(self, drug: str | None, keys: list[str], limit: int = 50, order: str = "a DESC") -> list[dict]:
        where, params = [], []
        if drug:
            where.append("drug = ?")
            params.append(drug)
        if keys:
            where.append(f"pt_key IN ({','.join('?' * len(keys))})")
            params += keys
        w = " AND ".join(where) or "TRUE"
        return self.db.query(
            f"""SELECT drug, pt, pt_key, a, b, c, d, n_drug, n_pt, expected, ror, ror_lower, ror_upper, prr, chi2,
                       ic, ic025, q_value, methods_flagging, ror_signal, prr_signal, ic_signal, haldane_applied,
                       indication_overlap, possible_indication_bias
                FROM signals WHERE {w} ORDER BY {order} LIMIT ?""", params + [limit])

    def _raw_count(self, drug: str, pt_key: str) -> int:
        return int(self.db.scalar(
            """SELECT count(DISTINCT r.primaryid) FROM reactions r
               JOIN (SELECT DISTINCT primaryid FROM drugs WHERE drug = ? AND role_cod IN ('PS','SS')) d USING (primaryid)
               WHERE r.pt_key = ?""", [drug, pt_key]) or 0)

    # ------------------------------------------------------------------ composing
    @staticmethod
    def _methods_text(r: dict) -> str:
        names = [n for n, f in (("ROR", r["ror_signal"]), ("PRR", r["prr_signal"]), ("IC", r["ic_signal"])) if f]
        if len(names) == 3:
            return "All three screening methods (ROR, PRR, IC) flag this pair as a disproportionality signal"
        if names:
            return f"Only {', '.join(names)} flag{'s' if len(names) == 1 else ''} this pair; the other method(s) do not"
        return "None of the three screening methods flags this pair as a disproportionality signal"

    def answer(self, question: str, k_passages: int = 3, generate: bool = False, external: bool = True,
               summarise_faers: bool = False) -> dict:
        """Answer a question. ``generate`` lets the local generative model write the established-evidence
        answer (verified against the retrieved passages); ``summarise_faers`` additionally asks it to paraphrase
        the FAERS statistics; ``external`` adds FDA label / FDA communication / literature evidence."""
        t0 = time.perf_counter()
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        if len(question) > MAX_QUESTION_LEN:
            raise ValueError(f"question must be at most {MAX_QUESTION_LEN} characters")
        with self._lock:
            parsed = self.parse(question)
            evidence: list[dict] = []
            sentences: list[dict] = []
            status = "answered"

            def add_ev(kind: str, payload: dict) -> str:
                prefix = {"statistic": "S", "aggregate": "S", "document": "D"}[kind]
                eid = f"{prefix}{sum(1 for e in evidence if e['id'].startswith(prefix)) + 1}"
                evidence.append({"id": eid, "kind": kind, **payload})
                return eid

            def say(text: str, *cites: str) -> None:
                sentences.append({"text": text, "cites": [c for c in cites if c]})

            if parsed.ambiguous_drugs and parsed.intent in ("reaction", "drug", "pair"):
                for amb in parsed.ambiguous_drugs:
                    names = ", ".join(c["drug"] for c in amb["candidates"])
                    say(f"“{amb['text']}” matches several FAERS drug names ({names}); it was not used. "
                        "Name a specific product for a drug-level answer.")
            if parsed.intent == "pair":
                self._compose_pair(parsed, add_ev, say)
            elif parsed.intent == "drug":
                self._compose_drug(parsed, add_ev, say)
            elif parsed.intent == "reaction":
                self._compose_reaction(parsed, add_ev, say)
            elif parsed.intent == "clarify":
                status = "needs_clarification"
                amb = parsed.ambiguous_drugs[0]
                names = ", ".join(c["drug"] for c in amb["candidates"])
                say(f"“{amb['text']}” matches several FAERS drug names ({names}). Please name the specific product.")
            if parsed.intent == "documentation":
                hits = self.passages.search(question, k_passages, DOC_SOURCES)
                if not hits or hits[0]["score"] < 3.0:
                    status = "abstained"
                    say("I could not find a drug, a reaction term or a relevant documentation passage for this question "
                        "in FAERS 2025 Q1, so I am not answering it. Try naming a drug and/or a reaction, e.g. "
                        "“Is semaglutide associated with pancreatitis?”")
                else:
                    hits = [h for h in hits if h["score"] >= 0.5 * hits[0]["score"]]
                    quoted: set[str] = set()
                    for h in hits:
                        eid = add_ev("document", h)
                        src_text = h.pop("answer", "") or h["text"]  # FAQ: quote the FDA answer, not the question
                        for s in self._best_sentences(question, src_text, 2 if h is hits[0] else 1):
                            norm = re.sub(r"\W+", " ", s.lower()).replace("therefore ", "").strip()
                            if norm in quoted:
                                continue
                            quoted.add(norm)
                            say(f"“{s}”", eid)
            elif status == "answered":
                # supporting documentation for statistical answers: caveats + definition of the linked reaction
                for r in parsed.reactions[:1]:
                    d = self.passages.ctcae_definition(r.pt_key)
                    if d:
                        say(f"Definition (NCI CTCAE v5.0): “{d['text']}”", add_ev("document", d))
            causal_q = re.search(r"caus|risk|signal|\bror\b|\bprr\b|disproportion|prove", question, re.I)
            if status == "answered" and (parsed.intent != "documentation" or causal_q):
                cites = [add_ev("document", c) for c in (self.caveat_project, self.caveat_fda) if c]
                say("This is a reporting association in a spontaneous-reporting database. It does not show that the drug "
                    "causes the reaction and is not an incidence or absolute risk; reports are unverified and "
                    "subject to reporting biases.", *cites)

        if status == "answered" and not sentences:
            status = "abstained"
        if summarise_faers and status == "answered":
            from .generator import summarise

            generated = summarise(parsed.question, evidence, " ".join(x["text"] for x in sentences))
        else:
            from .generator import status as gen_status

            st = gen_status()
            generated = {"status": "not_requested" if st["available"] else "unavailable",
                         "reason": None if st["available"] else "no local generator model installed"}
        return {
            "question": parsed.question,
            "status": status,
            "intent": parsed.intent,
            "answer": sentences,
            "understanding": {
                "drugs": [d.__dict__ for d in parsed.drugs],
                "reactions": [r.__dict__ for r in parsed.reactions],
                "unlinked_phrases": parsed.unlinked_phrases,
                "ambiguous_drugs": parsed.ambiguous_drugs,
                "link_mode": parsed.link_mode,
            },
            "evidence": evidence,
            "generated_summary": generated,
            "established": (self._established(parsed, question, generate) if external else
                            {"status": "not_requested"}),
            "provenance": self.provenance,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "disclaimer": ("Answers are assembled automatically from retrieved FAERS statistics and documents. They are "
                           "provisional, not medical advice, and do not establish causality."),
        }

    # ------------------------------------------------------------------ established clinical evidence
    def _established(self, parsed: Parsed, question: str, generate: bool) -> dict:
        """External evidence (FDA labels, FDA Drug Safety Communications, PubMed reviews) for the question.

        Kept separate from the FAERS statistics: FAERS shows what was *reported*; labels and literature state what
        is *established*. Never raises - a missing index or model degrades to an explanatory status."""
        if parsed.intent not in ("pair", "drug", "reaction"):
            return {"status": "not_applicable", "reason": "no drug or reaction in the question"}
        try:
            from ..evidence.evaluate import sufficiency_threshold
            from ..evidence.index import get_evidence_index

            idx = get_evidence_index()
        except FileNotFoundError as exc:
            return {"status": "unavailable", "reason": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive: never break the FAERS answer
            log.exception("evidence index failed")
            return {"status": "unavailable", "reason": f"evidence index error: {exc}"}
        items: list[dict] = []
        sentences: list[dict] = []
        prefix = {"label": "L", "literature": "P", "regulatory": "R"}

        def add(hit: dict) -> str:
            for it in items:
                if it["chunk_id"] == hit["chunk_id"]:
                    return it["id"]
            p = prefix.get(hit["source_type"], "X")
            eid = f"{p}{sum(1 for i in items if i['id'].startswith(p)) + 1}"
            keep = ("chunk_id", "doc_id", "source_type", "drugs", "section_key", "section", "title", "publisher", "date",
                    "version", "url", "identifier", "text", "score", "dense_similarity", "bm25")
            items.append({"id": eid, "kind": "external", **{k: hit.get(k) for k in keep}})
            return eid

        drugs = [d.drug for d in parsed.drugs]
        not_indexed = [d for d in drugs if d not in idx.indexed_drugs]
        searchable = [d for d in drugs if d in idx.indexed_drugs]
        tau = sufficiency_threshold(self.paths)
        out = {"retrieval_mode": idx.mode, "threshold": tau, "drugs_searched": searchable, "not_indexed": not_indexed,
               "index": {k: idx.manifest.get(k) for k in ("index_version", "created_at", "documents", "chunks",
                                                          "content_sha256")},
               "label_mentions": []}
        if drugs and not searchable:
            names = ", ".join(not_indexed)
            sentences.append({"text": f"No FDA label, safety communication or review has been indexed for {names}, so "
                                      "no established-evidence answer can be given. (Add it with "
                                      "`python -m pvsys.pipeline evidence-fetch --drug \"<name>\"`.)", "cites": []})
            return {**out, "status": "insufficient", "sentences": sentences, "evidence": items,
                    "generated": {"status": "skipped", "reason": "drug not indexed"}}
        # 1. deterministic label check: does the label text mention the reaction (or a closely related term)?
        for dm in [d for d in parsed.drugs if d.drug in searchable]:
            for rm in parsed.reactions:
                terms = [rm.pt.lower()] + [r["pt"].lower() for r in rm.related]
                found = idx.label_mentions(dm.drug, terms)
                rec = {"drug": dm.drug, "pt": rm.pt, "found": bool(found),
                       "sections": sorted({f["section"] for f in found})}
                has_label = bool(idx.mask([dm.drug], ("label",)).any())
                rec["label_indexed"] = has_label
                if found:
                    f0 = found[0]
                    eid = add({**f0, "score": None, "dense_similarity": None, "bm25": None})
                    term = f0["matched_terms"][0]
                    sent = next((s for s in self._best_sentences(term, f0["text"], 3) if term in s.lower()), None)
                    secs = ", ".join(rec["sections"])
                    txt = f"The FDA label for {dm.drug} mentions \u201c{term}\u201d (sections: {secs})."
                    sentences.append({"text": txt, "cites": [eid]})
                    if sent:
                        sentences.append({"text": f"Label text: \u201c{sent}\u201d", "cites": [eid]})
                    rec["cite"] = eid
                elif has_label:
                    sentences.append({"text": f"\u201c{rm.pt}\u201d (or the related terms found by the language model) "
                                              f"was not found in the indexed sections of the {dm.drug} label. Absence "
                                              "from these sections is not proof that the reaction is unlisted.",
                                      "cites": []})
                out["label_mentions"].append(rec)
        # 2. retrieval over labels, FDA communications and literature
        q = question
        if parsed.reactions and parsed.drugs:
            q = f"{question} {parsed.drugs[0].drug.lower()} {parsed.reactions[0].pt.lower()}"
        res = idx.search(q, k=5, drugs=searchable or None)
        hits = res["hits"]
        top = max((h["dense_similarity"] for h in hits if h["dense_similarity"] is not None), default=None)
        out["top_similarity"] = top
        out["retrieval_mode"] = res["mode"]
        sufficient = bool(hits) and (top is None or tau is None or top >= tau)
        if not sufficient:
            sentences.append({"text": "The retrieved label, FDA-communication and literature passages are not close enough "
                                      f"to the question (best similarity {fmt(top)} < threshold {fmt(tau)}), so no "
                                      "further established-evidence statement is made.", "cites": []})
            return {**out, "status": "insufficient" if not out["label_mentions"] or
                    not any(m["found"] for m in out["label_mentions"]) else "partial",
                    "sentences": sentences, "evidence": items, "generated": {"status": "skipped",
                                                                            "reason": "insufficient evidence"}}
        hit_ids = [add(h) for h in hits]
        generated = {"status": "not_requested"}
        if generate:
            from .generator import summarise_external

            generated = summarise_external(parsed.question, [i for i in items if i["id"] in hit_ids][:3])
        if generated.get("status") == "verified":
            sentences.append({"text": generated["text"], "cites": [], "generated": True})
        else:  # extractive fallback: the best-matching sentence of the top passages, quoted verbatim
            seen: set[str] = set()
            for h, eid in list(zip(hits, hit_ids, strict=True))[:3]:
                body = h["text"]
                if h["source_type"] == "literature" and body.startswith(h["title"]):
                    body = body[len(h["title"]):].strip()  # abstracts are indexed with their title in front
                cands = [x for x in self._best_sentences(question, body, 3) if len(x) >= 50] or \
                    self._best_sentences(question, body, 1)
                for s in cands[:1]:
                    if s in seen:
                        continue
                    seen.add(s)
                    src = {"label": f"{h['title'].split(' - ')[0]} label, {h['section']}",
                           "literature": f"{h['publisher']} ({h['date']})",
                           "regulatory": f"FDA Drug Safety Communication ({h['date'][:16]})"}[h["source_type"]]
                    sentences.append({"text": f"{src}: \u201c{s}\u201d", "cites": [eid]})
        status = "answered" if generated.get("status") != "insufficient" else "insufficient"
        return {**out, "status": status, "sentences": sentences, "evidence": items, "generated": generated}

    def _stat_payload(self, r: dict) -> dict:
        keep = ("drug", "pt", "pt_key", "a", "b", "c", "d", "n_drug", "n_pt", "expected", "ror", "ror_lower", "ror_upper",
                "prr", "chi2", "ic", "ic025", "q_value", "methods_flagging", "ror_signal", "prr_signal", "ic_signal",
                "haldane_applied", "indication_overlap", "possible_indication_bias")
        p = {k: r.get(k) for k in keep}
        if r.get("indication_overlap") is not None:
            p["indication_overlap_pct"] = int(round(100 * float(r["indication_overlap"])))
        if r.get("haldane_applied"):
            p["haldane_continuity"] = self.cfg.continuity
        p["tested"] = True
        p["source"] = "data/processed/signals.parquet"
        return p

    def _compose_pair(self, parsed: Parsed, add_ev, say) -> None:
        for dm in parsed.drugs:
            for rm in parsed.reactions:
                keys = [rm.pt_key] + [x["pt_key"] for x in rm.related]
                rows = {r["pt_key"]: r for r in self._signal_rows(dm.drug, keys, limit=len(keys))}
                main = rows.get(rm.pt_key)
                label = f"{dm.drug} and “{rm.pt}”"
                if main:
                    eid = add_ev("statistic", self._stat_payload(main))
                    say(f"In FAERS 2025 Q1, {label} were reported together in {main['a']:,} cases "
                        f"(about {fmt(main['expected'])} expected if they were unrelated).", eid)
                    say(f"Reporting odds ratio {fmt(main['ror'])} (95% CI {fmt(main['ror_lower'])}–"
                        f"{fmt(main['ror_upper'])}), PRR {fmt(main['prr'])}, IC025 {fmt(main['ic025'])}.", eid)
                    say(f"{self._methods_text(main)}.", eid)
                    if main["possible_indication_bias"]:
                        say(f"Caution: {_pct(main['indication_overlap'])} of these cases also list the reaction as an "
                            "indication, so the association may reflect why the drug was given (indication bias).", eid)
                    if main["haldane_applied"]:
                        say(f"A zero cell required the Haldane +{self.cfg.continuity} correction; the estimate is imprecise.", eid)
                else:
                    a = self._raw_count(dm.drug, rm.pt_key)
                    eid = add_ev("statistic", {"drug": dm.drug, "pt": rm.pt, "pt_key": rm.pt_key, "a": a,
                                               "tested": False, "min_cases": self.cfg.min_cases,
                                               "source": "data/processed/{drugs,reactions}.parquet"})
                    if a == 0:
                        say(f"In FAERS 2025 Q1 no case reports {label} together, so there is no disproportionality "
                            "estimate. Absence of reports is not evidence of safety.", eid)
                    else:
                        say(f"In FAERS 2025 Q1, {label} were reported together in only {a} case{'s' if a != 1 else ''}, "
                            f"below the minimum of {self.cfg.min_cases} needed for disproportionality testing.", eid)
                rel = [rows[x["pt_key"]] for x in rm.related if x["pt_key"] in rows]
                if rel:
                    flagged = [r for r in rel if r["methods_flagging"] == 3]
                    ids = [add_ev("statistic", self._stat_payload(r)) for r in rel]
                    names = ", ".join(f"{r['pt']} (ROR {fmt(r['ror'])}, {r['a']} cases)" for r in rel[:4])
                    say(f"Closely related terms found by the language model: {names}; {len(flagged)} of "
                        f"{len(rel)} tested related terms are flagged by all three methods.", *ids)

    def _compose_drug(self, parsed: Parsed, add_ev, say) -> None:
        for dm in parsed.drugs:
            rows = self.db.query(
                """SELECT * FROM signals WHERE drug = ? AND methods_flagging = 3 AND NOT possible_indication_bias
                   ORDER BY a DESC, ror_lower DESC LIMIT 8""", [dm.drug])
            n_flag = self.db.scalar("SELECT count(*) FROM signals WHERE drug = ? AND methods_flagging = 3", [dm.drug]) or 0
            n_test = self.db.scalar("SELECT count(*) FROM signals WHERE drug = ?", [dm.drug]) or 0
            if not rows:
                agg = add_ev("aggregate", {"drug": dm.drug, "suspect_cases": dm.cases, "pairs_tested": int(n_test),
                                           "pairs_flagged_all_three": int(n_flag), "source": "data/processed/signals.parquet"})
                say(f"{dm.drug} appears as a suspect drug in {dm.cases:,} cases, but none of its tested drug–reaction "
                    f"pairs ({n_test}) is flagged by all three methods without an indication-bias flag.", agg)
                continue
            agg = add_ev("aggregate", {"drug": dm.drug, "suspect_cases": dm.cases, "pairs_tested": int(n_test),
                                       "pairs_flagged_all_three": int(n_flag), "source": "data/processed/signals.parquet"})
            ids = [add_ev("statistic", self._stat_payload(r)) for r in rows]
            say(f"{dm.drug} is a suspect drug in {dm.cases:,} FAERS 2025 Q1 cases; {n_flag} of its {n_test} tested "
                "reaction terms are flagged by all three methods.", agg)
            listing = "; ".join(f"{r['pt']} ({r['a']} cases, ROR {fmt(r['ror'])} [{fmt(r['ror_lower'])}–"
                                f"{fmt(r['ror_upper'])}])" for r in rows)
            say(f"Most frequently reported flagged reactions (excluding possible indication bias): {listing}.", *ids)

    def _compose_reaction(self, parsed: Parsed, add_ev, say) -> None:
        for rm in parsed.reactions:
            keys = [rm.pt_key]
            rows = self._signal_rows(None, keys, limit=50, order="a DESC")
            rows = [r for r in rows if r["methods_flagging"] == 3][:8]
            agg = add_ev("aggregate", {"pt": rm.pt, "pt_key": rm.pt_key, "reaction_cases": rm.cases,
                                       "link": rm.match, "similarity": round(rm.similarity, 2),
                                       "source": "data/nlp/predictions.parquet (term case counts)"})
            if rm.match == "semantic":
                say(f"I read “{rm.phrase}” as the MedDRA term “{rm.pt}” (similarity "
                    f"{rm.similarity:.2f}; provisional language-model link).", agg)
            if not rows:
                say(f"No suspect drug is flagged by all three methods for “{rm.pt}” ({rm.cases:,} cases).", agg)
                continue
            ids = [agg] + [add_ev("statistic", self._stat_payload(r)) for r in rows]
            listing = "; ".join(f"{r['drug']} ({r['a']} cases, ROR {fmt(r['ror'])})" for r in rows)
            say(f"“{rm.pt}” appears in {rm.cases:,} cases. Drugs most often co-reported with it and flagged "
                f"by all three methods: {listing}.", *ids)
            if any(r["possible_indication_bias"] for r in rows):
                say("Some of these pairs carry the possible-indication-bias flag (see evidence).",
                    *[i for i, r in zip(ids[1:], rows, strict=True) if r["possible_indication_bias"]])

    @staticmethod
    def _best_sentences(question: str, text: str, n: int) -> list[str]:
        protected = re.sub(r"\b(et al|e\.g|i\.e|etc|vs|approx|no)\.", lambda m: m.group(0)[:-1] + "\u2024", text)
        sents = [s.strip().replace("\u2024", ".") for s in re.split(r"(?<=[.?!])\s+(?=[A-Z(])", protected)
                 if len(s.strip()) > 20]
        if not sents:
            return [text[:400]]
        qt = set(tokenize(question))
        scored = [(len(qt & set(tokenize(s))) / (1 + 0.01 * len(s)), i) for i, s in enumerate(sents)]
        best = sorted(sorted(scored, reverse=True)[:n], key=lambda x: x[1])
        return [sents[i][:500] for _, i in best]


_engine: RAGEngine | None = None
_engine_lock = threading.Lock()


def get_engine(db=None) -> RAGEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = RAGEngine(db=db)
        return _engine


def reset_engine() -> None:
    global _engine
    with _engine_lock:
        _engine = None
