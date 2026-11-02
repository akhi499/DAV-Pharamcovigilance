"""Download external evidence with source metadata, caching every raw response.

Layout (``data/evidence/``)::

    raw/labels/<DRUG>.json        openFDA label record + DailyMed metadata (or a 'not_found' marker)
    raw/dsc/<slug>.json           one FDA Drug Safety Communication (text + dates + URL)
    raw/pubmed/<DRUG>.json        PubMed records (pmid, title, abstract, journal, year, doi, types)
    documents.parquet             one row per document with full metadata (built from raw/)
    sources_manifest.json         what was fetched, when, from where, counts and failures

Fetching is idempotent: cached files are reused unless ``refresh=True``. Network errors are
recorded per item and never abort the whole run. Requests are rate-limited (openFDA 240/min,
NCBI 3/s without an API key).
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from ..config import Paths, get_paths

log = logging.getLogger(__name__)

USER_AGENT = "pvsys-faers-workbench/1.0 (academic project; contact: local user)"
OPENFDA = "https://api.fda.gov/drug/label.json"
DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
DAILYMED_PAGE = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"
FDA = "https://www.fda.gov"
DSC_INDEX = FDA + "/drugs/drug-safety-and-availability/drug-safety-communications"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_PAGE = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

# Label sections used as evidence (openFDA field -> display name). Order = clinical priority.
LABEL_SECTIONS = {
    "boxed_warning": "Boxed Warning",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "pregnancy": "Pregnancy",
    "overdosage": "Overdosage",
    "indications_and_usage": "Indications and Usage",
}
REPACKAGER_HINTS = re.compile(r"A-S MEDICATION|BRYANT RANCH|PROFICIENT RX|NUCARE|DIRECT[_ ]RX|QUALITY CARE|"
                              r"PREFERRED PHARMACEUTICALS|REMEDYREPACK|HENRY SCHEIN|MEDSOURCE|PD-RX|RPK|"
                              r"NORTHWIND|ASCLEMED|LAKE ERIE|STAT RX|PHYSICIANS TOTAL CARE", re.I)


class Fetcher:
    """Small HTTP client with per-host rate limiting and retries (httpx)."""

    def __init__(self, min_interval: dict[str, float] | None = None, timeout: float = 30.0, retries: int = 3):
        import httpx

        self.client = httpx.Client(timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        self.min_interval = min_interval or {"api.fda.gov": 0.3, "eutils.ncbi.nlm.nih.gov": 0.36,
                                             "dailymed.nlm.nih.gov": 0.2, "www.fda.gov": 0.5}
        self.last: dict[str, float] = {}
        self.retries = retries
        self.n_requests = 0

    def get(self, url: str) -> tuple[int, str]:
        host = url.split("/")[2]
        for attempt in range(self.retries):
            wait = self.min_interval.get(host, 0.2) - (time.monotonic() - self.last.get(host, 0))
            if wait > 0:
                time.sleep(wait)
            self.last[host] = time.monotonic()
            self.n_requests += 1
            try:
                r = self.client.get(url)
            except Exception as exc:  # network error: retry with back-off
                if attempt == self.retries - 1:
                    raise
                log.warning("GET %s failed (%s); retrying", url, exc)
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            return r.status_code, r.text
        return 0, ""

    def json(self, url: str) -> tuple[int, dict | None]:
        code, text = self.get(url)
        if code != 200:
            return code, None
        try:
            return code, json.loads(text)
        except json.JSONDecodeError:
            return code, None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")[:120] or "item"


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def evidence_dir(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).root / "evidence"


# --------------------------------------------------------------------------- drug list

def default_drugs(paths: Paths, n: int = 150) -> list[str]:
    """The n most-reported suspect drugs in the processed FAERS data (deterministic order)."""
    import duckdb

    f = str(paths.processed / "drugs.parquet").replace("\\", "/")
    rows = duckdb.sql(
        f"""SELECT drug, count(DISTINCT primaryid) AS n FROM read_parquet('{f}')
            WHERE role_cod IN ('PS','SS') AND drug IS NOT NULL AND drug NOT IN ('UNKNOWN','PLACEBO')
            GROUP BY 1 ORDER BY n DESC, drug LIMIT {int(n)}""").fetchall()
    return [r[0] for r in rows]


# --------------------------------------------------------------------------- labels

def _dailymed_candidates(f: Fetcher, drug: str) -> list[dict]:
    code, d = f.json(f"{DAILYMED}?drug_name={quote(drug.lower())}&pagesize=50")
    return (d or {}).get("data", []) if code == 200 else []


def choose_setid(cands: list[dict], label_rows: list[dict]) -> str | None:
    """Prefer the manufacturer's (non-repackager) label set with the most revisions."""
    have = {r.get("set_id") for r in label_rows}
    usable = [c for c in cands if c.get("setid") in have] or [c for c in cands]
    if not usable:
        return None
    usable.sort(key=lambda c: (bool(REPACKAGER_HINTS.search(c.get("title", ""))), -int(c.get("spl_version") or 0)))
    return usable[0].get("setid")


def fetch_label(f: Fetcher, drug: str) -> dict:
    """openFDA label for a normalised FAERS drug name (generic name, then substance name)."""
    rows: list[dict] = []
    for field in ("openfda.generic_name", "openfda.substance_name"):
        q = f'{field}:"{drug}"'
        code, d = f.json(f"{OPENFDA}?search={quote(q)}&limit=100")
        if code == 200 and d and d.get("results"):
            rows = d["results"]
            break
    if not rows:
        return {"drug": drug, "status": "not_found", "fetched_at": _now()}
    cands = _dailymed_candidates(f, drug)
    setid = choose_setid(cands, rows)
    best = next((r for r in rows if r.get("set_id") == setid), None)
    if best is None:  # DailyMed set not among openFDA hits: fall back to most-revised non-repackager openFDA row
        rows.sort(key=lambda r: (bool(REPACKAGER_HINTS.search(" ".join(r.get("openfda", {}).get("manufacturer_name", [])))),
                                 -int(r.get("version") or 0), r.get("effective_time", "")), reverse=False)
        best = rows[0]
    dm = next((c for c in cands if c.get("setid") == best.get("set_id")), None)
    of = best.get("openfda", {})
    return {
        "drug": drug, "status": "ok", "fetched_at": _now(),
        "set_id": best.get("set_id"), "spl_id": best.get("id"), "spl_version": best.get("version"),
        "effective_time": best.get("effective_time"),
        "brand_names": of.get("brand_name", []), "generic_names": of.get("generic_name", []),
        "manufacturer": (of.get("manufacturer_name") or [""])[0], "route": of.get("route", []),
        "dailymed_title": dm.get("title") if dm else None, "dailymed_published": dm.get("published_date") if dm else None,
        "dailymed_spl_version": dm.get("spl_version") if dm else None,
        "url": DAILYMED_PAGE.format(setid=best.get("set_id")),
        "api_url": f"{OPENFDA}?search=set_id:%22{best.get('set_id')}%22",
        "sections": {k: "\n".join(best[k]) for k in LABEL_SECTIONS if best.get(k)},
    }


# --------------------------------------------------------------------------- FDA Drug Safety Communications

_NAV_RE = re.compile(r"<(nav|aside|script|style|noscript|header|footer)\b.*?</\1>", re.S | re.I)


def html_to_text(fragment: str) -> str:
    t = _NAV_RE.sub(" ", fragment)
    t = re.sub(r"<(br|/p|/li|/h\d|/tr|/div)[^>]*>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    return re.sub(r"\n\s*\n+", "\n", t).strip()


def parse_dsc_page(page: str, url: str) -> dict:
    m = re.search(r'<article[^>]*id="main-content".*?</article>', page, re.S)
    body = m.group(0) if m else page
    title = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    text = html_to_text(body)
    # drop everything before the title (breadcrumbs / side menu that live inside the article element)
    t = html_to_text(title.group(1)) if title else ""
    if t and t in text:
        text = text[text.index(t):]
    text = text.split("Content current as of")[0].strip()
    pub = re.search(r'article:published_time" content="([^"]+)"', page)
    mod = re.search(r'article:modified_time" content="([^"]+)"', page)
    return {"url": url, "title": t or url.rsplit("/", 1)[-1], "text": text,
            "published": pub.group(1) if pub else None, "modified": mod.group(1) if mod else None}


def dsc_links(index_html: str) -> list[str]:
    links = set(re.findall(r'href="(/drugs/drug-safety-communications/[^"#?]+)"', index_html))
    links |= {h for h in re.findall(r'href="(/drugs/drug-safety-and-availability/[^"#?]+)"', index_html)
              if re.search(r"fda-|risk|warning|safety-communication", h) and "podcast" not in h
              and not h.endswith("drug-safety-communications") and "espanol" not in h}
    return sorted(links)


# --------------------------------------------------------------------------- PubMed

def pubmed_query(drug: str) -> str:
    return (f'"{drug.lower()}"[tiab] AND ("adverse effects"[sh] OR "drug-related side effects and adverse reactions"[mh]) '
            "AND (review[pt] OR systematic review[pt] OR meta-analysis[pt]) AND english[la] AND hasabstract "
            "AND 2015:3000[dp]")


def parse_pubmed_xml(xml: str) -> list[dict]:
    import xml.etree.ElementTree as ET

    out = []
    # NCBI XML carries a DOCTYPE; drop it so no entity declarations are processed, then parse
    # (expat in Python >= 3.7.1 also guards against entity-expansion attacks).
    xml = re.sub(r"<!DOCTYPE[^>]*(\[.*?\])?\s*>", "", xml, count=1, flags=re.S)
    root = ET.fromstring(xml)  # noqa: S314 - DOCTYPE removed above; source is NCBI E-utilities
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID")
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        parts = []
        for ab in art.findall(".//Abstract/AbstractText"):
            label = ab.get("Label")
            txt = "".join(ab.itertext()).strip()
            parts.append(f"{label}: {txt}" if label else txt)
        year = art.findtext(".//JournalIssue/PubDate/Year") or (art.findtext(".//JournalIssue/PubDate/MedlineDate") or "")[:4]
        doi = next((e.text for e in art.findall(".//ArticleId") if e.get("IdType") == "doi"), None)
        types = [e.text for e in art.findall(".//PublicationType") if e.text]
        out.append({"pmid": pmid, "title": title.strip(), "abstract": "\n".join(parts), "journal": art.findtext(".//Journal/Title"),
                    "year": year, "doi": doi, "publication_types": types, "url": PUBMED_PAGE.format(pmid=pmid)})
    return out


def fetch_pubmed(f: Fetcher, drug: str, k: int) -> dict:
    q = pubmed_query(drug)
    code, d = f.json(f"{EUTILS}/esearch.fcgi?db=pubmed&retmode=json&sort=relevance&retmax={k}&term={quote(q)}")
    ids = (d or {}).get("esearchresult", {}).get("idlist", []) if code == 200 else []
    if code != 200:
        return {"drug": drug, "status": f"error_{code}", "query": q, "fetched_at": _now()}
    if not ids:
        return {"drug": drug, "status": "no_results", "query": q, "fetched_at": _now(), "records": []}
    code, xml = f.get(f"{EUTILS}/efetch.fcgi?db=pubmed&retmode=xml&id={','.join(ids)}")
    if code != 200:
        return {"drug": drug, "status": f"error_{code}", "query": q, "fetched_at": _now()}
    return {"drug": drug, "status": "ok", "query": q, "fetched_at": _now(), "records": parse_pubmed_xml(xml)}


# --------------------------------------------------------------------------- orchestration

def fetch_all(paths: Paths | None = None, drugs: list[str] | None = None, n_drugs: int = 150, pubmed_k: int = 4,
              max_dsc: int = 80, refresh: bool = False, fetcher: Fetcher | None = None) -> dict:
    paths = (paths or get_paths()).ensure()
    root = evidence_dir(paths)
    raw = root / "raw"
    drugs = drugs or default_drugs(paths, n_drugs)
    f = fetcher or Fetcher()
    stats = {"labels": {"ok": 0, "not_found": 0, "error": 0, "cached": 0},
             "pubmed": {"ok": 0, "no_results": 0, "error": 0, "cached": 0}, "dsc": {"ok": 0, "error": 0, "cached": 0}}
    failures: list[dict] = []
    started = _now()
    for i, drug in enumerate(drugs, 1):
        lf = raw / "labels" / f"{_safe(drug)}.json"
        if lf.exists() and not refresh:
            stats["labels"]["cached"] += 1
        else:
            try:
                rec = fetch_label(f, drug)
                stats["labels"]["ok" if rec["status"] == "ok" else "not_found"] += 1
                _write(lf, rec)
            except Exception as exc:
                stats["labels"]["error"] += 1
                failures.append({"source": "label", "drug": drug, "error": str(exc)[:200]})
        pf = raw / "pubmed" / f"{_safe(drug)}.json"
        if pf.exists() and not refresh:
            stats["pubmed"]["cached"] += 1
        else:
            try:
                rec = fetch_pubmed(f, drug, pubmed_k)
                key = "ok" if rec["status"] == "ok" else "no_results" if rec["status"] == "no_results" else "error"
                stats["pubmed"][key] += 1
                if key != "error":
                    _write(pf, rec)
                else:
                    failures.append({"source": "pubmed", "drug": drug, "error": rec["status"]})
            except Exception as exc:
                stats["pubmed"]["error"] += 1
                failures.append({"source": "pubmed", "drug": drug, "error": str(exc)[:200]})
        if i % 10 == 0:
            log.info("evidence fetch: %d/%d drugs", i, len(drugs))
    try:
        code, idx = f.get(DSC_INDEX)
        links = dsc_links(idx)[:max_dsc] if code == 200 else []
        if code != 200:
            failures.append({"source": "dsc", "item": "index", "error": f"HTTP {code}"})
    except Exception as exc:
        links = []
        failures.append({"source": "dsc", "item": "index", "error": str(exc)[:200]})
    for link in links:
        df = raw / "dsc" / f"{_safe(link.rsplit('/', 1)[-1])}.json"
        if df.exists() and not refresh:
            stats["dsc"]["cached"] += 1
            continue
        try:
            code, page = f.get(FDA + link)
            if code != 200:
                raise RuntimeError(f"HTTP {code}")
            rec = parse_dsc_page(page, FDA + link)
            rec["fetched_at"] = _now()
            if len(rec["text"]) > 200:
                _write(df, rec)
                stats["dsc"]["ok"] += 1
        except Exception as exc:
            stats["dsc"]["error"] += 1
            failures.append({"source": "dsc", "item": link, "error": str(exc)[:200]})
    manifest = {"started_at": started, "finished_at": _now(), "drugs_requested": len(drugs), "drug_list": drugs,
                "requests": f.n_requests, "stats": stats, "failures": failures,
                "sources": {"labels": "openFDA /drug/label (FDA SPL) + DailyMed set selection",
                            "dsc": DSC_INDEX, "pubmed": "NCBI E-utilities esearch/efetch",
                            "pubmed_query_template": pubmed_query("<drug>")}}
    _write(root / "sources_manifest.json", manifest)
    build_documents(paths)
    return manifest


def build_documents(paths: Paths | None = None) -> pd.DataFrame:
    """Flatten cached raw files into documents.parquet (one row per document) - fully offline."""
    paths = paths or get_paths()
    root = evidence_dir(paths)
    raw = root / "raw"
    rows: list[dict] = []
    for p in sorted((raw / "labels").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("status") != "ok":
            continue
        brand = ", ".join(d.get("brand_names", [])[:3])
        rows.append({"doc_id": f"label:{d['set_id']}", "source_type": "label", "drug": d["drug"],
                     "title": f"{brand or d['drug']} ({d['drug'].title()}) - FDA prescribing information",
                     "publisher": d.get("manufacturer") or "", "date": d.get("effective_time") or "",
                     "version": str(d.get("spl_version") or ""), "url": d["url"], "identifier": d["set_id"],
                     "aliases": "|".join(sorted({*(x.upper() for x in d.get("brand_names", [])),
                                                 *(x.upper() for x in d.get("generic_names", []))})),
                     "sections_json": json.dumps(d["sections"], ensure_ascii=False), "text": "",
                     "fetched_at": d.get("fetched_at")})
    for p in sorted((raw / "pubmed").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d.get("records", []):
            if not r.get("abstract"):
                continue
            rows.append({"doc_id": f"pubmed:{r['pmid']}", "source_type": "literature", "drug": d["drug"],
                         "title": r["title"], "publisher": r.get("journal") or "", "date": r.get("year") or "",
                         "version": ", ".join(t for t in r.get("publication_types", []) if t in
                                              ("Review", "Systematic Review", "Meta-Analysis")),
                         "url": r["url"], "identifier": f"PMID {r['pmid']}" + (f"; doi:{r['doi']}" if r.get("doi") else ""),
                         "aliases": "", "sections_json": "", "text": r["abstract"], "fetched_at": d.get("fetched_at")})
    for p in sorted((raw / "dsc").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"doc_id": f"dsc:{p.stem}", "source_type": "regulatory", "drug": "", "title": d["title"],
                     "publisher": "U.S. FDA Drug Safety Communication", "date": d.get("modified") or d.get("published") or "",
                     "version": "", "url": d["url"], "identifier": d["url"].rsplit("/", 1)[-1], "aliases": "",
                     "sections_json": "", "text": d["text"], "fetched_at": d.get("fetched_at")})
    df = pd.DataFrame(rows, columns=["doc_id", "source_type", "drug", "title", "publisher", "date", "version", "url",
                                     "identifier", "aliases", "sections_json", "text", "fetched_at"])
    df = df.drop_duplicates("doc_id").reset_index(drop=True)
    root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "documents.parquet", index=False)
    return df
