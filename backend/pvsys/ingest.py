"""Stage 1 - Ingestion of FAERS quarterly ASCII extracts.

The FDA publishes each quarter as a zip containing '$'-delimited text tables
(DEMO, DRUG, REAC, OUTC, RPSR, THER, INDI) plus a list of deleted case IDs.
This module extracts the zip, loads every table as text (no type coercion, so
nothing is silently lost) and writes one Parquet file per table together with
a manifest recording checksums and row counts for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .config import FAERS_TABLES, FAERS_URL_TEMPLATE, QUARTER, Paths, get_paths

log = logging.getLogger(__name__)


def quarter_code(quarter: str = QUARTER) -> str:
    """'2025Q1' -> '25Q1' (the suffix FDA uses in file names)."""
    m = re.fullmatch(r"(\d{4})[Qq]([1-4])", quarter)
    if not m:
        raise ValueError(f"Quarter must look like 2025Q1, got {quarter!r}")
    return f"{m.group(1)[2:]}Q{m.group(2)}"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def zip_path(paths: Paths, quarter: str = QUARTER) -> Path:
    return paths.raw / f"faers_ascii_{quarter.lower()}.zip"


def download(paths: Paths | None = None, quarter: str = QUARTER, force: bool = False) -> Path:
    """Download the quarterly zip from FDA if it is not already present."""
    paths = (paths or get_paths()).ensure()
    target = zip_path(paths, quarter)
    if target.exists() and not force:
        log.info("Using existing %s", target)
        return target
    url = FAERS_URL_TEMPLATE.format(q=quarter.lower())
    log.info("Downloading %s", url)
    tmp = target.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed FDA https URL
    tmp.replace(target)
    return target


def extract(zpath: Path, dest: Path) -> Path:
    """Extract the zip safely (rejecting path traversal) and return dest."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        bad = zf.testzip()
        if bad:
            raise ValueError(f"Corrupt member in {zpath}: {bad}")
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(dest)
    return dest


def find_table_file(root: Path, table: str, qcode: str) -> Path:
    candidates = [p for p in root.rglob("*") if p.is_file() and p.name.upper() == f"{table}{qcode}.TXT"]
    if not candidates:
        raise FileNotFoundError(f"{table}{qcode}.txt not found under {root}")
    return candidates[0]


def find_delete_file(root: Path, qcode: str) -> Path | None:
    for p in root.rglob("*"):
        if p.is_file() and p.name.upper() == f"DELETE{qcode}.TXT":
            return p
    return None


def load_table_to_parquet(con: duckdb.DuckDBPyConnection, src: Path, dest: Path) -> dict:
    """Load one '$'-delimited FAERS table as VARCHAR columns into Parquet."""
    src_s = str(src).replace("'", "''")
    dest_s = str(dest).replace("'", "''")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _t AS
        SELECT * FROM read_csv('{src_s}', delim='$', header=true, all_varchar=true,
                               quote='', escape='', null_padding=true,
                               strict_mode=false, encoding='latin-1',
                               store_rejects=true)
        """
    )
    cols = [r[0] for r in con.execute("DESCRIBE _t").fetchall()]
    # normalise column names to lower case
    select = ", ".join(f'"{c}" AS "{c.lower()}"' for c in cols)
    con.execute(f"COPY (SELECT {select} FROM _t) TO '{dest_s}' (FORMAT parquet, COMPRESSION zstd)")
    n = con.execute("SELECT count(*) FROM _t").fetchone()[0]
    try:
        rejects = con.execute("SELECT count(*) FROM reject_errors").fetchone()[0]
    except duckdb.Error:
        rejects = 0
    with open(src, "rb") as fh:
        physical_lines = sum(1 for _ in fh) - 1
    return {
        "rows": int(n),
        "physical_lines": int(physical_lines),
        "rejected_lines": int(rejects),
        "columns": [c.lower() for c in cols],
    }


def ingest(paths: Paths | None = None, quarter: str = QUARTER, zpath: Path | None = None) -> dict:
    """Run ingestion; returns the manifest dictionary."""
    paths = (paths or get_paths()).ensure()
    zpath = zpath or zip_path(paths, quarter)
    if not zpath.exists():
        raise FileNotFoundError(
            f"{zpath} not found. Download it from the FDA FAERS/AEMS Quarterly Data Extract page "
            f"or run `python -m pvsys.pipeline download`."
        )
    qcode = quarter_code(quarter)
    extract_dir = extract(zpath, paths.raw / f"faers_{quarter.lower()}")
    raw_dir = paths.processed / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "quarter": quarter,
        "source_zip": zpath.name,
        "source_sha256": sha256(zpath),
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tables": {},
    }
    con = duckdb.connect()
    for table in FAERS_TABLES:
        src = find_table_file(extract_dir, table, qcode)
        info = load_table_to_parquet(con, src, raw_dir / f"{table.lower()}.parquet")
        manifest["tables"][table.lower()] = info
        log.info("%s: %s rows", table, info["rows"])

    deleted: list[str] = []
    dfile = find_delete_file(extract_dir, qcode)
    if dfile is not None:
        deleted = [ln.strip() for ln in dfile.read_text(encoding="latin-1").splitlines() if ln.strip()]
    con.execute("CREATE OR REPLACE TEMP TABLE _del(caseid VARCHAR)")
    if deleted:
        con.executemany("INSERT INTO _del VALUES (?)", [(d,) for d in deleted])
    out = str(raw_dir / "deleted_cases.parquet").replace("'", "''")
    con.execute(f"COPY _del TO '{out}' (FORMAT parquet)")
    manifest["deleted_case_ids"] = len(deleted)
    con.close()

    (paths.processed / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
