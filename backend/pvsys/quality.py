"""Stage 3 - Data-quality assessment.

Checks are grouped into the classic dimensions (completeness, validity,
consistency, uniqueness). Each check reports a measured value and a status:
'pass', 'warn' (worth disclosing / handling in analysis) or 'info'.
Spontaneous-report data are expected to be incomplete, so most findings are
disclosed limitations rather than errors to be "fixed".
"""

from __future__ import annotations

import json
from datetime import date

from .config import Paths, get_paths
from .db import connect


def _check(name, dimension, value, unit, status, detail) -> dict:
    return {
        "check": name,
        "dimension": dimension,
        "value": None if value is None else round(float(value), 4),
        "unit": unit,
        "status": status,
        "detail": detail,
    }


def run(paths: Paths | None = None) -> dict:
    paths = paths or get_paths()
    con = connect(paths)
    raw = paths.processed / "raw"
    for t in ("demo", "drug", "reac"):
        con.execute(f"CREATE VIEW raw_{t} AS SELECT * FROM read_parquet('{str(raw / (t + '.parquet')).replace(chr(92), '/')}')")

    n = con.execute("SELECT count(*) FROM cases").fetchone()[0]
    checks: list[dict] = []

    # ---- completeness ---------------------------------------------------
    completeness_fields = {
        "age_years": "Age (valid, converted to years)",
        "weight_kg": "Weight (valid, converted to kg)",
        "event_date": "Full event date",
        "sex": "Sex",
        "reporter_type": "Reporter occupation",
        "occr_country": "Occurrence country",
    }
    field_missing = {}
    for col, label in completeness_fields.items():
        if col in ("sex", "reporter_type", "occr_country"):
            miss = con.execute(f"SELECT count(*) FILTER (WHERE {col} = 'Unknown') FROM cases").fetchone()[0]
        else:
            miss = con.execute(f"SELECT count(*) FILTER (WHERE {col} IS NULL) FROM cases").fetchone()[0]
        pct = 100 * miss / n
        field_missing[col] = round(pct, 2)
        checks.append(
            _check(
                f"Missing {label}",
                "completeness",
                pct,
                "% of cases",
                "warn" if pct > 20 else "pass",
                f"{miss:,} of {n:,} cases",
            )
        )

    # ---- validity -------------------------------------------------------
    implaus_age = con.execute("SELECT count(*) FROM cases WHERE age_implausible").fetchone()[0]
    checks.append(_check("Implausible age (outside 0-120 y) set to missing", "validity", implaus_age, "cases",
                         "pass" if implaus_age < 0.001 * n else "warn", "Converted from age + age_cod"))
    implaus_wt = con.execute("SELECT count(*) FROM cases WHERE weight_implausible").fetchone()[0]
    checks.append(_check("Implausible weight (outside 0.2-400 kg) set to missing", "validity", implaus_wt, "cases",
                         "pass" if implaus_wt < 0.001 * n else "warn", "Converted from wt + wt_cod"))
    partial_dates = con.execute(
        "SELECT count(*) FROM raw_demo WHERE length(trim(event_dt)) IN (4, 6)"
    ).fetchone()[0]
    checks.append(_check("Partial event dates (YYYY or YYYYMM only)", "validity", partial_dates, "reports", "info",
                         "Kept as missing full date; not imputed"))
    future = con.execute(f"SELECT count(*) FROM cases WHERE event_date > DATE '{date.today().isoformat()}'").fetchone()[0]
    checks.append(_check("Event date in the future", "validity", future, "cases", "pass" if future < 0.001 * n else "warn",
                         "Likely data-entry error; kept but excluded from date-based summaries"))
    after_receipt = con.execute("SELECT count(*) FROM cases WHERE event_date > fda_date").fetchone()[0]
    checks.append(_check("Event date after FDA receipt date", "validity", after_receipt, "cases",
                         "pass" if after_receipt < 0.001 * n else "warn", "Likely data-entry error"))
    old_events = con.execute("SELECT count(*) FROM cases WHERE event_date < DATE '1970-01-01'").fetchone()[0]
    checks.append(_check("Event date before 1970", "validity", old_events, "cases", "info", "Possible data-entry error"))

    # ---- consistency ----------------------------------------------------
    no_suspect = con.execute(
        "SELECT count(*) FROM cases WHERE primaryid NOT IN (SELECT primaryid FROM drugs WHERE role_cod IN ('PS','SS'))"
    ).fetchone()[0]
    checks.append(_check("Cases without any suspect drug", "consistency", no_suspect, "cases",
                         "pass" if no_suspect < 0.01 * n else "warn", "Excluded from disproportionality"))
    no_reac = con.execute("SELECT count(*) FROM cases WHERE primaryid NOT IN (SELECT primaryid FROM reactions)").fetchone()[0]
    checks.append(_check("Cases without any reaction term", "consistency", no_reac, "cases",
                         "pass" if no_reac < 0.01 * n else "warn", "Excluded from disproportionality"))
    orphan_drug = con.execute(
        "SELECT count(*) FROM raw_drug WHERE trim(primaryid) NOT IN (SELECT trim(primaryid) FROM raw_demo)"
    ).fetchone()[0]
    checks.append(_check("Orphan DRUG rows (no DEMO record)", "consistency", orphan_drug, "rows",
                         "pass" if orphan_drug == 0 else "warn", ""))
    orphan_reac = con.execute(
        "SELECT count(*) FROM raw_reac WHERE trim(primaryid) NOT IN (SELECT trim(primaryid) FROM raw_demo)"
    ).fetchone()[0]
    checks.append(_check("Orphan REAC rows (no DEMO record)", "consistency", orphan_reac, "rows",
                         "pass" if orphan_reac == 0 else "warn", ""))
    no_ps = con.execute(
        "SELECT count(*) FROM cases WHERE primaryid NOT IN (SELECT primaryid FROM drugs WHERE role_cod = 'PS')"
    ).fetchone()[0]
    checks.append(_check("Cases without a primary-suspect (PS) drug", "consistency", no_ps, "cases",
                         "pass" if no_ps < 0.01 * n else "warn", ""))
    unnamed = con.execute("SELECT count(*) FROM drugs WHERE drug IS NULL").fetchone()[0]
    checks.append(_check("Drug rows with no usable name", "consistency", unnamed, "rows", "info", ""))
    no_ai = con.execute("SELECT 100.0 * count(*) FILTER (WHERE prod_ai IS NULL) / count(*) FROM drugs").fetchone()[0]
    checks.append(_check("Drug rows lacking active ingredient (fallback to verbatim name)", "consistency", no_ai,
                         "% of drug rows", "info", "Verbatim names are normalised by regex only (no RxNorm)"))

    # ---- uniqueness -----------------------------------------------------
    dup_cases = con.execute("SELECT count(*) - count(DISTINCT caseid) FROM raw_demo").fetchone()[0]
    checks.append(_check("Duplicate CASEID versions within the quarter", "uniqueness", dup_cases, "rows", "info",
                         "Resolved by keeping latest FDA_DT / highest PRIMARYID"))
    dup_reac = con.execute(
        "SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT trim(primaryid), lower(trim(pt)) FROM raw_reac)) FROM raw_reac"
    ).fetchone()[0]
    checks.append(_check("Repeated reaction term within the same report", "uniqueness", dup_reac, "rows", "info",
                         "Collapsed to one row per case-term"))
    probable = con.execute("SELECT count(*) FROM cases WHERE probable_duplicate").fetchone()[0]
    checks.append(_check("Probable duplicate reports (same clinical fingerprint, different CASEID)", "uniqueness",
                         probable, "cases", "warn" if probable > 0 else "pass",
                         "Excluded from disproportionality; kept in descriptive counts"))

    result = {
        "n_cases": int(n),
        "field_missing_pct": field_missing,
        "checks": checks,
        "summary": {s: sum(1 for c in checks if c["status"] == s) for s in ("pass", "warn", "info")},
    }
    (paths.processed / "quality.json").write_text(json.dumps(result, indent=2))
    con.close()
    return result
