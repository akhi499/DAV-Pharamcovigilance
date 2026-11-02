"""Exploratory data analysis, demographic and subgroup summaries.

Every function takes a `Database` (see db.py) and returns JSON-serialisable
structures so the same code feeds the API, the report and the tests.
Descriptive counts use all de-duplicated cases; disproportionality uses the
stricter analysis population defined in signals.py.
"""

from __future__ import annotations

import json
import math

from scipy import stats as sstats

from .clean import AGE_GROUP_ORDER
from .config import OUTCOME_LABELS, REPORTER_LABELS, Paths, get_paths
from .db import Database


def overview(db: Database) -> dict:
    row = db.query(
        """
        SELECT count(*) AS cases,
               count(*) FILTER (WHERE serious) AS serious,
               count(*) FILTER (WHERE death) AS deaths,
               count(*) FILTER (WHERE hospitalization) AS hospitalizations,
               count(*) FILTER (WHERE probable_duplicate) AS probable_duplicates,
               count(*) FILTER (WHERE rept_cod = 'EXP') AS expedited,
               count(*) FILTER (WHERE literature) AS literature,
               min(fda_date) AS first_fda_date, max(fda_date) AS last_fda_date,
               median(age_years) AS median_age
        FROM cases
        """
    )[0]
    row["drug_rows"] = db.scalar("SELECT count(*) FROM drugs")
    row["distinct_suspect_drugs"] = db.scalar(
        "SELECT count(DISTINCT drug) FROM drugs WHERE role_cod IN ('PS','SS') AND drug IS NOT NULL"
    )
    row["reaction_rows"] = db.scalar("SELECT count(*) FROM reactions")
    row["distinct_reaction_terms"] = db.scalar("SELECT count(DISTINCT pt_key) FROM reactions")
    row["first_fda_date"] = str(row["first_fda_date"])
    row["last_fda_date"] = str(row["last_fda_date"])
    return row


def distribution(db: Database, column: str) -> list[dict]:
    allowed = {"sex", "age_group", "reporter_type", "occr_country", "reporter_country", "rept_cod"}
    if column not in allowed:
        raise ValueError(f"column must be one of {sorted(allowed)}")
    rows = db.query(
        f"""
        SELECT {column} AS level, count(*) AS cases,
               count(*) FILTER (WHERE serious) AS serious,
               count(*) FILTER (WHERE death) AS deaths
        FROM cases GROUP BY 1 ORDER BY cases DESC
        """
    )
    for r in rows:
        r["serious_pct"] = round(100 * r["serious"] / r["cases"], 2) if r["cases"] else None
        r["death_pct"] = round(100 * r["deaths"] / r["cases"], 2) if r["cases"] else None
        if column == "reporter_type":
            r["label"] = REPORTER_LABELS.get(r["level"], r["level"])
    if column == "age_group":
        rows.sort(key=lambda r: AGE_GROUP_ORDER.index(r["level"]) if r["level"] in AGE_GROUP_ORDER else 99)
    return rows


def outcomes(db: Database) -> list[dict]:
    rows = db.query("SELECT outc_cod AS code, count(DISTINCT primaryid) AS cases FROM outcomes GROUP BY 1 ORDER BY 2 DESC")
    for r in rows:
        r["label"] = OUTCOME_LABELS.get(r["code"], r["code"])
    return rows


def age_sex_pyramid(db: Database) -> list[dict]:
    return db.query(
        """
        SELECT age_group, sex, count(*) AS cases FROM cases
        WHERE sex IN ('F','M') GROUP BY ALL ORDER BY ALL
        """
    )


def age_histogram(db: Database, bin_width: int = 5) -> list[dict]:
    return db.query(
        f"""
        SELECT floor(age_years / {int(bin_width)}) * {int(bin_width)} AS bin_start, count(*) AS cases
        FROM cases WHERE age_years IS NOT NULL GROUP BY 1 ORDER BY 1
        """
    )


def receipt_trend(db: Database) -> list[dict]:
    rows = db.query(
        """
        SELECT fda_date AS date, count(*) AS cases, count(*) FILTER (WHERE serious) AS serious
        FROM cases WHERE fda_date IS NOT NULL GROUP BY 1 ORDER BY 1
        """
    )
    for r in rows:
        r["date"] = str(r["date"])
    return rows


def top_drugs(db: Database, limit: int = 20, role: str = "suspect") -> list[dict]:
    roles = "('PS','SS')" if role == "suspect" else "('PS')" if role == "primary" else "('PS','SS','C','I')"
    return db.query(
        f"""
        SELECT d.drug, count(DISTINCT d.primaryid) AS cases,
               count(DISTINCT d.primaryid) FILTER (WHERE c.serious) AS serious,
               count(DISTINCT d.primaryid) FILTER (WHERE c.death) AS deaths
        FROM drugs d JOIN cases c USING (primaryid)
        WHERE d.role_cod IN {roles} AND d.drug IS NOT NULL
        GROUP BY 1 ORDER BY 2 DESC LIMIT ?
        """,
        [limit],
    )


def top_reactions(db: Database, limit: int = 20) -> list[dict]:
    return db.query(
        """
        SELECT r.pt, r.pt_key, count(*) AS cases,
               count(*) FILTER (WHERE c.serious) AS serious,
               count(*) FILTER (WHERE c.death) AS deaths
        FROM reactions r JOIN cases c USING (primaryid)
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT ?
        """,
        [limit],
    )


def countries(db: Database, limit: int = 15) -> list[dict]:
    return db.query(
        "SELECT occr_country AS country, count(*) AS cases FROM cases GROUP BY 1 ORDER BY 2 DESC LIMIT ?", [limit]
    )


def serious_by_sex_test(db: Database) -> dict:
    """Chi-square test of independence: sex (F/M) x serious outcome."""
    rows = db.query(
        """
        SELECT sex, count(*) FILTER (WHERE serious) AS s, count(*) FILTER (WHERE NOT serious) AS ns
        FROM cases WHERE sex IN ('F','M') GROUP BY sex ORDER BY sex
        """
    )
    table = [[r["s"], r["ns"]] for r in rows]
    chi2, p, dof, _ = sstats.chi2_contingency(table)
    n = sum(map(sum, table))
    cramers_v = math.sqrt(chi2 / n) if n else None
    return {
        "table": rows,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "note": "Association in reported cases only; reflects reporting patterns, not population risk.",
    }


def drug_profile(db: Database, drug: str, limit: int = 25) -> dict:
    drug = drug.strip().upper()
    n = db.scalar(
        "SELECT count(DISTINCT primaryid) FROM drugs WHERE drug = ? AND role_cod IN ('PS','SS')", [drug]
    )
    if not n:
        return {"drug": drug, "cases": 0}
    base = "SELECT DISTINCT primaryid FROM drugs WHERE drug = ? AND role_cod IN ('PS','SS')"
    demo = db.query(
        f"""
        SELECT sex, age_group, count(*) AS cases FROM cases
        WHERE primaryid IN ({base}) GROUP BY ALL ORDER BY ALL
        """,
        [drug],
    )
    serious = db.query(
        f"""
        SELECT count(*) FILTER (WHERE serious) AS serious, count(*) FILTER (WHERE death) AS deaths,
               count(*) FILTER (WHERE hospitalization) AS hospitalizations, median(age_years) AS median_age
        FROM cases WHERE primaryid IN ({base})
        """,
        [drug],
    )[0]
    reactions = db.query(
        f"""
        SELECT pt, pt_key, count(*) AS cases FROM reactions WHERE primaryid IN ({base})
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT ?
        """,
        [drug, limit],
    )
    indications = db.query(
        """
        SELECT i.indi_pt, count(DISTINCT i.primaryid) AS cases
        FROM indications i JOIN drugs d ON d.primaryid = i.primaryid AND d.drug_seq = i.drug_seq
        WHERE d.drug = ? AND d.role_cod IN ('PS','SS') GROUP BY 1 ORDER BY 2 DESC LIMIT 10
        """,
        [drug],
    )
    return {"drug": drug, "cases": n, **serious, "demographics": demo, "top_reactions": reactions,
            "top_indications": indications}


def snapshot(paths: Paths | None = None) -> dict:
    """Persist a JSON snapshot of the main descriptive results (used in the report)."""
    paths = paths or get_paths()
    db = Database(paths)
    snap = {
        "overview": overview(db),
        "sex": distribution(db, "sex"),
        "age_group": distribution(db, "age_group"),
        "reporter_type": distribution(db, "reporter_type"),
        "outcomes": outcomes(db),
        "top_drugs": top_drugs(db, 20),
        "top_reactions": top_reactions(db, 20),
        "countries": countries(db, 15),
        "serious_by_sex": serious_by_sex_test(db),
    }
    (paths.processed / "eda_snapshot.json").write_text(json.dumps(snap, indent=2, default=str))
    return snap
