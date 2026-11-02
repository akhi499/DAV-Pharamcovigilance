"""Stage 2 - Cleaning, normalisation, case-version handling and de-duplication.

Case-version handling follows the FDA FAERS README recommendation:
  1. drop every CASEID listed in the quarter's DELETE file;
  2. for each CASEID keep only the most recent version - the row with the
     latest FDA_DT, breaking ties with the highest PRIMARYID
     (PRIMARYID = CASEID || CASEVERSION).
All child tables (DRUG, REAC, OUTC, ...) are then restricted to the retained
PRIMARYIDs so every downstream count is at the de-duplicated case level.
"""

from __future__ import annotations

import json
import logging

import duckdb

from .config import Paths, get_paths

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL building blocks (kept as constants so they can be unit tested)
# ---------------------------------------------------------------------------

AGE_YEARS_SQL = """
CASE upper(trim(age_cod))
    WHEN 'YR'  THEN try_cast(age AS DOUBLE)
    WHEN 'DEC' THEN try_cast(age AS DOUBLE) * 10
    WHEN 'MON' THEN try_cast(age AS DOUBLE) / 12
    WHEN 'WK'  THEN try_cast(age AS DOUBLE) / 52.1775
    WHEN 'DY'  THEN try_cast(age AS DOUBLE) / 365.25
    WHEN 'HR'  THEN try_cast(age AS DOUBLE) / 8766
    ELSE NULL
END
"""

WEIGHT_KG_SQL = """
CASE upper(trim(wt_cod))
    WHEN 'KG'  THEN try_cast(wt AS DOUBLE)
    WHEN 'LBS' THEN try_cast(wt AS DOUBLE) * 0.45359237
    WHEN 'GMS' THEN try_cast(wt AS DOUBLE) / 1000
    ELSE NULL
END
"""

AGE_GROUP_SQL = """
CASE
    WHEN age_years IS NULL THEN 'Unknown'
    WHEN age_years < 2  THEN '0-1'
    WHEN age_years < 12 THEN '2-11'
    WHEN age_years < 18 THEN '12-17'
    WHEN age_years < 45 THEN '18-44'
    WHEN age_years < 65 THEN '45-64'
    WHEN age_years < 75 THEN '65-74'
    ELSE '75+'
END
"""

AGE_GROUP_ORDER = ["0-1", "2-11", "12-17", "18-44", "45-64", "65-74", "75+", "Unknown"]


def date_sql(col: str) -> str:
    """Parse FAERS YYYYMMDD strings; partial dates (YYYY or YYYYMM) become NULL dates."""
    return f"CASE WHEN length(trim({col})) = 8 THEN try_strptime(trim({col}), '%Y%m%d')::DATE ELSE NULL END"


DRUG_NAME_SQL = r"""
nullif(
  regexp_replace(
    regexp_replace(
      upper(trim(coalesce(nullif(trim(prod_ai), ''),
        regexp_replace(trim(drugname), '\s+\d+(\.\d+)?\s*(MG|MCG|G|ML|IU|UNITS?|%)\b.*$', '', 'i')))),
      '\s+', ' ', 'g'),
    '[\.\,;:]+$', ''),
  '')
"""


def _q(p) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


def clean(paths: Paths | None = None) -> dict:
    """Build cleaned, de-duplicated Parquet tables from the raw Parquet files."""
    paths = (paths or get_paths()).ensure()
    raw = paths.processed / "raw"
    out = paths.processed
    con = duckdb.connect()
    for t in ("demo", "drug", "reac", "outc", "rpsr", "ther", "indi", "deleted_cases"):
        con.execute(f"CREATE VIEW raw_{t} AS SELECT * FROM read_parquet('{_q(raw / (t + '.parquet'))}')")

    stats: dict = {}
    stats["demo_rows_raw"] = con.execute("SELECT count(*) FROM raw_demo").fetchone()[0]
    stats["distinct_caseids_raw"] = con.execute("SELECT count(DISTINCT caseid) FROM raw_demo").fetchone()[0]
    stats["deleted_caseids_listed"] = con.execute("SELECT count(DISTINCT caseid) FROM raw_deleted_cases").fetchone()[0]
    stats["rows_removed_deleted"] = con.execute(
        "SELECT count(*) FROM raw_demo WHERE trim(caseid) IN (SELECT trim(caseid) FROM raw_deleted_cases)"
    ).fetchone()[0]

    # --- case-version de-duplication ------------------------------------
    con.execute(
        """
        CREATE TEMP TABLE demo_dedup AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (
                PARTITION BY trim(caseid)
                ORDER BY try_cast(fda_dt AS BIGINT) DESC NULLS LAST,
                         try_cast(primaryid AS BIGINT) DESC
            ) AS rn
            FROM raw_demo
            WHERE trim(caseid) NOT IN (SELECT trim(caseid) FROM raw_deleted_cases)
        ) WHERE rn = 1
        """
    )
    stats["cases_after_dedup"] = con.execute("SELECT count(*) FROM demo_dedup").fetchone()[0]
    stats["rows_removed_older_versions"] = (
        stats["demo_rows_raw"] - stats["rows_removed_deleted"] - stats["cases_after_dedup"]
    )

    # --- cleaned case table ------------------------------------------------
    con.execute(
        f"""
        CREATE TEMP TABLE cases_base AS
        SELECT
            trim(primaryid) AS primaryid,
            trim(caseid) AS caseid,
            try_cast(caseversion AS INTEGER) AS caseversion,
            upper(trim(i_f_code)) AS i_f_code,
            {date_sql('event_dt')} AS event_date,
            {date_sql('init_fda_dt')} AS init_fda_date,
            {date_sql('fda_dt')} AS fda_date,
            {date_sql('rept_dt')} AS rept_date,
            upper(trim(rept_cod)) AS rept_cod,
            nullif(trim(mfr_sndr), '') AS mfr_sndr,
            (nullif(trim(lit_ref), '') IS NOT NULL) AS literature,
            {AGE_YEARS_SQL} AS age_years_raw,
            {WEIGHT_KG_SQL} AS weight_kg_raw,
            CASE upper(trim(sex)) WHEN 'F' THEN 'F' WHEN 'M' THEN 'M' ELSE 'Unknown' END AS sex,
            upper(trim(e_sub)) = 'Y' AS e_sub,
            coalesce(nullif(upper(trim(occp_cod)), ''), 'Unknown') AS reporter_type,
            coalesce(nullif(upper(trim(reporter_country)), ''), 'Unknown') AS reporter_country,
            coalesce(nullif(upper(trim(occr_country)), ''), 'Unknown') AS occr_country
        FROM demo_dedup
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE outc_c AS
        SELECT DISTINCT trim(o.primaryid) AS primaryid, upper(trim(o.outc_cod)) AS outc_cod
        FROM raw_outc o JOIN cases_base c ON trim(o.primaryid) = c.primaryid
        WHERE nullif(trim(o.outc_cod), '') IS NOT NULL;

        CREATE TEMP TABLE cases AS
        SELECT c.* EXCLUDE (age_years_raw, weight_kg_raw),
            CASE WHEN c.age_years_raw BETWEEN 0 AND 120 THEN round(c.age_years_raw, 2) END AS age_years,
            (c.age_years_raw IS NOT NULL AND NOT c.age_years_raw BETWEEN 0 AND 120) AS age_implausible,
            CASE WHEN c.weight_kg_raw BETWEEN 0.2 AND 400 THEN round(c.weight_kg_raw, 2) END AS weight_kg,
            (c.weight_kg_raw IS NOT NULL AND NOT c.weight_kg_raw BETWEEN 0.2 AND 400) AS weight_implausible,
            coalesce(o.serious, false) AS serious,
            coalesce(o.death, false) AS death,
            coalesce(o.hosp, false) AS hospitalization,
            coalesce(o.lt, false) AS life_threatening
        FROM cases_base c
        LEFT JOIN (
            SELECT primaryid, true AS serious,
                   bool_or(outc_cod = 'DE') AS death,
                   bool_or(outc_cod = 'HO') AS hosp,
                   bool_or(outc_cod = 'LT') AS lt
            FROM outc_c GROUP BY primaryid
        ) o USING (primaryid);
        """
    )
    con.execute(f"CREATE TEMP TABLE cases2 AS SELECT *, {AGE_GROUP_SQL} AS age_group FROM cases")

    # --- child tables restricted to retained reports ---------------------
    con.execute(
        f"""
        CREATE TEMP TABLE drugs AS
        SELECT trim(d.primaryid) AS primaryid, trim(d.caseid) AS caseid,
               try_cast(d.drug_seq AS INTEGER) AS drug_seq,
               upper(trim(d.role_cod)) AS role_cod,
               upper(trim(d.drugname)) AS drugname,
               nullif(upper(trim(d.prod_ai)), '') AS prod_ai,
               {DRUG_NAME_SQL} AS drug,
               nullif(trim(d.route), '') AS route,
               nullif(upper(trim(d.dechal)), '') AS dechal,
               nullif(upper(trim(d.rechal)), '') AS rechal,
               nullif(trim(d.dose_form), '') AS dose_form
        FROM raw_drug d JOIN cases2 c ON trim(d.primaryid) = c.primaryid;

        CREATE TEMP TABLE reactions AS
        SELECT DISTINCT trim(r.primaryid) AS primaryid, trim(r.caseid) AS caseid,
               regexp_replace(trim(r.pt), '\\s+', ' ', 'g') AS pt,
               lower(regexp_replace(trim(r.pt), '\\s+', ' ', 'g')) AS pt_key
        FROM raw_reac r JOIN cases2 c ON trim(r.primaryid) = c.primaryid
        WHERE nullif(trim(r.pt), '') IS NOT NULL;

        CREATE TEMP TABLE indications AS
        SELECT DISTINCT trim(i.primaryid) AS primaryid,
               try_cast(i.indi_drug_seq AS INTEGER) AS drug_seq, trim(i.indi_pt) AS indi_pt
        FROM raw_indi i JOIN cases2 c ON trim(i.primaryid) = c.primaryid
        WHERE nullif(trim(i.indi_pt), '') IS NOT NULL;

        CREATE TEMP TABLE therapy AS
        SELECT trim(t.primaryid) AS primaryid, try_cast(t.dsg_drug_seq AS INTEGER) AS drug_seq,
               {date_sql('t.start_dt')} AS start_date, {date_sql('t.end_dt')} AS end_date,
               try_cast(t.dur AS DOUBLE) AS dur, nullif(upper(trim(t.dur_cod)), '') AS dur_cod
        FROM raw_ther t JOIN cases2 c ON trim(t.primaryid) = c.primaryid;

        CREATE TEMP TABLE report_sources AS
        SELECT DISTINCT trim(s.primaryid) AS primaryid, upper(trim(s.rpsr_cod)) AS rpsr_cod
        FROM raw_rpsr s JOIN cases2 c ON trim(s.primaryid) = c.primaryid;
        """
    )

    # Use the most frequent original spelling for each lower-cased PT key
    con.execute(
        """
        CREATE TEMP TABLE pt_display AS
        SELECT pt_key, arg_max(pt, n) AS pt_display FROM (
            SELECT pt_key, pt, count(*) n FROM reactions GROUP BY ALL) GROUP BY pt_key;
        CREATE TEMP TABLE reactions2 AS
        SELECT r.primaryid, r.caseid, d.pt_display AS pt, r.pt_key
        FROM (SELECT DISTINCT primaryid, caseid, pt_key FROM reactions) r JOIN pt_display d USING (pt_key);
        """
    )

    # --- probable duplicate detection (different CASEIDs, same clinical fingerprint)
    # A report is flagged when another report shares sex, age, event date,
    # occurrence country, the full set of normalised suspect drugs and the full
    # set of reaction PTs. Event date and age must be present to avoid
    # over-matching sparse reports. The most recently received report (latest
    # fda_date, then highest primaryid) of each group is kept un-flagged.
    con.execute(
        """
        CREATE TEMP TABLE fp AS
        WITH dr AS (
            SELECT primaryid, string_agg(DISTINCT drug, '|' ORDER BY drug) AS drug_set
            FROM drugs WHERE role_cod IN ('PS','SS') AND drug IS NOT NULL GROUP BY primaryid),
        re AS (
            SELECT primaryid, string_agg(DISTINCT pt_key, '|' ORDER BY pt_key) AS pt_set
            FROM reactions2 GROUP BY primaryid)
        SELECT c.primaryid,
               md5(concat_ws('#', c.sex, c.age_years::VARCHAR, c.event_date::VARCHAR,
                             c.occr_country, dr.drug_set, re.pt_set)) AS fingerprint,
               row_number() OVER (
                   PARTITION BY c.sex, c.age_years, c.event_date, c.occr_country, dr.drug_set, re.pt_set
                   ORDER BY c.fda_date DESC NULLS LAST, try_cast(c.primaryid AS BIGINT) DESC) AS rn
        FROM cases2 c JOIN dr USING (primaryid) JOIN re USING (primaryid)
        WHERE c.event_date IS NOT NULL AND c.age_years IS NOT NULL;

        CREATE TEMP TABLE cases3 AS
        SELECT c.*, coalesce(fp.rn > 1, false) AS probable_duplicate, fp.fingerprint
        FROM cases2 c LEFT JOIN fp USING (primaryid);
        """
    )
    stats["probable_duplicates_flagged"] = con.execute(
        "SELECT count(*) FROM cases3 WHERE probable_duplicate"
    ).fetchone()[0]
    stats["reaction_rows_raw"] = con.execute("SELECT count(*) FROM raw_reac").fetchone()[0]

    exports = {
        "cases": "SELECT * FROM cases3",
        "drugs": "SELECT * FROM drugs",
        "reactions": "SELECT * FROM reactions2",
        "outcomes": "SELECT * FROM outc_c",
        "indications": "SELECT * FROM indications",
        "therapy": "SELECT * FROM therapy",
        "report_sources": "SELECT * FROM report_sources",
    }
    for name, sql in exports.items():
        con.execute(f"COPY ({sql}) TO '{_q(out / (name + '.parquet'))}' (FORMAT parquet, COMPRESSION zstd)")
        stats[f"{name}_rows"] = con.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
    con.close()

    stats = {k: int(v) for k, v in stats.items()}
    (out / "dedup_stats.json").write_text(json.dumps(stats, indent=2))
    log.info("Cleaning done: %s", stats)
    return stats
