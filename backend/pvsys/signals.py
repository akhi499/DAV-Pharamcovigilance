"""Stage 4 - Disproportionality analysis (signal detection).

For every suspect-drug x reaction (PT) pair a 2x2 contingency table is built at
the *case* level over the de-duplicated analysis population:

                       reaction R    all other reactions
    drug D                  a               b
    all other drugs         c               d

Measures
--------
* ROR  = (a/b) / (c/d),   95% CI = exp(ln ROR +/- z * sqrt(1/a + 1/b + 1/c + 1/d))
* PRR  = (a/(a+b)) / (c/(c+d)),  95% CI = exp(ln PRR +/- z * sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d)))
* chi-square with Yates continuity correction (1 d.f.)
* IC (BCPNN information component, Noren et al. 2013 shrinkage approximation)
  IC = log2((a + 0.5) / (E + 0.5)), E = (a+b)(a+c)/N, IC025 ~ IC - 3.3(a+0.5)^-1/2 - 2(a+0.5)^-3/2

Sparse cells: when any cell of a table is zero, the Haldane-Anscombe
correction (+0.5 to every cell) is applied before computing ratios and CIs so
estimates stay finite; the flag `haldane_applied` records this per row.

IMPORTANT: these are measures of *disproportional reporting* in a spontaneous
reporting database. They are not incidence rates, do not estimate absolute
risk (there is no exposure denominator), and do not establish causality.
"""

from __future__ import annotations

import json
import logging

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sstats

from .config import Paths, SignalConfig, get_paths

log = logging.getLogger(__name__)

SUSPECT_ROLES = ("PS", "SS")


def disproportionality(a, b, c, d, cfg: SignalConfig | None = None) -> pd.DataFrame:
    """Vectorised ROR/PRR/chi2/IC for arrays of 2x2 cell counts."""
    cfg = cfg or SignalConfig()
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    if np.any((a < 0) | (b < 0) | (c < 0) | (d < 0)):
        raise ValueError("2x2 cell counts must be non-negative")

    n = a + b + c + d
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    k = np.where(zero, cfg.continuity, 0.0)
    ac, bc, cc, dc = a + k, b + k, c + k, d + k

    with np.errstate(divide="ignore", invalid="ignore"):
        ror = (ac / bc) / (cc / dc)
        se_ror = np.sqrt(1 / ac + 1 / bc + 1 / cc + 1 / dc)
        ror_lo = np.exp(np.log(ror) - cfg.z * se_ror)
        ror_hi = np.exp(np.log(ror) + cfg.z * se_ror)

        prr = (ac / (ac + bc)) / (cc / (cc + dc))
        se_prr = np.sqrt(np.clip(1 / ac - 1 / (ac + bc) + 1 / cc - 1 / (cc + dc), 0, None))
        prr_lo = np.exp(np.log(prr) - cfg.z * se_prr)
        prr_hi = np.exp(np.log(prr) + cfg.z * se_prr)

        num = n * np.square(np.maximum(np.abs(a * d - b * c) - n / 2, 0))
        den = (a + b) * (c + d) * (a + c) * (b + d)
        chi2 = np.where(den > 0, num / den, 0.0)
        p = sstats.chi2.sf(chi2, df=1)

        expected = (a + b) * (a + c) / np.where(n > 0, n, np.nan)
        ic = np.log2((a + 0.5) / (expected + 0.5))
        ic025 = ic - 3.3 * np.power(a + 0.5, -0.5) - 2 * np.power(a + 0.5, -1.5)

    df = pd.DataFrame(
        {
            "ror": ror,
            "ror_lower": ror_lo,
            "ror_upper": ror_hi,
            "prr": prr,
            "prr_lower": prr_lo,
            "prr_upper": prr_hi,
            "chi2": chi2,
            "p_value": p,
            "expected": expected,
            "ic": ic,
            "ic025": ic025,
            "haldane_applied": zero,
        }
    )
    df["ror_signal"] = (a >= cfg.min_cases) & (df["ror_lower"] > cfg.ror_lower_ci_threshold)
    df["prr_signal"] = (a >= cfg.min_cases) & (df["prr"] >= cfg.prr_threshold) & (df["chi2"] >= cfg.chi2_threshold)
    df["ic_signal"] = (a >= cfg.min_cases) & (df["ic025"] > 0)
    df["methods_flagging"] = df[["ror_signal", "prr_signal", "ic_signal"]].sum(axis=1).astype(int)
    return df


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(q, 0, 1)
    return out


ANALYSIS_POPULATION_SQL = f"""
    SELECT c.primaryid FROM cases c
    WHERE NOT c.probable_duplicate
      AND c.primaryid IN (SELECT primaryid FROM drugs WHERE role_cod IN {SUSPECT_ROLES} AND drug IS NOT NULL)
      AND c.primaryid IN (SELECT primaryid FROM reactions)
"""


def build_pair_counts(con: duckdb.DuckDBPyConnection, where_cases: str = "") -> pd.DataFrame:
    """Return per-pair a, n_drug, n_pt and N for an (optionally filtered) population."""
    filt = f"AND c.primaryid IN (SELECT primaryid FROM cases WHERE {where_cases})" if where_cases else ""
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE pop AS
        SELECT primaryid FROM ({ANALYSIS_POPULATION_SQL}) c WHERE true {filt};
        CREATE OR REPLACE TEMP TABLE dp AS
        SELECT DISTINCT d.primaryid, d.drug FROM drugs d JOIN pop USING (primaryid)
        WHERE d.role_cod IN {SUSPECT_ROLES} AND d.drug IS NOT NULL;
        CREATE OR REPLACE TEMP TABLE rp AS
        SELECT DISTINCT r.primaryid, r.pt_key, r.pt FROM reactions r JOIN pop USING (primaryid);
        """
    )
    n_total = con.execute("SELECT count(*) FROM pop").fetchone()[0]
    df = con.execute(
        """
        WITH ind AS (
            SELECT DISTINCT i.primaryid, lower(regexp_replace(trim(i.indi_pt), '\\s+', ' ', 'g')) AS pt_key
            FROM indications i JOIN pop USING (primaryid)),
        pairs AS (
            SELECT dp.drug, rp.pt_key, count(*) AS a,
                   count(ind.primaryid) AS a_reaction_is_indication
            FROM dp JOIN rp USING (primaryid)
            LEFT JOIN ind ON ind.primaryid = rp.primaryid AND ind.pt_key = rp.pt_key
            GROUP BY ALL),
        nd AS (SELECT drug, count(*) AS n_drug FROM dp GROUP BY drug),
        np AS (SELECT pt_key, any_value(pt) AS pt, count(*) AS n_pt FROM rp GROUP BY pt_key)
        SELECT pairs.drug, np.pt, pairs.pt_key, pairs.a, nd.n_drug, np.n_pt,
               pairs.a_reaction_is_indication
        FROM pairs JOIN nd USING (drug) JOIN np USING (pt_key)
        """
    ).df()
    df["n_total"] = n_total
    return df


def compute_signals(pairs: pd.DataFrame, cfg: SignalConfig | None = None) -> pd.DataFrame:
    cfg = cfg or SignalConfig()
    pairs = pairs[pairs["a"] >= cfg.min_cases].copy()
    a = pairs["a"].to_numpy()
    b = pairs["n_drug"].to_numpy() - a
    c = pairs["n_pt"].to_numpy() - a
    d = pairs["n_total"].to_numpy() - a - b - c
    pairs["b"], pairs["c"], pairs["d"] = b, c, d
    res = disproportionality(a, b, c, d, cfg)
    out = pd.concat([pairs.reset_index(drop=True), res], axis=1)
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    # Share of the a cases in which the reaction term was also reported as an
    # indication in the same case: a high share suggests confounding by
    # indication / disease progression rather than an adverse reaction.
    out["indication_overlap"] = out["a_reaction_is_indication"] / out["a"]
    out["possible_indication_bias"] = out["indication_overlap"] >= 0.2
    return out.sort_values(["methods_flagging", "ror_lower"], ascending=[False, False]).reset_index(drop=True)


def run(paths: Paths | None = None, cfg: SignalConfig | None = None) -> dict:
    from .db import connect

    paths = paths or get_paths()
    cfg = cfg or SignalConfig()
    con = connect(paths, tables=("cases", "drugs", "reactions", "indications"))
    pairs = build_pair_counts(con)
    n_pairs_all = len(pairs)
    sig = compute_signals(pairs, cfg)
    float_cols = sig.select_dtypes("float").columns
    sig[float_cols] = sig[float_cols].astype("float64")
    sig.to_parquet(paths.processed / "signals.parquet", index=False)
    summary = {
        "analysis_population_cases": int(pairs["n_total"].iloc[0]) if len(pairs) else 0,
        "drug_event_pairs_observed": int(n_pairs_all),
        "pairs_tested_min_cases": int(len(sig)),
        "min_cases": cfg.min_cases,
        "ror_signals": int(sig["ror_signal"].sum()),
        "prr_signals": int(sig["prr_signal"].sum()),
        "ic_signals": int(sig["ic_signal"].sum()),
        "all_three_methods": int((sig["methods_flagging"] == 3).sum()),
        "haldane_corrected_rows": int(sig["haldane_applied"].sum()),
        "q_below_0_05": int((sig["q_value"] < 0.05).sum()),
        "possible_indication_bias_rows": int(sig["possible_indication_bias"].sum()),
        "distinct_suspect_drugs": int(pairs["drug"].nunique()),
        "distinct_pts": int(pairs["pt_key"].nunique()),
    }
    (paths.processed / "signals_summary.json").write_text(json.dumps(summary, indent=2))
    con.close()
    log.info("Signals: %s", summary)
    return summary


def stratified(con: duckdb.DuckDBPyConnection, drug: str, pt_key: str, by: str, cfg: SignalConfig | None = None) -> list[dict]:
    """Recompute the 2x2 table for one pair within each level of `by` (sex or age_group)."""
    if by not in ("sex", "age_group", "reporter_type"):
        raise ValueError("by must be one of sex, age_group, reporter_type")
    cfg = cfg or SignalConfig()
    df = con.execute(
        f"""
        WITH pop AS (
            SELECT c.primaryid, c.{by} AS stratum FROM cases c
            WHERE c.primaryid IN ({ANALYSIS_POPULATION_SQL})),
        hasd AS (SELECT DISTINCT primaryid FROM drugs WHERE role_cod IN {SUSPECT_ROLES} AND drug = ?),
        hasr AS (SELECT DISTINCT primaryid FROM reactions WHERE pt_key = ?)
        SELECT stratum,
               count(*) FILTER (WHERE d.primaryid IS NOT NULL AND r.primaryid IS NOT NULL) AS a,
               count(*) FILTER (WHERE d.primaryid IS NOT NULL AND r.primaryid IS NULL) AS b,
               count(*) FILTER (WHERE d.primaryid IS NULL AND r.primaryid IS NOT NULL) AS c,
               count(*) FILTER (WHERE d.primaryid IS NULL AND r.primaryid IS NULL) AS d
        FROM pop LEFT JOIN hasd d USING (primaryid) LEFT JOIN hasr r USING (primaryid)
        GROUP BY stratum ORDER BY stratum
        """,
        [drug, pt_key],
    ).df()
    if df.empty:
        return []
    res = disproportionality(df["a"], df["b"], df["c"], df["d"], cfg)
    out = pd.concat([df, res[["ror", "ror_lower", "ror_upper", "prr", "chi2", "haldane_applied"]]], axis=1)
    out["estimable"] = out["a"] >= cfg.min_cases
    return out.replace({np.nan: None}).to_dict(orient="records")
