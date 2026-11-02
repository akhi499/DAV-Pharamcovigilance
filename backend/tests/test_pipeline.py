"""End-to-end tests of ingestion, cleaning/de-duplication, quality and signals on synthetic data."""

from __future__ import annotations

import json

import duckdb
import pytest

from pvsys.ingest import quarter_code


def q(paths, sql):
    return duckdb.sql(sql.replace("{p}", str(paths.processed).replace("\\", "/"))).fetchall()


def test_quarter_code():
    assert quarter_code("2025Q1") == "25Q1"
    with pytest.raises(ValueError):
        quarter_code("2025-1")


def test_manifest_counts(pipeline_paths):
    m = json.loads((pipeline_paths.processed / "manifest.json").read_text())
    assert m["tables"]["demo"]["rows"] == 31
    assert m["tables"]["demo"]["rejected_lines"] == 0
    assert m["deleted_case_ids"] == 1
    assert len(m["source_sha256"]) == 64


def test_case_version_dedup(pipeline_paths):
    stats = json.loads((pipeline_paths.processed / "dedup_stats.json").read_text())
    assert stats["demo_rows_raw"] == 31
    assert stats["rows_removed_deleted"] == 1
    assert stats["rows_removed_older_versions"] == 2
    assert stats["cases_after_dedup"] == 28
    kept = dict(q(pipeline_paths, "SELECT caseid, primaryid FROM '{p}/cases.parquet' WHERE caseid IN ('1001','1002')"))
    assert kept == {"1001": "10012", "1002": "10022"}  # latest fda_dt, then highest primaryid
    assert q(pipeline_paths, "SELECT count(*) FROM '{p}/cases.parquet' WHERE caseid = '1003'")[0][0] == 0


def test_child_tables_follow_retained_versions(pipeline_paths):
    pids = {r[0] for r in q(pipeline_paths, "SELECT DISTINCT primaryid FROM '{p}/drugs.parquet'")}
    assert "10011" not in pids and "10031" not in pids and "10012" in pids


def test_normalisation(pipeline_paths):
    row = q(pipeline_paths, "SELECT age_years, age_group, sex FROM '{p}/cases.parquet' WHERE caseid='1002'")[0]
    assert row == (1.5, "0-1", "M")  # 18 months -> 1.5 years
    wt = q(pipeline_paths, "SELECT weight_kg FROM '{p}/cases.parquet' WHERE caseid='4001'")[0][0]
    assert wt == pytest.approx(68.04, abs=0.01)  # 150 lbs
    bad = q(pipeline_paths, "SELECT age_years, age_implausible FROM '{p}/cases.parquet' WHERE caseid='4000'")[0]
    assert bad == (None, True)
    names = {r[0] for r in q(pipeline_paths, "SELECT DISTINCT drug FROM '{p}/drugs.parquet'")}
    assert {"DRUGX", "DRUGY", "DRUGZ", "ASPIRIN"} == names


def test_repeated_reaction_terms_collapsed(pipeline_paths):
    n = q(pipeline_paths, "SELECT count(*) FROM '{p}/reactions.parquet' WHERE caseid='3000'")[0][0]
    assert n == 1


def test_probable_duplicate_flag(pipeline_paths):
    rows = dict(q(pipeline_paths, "SELECT caseid, probable_duplicate FROM '{p}/cases.parquet' WHERE caseid IN ('2001','2002')"))
    assert rows == {"2001": True, "2002": False}  # older report flagged, newest kept


def test_signal_two_by_two_and_haldane(pipeline_paths):
    import pandas as pd

    s = pd.read_parquet(pipeline_paths.processed / "signals.parquet")
    row = s[(s.drug == "DRUGX") & (s.pt_key == "nausea")].iloc[0]
    assert (row.a, row.b, row.c, row.d) == (6, 0, 5, 16)
    assert bool(row.haldane_applied)
    assert row.ror == pytest.approx((6.5 / 0.5) / (5.5 / 16.5))
    summary = json.loads((pipeline_paths.processed / "signals_summary.json").read_text())
    assert summary["analysis_population_cases"] == 27  # 28 cases minus 1 probable duplicate


def test_indication_bias_flag(pipeline_paths):
    import pandas as pd

    s = pd.read_parquet(pipeline_paths.processed / "signals.parquet")
    row = s[(s.drug == "DRUGZ") & (s.pt_key == "rash")].iloc[0]
    assert row.a == 15
    assert row.indication_overlap == pytest.approx(5 / 15)
    assert bool(row.possible_indication_bias)


def test_quality_report(pipeline_paths):
    qr = json.loads((pipeline_paths.processed / "quality.json").read_text())
    checks = {c["check"]: c for c in qr["checks"]}
    assert checks["Implausible age (outside 0-120 y) set to missing"]["value"] == 1
    assert checks["Probable duplicate reports (same clinical fingerprint, different CASEID)"]["value"] == 1
    assert qr["n_cases"] == 28
    assert {"pass", "warn", "info"} >= {c["status"] for c in qr["checks"]}


def test_ingest_rejects_missing_zip(tmp_path):
    from pvsys.config import Paths
    from pvsys.ingest import ingest

    with pytest.raises(FileNotFoundError):
        ingest(Paths(root=tmp_path), "2025Q1")


def test_extract_rejects_path_traversal(tmp_path):
    import zipfile

    from pvsys.ingest import extract

    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError):
        extract(z, tmp_path / "out")
