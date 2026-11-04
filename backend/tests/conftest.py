"""Shared fixtures: a tiny synthetic FAERS quarter with hand-checkable answers.

Design of the synthetic data (quarter 2025Q1):
* case 1001 has two versions (primaryids 10011, 10012) -> only 10012 kept
* case 1002 has two versions with the same fda_dt -> highest primaryid kept
* case 1003 is listed in the DELETE file -> removed
* cases 2001 and 2002 are identical clinical fingerprints -> 2001 (older) flagged duplicate
* age codes cover YR, MON, DEC and an implausible value
* drug X ("DRUGX") is co-reported with reaction "Nausea" in 6 cases
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

DEMO_HEADER = ("primaryid$caseid$caseversion$i_f_code$event_dt$mfr_dt$init_fda_dt$fda_dt$rept_cod$auth_num$mfr_num$"
               "mfr_sndr$lit_ref$age$age_cod$age_grp$sex$e_sub$wt$wt_cod$rept_dt$to_mfr$occp_cod$reporter_country$"
               "occr_country")
DRUG_HEADER = ("primaryid$caseid$drug_seq$role_cod$drugname$prod_ai$val_vbm$route$dose_vbm$cum_dose_chr$cum_dose_unit$"
               "dechal$rechal$lot_num$exp_dt$nda_num$dose_amt$dose_unit$dose_form$dose_freq")


def demo_row(pid, cid, ver, fda_dt, age="", age_cod="", sex="F", event_dt="20250105", wt="", wt_cod="",
             occp="MD", country="US"):
    return (f"{pid}${cid}${ver}$I${event_dt}$$20250101${fda_dt}$EXP$$$MFR$$"
            f"{age}${age_cod}$${sex}$Y${wt}${wt_cod}${fda_dt}$$" f"{occp}${country}${country}")


def drug_row(pid, cid, seq, role, name, ai=""):
    return f"{pid}${cid}${seq}${role}${name}${ai}$1$Oral$$$$$$$$$$$$"


def build_rows():
    demo, drug, reac, outc, indi, ther, rpsr = [], [], [], [], [], [], []
    # versioned / deleted cases
    demo += [demo_row(10011, 1001, 1, "20250110", 50, "YR"), demo_row(10012, 1001, 2, "20250220", 50, "YR"),
             demo_row(10021, 1002, 1, "20250301", 18, "MON", sex="M"),
             demo_row(10022, 1002, 2, "20250301", 18, "MON", sex="M"),
             demo_row(10031, 1003, 1, "20250115", 7, "DEC")]
    for pid, cid in ((10011, 1001), (10012, 1001), (10021, 1002), (10022, 1002), (10031, 1003)):
        drug.append(drug_row(pid, cid, 1, "PS", "DRUGX 10 MG", "DRUGX"))
        reac.append(f"{pid}${cid}$Nausea$")
    outc.append("10012$1001$HO")
    # duplicate fingerprints
    for pid, cid, fda in ((20011, 2001, "20250110"), (20021, 2002, "20250210")):
        demo.append(demo_row(pid, cid, 1, fda, 40, "YR", sex="F", event_dt="20250101"))
        drug.append(drug_row(pid, cid, 1, "PS", "DRUGY", "DRUGY"))
        reac.append(f"{pid}${cid}$Headache$")
    # background cases: drug X + nausea (4 more), drug Z + rash, implausible age
    for i in range(4):
        pid, cid = 30000 + i * 10 + 1, 3000 + i
        demo.append(demo_row(pid, cid, 1, "20250201", 30 + i, "YR", sex="M", event_dt=f"2025010{i + 1}"))
        drug.append(drug_row(pid, cid, 1, "PS", "DrugX", "DRUGX"))
        drug.append(drug_row(pid, cid, 2, "C", "ASPIRIN", "ASPIRIN"))
        reac.append(f"{pid}${cid}$Nausea$")
        reac.append(f"{pid}${cid}$nausea$")  # repeated term, different case -> collapsed
    for i in range(20):
        pid, cid = 40000 + i * 10 + 1, 4000 + i
        age, cod = (300, "YR") if i == 0 else (25 + i, "YR")
        demo.append(demo_row(pid, cid, 1, "20250205", age, cod, sex="F" if i % 2 else "M", event_dt="",
                             wt=150 if i == 1 else "", wt_cod="LBS" if i == 1 else ""))
        drug.append(drug_row(pid, cid, 1, "PS", "DRUGZ", "DRUGZ"))
        reac.append(f"{pid}${cid}${'Rash' if i < 15 else 'Nausea'}$")
        if i < 3:
            outc.append(f"{pid}${cid}$DE")
        indi.append(f"{pid}${cid}$1$Psoriasis")
    # indication = reaction for DRUGZ / Rash in a few cases (indication-bias flag)
    for i in range(5):
        pid, cid = 40000 + i * 10 + 1, 4000 + i
        indi.append(f"{pid}${cid}$1$Rash")
    ther.append("30001$3000$1$20250101$20250102$1$DAY")
    rpsr.append("30001$3000$HP")
    return demo, drug, reac, outc, indi, ther, rpsr


@pytest.fixture(scope="session")
def synthetic_zip(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("faers_src")
    demo, drug, reac, outc, indi, ther, rpsr = build_rows()
    files = {
        "ASCII/DEMO25Q1.txt": [DEMO_HEADER] + demo,
        "ASCII/DRUG25Q1.txt": [DRUG_HEADER] + drug,
        "ASCII/REAC25Q1.txt": ["primaryid$caseid$pt$drug_rec_act"] + reac,
        "ASCII/OUTC25Q1.txt": ["primaryid$caseid$outc_cod"] + outc,
        "ASCII/INDI25Q1.txt": ["primaryid$caseid$indi_drug_seq$indi_pt"] + indi,
        "ASCII/THER25Q1.txt": ["primaryid$caseid$dsg_drug_seq$start_dt$end_dt$dur$dur_cod"] + ther,
        "ASCII/RPSR25Q1.txt": ["primaryid$caseid$rpsr_cod"] + rpsr,
        "Deleted/DELETE25Q1.txt": ["", "1003"],
    }
    z = root / "faers_ascii_2025q1.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, lines in files.items():
            zf.writestr(name, "\n".join(lines) + "\n")
    return z


@pytest.fixture(scope="session")
def pipeline_paths(synthetic_zip, tmp_path_factory):
    """Run the statistical pipeline on the synthetic quarter in an isolated data dir."""
    import os

    data = tmp_path_factory.mktemp("data")
    (data / "raw").mkdir()
    (data / "raw" / "faers_ascii_2025q1.zip").write_bytes(synthetic_zip.read_bytes())
    old = os.environ.get("PV_DATA_DIR")
    os.environ["PV_DATA_DIR"] = str(data)
    os.environ["PV_SAPBERT_DIR"] = str(data / "no-model")  # force graceful NLP fallback in tests
    os.environ["PV_WARMUP"] = "0"  # no background model loading inside the test client
    os.environ["PV_GENERATOR_DIR"] = str(data / "no-generator")  # tests never load the real local models
    os.environ["PV_EMBED_DIR"] = str(data / "no-embedder")
    os.environ["PV_RERANKER_DIR"] = str(data / "no-reranker")
    os.environ["PV_NLI_DIR"] = str(data / "no-nli")
    from pvsys import clean, eda, ingest, quality, signals
    from pvsys.config import get_paths

    paths = get_paths().ensure()
    ingest.ingest(paths, "2025Q1")
    clean.clean(paths)
    quality.run(paths)
    signals.run(paths)
    eda.snapshot(paths)
    yield paths
    if old is None:
        os.environ.pop("PV_DATA_DIR", None)
    else:
        os.environ["PV_DATA_DIR"] = old
