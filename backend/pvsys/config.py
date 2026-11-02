"""Central configuration: paths and analysis constants.

All paths can be overridden with environment variables so tests can run
against a temporary directory without touching real outputs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUARTER = os.environ.get("PV_QUARTER", "2025Q1")
FAERS_URL_TEMPLATE = "https://fis.fda.gov/content/Exports/faers_ascii_{q}.zip"

# FAERS ASCII tables used by the pipeline (FDA file prefixes).
FAERS_TABLES = ("DEMO", "DRUG", "REAC", "OUTC", "RPSR", "THER", "INDI")

OUTCOME_LABELS = {
    "DE": "Death",
    "LT": "Life-threatening",
    "HO": "Hospitalization",
    "DS": "Disability",
    "CA": "Congenital anomaly",
    "RI": "Required intervention",
    "OT": "Other serious",
}

REPORTER_LABELS = {
    "MD": "Physician",
    "PH": "Pharmacist",
    "OT": "Other health professional",
    "LW": "Lawyer",
    "CN": "Consumer",
    "HP": "Health professional",
}

ROLE_LABELS = {
    "PS": "Primary suspect",
    "SS": "Secondary suspect",
    "C": "Concomitant",
    "I": "Interacting",
}


@dataclass(frozen=True)
class Paths:
    root: Path = field(default_factory=lambda: Path(os.environ.get("PV_DATA_DIR", PROJECT_ROOT / "data")))

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def nlp(self) -> Path:
        return self.root / "nlp"

    @property
    def reviews(self) -> Path:
        return self.root / "reviews"

    @property
    def models(self) -> Path:
        return Path(os.environ.get("PV_MODEL_DIR", PROJECT_ROOT / "models"))

    @property
    def duckdb_file(self) -> Path:
        return self.processed / "faers.duckdb"

    def ensure(self) -> Paths:
        for p in (self.raw, self.processed, self.nlp, self.reviews):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    """Return paths, re-reading the environment (useful for tests)."""
    return Paths()


@dataclass(frozen=True)
class SignalConfig:
    """Thresholds for disproportionality screening.

    Defaults follow commonly cited screening rules:
    * PRR >= 2, chi-square (Yates) >= 4 and a >= 3  (Evans et al., 2001)
    * lower 95% CI of ROR > 1 and a >= 3            (van Puijenbroek et al., 2002)
    These are screening heuristics that flag hypotheses for review, not proof
    of causality.
    """

    min_cases: int = 3
    prr_threshold: float = 2.0
    chi2_threshold: float = 4.0
    ror_lower_ci_threshold: float = 1.0
    z: float = 1.959963984540054  # 95% two-sided
    continuity: float = 0.5  # Haldane-Anscombe correction applied when any cell is zero
