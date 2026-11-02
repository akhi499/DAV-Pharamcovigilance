"""Unit tests of the disproportionality maths against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pvsys.config import SignalConfig
from pvsys.signals import benjamini_hochberg, disproportionality


def test_ror_prr_textbook_example():
    # a=20, b=80, c=100, d=9800
    r = disproportionality([20], [80], [100], [9800]).iloc[0]
    ror = (20 / 80) / (100 / 9800)
    assert r.ror == pytest.approx(ror)
    se = math.sqrt(1 / 20 + 1 / 80 + 1 / 100 + 1 / 9800)
    assert r.ror_lower == pytest.approx(math.exp(math.log(ror) - 1.959963984540054 * se))
    assert r.ror_upper == pytest.approx(math.exp(math.log(ror) + 1.959963984540054 * se))
    prr = (20 / 100) / (100 / 9900)
    assert r.prr == pytest.approx(prr)
    se_p = math.sqrt(1 / 20 - 1 / 100 + 1 / 100 - 1 / 9900)
    assert r.prr_lower == pytest.approx(math.exp(math.log(prr) - 1.959963984540054 * se_p))
    n = 10000
    chi2 = n * (abs(20 * 9800 - 80 * 100) - n / 2) ** 2 / (100 * 9900 * 120 * 9880)
    assert r.chi2 == pytest.approx(chi2)
    assert not r.haldane_applied
    assert r.ror_signal and r.prr_signal and r.ic_signal and r.methods_flagging == 3


def test_haldane_only_when_zero_cell():
    r = disproportionality([5, 5], [0, 10], [10, 10], [100, 100])
    assert list(r.haldane_applied) == [True, False]
    assert np.isfinite(r.ror.iloc[0]) and np.isfinite(r.ror_upper.iloc[0])
    assert r.ror.iloc[0] == pytest.approx((5.5 / 0.5) / (10.5 / 100.5))


def test_min_case_rule_blocks_signal():
    r = disproportionality([2], [1], [10], [10000]).iloc[0]
    assert r.ror > 1 and not r.ror_signal and not r.prr_signal


def test_null_association_not_flagged():
    r = disproportionality([10], [90], [100], [900]).iloc[0]  # identical proportions
    assert r.ror == pytest.approx(1.0)
    assert r.prr == pytest.approx(1.0)
    assert r.methods_flagging == 0


def test_negative_counts_rejected():
    with pytest.raises(ValueError):
        disproportionality([-1], [1], [1], [1])


def test_custom_thresholds():
    cfg = SignalConfig(min_cases=10)
    r = disproportionality([5], [10], [10], [1000], cfg).iloc[0]
    assert not r.ror_signal


def test_bh_qvalues_monotone_and_bounded():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    q = benjamini_hochberg(p)
    assert np.all((q >= p) & (q <= 1))
    assert q[0] == pytest.approx(0.04)  # 0.01 * 4 / 1 = 0.04
    assert benjamini_hochberg(np.array([])).size == 0
