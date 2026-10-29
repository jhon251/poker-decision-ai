"""Phase 5B 数学工具单测（betainc/beta_quantile/beta_binomial/pearson/bootstrap）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase5b_math import (  # noqa: E402
    beta_binomial_interval, beta_quantile, betainc, bootstrap_pct,
    pearson, spearman_brown, wilson_interval)


def test_betainc_uniform():
    assert betainc(1, 1, 0.3) == pytest.approx(0.3, abs=1e-9)
    assert betainc(1, 1, 0.0) == 0.0
    assert betainc(1, 1, 1.0) == 1.0


def test_betainc_symmetry():
    # I_x(a,b) + I_{1-x}(b,a) = 1
    assert betainc(3, 7, 0.3) + betainc(7, 3, 0.7) == pytest.approx(1.0, abs=1e-9)


def test_beta_quantile_uniform():
    assert beta_quantile(1, 1, 0.025) == pytest.approx(0.025, abs=1e-9)
    assert beta_quantile(1, 1, 0.975) == pytest.approx(0.975, abs=1e-9)


def test_beta_quantile_symmetric_prior():
    # Beta(5,5) 中位数 = 0.5
    assert beta_quantile(5, 5, 0.5) == pytest.approx(0.5, abs=1e-9)


def test_beta_quantile_known_prior():
    # Beta(3.02, 7.71): 中位数 ~0.267
    med = beta_quantile(3.02, 7.71, 0.5)
    assert 0.24 < med < 0.29
    q025 = beta_quantile(3.02, 7.71, 0.025)
    q975 = beta_quantile(3.02, 7.71, 0.975)
    assert q025 < med < q975
    assert q025 > 0.05 and q975 < 0.60


def test_beta_binomial_interval_contains_center():
    # Beta-Binomial(n=100, a=3, b=7): 95% 区间应包含均值附近
    lo, hi = beta_binomial_interval(3, 7, 100)
    mean = 100 * 3 / 10
    assert lo <= round(mean) <= hi
    assert lo >= 0 and hi <= 100


def test_beta_binomial_uniform_prior():
    # 先验 Uniform(1,1): Beta-Binomial 是 {0..100} 离散均匀，95% 中央区间 ≈ [2, 98]
    lo, hi = beta_binomial_interval(1, 1, 100)
    assert lo == 2
    assert hi == 98


def test_wilson_monotone():
    lo1, hi1 = wilson_interval(10, 20)
    lo2, hi2 = wilson_interval(100, 200)
    assert (hi1 - lo1) > (hi2 - lo2)   # 样本越大区间越窄


def test_pearson():
    assert pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 2, 3], [1, 1, 1]) == 0.0


def test_spearman_brown():
    assert spearman_brown(0.5) == pytest.approx(2 * 0.5 / 1.5)
    assert spearman_brown(-0.2) == -0.2
    assert spearman_brown(1.0) == 1.0


def test_bootstrap_reproducible():
    r1 = bootstrap_pct(lambda xs: sum(xs) / len(xs), list(range(20)),
                       n_resample=200, seed=7)
    r2 = bootstrap_pct(lambda xs: sum(xs) / len(xs), list(range(20)),
                       n_resample=200, seed=7)
    assert r1[0] == r2[0] and r1[1] == r2[1]
    lo, hi, _ = r1
    assert lo <= hi
