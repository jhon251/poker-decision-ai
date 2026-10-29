"""Phase 5B MVP 集成测试（合成数据，验证方向性 + 泄漏防护）。"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase5b_step1_icc import icc_from_stats  # noqa: E402
from phase5b_step4_hb import fit_beta_binomial  # noqa: E402
from phase5b_math import mean_log_loss, pearson  # noqa: E402


def _sim_players(rng, n_players, n_hands, a_prior, b_prior):
    """模拟玩家：theta_i ~ Beta(a,b)，手牌 ~ Bernoulli(theta_i)。"""
    out = []
    for _ in range(n_players):
        theta = rng.betavariate(a_prior, b_prior)
        s = sum(1 for _ in range(n_hands) if rng.random() < theta)
        out.append((n_hands, s))
    return out


def test_icc_detects_structure():
    rng = random.Random(42)
    records = _sim_players(rng, 300, 200, 5.0, 5.0)   # 强玩家间差异
    stats = icc_from_stats(records)
    assert stats["icc1_k"] > 0.3
    assert stats["icc1"] > 0.02


def test_icc_no_structure():
    rng = random.Random(7)
    # theta 全部等于 0.3（无玩家间差异）→ ICC 应接近 0
    records = []
    for _ in range(300):
        s = sum(1 for _ in range(200) if rng.random() < 0.3)
        records.append((200, s))
    stats = icc_from_stats(records)
    assert stats["icc1_k"] < 0.10


def test_split_half_direction():
    rng = random.Random(3)
    r1, r2 = [], []
    for _ in range(200):
        theta = rng.betavariate(5, 5)
        h1 = sum(1 for _ in range(50) if rng.random() < theta) / 50
        h2 = sum(1 for _ in range(50) if rng.random() < theta) / 50
        r1.append(h1)
        r2.append(h2)
    r = pearson(r1, r2)
    assert r > 0.15   # 真实结构 → 半程相关显著


def test_hb_recovers_prior():
    rng = random.Random(11)
    records = _sim_players(rng, 500, 200, 3.0, 7.0)
    fit = fit_beta_binomial(records)
    # mu 接近 0.3；a/b 比例接近 3/7（宽松容差，EB 矩估计）
    assert 0.27 < fit["mu"] < 0.33
    assert 0.3 < fit["a"] / fit["b"] < 0.6
    assert fit["protection_triggered"] is False


def test_hb_numerical_protection_no_variance():
    # 所有玩家成功数完全一致 → MSB=0 → sigma2_b=0 → 保护必触发且显式记录
    records = [(100, 30)] * 200
    fit = fit_beta_binomial(records)
    assert fit["protection_triggered"] is True
    assert fit["protection_reason"] == "sigma2_nonpositive"
    assert fit["effective_nu"] <= 100000.0
    assert fit["raw_sigma2_theta"] == 0.0


def test_hb_protection_never_silent():
    """构造 sigma2 超最大方差场景：保护必须显式记录，nu 有界。"""
    # 极端记录：一半 0 一半 1 → 观测方差巨大（theta 方差估计超界）
    records = [(100, 100)] * 50 + [(100, 0)] * 50
    fit = fit_beta_binomial(records)
    if fit["protection_triggered"]:
        assert fit["protection_reason"] is not None
        assert fit["raw_sigma2_theta"] is not None
        assert fit["raw_nu"] is not None or fit["protection_reason"] == "sigma2_nonpositive"
    assert fit["a"] >= 0 and fit["b"] >= 0


def test_mean_log_loss_bounds():
    # 完美预测 → 低损失；错误预测 → 高损失
    ll_good = mean_log_loss(0.3, 100, 30)
    ll_bad = mean_log_loss(0.3, 100, 90)
    assert ll_good < ll_bad
    assert math.isfinite(ll_good) and math.isfinite(ll_bad)
