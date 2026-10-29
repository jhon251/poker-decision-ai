"""Phase 6C 测试：分层画像（tier 边界/收缩/零机会/fallback/consumable 硬过滤/合成验证）。"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase6c_profiles import (  # noqa: E402
    consumable_for, features_for, pick_prior, posterior_feature, tier_of)
from phase6b_tiers import reliability_rule  # noqa: E402


# ---------------- tier 边界 ----------------
@pytest.mark.parametrize("n,expected", [
    (199, "T2"), (200, "T1"), (49, "T3"), (50, "T2"), (9, "T4"), (10, "T3")])
def test_tier_boundaries(n, expected):
    assert tier_of(n) == expected


def test_tier_feature_sets():
    schema = json.load(open("data/uoft/models/phase6/feature_schema.json",
                            encoding="utf-8"))["features"]
    names = [f["name"] for f in features_for("T1", schema)]
    assert len(names) == 17
    t2 = [f["name"] for f in features_for("T2", schema)]
    assert set(t2) == {"vpip", "pfr", "three_bet", "cbet",
                       "position_sensitivity", "fold_to_3bet"}
    t3 = [f["name"] for f in features_for("T3", schema)]
    assert set(t3) == {"vpip", "pfr", "three_bet"}


# ---------------- 后验收缩 ----------------
def test_posterior_shrinks_with_samples():
    prior = {"a": 3.0, "b": 7.0, "mu": 0.3, "protection_triggered": False}
    p_small = posterior_feature(5, 10, prior)
    p_large = posterior_feature(100, 200, prior)
    # 样本越多 shrinkage 越小
    assert p_large["shrinkage"] < p_small["shrinkage"]


def test_posterior_manual_computation():
    prior = {"a": 3.0, "b": 7.0, "mu": 0.3, "protection_triggered": False}
    S, n = 40, 100
    p = posterior_feature(S, n, prior)
    ap, bp = 3.0 + S, 7.0 + n - S
    assert p["value"] == pytest.approx(ap / (ap + bp), abs=1e-6)
    assert p["shrinkage"] == pytest.approx(10.0 / (10.0 + n), abs=1e-6)
    # CI 含均值且在 [0,1]
    assert p["ci95"][0] <= p["value"] <= p["ci95"][1]
    assert 0.0 <= p["ci95"][0] and p["ci95"][1] <= 1.0


def test_posterior_ci_narrows_with_samples():
    prior = {"a": 3.0, "b": 7.0, "mu": 0.3, "protection_triggered": False}
    w_small = posterior_feature(5, 10, prior)["ci_width"]
    w_large = posterior_feature(100, 200, prior)["ci_width"]
    assert w_large < w_small


# ---------------- 零机会 ----------------
def test_zero_opportunity_not_interpreted_as_never():
    prior = {"a": 3.0, "b": 7.0, "mu": 0.3, "protection_triggered": False}
    p = posterior_feature(0, 0, prior)
    assert p["zero_opportunity"] is True
    # 值 = 先验均值（不是 0），CI = 先验区间，shrinkage=1
    assert p["value"] == pytest.approx(0.3, abs=1e-6)
    assert p["shrinkage"] == 1.0
    assert p["samples"] == 0


# ---------------- prior fallback ----------------
def test_prior_pick_strata_no_fallback():
    priors = {
        "global": {"vpip": {"a": 3.0, "b": 7.0, "mu": 0.3,
                            "fallback_to_global": False}},
        "strata": {"PokerStars|low": {"vpip": {"a": 5.0, "b": 5.0, "mu": 0.5,
                                               "fallback_to_global": False}}},
    }
    fit, src, fb = pick_prior("vpip", "PokerStars", "low", priors)
    assert fit["a"] == 5.0 and fb is False and src == "PokerStars|low"


def test_prior_pick_fallback_to_global():
    priors = {
        "global": {"vpip": {"a": 3.0, "b": 7.0, "mu": 0.3,
                            "fallback_to_global": False}},
        "strata": {"SmallVenue|high": {"vpip": {"a": 3.0, "b": 7.0, "mu": 0.3,
                                                "fallback_to_global": True}}},
    }
    fit, src, fb = pick_prior("vpip", "SmallVenue", "high", priors)
    assert fb is True


# ---------------- consumable 硬过滤 ----------------
def test_consumable_tier_c_always_false():
    feat = {"name": "won_hand", "category": "outcome", "alias_of": None}
    assert consumable_for(feat, "C", "reliable", 0.1) is False


def test_consumable_alias_always_false():
    feat = {"name": "ep_open", "category": "alias", "alias_of": "rfi_ep"}
    # 即使 tier B + reliable + shrinkage<0.5 也不放行
    assert consumable_for(feat, "B", "reliable", 0.1) is False
    assert consumable_for(feat, "A", "reliable", 0.1) is False


def test_consumable_tier_a_reliable_true():
    feat = {"name": "vpip", "category": "behavioral", "alias_of": None}
    assert consumable_for(feat, "A", "reliable", 0.1) is True
    assert consumable_for(feat, "A", "unreliable", 0.1) is False


def test_consumable_tier_b_needs_shrinkage():
    feat = {"name": "wtsd", "category": "behavioral", "alias_of": None}
    assert consumable_for(feat, "B", "reliable", 0.3) is True
    assert consumable_for(feat, "B", "reliable", 0.6) is False


def test_tier_a_still_needs_reliability():
    """agg_freq/cold_call 是 Tier A 但必须经过 reliability 才能 consumable。"""
    feat = {"name": "agg_freq", "category": "behavioral", "alias_of": None}
    assert consumable_for(feat, "A", "unreliable", 0.1) is False
    assert consumable_for(feat, "A", "marginal", 0.1) is False
    assert consumable_for(feat, "A", "reliable", 0.1) is True


# ---------------- 输出完整性 ----------------
def test_profiles_schema_complete():
    n = 0
    for line in open("data/uoft/models/phase6/profiles.jsonl", encoding="utf-8"):
        p = json.loads(line)
        n += 1
        assert p["tier"] in ("T1", "T2", "T3")
        for name, f in p["features"].items():
            for k in ("value", "ci95", "samples", "shrinkage", "reliability",
                      "tier", "consumable", "zero_opportunity"):
                assert k in f, f"{name} missing {k}"
            assert f["ci95"][0] <= f["ci95"][1]
            if name == "position_sensitivity":
                # 差量特征 CI 允许为负（值域 [-1,1]）
                assert -1.0 <= f["ci95"][0] and f["ci95"][1] <= 1.0
            else:
                assert 0.0 <= f["ci95"][0] and f["ci95"][1] <= 1.0
            assert f["tier"] in ("A", "B", "C")
        assert n <= 30000, "仅抽样验证前 3 万行（全量 24 万行）"
        if n >= 30000:
            break
    assert n > 0


def test_profiles_hard_filters_on_output():
    """抽样验证：ep_open/Tier C 在输出中 consumable 恒 False。"""
    n = 0
    for line in open("data/uoft/models/phase6/profiles.jsonl", encoding="utf-8"):
        p = json.loads(line)
        n += 1
        for name, f in p["features"].items():
            if name == "ep_open":
                assert f["consumable"] is False
            if f["tier"] == "C":
                assert f["consumable"] is False
            if f["consumable"] is True:
                assert f["tier"] in ("A", "B")
                assert f["reliability"] == "reliable"
                if f["tier"] == "B":
                    assert f["shrinkage"] < 0.5
        if n >= 50000:
            break


def test_profiles_tier_counts():
    s = json.load(open("data/uoft/models/phase6/profiles_summary.json",
                       encoding="utf-8"))
    assert s["n_by_tier"]["T1"] > 60000
    assert s["n_by_tier"]["T2"] > 70000
    assert s["n_by_tier"]["T3"] > 70000
    assert s["n_by_tier"]["T4"] == 0    # T4 无个体行


def test_no_future_data_dependency():
    """后验计算只依赖 (S,n,prior)，不读取任何时间字段（接口无 ts 参数）。"""
    import inspect
    sig = inspect.signature(posterior_feature)
    assert set(sig.parameters) == {"S", "n", "prior"}
