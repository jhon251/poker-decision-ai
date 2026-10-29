"""Phase 6B 测试：schema 完整性、tier 规则、reliability 规则、priors 数值保护。"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase6b_schema import FEATURE_SCHEMA  # noqa: E402
from phase6b_tiers import reliability_rule, tier_rule  # noqa: E402


def test_schema_17_features():
    assert len(FEATURE_SCHEMA) == 17
    names = [f["name"] for f in FEATURE_SCHEMA]
    assert len(set(names)) == 17, "feature 名重复"


def test_schema_alias_valid():
    by_name = {f["name"]: f for f in FEATURE_SCHEMA}
    for f in FEATURE_SCHEMA:
        if f["alias_of"]:
            assert f["alias_of"] in by_name
        if f["derived_from"]:
            targets = f["derived_from"] if isinstance(f["derived_from"], list) \
                else [f["derived_from"]]
            for t in targets:
                assert t in by_name


def test_ep_open_alias_of_rfi_ep():
    ep = next(f for f in FEATURE_SCHEMA if f["name"] == "ep_open")
    assert ep["alias_of"] == "rfi_ep"
    assert ep["category"] == "alias"


def test_ps_derived_from_components():
    ps = next(f for f in FEATURE_SCHEMA if f["name"] == "position_sensitivity")
    assert ps["category"] == "derived"
    assert set(ps["derived_from"]) == {"rfi_ep", "rfi_lp"}


# ---------------- tier 规则（合成证据） ----------------
def test_tier_outcome_to_c():
    feat = {"name": "won_hand", "category": "outcome", "conditional": False,
            "alias_of": None}
    tier, path = tier_rule(feat, {"rate_global": 0.5, "median_success": 10,
                                  "median_opp": 100}, {}, set(), set())
    assert tier == "C" and path == "category=outcome"


def test_tier_mvp_core_to_a():
    feat = {"name": "vpip", "category": "behavioral", "conditional": False,
            "alias_of": None}
    tier, _ = tier_rule(feat, {}, {}, {"vpip"}, set())
    assert tier == "A"


def test_tier_rare_rate_to_c():
    feat = {"name": "four_bet", "category": "behavioral", "conditional": False,
            "alias_of": None}
    tier, _ = tier_rule(feat, {"rate_global": 0.0004, "median_success": 0,
                               "median_opp": 154}, {}, set(), set())
    assert tier == "C"


def test_tier_conditional_thin_to_c():
    feat = {"name": "btn_steal", "category": "behavioral", "conditional": True,
            "alias_of": None}
    tier, _ = tier_rule(feat, {"rate_global": 0.3, "median_success": 3,
                               "median_opp": 10}, {}, set(), set())
    assert tier == "C"


def test_tier_default_to_b():
    feat = {"name": "fold_to_cbet", "category": "behavioral",
            "conditional": False, "alias_of": None}
    tier, _ = tier_rule(feat, {"rate_global": 0.5, "median_success": 5,
                               "median_opp": 10}, {}, set(), set())
    assert tier == "B"


def test_tier_high_evidence_to_a():
    feat = {"name": "fold_to_3bet", "category": "behavioral",
            "conditional": False, "alias_of": None}
    tier, _ = tier_rule(feat, {"rate_global": 0.86, "median_success": 63,
                               "median_opp": 85},
                        {"fold_to_3bet": {"icc1_k": 0.975}}, set(), set())
    assert tier == "A"


def test_tier_alias_follows_target():
    feat = {"name": "ep_open", "category": "alias", "conditional": True,
            "alias_of": "rfi_ep"}
    tier, path = tier_rule(feat, {"rate_global": 0.13, "median_success": 5,
                                  "median_opp": 44}, {}, set(), set())
    # alias 跟随目标（第一遍由缓存决定，这里只验证走 alias 分支）
    assert path == "alias_of"


# ---------------- reliability 规则（samples + shrinkage + CI 宽度） ----------------
def test_reliability_reliable():
    assert reliability_rule(samples=100, shrinkage=0.2, ci_width=0.10) == "reliable"


def test_reliability_marginal_ci_too_wide():
    # 样本与收缩合格但 CI 太宽 → marginal
    assert reliability_rule(samples=100, shrinkage=0.2, ci_width=0.5) == "marginal"


def test_reliability_marginal_low_samples():
    assert reliability_rule(samples=20, shrinkage=0.2, ci_width=0.10) == "marginal"


def test_reliability_unreliable():
    assert reliability_rule(samples=10, shrinkage=0.8, ci_width=0.5) == "unreliable"
    assert reliability_rule(samples=5, shrinkage=0.1, ci_width=0.1) == "unreliable"


# ---------------- priors 数值保护（合成） ----------------
def test_priors_protection_zero_variance():
    from phase5b_step4_hb import fit_beta_binomial
    records = [(100, 30)] * 200
    fit = fit_beta_binomial(records)
    assert fit["protection_triggered"] is True
    assert fit["protection_reason"] == "sigma2_nonpositive"


def test_priors_no_protection_healthy():
    from phase5b_step4_hb import fit_beta_binomial
    rng = random.Random(7)
    records = []
    for _ in range(500):
        theta = rng.betavariate(3, 7)
        s = sum(1 for _ in range(200) if rng.random() < theta)
        records.append((200, s))
    fit = fit_beta_binomial(records)
    assert fit["protection_triggered"] is False
    assert 0.27 < fit["mu"] < 0.33


# ---------------- 产物一致性 ----------------
def test_tier_output_file_has_17_features():
    d = json.load(open("data/uoft/models/phase6/tier_reliability_rules.json",
                       encoding="utf-8"))
    assert len(d["features"]) == 17
    for name, info in d["features"].items():
        assert info["tier"] in ("A", "B", "C")
        assert "rule_path" in info


def test_tier_output_c_non_consumable():
    d = json.load(open("data/uoft/models/phase6/tier_reliability_rules.json",
                       encoding="utf-8"))
    for name, info in d["features"].items():
        if info["tier"] == "C":
            assert info["consumable"] is False
        if name == "ep_open":    # alias 永不消费
            assert info["consumable"] is False


def test_priors_output_integrity():
    d = json.load(open("data/uoft/models/phase6/population_priors.json",
                       encoding="utf-8"))
    for k, fit in d["global"].items():
        assert fit["a"] > 0 and fit["b"] > 0
        assert 0.0 < fit["mu"] < 1.0
        assert "protection_triggered" in fit
        assert "protection_reason" in fit
