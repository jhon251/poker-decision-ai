"""Phase 7A Exploit Advisor 决策层测试：§10.4 全部硬不变量 + 合成输入样例。

覆盖：顾问模式恒 0 / gate 失败零建议 / 确定性 / 上下文黑白名单 /
      as_of-ts 边界 / train-only provenance / 未知玩家与 T4 / zero_opportunity /
      time_sensitive 透传 / 白名单 / 配置红线 / 强度假设臂 / cbet n_eff / 源码红线。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import phase7a_decision as p7  # noqa: E402
from phase7a_decision import DECISION_FEATURES, decide  # noqa: E402


# ---------------- 合成输入构造 ----------------
def _ts(i: int) -> str:
    # 固定宽度计数器（字符串序 = 时间序，保证升序）
    return f"2009-07-05T00:00:{i:06d}"


def mk_priors() -> dict:
    specs = {"vpip": 0.25, "pfr": 0.15, "three_bet": 0.05, "cbet": 0.65}
    out = {}
    for f, mu in specs.items():
        out[f] = {"a": mu * 20.0, "b": (1.0 - mu) * 20.0,
                  "mu": mu, "provenance": "hb_fit"}
    out["fold_to_3bet"] = {"a": 6.0, "b": 4.0, "mu": 0.60,
                           "provenance": "scheme_b"}
    return out


def mk_events_ft3b(S: int, n: int, total: int | None = None) -> list:
    """fold_to_3bet 机会门控事件：n 个 faced_3bet，其中前 S 个 folded_to_3bet。"""
    total = n if total is None else total
    return [{"ts": _ts(i), "faced_3bet": i < n,
             "folded_to_3bet": i < S} for i in range(total)]


def mk_events_full(feature: str, S: int, n: int) -> list:
    """全手分母特征（vpip/pfr/three_bet）：n 手，前 S 手该行为发生。"""
    assert feature in ("vpip", "pfr", "three_bet")
    return [{"ts": _ts(i), feature: i < S} for i in range(n)]


def mk_input(feature: str, events: list, **over) -> dict:
    priors = mk_priors()
    ctx = ({"faced_3bet": True, "position": "LP"} if feature == "fold_to_3bet"
           else {"position": "LP"})
    inp = {
        "opponent_id": "test-opp",
        "as_of": "2009-07-06T00:00:00",
        "history": events,
        "priors": priors,
        "population_mu": {f: p["mu"] for f, p in priors.items()},
        "spot": {"feature": feature, "trigger": True, "context": ctx},
        "baseline_decision": {"action": "check", "size": None},
        "config": {"p7_effect_size": 0.02, "strength_arm": "k=0",
                   "cluster": "disabled", "time_decay": "none"},
    }
    inp.update(over)
    return inp


# ---------------- 顾问模式红字（§10.4-1） ----------------
def test_advisor_mode_redlines_strong_case():
    out = decide(mk_input("fold_to_3bet", mk_events_ft3b(S=680, n=800)))
    assert out["exploit_status"] == "strong"
    assert out["adjustment_applied"] is False
    assert out["meta"]["ev_claim"] is False
    assert out["meta"]["advisor_mode"] is True
    assert out["meta"]["cluster_probs"] is None
    cand = out["candidates"][0]
    assert cand["strength_applied"] == 0.0
    assert cand["strength_recommended"] == 0.0   # k=0 臂恒 0
    assert cand["recommended_adjustment"]["value"] is None


def test_strong_exploit_fold_to_3bet_over():
    out = decide(mk_input("fold_to_3bet", mk_events_ft3b(S=680, n=800)))
    assert out["exploit_status"] == "strong"
    assert out["fallback_level"] is None
    assert out["degradation"] == []
    cand = out["candidates"][0]
    assert cand["direction"] == "over"
    assert cand["hypothesis_id"] == "H-ft3b-over"
    assert cand["effect_gate_pass"] is True
    assert cand["confidence"] >= 0.95
    feat = out["profile_image"]["features"]["fold_to_3bet"]
    assert feat["consumable"] is True
    assert feat["reliability"] == "reliable"
    assert feat["samples"] == 800


def test_weak_exploit_fold_to_3bet_over():
    out = decide(mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625)))
    assert out["exploit_status"] == "weak"
    cand = out["candidates"][0]
    assert cand["direction"] == "over"
    assert 0.5 < cand["confidence"] < 0.95
    assert cand["strength_recommended"] == 0.0
    assert cand["strength_applied"] == 0.0


def test_strong_exploit_fold_to_3bet_under():
    out = decide(mk_input("fold_to_3bet", mk_events_ft3b(S=45, n=100)))
    assert out["exploit_status"] == "strong"
    cand = out["candidates"][0]
    assert cand["direction"] == "under"
    assert cand["hypothesis_id"] == "H-ft3b-under"
    assert cand["confidence"] >= 0.95


def test_vpip_over_hypothesis():
    out = decide(mk_input("vpip", mk_events_full("vpip", S=180, n=400)))
    assert out["exploit_status"] == "strong"
    cand = out["candidates"][0]
    assert cand["direction"] == "over"
    assert cand["hypothesis_id"] == "H-vpip-over"
    assert out["tier"] == "T1"
    feat = out["profile_image"]["features"]["vpip"]
    assert feat["samples"] == 400
    assert feat["consumable"] is True


# ---------------- gate 失败 → 零建议 + confidence null（§10.4-2） ----------------
def test_effect_gate_fail_confidence_null():
    out = decide(mk_input("fold_to_3bet", mk_events_ft3b(S=4090, n=6650)))
    assert out["exploit_status"] == "none"
    assert out["fallback_level"] == "baseline"
    cand = out["candidates"][0]
    assert cand["effect_gate_pass"] is False
    assert cand["confidence"] is None
    assert cand["strength_recommended"] == 0.0
    assert cand["hypothesis_id"] == "H-ft3b-over"   # 仅标注映射假设（§10.3 示例）
    assert cand["recommended_adjustment"]["value"] is None
    assert any(d["path"] == "effect_magnitude" for d in out["degradation"])


def test_direction_tail_fail():
    out = decide(mk_input("fold_to_3bet", mk_events_ft3b(S=54, n=100)))
    assert out["exploit_status"] == "none"
    cand = out["candidates"][0]
    assert cand["direction"] == "under"
    assert cand["confidence"] is None
    assert any(d["path"] == "direction_tail" for d in out["degradation"])


def test_unreliable_not_consumable():
    out = decide(mk_input("fold_to_3bet", mk_events_ft3b(S=8, n=10)))
    assert out["exploit_status"] == "none"
    feat = out["profile_image"]["features"]["fold_to_3bet"]
    assert feat["reliability"] == "unreliable"
    assert feat["consumable"] is False
    assert any(d["path"] == "consumable" for d in out["degradation"])


def test_trigger_false_no_candidate():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=680, n=800))
    inp["spot"]["trigger"] = False
    out = decide(inp)
    assert out["exploit_status"] == "none"
    assert out["candidates"] == []
    assert out["fallback_level"] == "baseline"
    assert any(d["path"] == "spot_trigger" for d in out["degradation"])


# ---------------- 未知玩家 / T4 / zero_opportunity（§10.4-7/8） ----------------
def test_unknown_player_population_only():
    out = decide(mk_input("fold_to_3bet", []))
    assert out["tier"] == "unknown"
    assert out["exploit_status"] == "none"
    assert out["fallback_level"] == "population_prior"
    assert out["candidates"] == []
    assert out["profile_image"]["features"] == {}
    assert any(d["path"] == "population_only" for d in out["degradation"])


def test_t4_population_only():
    out = decide(mk_input("vpip", mk_events_full("vpip", S=3, n=5)))
    assert out["tier"] == "T4"
    assert out["fallback_level"] == "population_prior"
    assert out["candidates"] == []
    assert out["profile_image"]["features"] == {}


def test_zero_opportunity_population_prior_fallback():
    events = [{"ts": _ts(i), "vpip": i < 30} for i in range(100)]
    out = decide(mk_input("fold_to_3bet", events))
    assert out["exploit_status"] == "none"
    assert out["fallback_level"] == "population_prior"
    cand = out["candidates"][0]
    assert cand["confidence"] is None
    assert cand["direction"] == "none"
    assert cand["strength_recommended"] == 0.0
    feat = out["profile_image"]["features"]["fold_to_3bet"]
    assert feat["zero_opportunity"] is True
    assert feat["samples"] == 0
    assert feat["shrinkage"] == 1.0
    assert feat["value"] == pytest.approx(0.6)   # 值=先验均值，不猜测
    assert feat["consumable"] is False
    assert any(d["path"] == "zero_opportunity" for d in out["degradation"])


# ---------------- cbet n_eff（§8-11，6.5 冻结口径） ----------------
def test_cbet_n_eff_quirk():
    events = [{"ts": _ts(i), "cbet_opportunity": False, "cbet": True}
              for i in range(5)]
    events += [{"ts": _ts(5 + i), "cbet_opportunity": False, "cbet": False}
               for i in range(100)]
    out = decide(mk_input("cbet", events))
    feat = out["profile_image"]["features"]["cbet"]
    assert feat["samples"] == 5          # n_eff = max(S=5, n=0)
    assert feat["zero_opportunity"] is False
    assert feat["consumable"] is False   # 5 < 30 → unreliable → 不消费
    assert out["exploit_status"] == "none"


# ---------------- 确定性（§10.4-3） ----------------
def test_determinism():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    a = json.dumps(decide(inp), sort_keys=True)
    b = json.dumps(decide(json.loads(json.dumps(inp))), sort_keys=True)
    assert a == b


# ---------------- as_of / ts 边界（§10.4-5，泄漏硬失败） ----------------
def test_future_in_history_hard_fail():
    events = mk_events_ft3b(S=10, n=20)
    events[5]["ts"] = "2009-07-07T00:00:00"   # 晚于 as_of
    events.sort(key=lambda e: e["ts"])
    with pytest.raises(AssertionError):
        decide(mk_input("fold_to_3bet", events))


def test_unsorted_history_hard_fail():
    events = mk_events_ft3b(S=10, n=20)
    events[0], events[1] = events[1], events[0]
    with pytest.raises(AssertionError):
        decide(mk_input("fold_to_3bet", events))


def test_missing_ts_hard_fail():
    events = [{"faced_3bet": True, "folded_to_3bet": True} for _ in range(20)]
    with pytest.raises(AssertionError):
        decide(mk_input("fold_to_3bet", events))


def test_hu_hand_in_history_hard_fail():
    events = mk_events_ft3b(S=10, n=20)
    events[0]["table_type"] = "hu"
    with pytest.raises(AssertionError):
        decide(mk_input("fold_to_3bet", events))


# ---------------- train-only prior provenance（§10.4-6，§8-2/§8-3） ----------------
def test_prior_provenance_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["priors"]["fold_to_3bet"]["provenance"] = "population_priors"
    with pytest.raises(AssertionError):
        decide(inp)


def test_prior_missing_provenance_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    del inp["priors"]["fold_to_3bet"]["provenance"]
    with pytest.raises(AssertionError):
        decide(inp)


def test_prior_mu_inconsistent_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["priors"]["fold_to_3bet"]["mu"] = 0.9
    with pytest.raises(AssertionError):
        decide(inp)


def test_prior_missing_feature_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    del inp["priors"]["vpip"]
    with pytest.raises(AssertionError):
        decide(inp)


def test_population_mu_mismatch_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["population_mu"]["vpip"] = 0.3
    with pytest.raises(AssertionError):
        decide(inp)


def test_population_mu_missing_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    del inp["population_mu"]["three_bet"]
    with pytest.raises(AssertionError):
        decide(inp)


# ---------------- 决策特征白名单（§10.4-4，§8-10） ----------------
@pytest.mark.parametrize("bad_feature", ["agg_freq", "position_sensitivity",
                                         "won_hand", "ep_open", "rfi_ep", "wtsd"])
def test_feature_whitelist_reject(bad_feature):
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["spot"]["feature"] = bad_feature
    with pytest.raises(ValueError):
        decide(inp)


def test_context_outcome_field_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["spot"]["context"]["folded_to_3bet"] = True
    with pytest.raises(AssertionError):
        decide(inp)


def test_context_outcome_field_won_hand_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["spot"]["context"]["won_hand"] = 1
    with pytest.raises(AssertionError):
        decide(inp)


def test_context_unknown_key_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["spot"]["context"]["equity"] = 0.5
    with pytest.raises(AssertionError):
        decide(inp)


# ---------------- 配置红线（cluster / time_decay / 臂 / δ） ----------------
def test_cluster_enabled_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["config"]["cluster"] = "enabled"
    with pytest.raises(AssertionError):
        decide(inp)


def test_time_decay_enabled_hard_fail():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["config"]["time_decay"] = "exponential"
    with pytest.raises(AssertionError):
        decide(inp)


def test_strength_arm_not_preregistered():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["config"]["strength_arm"] = "k=2"
    with pytest.raises(ValueError):
        decide(inp)


def test_effect_size_not_preregistered():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["config"]["p7_effect_size"] = 0.05
    with pytest.raises(ValueError):
        decide(inp)


# ---------------- baseline 透传不修改 ----------------
def test_baseline_passthrough_no_mutation():
    baseline = {"action": "raise", "size": 2.5}
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["baseline_decision"] = baseline
    out = decide(inp)
    assert out["baseline_decision"] == {"action": "raise", "size": 2.5}
    assert baseline == {"action": "raise", "size": 2.5}


# ---------------- strength 假设臂（§七：只作 recommendation strength） ----------------
def test_strength_hypothesis_arm_not_applied():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=680, n=800))
    inp["config"]["strength_arm"] = "k=0.5"
    out = decide(inp)
    cand = out["candidates"][0]
    c = cand["confidence"]
    m_mu = cand["deviation_magnitude"]
    assert cand["strength_recommended"] == pytest.approx(
        round(0.5 * c * m_mu * 1.0, 6))
    assert cand["strength_applied"] == 0.0          # 永不应用
    assert out["adjustment_applied"] is False
    assert any(w["code"] == "STRENGTH_HYPOTHESIS_ARM" for w in out["warnings"])
    assert out["meta"]["ev_claim"] is False


def test_strength_k0_arm_is_zero():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=680, n=800))
    out = decide(inp)
    assert out["candidates"][0]["strength_recommended"] == 0.0


# ---------------- time_sensitive 透传（§10.4-9，五） ----------------
def test_time_sensitive_passthrough_does_not_change_gating(monkeypatch):
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))

    monkeypatch.setattr(p7, "_time_cache",
                        {f: {"status": "stable"} for f in DECISION_FEATURES})
    out_stable = decide(json.loads(json.dumps(inp)))

    monkeypatch.setattr(p7, "_time_cache",
                        {f: {"status": "unstable"} for f in DECISION_FEATURES})
    out_unstable = decide(json.loads(json.dumps(inp)))

    for field in ("exploit_status", "fallback_level"):
        assert out_stable[field] == out_unstable[field]
    c_s, c_u = out_stable["candidates"][0], out_unstable["candidates"][0]
    assert c_s["confidence"] == c_u["confidence"]
    assert c_s["strength_recommended"] == c_u["strength_recommended"]
    feat_s = out_stable["profile_image"]["features"]["fold_to_3bet"]
    feat_u = out_unstable["profile_image"]["features"]["fold_to_3bet"]
    for key in ("value", "reliability", "consumable", "samples", "shrinkage"):
        assert feat_s[key] == feat_u[key]
    assert feat_s["time_sensitive"] is False
    assert feat_u["time_sensitive"] is True


def test_time_status_join_from_frozen_6e():
    out = decide(mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625)))
    feats = out["profile_image"]["features"]
    assert feats["fold_to_3bet"]["time_sensitive"] is True      # 6E: unstable
    assert feats["fold_to_3bet"]["time_status"] == "unstable"
    assert feats["cbet"]["time_sensitive"] is None              # 6E: insufficient
    assert feats["cbet"]["time_status"] == "insufficient_evidence"


# ---------------- 冻结 tier 规则消费 ----------------
def test_tier_from_frozen_rules_all_a():
    out = decide(mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625)))
    feats = out["profile_image"]["features"]
    assert all(feats[f]["tier"] == "A" for f in DECISION_FEATURES)


# ---------------- δ 敏感性口径（§5.8，预注册 0.0 仅报告） ----------------
def test_delta_zero_sensitivity_flagged():
    inp = mk_input("fold_to_3bet", mk_events_ft3b(S=403, n=625))
    inp["config"]["p7_effect_size"] = 0.0
    out = decide(inp)
    assert any(w["code"] == "EFFECT_SIZE_SENSITIVITY" for w in out["warnings"])
    assert out["meta"]["p7_effect_size"] == 0.0


# ---------------- 源码红线（§8-3/§8-8/§8-9；F3/F4） ----------------
def test_source_redlines():
    src = open(Path(p7.__file__), encoding="utf-8").read()
    assert "population_priors.json" not in src   # F3：全 pool 先验禁止
    assert "phase6g" not in src                  # 不导入 6G 接口
    assert "get_profile" not in src              # 静态画像禁止作输入
    assert "half_life" not in src                # H：无时间衰减
    assert "gmm" not in src                      # G：cluster 不重启
    assert "ev_gain" not in src                  # 无 EV 表述
    assert "agents" not in src                   # F4：不继承旧 exploit 数字
