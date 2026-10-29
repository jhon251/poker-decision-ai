"""Phase 7C Recommendation Boundary Layer 测试。

覆盖：validated→eligible / insufficient→恒 null / gate fail→null /
      confidence null→null / unknown / T4 / zero opportunity / non-consumable /
      time_sensitive 不影响 eligibility / strength.applied≡0 / action≡null /
      H-* 不产生 action / cluster·time_decay·k≠0 硬失败 / 泄漏硬失败 /
      prior train-only / 禁静态画像 / 禁二次校准 / 禁 P2 语义。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import phase7a_decision as p7a  # noqa: E402
import phase7c_recommend as p7c  # noqa: E402
from phase7c_recommend import recommend  # noqa: E402

CHECK = Path("data/uoft/models/phase7/phase7c/contract_check.json")


# ---------------- 合成输入构造（与 7A 测试同构） ----------------
def _ts(i: int) -> str:
    return f"2009-07-05T00:00:{i:06d}"


def _priors() -> dict:
    specs = {"vpip": 0.25, "pfr": 0.15, "three_bet": 0.05, "cbet": 0.65}
    out = {}
    for f, mu in specs.items():
        out[f] = {"a": mu * 20.0, "b": (1.0 - mu) * 20.0,
                  "mu": mu, "provenance": "hb_fit"}
    out["fold_to_3bet"] = {"a": 6.0, "b": 4.0, "mu": 0.60,
                           "provenance": "scheme_b"}
    return out


def _ft3b(S: int, n: int, total: int | None = None) -> list:
    total = n if total is None else total
    return [{"ts": _ts(i), "faced_3bet": i < n,
             "folded_to_3bet": i < S} for i in range(total)]


def _full(feature: str, S: int, n: int) -> list:
    return [{"ts": _ts(i), feature: i < S} for i in range(n)]


def _cbet_ev(S: int, n: int, total: int) -> list:
    return [{"ts": _ts(i), "cbet_opportunity": i < n,
             "cbet": i < n and i < S} for i in range(total)]


def _input(feature: str, history: list, **over) -> dict:
    priors = _priors()
    ctx = ({"faced_3bet": True, "position": "LP"} if feature == "fold_to_3bet"
           else {"position": "LP"})
    inp = {
        "opponent_id": "t-opp", "as_of": "2009-07-06T00:00:00",
        "history": history, "priors": priors,
        "population_mu": {f: p["mu"] for f, p in priors.items()},
        "spot": {"feature": feature, "trigger": True, "context": ctx},
        "baseline_decision": {"action": "check", "size": None},
        "config": {"p7_effect_size": 0.02, "strength_arm": "k=0",
                   "cluster": "disabled", "time_decay": "none"},
    }
    inp.update(over)
    return inp


# ---------------- 1. validated → eligible ----------------
def test_validated_feature_eligible():
    out = recommend(_input("fold_to_3bet", _ft3b(S=680, n=800)))
    assert out["state"] == "eligible"
    rec = out["recommendation"]
    assert rec["direction"] == "over"
    assert rec["confidence"] is not None
    assert rec["evidence"]["p15_feature_status"] == "validated"
    assert rec["eligibility_checks"] == {
        "p15_validated": True, "five_gates": True,
        "confidence_not_null": True, "no_time_leakage_violation": True}


def test_eligible_for_all_three_validated_features():
    vpip = recommend(_input("vpip", _full("vpip", S=180, n=400)))
    pfr = recommend(_input("pfr", _full("pfr", S=160, n=400)))
    assert vpip["state"] == "eligible"
    assert pfr["state"] == "eligible"


# ---------------- 2. insufficient → 恒 null ----------------
@pytest.mark.parametrize("feature,events", [
    ("three_bet", _full("three_bet", S=120, n=400)),   # 强偏差也不得绕过
    ("cbet", _cbet_ev(S=120, n=200, total=800)),
])
def test_insufficient_features_always_null(feature, events):
    out = recommend(_input(feature, events))
    assert out["state"] == "feature_insufficient_evidence"
    assert out["recommendation"] is None
    assert "禁止" in out["reason"]


# ---------------- 3/4. gate 失败 / confidence null → null ----------------
def test_gate_failure_null():
    out = recommend(_input("fold_to_3bet", _ft3b(S=54, n=100)))   # tail 不显著
    assert out["recommendation"] is None
    assert out["state"] == "gate_failed"
    assert out["reason"]


def test_effect_gate_fail_confidence_null_recommendation_null():
    out = recommend(_input("fold_to_3bet", _ft3b(S=4090, n=6650)))
    assert out["recommendation"] is None
    assert out["state"] == "gate_failed"


# ---------------- 5/6. unknown / T4 ----------------
def test_unknown_player_population_only():
    out = recommend(_input("fold_to_3bet", []))
    assert out["state"] == "population_only"
    assert out["recommendation"] is None


def test_t4_player_population_only():
    out = recommend(_input("vpip", _full("vpip", S=3, n=5)))
    assert out["state"] == "population_only"
    assert out["recommendation"] is None


# ---------------- 7/8. zero opportunity / non-consumable ----------------
def test_zero_opportunity_null():
    out = recommend(_input("fold_to_3bet", _full("vpip", S=30, n=100)))
    assert out["state"] == "zero_opportunity"
    assert out["recommendation"] is None


def test_non_consumable_null():
    out = recommend(_input("fold_to_3bet", _ft3b(S=8, n=10)))   # unreliable
    assert out["state"] == "not_consumable"
    assert out["recommendation"] is None
    assert "consumable=false" in out["reason"]


# ---------------- 9. time_sensitive 不影响 eligibility ----------------
def test_time_sensitive_does_not_change_eligibility(monkeypatch):
    inp = _input("fold_to_3bet", _ft3b(S=680, n=800))
    monkeypatch.setattr(p7a, "_time_cache",
                        {f: {"status": "stable"} for f in p7a.DECISION_FEATURES})
    a = recommend(json.loads(json.dumps(inp)))
    monkeypatch.setattr(p7a, "_time_cache",
                        {f: {"status": "unstable"} for f in p7a.DECISION_FEATURES})
    b = recommend(json.loads(json.dumps(inp)))
    assert a["state"] == b["state"] == "eligible"
    assert a["recommendation"]["confidence"] == b["recommendation"]["confidence"]
    assert a["recommendation"]["strength"] == b["recommendation"]["strength"]
    assert a["recommendation"]["action"] is None


# ---------------- 10/11/12. strength/action/H-* 恒 null ----------------
def test_strength_applied_always_zero():
    out = recommend(_input("fold_to_3bet", _ft3b(S=680, n=800)))
    assert out["recommendation"]["strength"] == {"recommended": 0.0,
                                                 "applied": 0.0}
    assert out["meta"]["advisor_mode"] is True


def test_action_always_null():
    for f, ev in [("fold_to_3bet", _ft3b(S=680, n=800)),
                  ("vpip", _full("vpip", S=180, n=400))]:
        out = recommend(_input(f, ev))
        assert out["recommendation"]["action"] is None


def test_hypothesis_id_is_annotation_only():
    out = recommend(_input("fold_to_3bet", _ft3b(S=680, n=800)))
    rec = out["recommendation"]
    assert rec["hypothesis_id"] == "H-ft3b-over"
    assert rec["action"] is None                      # H-* 不产生实际 action
    assert "recommended_adjustment" not in rec        # 推荐层不携带调整量字段


# ---------------- 13. 配置违规硬失败 ----------------
def _bad_config(**bad):
    cfg = {"p7_effect_size": 0.02, "strength_arm": "k=0",
           "cluster": "disabled", "time_decay": "none"}
    cfg.update(bad)
    return cfg


def test_cluster_violation_hard_fail():
    with pytest.raises(AssertionError):
        recommend(_input("fold_to_3bet", _ft3b(S=680, n=800),
                         **{"config": _bad_config(cluster="enabled")}))


def test_time_decay_violation_hard_fail():
    with pytest.raises(AssertionError):
        recommend(_input("fold_to_3bet", _ft3b(S=680, n=800),
                         **{"config": _bad_config(time_decay="x")}))


def test_strength_arm_nonzero_hard_fail():
    """Phase 7 主口径 k=0；k=0.5/1.0 是 Phase 8 假设槽，推荐层强制拒绝。"""
    with pytest.raises(AssertionError):
        recommend(_input("fold_to_3bet", _ft3b(S=680, n=800),
                         **{"config": _bad_config(strength_arm="k=0.5")}))


# ---------------- 14. history/future 泄漏硬失败 ----------------
def test_future_in_history_hard_fail():
    evs = _ft3b(S=680, n=800)
    evs[5]["ts"] = "2009-07-07T00:00:00"
    with pytest.raises(AssertionError):
        recommend(_input("fold_to_3bet", evs))


def test_unsorted_history_hard_fail():
    evs = _ft3b(S=680, n=800)
    evs[0], evs[1] = evs[1], evs[0]
    with pytest.raises(AssertionError):
        recommend(_input("fold_to_3bet", evs))


# ---------------- 15. prior train-only（test player 不得进入） ----------------
def test_prior_provenance_hard_fail():
    inp = _input("fold_to_3bet", _ft3b(S=680, n=800))
    inp["priors"]["fold_to_3bet"]["provenance"] = "population"
    with pytest.raises(AssertionError):
        recommend(inp)


def test_prior_mu_inconsistent_hard_fail():
    inp = _input("fold_to_3bet", _ft3b(S=680, n=800))
    inp["priors"]["vpip"]["mu"] = 0.9
    with pytest.raises(AssertionError):
        recommend(inp)


# ---------------- 16. 禁止读取静态 full-history Profile ----------------
def test_no_static_profile_source():
    src = open(Path(p7c.__file__), encoding="utf-8").read()
    assert "phase6g" not in src
    assert "get_profile" not in src
    assert "profiles.jsonl" not in src
    assert "population_priors" not in src
    assert "gmm" not in src
    assert "half_life" not in src
    assert "agents" not in src


# ---------------- 17. 禁止二次校准 7B confidence ----------------
def test_no_confidence_recalibration():
    """recommendation.confidence 必须 == 7A candidate.confidence（原值透传）。"""
    inp = _input("fold_to_3bet", _ft3b(S=680, n=800))
    out7a = p7a.decide(json.loads(json.dumps(inp)))
    out7c = recommend(json.loads(json.dumps(inp)))
    assert out7c["recommendation"]["confidence"] == \
        out7a["candidates"][0]["confidence"]
    # 7B persistence 只出现在 evidence 元数据块，不参与 confidence 计算
    src = open(Path(p7c.__file__), encoding="utf-8").read()
    assert "persistence_rate" in src
    assert "cand[\"confidence\"]" in src     # confidence 直接取自 7A 候选


def test_evidence_persistence_from_frozen_7b():
    v = json.load(open(p7c.VALIDATION_JSON, encoding="utf-8"))
    per = v["per_horizon"]["200"]["per_feature"]["fold_to_3bet"]
    out = recommend(_input("fold_to_3bet", _ft3b(S=680, n=800)))
    ev = out["recommendation"]["evidence"]
    assert ev["persistence_rate"] == per["persistence_rate"]
    assert ev["persistence_ci95"] == per["persistence_ci95"]
    assert ev["horizon"] == 200


# ---------------- 18. 禁止 P2 / EV 语义 ----------------
def test_no_p2_semantics_source_and_output():
    src = open(Path(p7c.__file__), encoding="utf-8").read()
    for bad in ("exploit_utility_proven", "EV proven", "ROI proven",
                "profitable", "profitability", "winnings"):
        assert bad not in src
    out = recommend(_input("fold_to_3bet", _ft3b(S=680, n=800)))
    raw = json.dumps(out, ensure_ascii=False)
    for bad in ("exploit_utility_proven", "EV proven", "ROI proven",
                "profitable", "profitability", "winnings"):
        assert bad not in raw
    assert out["meta"]["ev_claim"] is False


# ---------------- 契约自检产物 ----------------
def test_contract_check_all_clean():
    assert CHECK.exists()
    res = json.load(open(CHECK, encoding="utf-8"))
    assert res["all_clean"] is True
    assert res["n_cases"] >= 10


def test_determinism():
    inp = _input("fold_to_3bet", _ft3b(S=680, n=800))
    a = json.dumps(recommend(inp), sort_keys=True)
    b = json.dumps(recommend(json.loads(json.dumps(inp))), sort_keys=True)
    assert a == b


def test_phase8_dependencies_metadata_only():
    out = recommend(_input("fold_to_3bet", _ft3b(S=680, n=800)))
    deps = out["meta"]["phase8_dependencies"]
    assert deps["declaration"]
    assert len(deps["required_new_data"]) >= 3
    assert "absent_datasets" in deps
    assert "7C 内禁止补造数据" in deps["declaration"] or \
        "不补造" in deps["declaration"] or "禁止补造" in deps["declaration"]
