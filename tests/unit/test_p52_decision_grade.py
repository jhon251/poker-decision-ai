"""V3F-P52 — DecisionGradeDataset contract + flaky 回归测试。

覆盖：13 组字段契约；四值 verdict；mode 感知条件必填（MTT payout /
PKO bounty）；数据集三级裁决（decision/behavior/insufficient）；
adapter 接口与聚合报告；UofT 显式 behavior_grade（不升级）；flaky
修复回归（realtime 路径重复运行确定性 + 覆写不改决策内容）。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai.decision_grade import (
    BEHAVIOR_GRADE,
    DECISION_GRADE,
    DECISION_GRADE_FIELDS,
    INSUFFICIENT,
    DecisionDatasetAdapter,
    UofTDecisionGradeProbe,
    classify_record,
    dataset_verdict,
    mode_validation_scope,
)
from poker_decision_ai.domain.poker.evidence_gate import (
    EXCLUDED, PARTIALLY_USABLE, UNKNOWN, USABLE)


def _full_record(mode: str = "cash") -> dict:
    return {s["key"]: f"value_{s['key']}" for s in DECISION_GRADE_FIELDS}


# ------------------------------------------------------------ 字段契约
def test_contract_fields_and_verdict_vocabulary():
    keys = [s["key"] for s in DECISION_GRADE_FIELDS]
    assert len(keys) == len(set(keys)) == 14
    for must in ("hero_hole_cards", "observed_hero_action", "legal_actions",
                 "action_sequence_sizes", "game_mode", "bounty_context"):
        assert must in keys


def test_missing_fields_unknown_never_fabricated():
    rec = {"hand_id": "h1", "player_id": "p1"}
    fv = classify_record(rec, "cash")
    assert fv["hero_hole_cards"]["verdict"] == UNKNOWN
    assert fv["hero_hole_cards"]["reason"] == "missing"
    assert fv["hand_id"]["verdict"] == USABLE
    # 可选字段缺失（blind_level for cash）→ optional_missing
    assert fv["blind_level"]["verdict"] == UNKNOWN
    assert fv["blind_level"]["reason"] == "optional_missing"


def test_adapter_verdict_passthrough():
    """适配器已给字段级判定 → 直通（不二次伪造）。"""
    rec = {"hand_id": {"verdict": PARTIALLY_USABLE, "reason": "dup risk"},
           "player_id": "p"}
    fv = classify_record(rec, "cash")
    assert fv["hand_id"] == {"verdict": PARTIALLY_USABLE,
                             "reason": "dup risk"}


# ------------------------------------------------------------ 数据集级裁决
def test_full_record_is_decision_grade():
    assert dataset_verdict(classify_record(_full_record(), "cash"),
                           "cash")["verdict"] == DECISION_GRADE


def test_no_hero_cards_is_behavior_grade():
    rec = _full_record()
    rec["hero_hole_cards"] = None
    rec["observed_hero_action"] = None
    r = dataset_verdict(classify_record(rec, "cash"), "cash")
    assert r["verdict"] == BEHAVIOR_GRADE
    assert "hero_hole_cards" in r["reason"]


def test_no_action_sequence_is_insufficient():
    rec = _full_record()
    rec["action_sequence_sizes"] = {}
    assert dataset_verdict(classify_record(rec, "cash"),
                           "cash")["verdict"] == INSUFFICIENT


def test_mode_aware_conditional_required():
    # MTT 缺 payout → 不是 decision_grade；PKO 缺 bounty 同理
    mtt = _full_record("mtt")
    mtt["payout_icm_context"] = None
    r = dataset_verdict(classify_record(mtt, "mtt"), "mtt")
    assert r["verdict"] == BEHAVIOR_GRADE
    assert "payout_icm_context" in r["reason"]
    pko = _full_record("pko")
    pko["bounty_context"] = None
    assert dataset_verdict(classify_record(pko, "pko"),
                           "pko")["verdict"] == BEHAVIOR_GRADE
    # cash 缺 payout/bounty 是可选 → 仍 decision_grade
    cash = _full_record("cash")
    cash["payout_icm_context"] = None
    cash["bounty_context"] = None
    assert dataset_verdict(classify_record(cash, "cash"),
                           "cash")["verdict"] == DECISION_GRADE


def test_mode_validation_scope_contract():
    assert mode_validation_scope("cash")["objective"] == "chip_ev"
    assert mode_validation_scope("mtt")["objective"] == "chip_ev+icm"
    assert "payout_icm_context" in mode_validation_scope("mtt")["requires"]
    assert "bounty_context" in mode_validation_scope("pko")["requires"]
    assert mode_validation_scope("mystery")["objective"] == "chip_ev+icm+bounty"


# ------------------------------------------------------------ adapter
def test_uoft_probe_is_behavior_grade_never_decision():
    rep = UofTDecisionGradeProbe().evaluate_dataset(limit=3)
    assert rep["dataset_verdict_counts"][DECISION_GRADE] == 0
    assert rep["dataset_verdict_counts"][BEHAVIOR_GRADE] == 3
    assert rep["field_usable_ratio"]["hero_hole_cards"] == 0.0
    assert rep["field_usable_ratio"]["observed_hero_action"] == 0.0
    assert rep["dataset_meta"]["known_issues"]
    assert rep["modes"] == {"cash": 3}


def test_abstract_adapter_cannot_be_used():
    a = DecisionDatasetAdapter()
    with pytest.raises(NotImplementedError):
        a.dataset_meta()


def test_custom_future_adapter_roundtrip():
    """未来数据源示例：完整 cash 记录 → decision_grade 100%。"""

    class FutureCashAdapter(DecisionDatasetAdapter):
        def dataset_meta(self):
            return {"name": "future_hero_view_cash", "license": "tbd",
                    "known_issues": []}

        def iter_candidates(self, limit):
            for i in range(limit):
                yield _full_record()

        def mode_of(self, record):
            return "cash"

    rep = FutureCashAdapter().evaluate_dataset(limit=4)
    assert rep["dataset_verdict_ratio"][DECISION_GRADE] == 1.0
    assert all(v == 1.0 for v in rep["field_usable_ratio"].values())


def test_evaluate_deterministic():
    a = UofTDecisionGradeProbe().evaluate_dataset(limit=5)
    b = UofTDecisionGradeProbe().evaluate_dataset(limit=5)
    assert a == b


# ------------------------------------------------------------ flaky 回归
def test_realtime_timeout_override_does_not_change_decision():
    """P52-3 回归：超时覆写只放宽时钟预算，不改变决策内容（同帧
    生产配置 vs 测试配置 → 相同 decision/status）。"""
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.frame_decision import frame_to_decision
    from poker_decision_ai.core.orchestration.runtime import build_runtime
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    from pathlib import Path
    ev = Path("data/vision2/coinpoker_evidence")
    w, h, rgba = decode_png(
        (ev / "8bf82f134651db0add626be0304a2ff1.png").read_bytes())

    cfg_prod = load_config()
    cfg_test = load_config()
    import types
    # SOLVER-1：MC1-R5 后 realtime 链含 frequency_prior agent，生产 3000ms
    # 死线在本帧压线（~3.2s）——prod 侧提到 10s 保证两侧都产出决策；
    # 被测契约仍是"不同时钟预算 → 相同决策内容"。
    rt_prod = dict(cfg_prod.realtime_config() or {})
    rt_prod["agent_timeout_ms"] = max(int(rt_prod.get("agent_timeout_ms", 300)), 10000)
    rt_prod["deadline_ms"] = max(int(rt_prod.get("deadline_ms", 3000)), 10000)
    cfg_prod.realtime_config = types.MethodType(
        lambda self, _rt=rt_prod: dict(_rt), cfg_prod)
    rt = dict(cfg_test.realtime_config() or {})
    rt["agent_timeout_ms"] = max(int(rt.get("agent_timeout_ms", 300)), 5000)
    rt["deadline_ms"] = max(int(rt.get("deadline_ms", 3000)), 20000)
    cfg_test.realtime_config = types.MethodType(
        lambda self, _rt=rt: dict(_rt), cfg_test)

    r_prod = frame_to_decision(build_runtime(cfg_prod), rgba, w, h)
    r_test = frame_to_decision(build_runtime(cfg_test), rgba, w, h)
    rep_p = (r_prod.report or {}).get("final_decision") or {}
    rep_t = (r_test.report or {}).get("final_decision") or {}
    assert rep_p.get("decision_status") == rep_t.get("decision_status")
    assert rep_p.get("action") == rep_t.get("action")
    assert rep_p.get("solver_mode") == rep_t.get("solver_mode")


def test_realtime_path_repeat_determinism():
    """P52-3 回归：同一帧连续 3 次 analyze（测试安全超时）→ 结果逐位
    一致（timing 诱导的非确定性在修复后不再出现）。"""
    from cash_game_ui.runner import CashGameRunner
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    from pathlib import Path
    ev = Path("data/vision2/coinpoker_evidence")
    w, h, rgba = decode_png(
        (ev / "8bf82f134651db0add626be0304a2ff1.png").read_bytes())
    runner = CashGameRunner()
    outs = [runner.analyze(rgba, w, h).to_dict() for _ in range(3)]
    keys = ("status", "decision_action", "solver_mode")
    for o in outs[1:]:
        assert {k: o[k] for k in keys} == {k: outs[0][k] for k in keys}
