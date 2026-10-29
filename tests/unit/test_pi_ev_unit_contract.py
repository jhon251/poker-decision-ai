"""PI-EV 单位契约测试（零成本；捕获 100× / 0.01× 标度缺陷）。

覆盖用户要求的六类：
  1. bb 归一化输入不得二次缩放
  2. chip 输入只允许转换一次
  3. pot/stack 100× 缩放回归
  4. 极小 pot 与正常 pot 的比例保持一致
  5. PI-EV 输入审计日志（raw → normalized → input）
  6. deterministic / no-lookahead / legality 保持
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker_decision_ai.research_unfreeze.pi_ev_contract import (  # noqa: E402
    MAX_POT_BB,
    UNIT_CONTRACT_VERSION,
    AuditLog,
    UnitContractError,
    assert_bb,
    pi_ev_inputs,
    sanity_check,
)


def _norm_state(pot_bb=6.0, to_call_bb=0.0, stack_bb=100.0, bb=1.0):
    """bb 归一化 state（`_state_for` 产物形态）。"""
    return {"bb": bb, "pot": pot_bb, "to_call": to_call_bb,
            "players": [{"player_id": "Hero", "is_hero": True,
                         "stack": stack_bb},
                        {"player_id": "V1", "is_hero": False,
                         "stack": stack_bb}]}


def _chip_state(pot_chips=600.0, to_call_chips=0.0, stack_chips=10000.0,
                bb=100.0):
    """筹码单位 state（仅用于「恰好转换一次」路径）。"""
    return {"bb": bb, "pot": pot_chips, "to_call": to_call_chips,
            "players": [{"player_id": "Hero", "is_hero": True,
                         "stack": stack_chips}]}


# 1. bb 归一化输入不得二次缩放
def test_bb_normalized_not_scaled_twice():
    st = _norm_state(pot_bb=6.0, stack_bb=100.0)
    got = pi_ev_inputs(st)
    assert got["pot_bb"] == pytest.approx(6.0)
    assert got["hero_stack_bb"] == pytest.approx(100.0)
    assert got["normalized_once"] is True
    assert got["conversion_factor"] == 1.0
    # 历史缺陷写法（再除 hand_bb=100）必须被显式区分出来
    assert got["pot_bb"] != pytest.approx(0.06)
    assert got["hero_stack_bb"] != pytest.approx(1.0)


# 2. chip 输入只允许转换一次
def test_chip_state_converts_exactly_once():
    st = _chip_state(pot_chips=600.0, stack_chips=10000.0, bb=100.0)
    got = pi_ev_inputs(st)
    assert got["pot_bb"] == pytest.approx(6.0)
    assert got["hero_stack_bb"] == pytest.approx(100.0)
    assert got["normalized_once"] is False
    assert got["conversion_factor"] == pytest.approx(0.01)
    # 转换两次会得到 0.06 / 1.0 —— 必须不同
    assert got["pot_bb"] != pytest.approx(0.06)


# 3. pot/stack 100× 缩放回归
def test_100x_scaling_regression_detected():
    raw = {"pot_before": 600.0, "to_call": 0.0}
    good = pi_ev_inputs(_norm_state(pot_bb=6.0), raw=raw, hand_bb=100.0)
    assert good["pot_bb"] == pytest.approx(6.0)          # 与 raw/hand_bb 一致
    sanity_check(good, raw=raw, hand_bb=100.0)           # 不应抛错
    bad = dict(good)
    bad["pot_bb"] = 0.06                                 # 复现历史缺陷值
    with pytest.raises(UnitContractError) as e:
        sanity_check(bad, raw=raw, hand_bb=100.0)
    assert "比例异常" in str(e.value)


def test_wrong_input_construction_raises():
    """确认历史写法（state.pot / hand_bb）会触发契约错误。"""
    st = _norm_state(pot_bb=6.0)
    raw = {"pot_before": 600.0, "to_call": 0.0}
    legacy_value = st["pot"] / 100.0                     # = 0.06（缺陷）
    with pytest.raises(UnitContractError):
        sanity_check({"pot_bb": legacy_value, "to_call_bb": 0.0,
                      "hero_stack_bb": 100.0}, raw=raw, hand_bb=100.0)


# 4. 极小 pot 与正常 pot 的比例保持一致
def test_tiny_vs_normal_pot_ratio_preserved():
    tiny = pi_ev_inputs(_norm_state(pot_bb=0.5))["pot_bb"]
    normal = pi_ev_inputs(_norm_state(pot_bb=50.0))["pot_bb"]
    assert tiny == pytest.approx(0.5)
    assert normal == pytest.approx(50.0)
    assert normal / tiny == pytest.approx(100.0)
    # 与原始记录比对：小 pot 也必须在比例带内（不是被压成 0.005）
    raw = {"pot_before": 50.0, "to_call": 0.0}
    sanity_check({"pot_bb": tiny, "to_call_bb": 0.0, "hero_stack_bb": 100.0},
                 raw={"pot_before": 50.0, "to_call": 0.0}, hand_bb=100.0)
    with pytest.raises(UnitContractError):
        sanity_check({"pot_bb": 0.005, "to_call_bb": 0.0,
                      "hero_stack_bb": 100.0}, raw=raw, hand_bb=100.0)


# 5. 审计日志
def test_audit_log_records_raw_normalized_input():
    log = AuditLog(capacity=4)
    st = _norm_state(pot_bb=6.0, to_call_bb=2.0, stack_bb=100.0)
    raw = {"pot_before": 600.0, "to_call": 200.0}
    inputs = pi_ev_inputs(st, raw=raw, hand_bb=100.0)
    log.record(hand_id="h1", street="flop", raw=raw,
               state_pot_bb=st["pot"], state_to_call_bb=st["to_call"],
               hero_stack_bb=st["players"][0]["stack"], hand_bb=100.0,
               inputs=inputs)
    assert len(log.rows) == 1
    row = log.rows[0]
    assert row["raw"]["pot_before_chips"] == 600.0
    assert row["state"]["pot_bb"] == 6.0
    assert row["pi_ev_input"]["pot_bb"] == pytest.approx(6.0)
    assert row["ratios"]["pot_input_over_state"] == pytest.approx(1.0)
    assert row["ratios"]["stack_input_over_state"] == pytest.approx(1.0)
    assert row["ratios"]["pot_state_over_raw_bb"] == pytest.approx(1.0)
    assert row["pi_ev_input"]["units" if "units" in row["pi_ev_input"]
                              else "pot_bb"] is not None
    # 容量上限
    for i in range(10):
        log.record(hand_id=f"h{i}", street="flop", raw=raw,
                   state_pot_bb=6.0, state_to_call_bb=2.0, hero_stack_bb=100.0,
                   hand_bb=100.0, inputs=inputs)
    assert len(log.rows) == 4


# 6. 非法/越界输入必须硬失败（不得静默继续）
def test_invalid_and_oversize_inputs_rejected():
    with pytest.raises(UnitContractError):
        assert_bb(float("nan"), name="pot_bb", maximum=MAX_POT_BB)
    with pytest.raises(UnitContractError):
        assert_bb(float("inf"), name="pot_bb", maximum=MAX_POT_BB)
    with pytest.raises(UnitContractError):
        assert_bb(-1.0, name="pot_bb", maximum=MAX_POT_BB)
    # 未归一化的筹码值（600 筹码当 bb 用）→ 超上界
    st = _norm_state(pot_bb=float(MAX_POT_BB) + 1.0)
    with pytest.raises(UnitContractError) as e:
        pi_ev_inputs(st)
    assert "上界" in str(e.value)
    # 缺 hero
    with pytest.raises(UnitContractError):
        pi_ev_inputs({"bb": 1.0, "pot": 6.0, "players": []})
    # state["bb"] 非法
    with pytest.raises(UnitContractError):
        pi_ev_inputs({"bb": 0.0, "pot": 6.0,
                      "players": [{"is_hero": True, "stack": 1.0}]})


def test_zero_pot_and_zero_to_call_rejected():
    with pytest.raises(UnitContractError):
        sanity_check({"pot_bb": 0.0, "to_call_bb": 0.0, "hero_stack_bb": 10.0})


# 7. deterministic / no-lookahead / legality 保持
def test_deterministic_and_purity():
    st = _norm_state(pot_bb=6.0, to_call_bb=2.0)
    before = copy.deepcopy(st)
    a = pi_ev_inputs(st)
    b = pi_ev_inputs(copy.deepcopy(st))
    assert a == b                     # 确定性
    assert st == before               # 纯函数，不改 state
    assert a["units" if "units" in a else "pot_bb"] is not None


def test_no_lookahead_and_legality_unchanged():
    # no-lookahead：契约模块不接触牌面/未来牌（只用数值字段）
    src = Path("poker_decision_ai/research_unfreeze/pi_ev_contract.py"
               ).read_text(encoding="utf-8")
    for token in ("evaluate7", "sample_runouts", "combinations", "board"):
        assert token not in src, f"契约模块不应涉及 {token}"
    # legality 映射未被触碰（仍由冻结核提供）
    from poker_decision_ai.pluribus_validation import legal_action_types
    la = legal_action_types(0.0, 1000.0)
    assert la is not None and "check" in la and "bet" in la
    la2 = legal_action_types(50.0, 1000.0)
    assert la2 is not None and "fold" in la2 and "call" in la2


def test_contract_version_exposed():
    assert UNIT_CONTRACT_VERSION.startswith("pi-ev-unit-contract-")
    assert math.isfinite(MAX_POT_BB)


# 8. S″ 重检准备（split/manifest 封存 + 运行护栏）
def test_s2_registry_players_disjoint_and_scope_declared():
    from poker_decision_ai.research_unfreeze import s2_recheck as s2
    reg = json.loads(s2.REGISTRY.read_text(encoding="utf-8"))
    used = set(s2.USED_PLAYERS)
    assert not (set(reg["s2_players"]) & used), "S″ 玩家必须与 S/S′ 完全不相交"
    assert len(reg["s2_players"]) == 3
    assert reg["counts"]["s2_decisions"] >= 400
    assert reg["frozen_before_model_run"] is True
    # 手级耦合若无法消除，必须在 scope_rule 中显式声明
    if reg["counts"]["hands_clean_no_S_no_Sprime"] == 0:
        assert "手级上下文耦合" in reg["scope_rule"]


def test_s2_manifests_sealed_and_run_guarded():
    from poker_decision_ai.research_unfreeze import s2_recheck as s2
    for path, arm in ((s2.MANIFEST_U1, "U1"), (s2.MANIFEST_R22, "R2.2")):
        v = s2.verify_s2_manifest(path)
        assert v["present"] and v["clean"] is True and v["checked"] >= 10
        man = json.loads(path.read_text(encoding="utf-8"))
        assert man["arm"] == arm
        assert "pi-ev-unit-contract" in man["config"]["unit_contract"]
        assert set(man["instrument_gates"]) >= {
            "IG1_raw_ratio", "IG2_anomaly_rate", "IG3_distribution",
            "IG4_audit_log", "IG5_no_approximation", "IG6_no_dry_run"}
        assert man["production_wiring"].startswith("NOT WIRED")
    # 运行护栏：未显式授权时禁止执行（本阶段不执行 S″）
    with pytest.raises(RuntimeError) as e:
        s2.run_s2_acceptance([], arm="U1")
    assert "已禁用" in str(e.value)
