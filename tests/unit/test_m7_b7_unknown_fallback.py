"""M7 / B7-3：UNKNOWN / 隐式 fallback 全路径（dealer / table_size / hero /
occupied / position）。

原则（与冻结契约一致）：**null = unknown**。
  * 缺失（missing）→ null + uncertainty，绝不填默认值；
  * 冲突（conflict）→ validation error（valid=False），绝不静默纠正；
  * 不完整（incomplete）→ UNKNOWN，绝不用子集凑一个"最像"的答案；
  * 隐式 fallback 一律禁止：不得回退到某个座位、不得默认 6 人桌、不得把
    UNKNOWN 当 EMPTY/OCCUPIED、不得在 dealer 缺失时反推 position。

不修改 Vision / State / positions_for；本文件只做观测性断言。
"""
from __future__ import annotations

import pytest

from poker_decision_ai.vision2.assembly import assemble
from poker_decision_ai.vision2.pixel.realclient.seat_position import (
    EMPTY, OCCUPIED, UNKNOWN, compute_positions, runtime_position_pipeline,
    to_assembly_contract)
from poker_decision_ai.vision2.schemas import RawSeat, RawVisionState
from poker_decision_ai.vision2.validator import validate_raw

#: 6-max 全环（S1..S6），dealer=S3 ⇒ 冻结 positions_for 的正常参照
_ALL = {f"S{i}": OCCUPIED for i in range(1, 7)}


def _raw(**kw) -> RawVisionState:
    """最小 raw 观测；未指定字段保持 None（缺失，不填默认）。"""
    base = {"source": "mock", "synthetic_test_only": True, "game_type": "cash",
            "poker_variant": "nlhe", "table_size": 6, "street": "preflop",
            "seats": [RawSeat(seat=1, occupied=True),
                      RawSeat(seat=3, occupied=True),
                      RawSeat(seat=6, occupied=True)]}
    base.update(kw)
    return RawVisionState(**base)


def _unc_fields(uncertainty) -> set:
    """uncertainty 条目 → 字段名集合（兼容 dataclass / dict 两种表示）。"""
    return {(u if isinstance(u, dict) else u.to_dict())["field"]
            for u in uncertainty}


# ── dealer（button_seat）──────────────────────────────────────────────
def test_missing_dealer_is_unknown_not_guessed():
    """dealer 缺失 → 全部 position=None + uncertainty，绝不默认某座位为 BTN。"""
    n = assemble(_raw(button_seat=None, hero_seat=1))
    assert "button_seat" in _unc_fields(n.uncertainty)
    assert [p["position"] for p in n.players] == [None, None, None]
    assert n.hero["position"] is None


def test_dealer_conflict_is_error_not_silently_fixed():
    """dealer 指向不存在座位 → validation error（valid=False）。"""
    errs, _, _ = validate_raw(_raw(button_seat=99, hero_seat=1))
    assert any(e["code"] == "button_seat_not_found" for e in errs)
    n = assemble(_raw(button_seat=99, hero_seat=1))
    assert n.validation.valid is False


def test_position_never_derived_without_dealer():
    """无 dealer 时 position 一律 null（禁止"隐式 fallback 到首位"）。"""
    n = assemble(_raw(button_seat=None, hero_seat=3))
    assert all(p["position"] is None for p in n.players)
    assert n.hero["position"] is None


def test_dealer_conflict_must_not_yield_position():
    """M8 闭环：dealer 冲突 → position 必须为 null（不得经冻结 positions_for
    的"回退固定顺序"分支产生非空 position）。

    修复位于 `vision2.assembly._positions_for`（本层不把无效 button_seat
    传给冻结函数）；冻结 `positions_for` 本身零修改（见独立回归测试）。
    """
    n = assemble(_raw(button_seat=99, hero_seat=1))
    assert n.validation.valid is False
    assert all(p["position"] is None for p in n.players)
    assert n.hero["position"] is None


# ── hero ──────────────────────────────────────────────────────────────
def test_missing_hero_seat_is_unknown_not_defaulted():
    n = assemble(_raw(button_seat=3, hero_seat=None))
    assert "hero.seat" in _unc_fields(n.uncertainty)
    assert n.hero["seat"] is None and n.hero["position"] is None


def test_hero_seat_absent_from_seats_is_error():
    """hero_seat 不对应任何 seat → error（退回"第一座位"属隐式 fallback，禁止）。"""
    errs, _, _ = validate_raw(_raw(button_seat=3, hero_seat=5))
    assert [e["code"] for e in errs] == ["hero_seat_not_found"]
    n = assemble(_raw(button_seat=3, hero_seat=5))
    assert n.hero["position"] is None


def test_multi_hero_conflict_is_error():
    errs, _, _ = validate_raw(_raw(
        button_seat=1, hero_seat=1,
        seats=[RawSeat(seat=1, occupied=True), RawSeat(seat=1, occupied=True)]))
    codes = {e["code"] for e in errs}
    assert {"duplicate_seat", "multiple_hero"} <= codes


# ── table_size ────────────────────────────────────────────────────────
def test_table_size_missing_stays_null_never_defaults_to_six():
    n = assemble(_raw(table_size=None))
    assert n.game["table_size"] is None          # 不默认 6


def test_table_size_conflict_with_seats_is_not_silently_reconciled():
    """table_size=2 却有 3 个座位 → 原样如实传播（不推断、不纠正）。"""
    n = assemble(_raw(table_size=2))
    assert n.game["table_size"] == 2
    assert len(n.players) == 3


def test_table_size_does_not_invent_positions():
    """table_size 不参与 position 推导：仅 table_size 变化不得改变 position。"""
    a = assemble(_raw(table_size=6, button_seat=3, hero_seat=1))
    b = assemble(_raw(table_size=None, button_seat=3, hero_seat=1))
    assert a.hero["position"] == b.hero["position"]


# ── occupied ──────────────────────────────────────────────────────────
def test_occupied_missing_stays_null_not_false():
    n = assemble(_raw(button_seat=1, hero_seat=3,
                      seats=[RawSeat(seat=1, occupied=None),
                             RawSeat(seat=3, occupied=None)]))
    assert [p["occupied"] for p in n.players] == [None, None]
    # 无占位证据（None ≠ True）：不得据此声称座位被占用
    assert all(p["occupied"] is not True for p in n.players)


def test_occupied_unknown_is_unknown_not_empty():
    r = compute_positions({"S1": "OCCUPIED", "S2": UNKNOWN}, "S1", "S1")
    assert r["uncertainty"] == "UNKNOWN"
    assert r["position_map"] is None and r["hero_position"] is None
    assert "occupied_incomplete" in r["reason"]


def test_position_incomplete_evidence_is_unknown():
    """occupancy 缺失/不完整 → UNKNOWN（不得用已见子集推导 position）。"""
    assert compute_positions({}, "S3", "S1")["uncertainty"] == "UNKNOWN"
    assert compute_positions({}, "S3", "S1")["reason"] == "no_occupied_seats"
    assert compute_positions(_ALL, "S3", None)[
        "reason"] == "dealer_or_hero_missing"
    assert compute_positions(_ALL, None, "S1")[
        "reason"] == "dealer_or_hero_missing"


# ── position：冲突 / 不完整 / 隐式 fallback ────────────────────────────
def test_position_conflict_dealer_not_occupied_is_unknown():
    """dealer 指向空座 → UNKNOWN（不得改判为"最近的占用座"）。"""
    r = compute_positions({"S1": OCCUPIED, "S4": OCCUPIED}, "S3", "S1")
    assert r["uncertainty"] == "UNKNOWN"
    assert r["reason"] == "dealer_not_occupied"
    assert r["hero_position"] is None


def test_position_conflict_hero_not_occupied_is_unknown():
    r = compute_positions({"S1": OCCUPIED, "S4": OCCUPIED}, "S4", "S6")
    assert r["uncertainty"] == "UNKNOWN"
    assert r["reason"] == "hero_not_occupied"


def test_position_illegal_seat_is_unknown_not_wrapped():
    """非法座位号不得回绕到合法座位（禁止隐式 fallback）。"""
    r = compute_positions({"S9": OCCUPIED, "S1": OCCUPIED}, "S9", "S1")
    assert r["uncertainty"] == "UNKNOWN"
    assert r["position_map"] is None
    assert compute_positions({"S7": OCCUPIED}, "S7", "S7")[
        "uncertainty"] == "UNKNOWN"


def test_known_good_reference_is_not_unknown():
    """对照：证据完整时必须给出真实结果（防止"一律 UNKNOWN"式伪通过）。"""
    r = compute_positions(_ALL, "S3", "S1")
    assert r["uncertainty"] == "RECOGNIZED"
    assert r["hero_position"] is not None
    assert set(r["position_map"]) == set(_ALL)


def test_assembly_contract_propagates_unknown_as_none():
    """视觉 UNKNOWN → 冻结 assembly int 域必须为 None（如实传播）。"""
    assert to_assembly_contract({"S1": UNKNOWN}, "S1", "S1")["occupied_ringed"] \
        == {}
    assert to_assembly_contract({"S1": EMPTY}, "S1", "S1")["occupied_ringed"] \
        == {}
    assert to_assembly_contract({}, None, None) == {
        "button_seat": None, "hero_seat": None, "occupied_ringed": {}}


def test_runtime_pipeline_unknown_evidence_never_defaults_position():
    """全路径：证据 UNKNOWN/不完整 → hero_position None + uncertainty UNKNOWN。"""
    r = runtime_position_pipeline(occupied={"S1": UNKNOWN}, dealer_seat="S1",
                                  hero_seat="S1")
    assert r["hero_position"] is None
    assert r["positions_ring"] is None
    assert r["runtime_result"]["uncertainty"] == "UNKNOWN"
    assert r["ground_truth_status"] == "runtime_state_non_gt"


def test_runtime_pipeline_recognized_is_not_unknown():
    """对照：证据完整 → RECOGNIZED 且 hero_position 非空（防伪 UNKNOWN）。"""
    r = runtime_position_pipeline(occupied=dict(_ALL), dealer_seat="S3",
                                  hero_seat="S1")
    assert r["runtime_result"]["uncertainty"] == "RECOGNIZED"
    assert r["hero_position"] is not None


def test_unsupported_evidence_states_are_all_unknown():
    """None / EMPTY / UNKNOWN 三种"无占用证据"一律 UNKNOWN，互不冒充。"""
    outs = [compute_positions({"S1": st}, "S1", "S1")
            for st in (None, EMPTY, UNKNOWN)]
    assert {o["uncertainty"] for o in outs} == {"UNKNOWN"}
    assert all(o["position_map"] is None for o in outs)
    assert all(o["reason"] != "positions_for_derived" for o in outs)
    assert outs[2]["reason"] == "occupied_incomplete_S1"   # UNKNOWN ≠ EMPTY
    assert outs[0]["reason"] == outs[1]["reason"] == "no_occupied_seats"


def test_null_never_silently_becomes_false_or_zero_in_normalized_state():
    n = assemble(_raw(button_seat=3, hero_seat=1, pot=None, hero_stack=None,
                      blinds=None))
    assert n.pot["pot"] is None
    assert n.hero["stack"] is None
    assert n.game["blinds"] is None
    assert n.hero["all_in"] is None
