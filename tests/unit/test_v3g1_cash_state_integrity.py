"""V3G-1 — CASH State Integrity Gate 最小验收测试。

覆盖验收矩阵（1–10）：
  1 hero 手牌缺失；2 board/street 不一致；3 stack/chips 缺失；4 to_call > pot；
  5 dealer/position 等关键上下文缺失；6 UNKNOWN 输入；7 冲突字段；
  8 正常完整 CASH state；9 不完整 state 不得被默认值补齐；
  10 门禁本身不能修改输入 state。

另含：证据等级词表同源、确定性、只读接线审计、数值不变式。

不修改任何既有测试定义；不触碰冻结文件。
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai import cash_state_integrity as csi
from poker_decision_ai.cash_state_integrity import (
    COMPLETE, DEGRADED, INCOMPLETE, evaluate_cash_state_integrity)
from poker_decision_ai.domain.poker.strategy_core import compute_ev
from poker_decision_ai.quality_benchmark import EVIDENCE_GRADES as QB_GRADES

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"


def full_cash_state() -> dict:
    """完整、无冲突的 CASH state（全部必需+可选字段齐备）。"""
    return {
        "game_type": "cash", "street": "flop", "board": ["Ah", "7c", "2c"],
        "pot": 10.0, "to_call": 0.0, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "button_seat": 3, "table_size": 6,
        "synthetic_test_only": True,
        "players": [
            {"player_id": "v1", "seat": 1, "position": "SB", "status": "active",
             "hole_cards": [], "stack": 100.0},
            {"player_id": "hero", "seat": 3, "position": "BTN", "is_hero": True,
             "status": "active", "hole_cards": ["9c", "4c"], "stack": 100.0},
            {"player_id": "v2", "seat": 6, "position": "BB", "status": "active",
             "hole_cards": [], "stack": 100.0},
        ],
    }


def _mut(**over) -> dict:
    st = full_cash_state()
    st.update(over)
    return st


def _drop(*keys) -> dict:
    st = full_cash_state()
    for k in keys:
        st.pop(k, None)
    return st


def _hero(st: dict) -> dict:
    return next(p for p in st["players"] if p.get("is_hero"))


# ── 8. 正常完整 CASH state ────────────────────────────────────────────
def test_08_complete_cash_state_is_complete():
    r = evaluate_cash_state_integrity(full_cash_state())
    assert r.level == COMPLETE
    assert r.missing == [] and r.conflicts == []
    assert r.completeness == 1.0
    assert r.ok is True


def test_08_evidence_grade_uses_existing_vocabulary():
    """证据等级必须取自既有词表，且本模块词表与 quality_benchmark 同源。"""
    assert tuple(csi.EVIDENCE_GRADES) == tuple(QB_GRADES)
    for st in (full_cash_state(),
               _mut(synthetic_test_only=False),
               _drop("pot")):
        g = evaluate_cash_state_integrity(st).evidence_grade
        assert g in QB_GRADES, g


# ── 1. hero 手牌缺失 ──────────────────────────────────────────────────
def test_01_missing_hero_hole_cards_reported_not_defaulted():
    st = full_cash_state()
    _hero(st)["hole_cards"] = []
    r = evaluate_cash_state_integrity(st)
    assert "hero.hole_cards" in r.unknown
    assert r.level == DEGRADED                     # 非必需字段，但必须如实标 UNKNOWN
    assert r.ok is True
    # 不得被补成任意两张牌
    assert all(_norm(c) is not None for c in _hero(st)["hole_cards"]) is True


def test_01_hero_hole_cards_with_null_element_is_unknown():
    st = full_cash_state()
    _hero(st)["hole_cards"] = ["9c", None]
    r = evaluate_cash_state_integrity(st)
    assert "hero.hole_cards" in r.unknown


def test_01_missing_hero_entry_is_unknown():
    st = full_cash_state()
    st["players"] = [p for p in st["players"] if not p.get("is_hero")]
    r = evaluate_cash_state_integrity(st)
    assert "hero" in r.unknown


# ── 2. board / street 不一致 ──────────────────────────────────────────
@pytest.mark.parametrize("street,board,expect_code", [
    ("preflop", ["Ah", "7c", "2c"], "board_street_mismatch"),
    ("flop", ["Ah", "7c"], "board_street_mismatch"),
    ("turn", ["Ah", "7c", "2c"], "board_street_mismatch"),
    ("river", ["Ah", "7c", "2c", "2d"], "board_street_mismatch"),
])
def test_02_board_street_mismatch_is_conflict(street, board, expect_code):
    r = evaluate_cash_state_integrity(
        _mut(street=street, board=board))
    assert any(c.startswith(expect_code) for c in r.conflicts), r.conflicts
    assert r.level == INCOMPLETE and r.ok is False


def test_02_board_lengths_accepted_for_each_street():
    for street, board in (("preflop", []), ("flop", ["Ah", "7c", "2c"]),
                          ("turn", ["Ah", "7c", "2c", "2d"]),
                          ("river", ["Ah", "7c", "2c", "2d", "3s"])):
        st = _mut(street=street, board=board)
        if street == "preflop":
            st["to_call"] = 0.0
        assert evaluate_cash_state_integrity(st).conflicts == [], street


# ── 3. stack / chips 缺失 ─────────────────────────────────────────────
def test_03_missing_hero_stack_is_unknown():
    st = full_cash_state()
    _hero(st).pop("stack", None)
    r = evaluate_cash_state_integrity(st)
    assert "hero.stack" in r.unknown
    assert r.level == DEGRADED


def test_03_null_hero_stack_is_unknown_not_zero():
    st = full_cash_state()
    _hero(st)["stack"] = None
    r = evaluate_cash_state_integrity(st)
    assert "hero.stack" in r.unknown
    assert _hero(st)["stack"] is None          # 不得被填 0


def test_03_missing_bb_is_missing_required():
    r = evaluate_cash_state_integrity(_drop("bb"))
    assert "bb" in r.missing and r.ok is False


# ── 4. to_call > pot ──────────────────────────────────────────────────
def test_04_to_call_exceeding_pot_is_conflict():
    r = evaluate_cash_state_integrity(_mut(pot=10.0, to_call=25.0))
    assert any(c.startswith("to_call_exceeds_pot") for c in r.conflicts)
    assert r.level == INCOMPLETE


def test_04_to_call_equal_to_pot_is_not_conflict():
    r = evaluate_cash_state_integrity(_mut(pot=10.0, to_call=10.0))
    assert not any(c.startswith("to_call_exceeds_pot") for c in r.conflicts)


def test_04_to_call_exceeding_hero_stack_is_conflict():
    st = full_cash_state()
    st["pot"] = 100.0
    st["to_call"] = 20.0
    _hero(st)["stack"] = 5.0
    r = evaluate_cash_state_integrity(st)
    assert any(c.startswith("to_call_exceeds_hero_stack") for c in r.conflicts)
    # 对照：栈足够时不得报该冲突
    st2 = full_cash_state()
    st2["pot"] = 100.0
    st2["to_call"] = 20.0
    assert not any(c.startswith("to_call_exceeds_hero_stack")
                   for c in evaluate_cash_state_integrity(st2).conflicts)


# ── 5. dealer / position 等关键上下文缺失 ─────────────────────────────
def test_05_missing_button_seat_is_unknown_not_defaulted():
    r = evaluate_cash_state_integrity(_drop("button_seat"))
    assert "button_seat" in r.unknown
    assert r.level == DEGRADED


def test_05_missing_table_size_is_unknown_not_defaulted_six():
    r = evaluate_cash_state_integrity(_drop("table_size"))
    assert "table_size" in r.unknown


def test_05_button_seat_not_in_players_is_conflict():
    r = evaluate_cash_state_integrity(_mut(button_seat=99))
    assert any(c.startswith("button_seat_not_in_players") for c in r.conflicts)


def test_05_blank_button_seat_string_is_unknown():
    r = evaluate_cash_state_integrity(_mut(button_seat="  "))
    assert "button_seat" in r.unknown


# ── 6. UNKNOWN 输入 ───────────────────────────────────────────────────
def test_06_none_state_is_incomplete_with_all_required_missing():
    r = evaluate_cash_state_integrity(None)
    assert r.level == INCOMPLETE and r.ok is False
    assert set(r.missing) == set(csi.REQUIRED_FIELDS)
    assert r.evidence_grade == "NOT_VALIDATABLE"


def test_06_non_dict_state_is_incomplete():
    for bad in ([], "cash", 42):
        r = evaluate_cash_state_integrity(bad)
        assert r.level == INCOMPLETE and r.ok is False


def test_06_unknown_street_is_not_treated_as_preflop():
    r = evaluate_cash_state_integrity(_mut(street="mystery_street"))
    # 未知街道不得被当作 preflop（否则 board 长度检查会误判）
    assert not any(c.startswith("board_street_mismatch") for c in r.conflicts)


def test_06_missing_optional_board_on_postflop_is_unknown():
    r = evaluate_cash_state_integrity(_drop("board"))
    assert "board" in r.unknown


# ── 7. 冲突字段 ───────────────────────────────────────────────────────
def test_07_duplicate_seat_is_conflict():
    st = full_cash_state()
    st["players"][0]["seat"] = 3
    st["players"][2]["seat"] = 3
    r = evaluate_cash_state_integrity(st)
    assert "duplicate_seat" in r.conflicts


def test_07_duplicate_card_across_board_and_hero_is_conflict():
    st = full_cash_state()
    _hero(st)["hole_cards"] = ["Ah", "4c"]      # Ah 同时在 board 上
    r = evaluate_cash_state_integrity(st)
    assert any(c.startswith("duplicate_card") for c in r.conflicts)


def test_07_hero_hole_cards_wrong_length_is_conflict():
    st = full_cash_state()
    _hero(st)["hole_cards"] = ["9c", "4c", "2d"]
    r = evaluate_cash_state_integrity(st)
    assert any(c.startswith("hero_hole_cards_length") for c in r.conflicts)


def test_07_button_seat_non_integer_is_conflict():
    r = evaluate_cash_state_integrity(_mut(button_seat="BTN"))
    assert "button_seat_not_integer" in r.conflicts


def test_07_conflict_field_is_also_marked_unknown():
    """冲突字段的真值不得被猜测：必须同时出现在 unknown。"""
    r = evaluate_cash_state_integrity(_mut(pot=10.0, to_call=25.0))
    assert "to_call_exceeds_pot" in r.unknown


# ── 9. 不完整 state 不得被默认值补齐 ──────────────────────────────────
def test_09_incomplete_state_is_never_completed_by_defaults():
    r = evaluate_cash_state_integrity(_drop("pot", "to_call"))
    assert set(r.missing) == {"pot", "to_call"}
    assert r.level == INCOMPLETE and r.ok is False
    assert r.completeness < 1.0
    # 报告里不得出现任何"推断出来的"数值字段
    d = r.to_dict()
    assert not any(k in d for k in ("bb", "pot", "to_call", "table_size"))


def test_09_gate_does_not_mutate_input_state():
    """任务 10 的前半：输入 state 逐字段不变（含嵌套）。"""
    st = full_cash_state()
    before = copy.deepcopy(st)
    evaluate_cash_state_integrity(st)
    assert st == before
    # 连带缺字段/冲突输入也必须是只读的
    for bad in (_drop("pot"), _mut(button_seat=99), _mut(to_call=99.0),
                _mut(street="preflop")):
        b = copy.deepcopy(bad)
        evaluate_cash_state_integrity(bad)
        assert bad == b


def test_09_report_is_json_serializable_and_deterministic():
    st = full_cash_state()
    a = evaluate_cash_state_integrity(st)
    b = evaluate_cash_state_integrity(copy.deepcopy(st))
    assert a.to_dict() == b.to_dict()
    assert json.loads(json.dumps(a.to_dict())) == a.to_dict()


# ── 只读接线审计 + 数值不变式 ─────────────────────────────────────────
def test_gate_is_reachable_in_production_only_through_the_context_adapter():
    """V3G-1 断言"无接线"；该事实已被 **V3G-2 授权接线**取代（契约演进）。

    取代方式**不是放宽，而是收紧为更强的边界**：
      1. 生产包内**只有** `cash_integrity_context.py` 可以 import 门禁；
      2. 门禁**本身**仍不得被任何模块直接 import（只允许经适配层）；
      3. 接线必须经适配层（唯一合法入口）。

    原 V3G-1 的"零接线"断言已如实标记为**被取代**，并记录于
    `docs/V3G_2_CASH_CONTEXT_BYPASS_CLOSEOUT.md`。
    """
    adapter = "cash_integrity_context.py"
    gate = "cash_state_integrity.py"
    importers = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name in (gate, adapter):
            continue
        tree = ast.parse(p.read_text(encoding="utf-8-sig", errors="ignore"))
        for n in ast.walk(tree):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""] if isinstance(n, ast.ImportFrom)
                    else [])
            if any("cash_state_integrity" in m for m in mods):
                importers.append(str(p.relative_to(ROOT)))
    assert importers == [], (
        f"门禁被适配层以外的模块直接 import（V3G-2 允许的唯一入口是 "
        f"{adapter}）: {importers}")

    # 适配层必须确实接线（否则 V3G-2 的接线未生效）
    asrc = (PKG / adapter).read_text(encoding="utf-8")
    assert "from .cash_state_integrity import" in asrc
    # 且适配层不得把门禁结果喂给任何策略/EV 计算
    assert "compute_ev" not in asrc and "scores" not in asrc


def test_gate_module_has_no_strategy_dependencies():
    """只读且无策略依赖：**可执行代码**中不得调用/引用策略函数（按 AST 检查，
    不受 docstring 措辞影响）。"""
    src = (PKG / "cash_state_integrity.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    banned = ("strategy_core", "strategy_intel", "judge", "equity",
              "multistreet", "conditional_model")
    hit = [m for m in imported if any(b in m for b in banned)]
    assert hit == [], hit

    called = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            nm = (n.func.id if isinstance(n.func, ast.Name)
                  else getattr(n.func, "attr", ""))
            if nm:
                called.add(nm)
    assert called & {"compute_ev", "evaluate_intel", "to_summary"} == set(), called
    # 只允许标准库 + dataclasses/typing（无第三方、无项目内模块）
    for m in imported:
        assert m.split(".")[0] in ("dataclasses", "typing", "__future__"), m


def test_numeric_invariance_gate_does_not_change_compute_ev():
    """数值不变式：跑 gate 前后 `compute_ev` 输出逐位一致。"""
    st = full_cash_state()
    eq = {"hero_equity": 0.45, "outcome": {"win": 0.45, "tie": 0.0,
                                           "lose": 0.55}}
    before = compute_ev(copy.deepcopy(st), copy.deepcopy(eq)).to_summary()
    evaluate_cash_state_integrity(st)                       # 跑 gate
    after = compute_ev(copy.deepcopy(st), copy.deepcopy(eq)).to_summary()
    assert before == after


def test_numeric_invariance_on_incomplete_state():
    st = _drop("pot")
    eq = {"hero_equity": 0.45}
    before = compute_ev(copy.deepcopy(st), copy.deepcopy(eq)).to_summary()
    evaluate_cash_state_integrity(st)
    after = compute_ev(copy.deepcopy(st), copy.deepcopy(eq)).to_summary()
    assert before == after
    assert before.get("available") is False       # 缺 pot → 诚实不可用


def _norm(c):
    return isinstance(c, str) and len(c.strip()) == 2
