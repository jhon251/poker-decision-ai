"""V3G-3 — CASH Preflop Decision Layer 测试（授权矩阵 1–20）。

覆盖：2/3/4/5/6 人、空座、hero cards 缺失、position UNKNOWN、stack UNKNOWN、
action history 不完整、非法 raise、all-in、call/check legality、已 fold 玩家不得
行动、CASH mode、MTT/PKO/Mystery 隔离、V3G-1 gate、V3G-2 evidence、
确定性/可复现。

**不修改**任何既有测试定义；**不触碰**冻结文件。
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.cash_preflop_decision import (
    PREFLOP_LAYER_VERSION, build_preflop_decision)
from poker_decision_ai.decision_contract import validate_action

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

POS_6MAX = ("SB", "BB", "UTG", "MP", "CO", "BTN")


def preflop_state(n: int = 3, **over) -> dict:
    """合法 n 人 preflop state（位置为**既有标签**，不由本层推导）。"""
    players = []
    for i in range(n):
        is_hero = (i == n - 1)                     # hero 在 BTN
        players.append({
            "player_id": "hero" if is_hero else f"v{i + 1}",
            "seat": i + 1,
            "position": POS_6MAX[i] if n == 6 else POS_6MAX[-n + i],
            "status": "active",
            "stack": 100.0,
            "is_hero": is_hero,
            "hole_cards": ["As", "Kd"] if is_hero else [],
        })
    st = {
        "game_type": "cash", "street": "preflop", "board": [],
        "pot": 3.0, "to_call": 2.5, "bb": 1.0, "sb": 0.5,
        "min_raise_to": 5.0, "hero_id": "hero", "button_seat": n,
        "table_size": n, "synthetic_test_only": True,
        "players": players,
        "action_history": [{"street": "preflop", "player_id": players[0]["player_id"],
                            "action": "raise", "amount": 2.5}],
    }
    st.update(over)
    return st


# ── 1/2/3/4/5. 各人数正常 preflop ─────────────────────────────────────
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_01_to_05_normal_preflop_by_table_size(n):
    r = build_preflop_decision(preflop_state(n))
    assert r["present"] is True and r["version"] == PREFLOP_LAYER_VERSION
    ctx = r["preflop_context"]
    assert ctx["street"] == "preflop" and ctx["mode"] == "cash"
    assert ctx["ready"] is True and ctx["incomplete"] == []
    assert ctx["effective_stack_bb"] == 100.0
    assert ctx["hero"]["hole_cards"] == ["As", "Kd"]
    assert len(ctx["seats_in_hand"]) == n
    acts = {c["action"] for c in r["candidates"]}
    assert acts and acts <= set(ctx["legal_actions"])


def test_06_empty_seats_are_not_seated_players():
    st = preflop_state(6)
    # 座位 2/4 空座（seat 存在但无 player 条目）→ seats_in_hand 只含真实玩家
    st["players"] = [p for p in st["players"] if p["seat"] not in (2, 4)]
    st["players"][-1]["is_hero"] = True
    for p in st["players"]:
        p["position"] = "BTN" if p["is_hero"] else "BB"
    r = build_preflop_decision(st)
    assert r["present"] is True
    assert len(r["preflop_context"]["seats_in_hand"]) == 4


# ── 7/8/9. hero cards / position / stack UNKNOWN → 不猜、不可用 ─────────
def test_07_missing_hero_cards_is_not_ready_and_no_candidates():
    st = preflop_state()
    st["players"][-1]["hole_cards"] = []
    r = build_preflop_decision(st)
    ctx = r["preflop_context"]
    assert ctx["ready"] is False
    assert "hero.hole_cards" in ctx["incomplete"]
    assert r["candidates"] == []                    # 不猜一个动作
    assert ctx["hero"]["hole_cards"] is None
    assert ctx["hero"]["hole_cards_unknown"] is True


def test_07_null_card_element_is_unknown():
    st = preflop_state()
    st["players"][-1]["hole_cards"] = ["As", None]
    r = build_preflop_decision(st)
    assert r["preflop_context"]["ready"] is False
    assert "hero.hole_cards" in r["preflop_context"]["incomplete"]


def test_08_position_unknown_is_not_ready_and_not_defaulted():
    st = preflop_state()
    st["players"][-1]["position"] = None
    r = build_preflop_decision(st)
    ctx = r["preflop_context"]
    assert ctx["ready"] is False
    assert "hero.position" in ctx["incomplete"]
    assert ctx["hero"]["position"] is None          # 不得默认 BTN/任何位置
    assert r["candidates"] == []


def test_09_stack_unknown_is_not_ready_and_not_defaulted():
    st = preflop_state()
    st["players"][-1]["stack"] = None
    r = build_preflop_decision(st)
    ctx = r["preflop_context"]
    assert ctx["ready"] is False
    assert "hero.stack" in ctx["incomplete"]
    assert ctx["hero"]["stack"] is None             # 不得默认 100
    assert ctx["effective_stack_bb"] is None
    assert r["candidates"] == []


# ── 10. action history 不完整 → 如实标注，不猜测 ──────────────────────
def test_10_incomplete_action_history_is_flagged_per_entry():
    st = preflop_state()
    st["action_history"] = [
        {"street": "preflop", "player_id": "v1", "action": "raise", "amount": 2.5},
        {"street": "preflop", "player_id": "v2"},          # action 缺失
        {"player_id": "v3", "action": "call", "amount": None},
    ]
    r = build_preflop_decision(st)
    hist = r["preflop_context"]["action_history"]
    assert len(hist) == 3
    assert hist[0]["evidence"] == "declared"
    assert hist[1]["action"] is None and hist[1]["evidence"] == "unknown"
    assert hist[2]["amount"] is None and hist[2]["amount_bb"] is None
    # 不得为缺失项编造 actor/position/amount
    assert hist[1]["actor"] == "v2"


def test_10_missing_action_history_is_empty_not_defaulted():
    st = preflop_state()
    st.pop("action_history", None)
    r = build_preflop_decision(st)
    assert r["preflop_context"]["action_history"] == []


# ── 11. 非法 raise ────────────────────────────────────────────────────
def test_11_raise_below_min_raise_to_is_rejected():
    """min_raise_to=14、to_call=2.5：2.5×to_call=6.25 < 14 必须被拒
    （拒绝路径被记录），3.0×to_call=7.5 亦 < 14 被拒 → 只剩 14 与 all-in。"""
    st = preflop_state(to_call=2.5, min_raise_to=14.0)
    r = build_preflop_decision(st)
    raises = [c for c in r["candidates"] if c["action"] == "raise"]
    assert raises, r["candidates"]
    assert sorted(c["amount_bb"] for c in raises) == [14.0, 100.0]
    assert all(c["amount_bb"] >= 14.0 - 1e-9 for c in raises)
    assert any(p.startswith("raise_below_min_raise_to") for p in r["problems"])


def test_11_all_raises_rejected_when_min_raise_to_exceeds_stack():
    """min_raise_to 超过有效栈 → 不存在合法加注：不得生成任何 raise 候选。"""
    st = preflop_state(to_call=2.5, min_raise_to=200.0)
    r = build_preflop_decision(st)
    assert not [c for c in r["candidates"] if c["action"] == "raise"]
    assert r["problems"].count("raise_below_min_raise_to:100.0") >= 1
    # fold/call 仍然合法且生成
    assert {c["action"] for c in r["candidates"]} == {"fold", "call"}


def test_11_min_raise_to_unknown_yields_no_raise_candidates():
    st = preflop_state()
    st.pop("min_raise_to", None)
    r = build_preflop_decision(st)
    assert not [c for c in r["candidates"] if c["action"] == "raise"]
    assert "min_raise_to_unknown_no_raise_candidates" in r["problems"]


def test_11_raise_never_exceeds_effective_stack():
    st = preflop_state()
    st["players"][-1]["stack"] = 20.0            # 20bb 深
    r = build_preflop_decision(st)
    for c in r["candidates"]:
        if c["amount_bb"] is not None:
            assert c["amount_bb"] <= 20.0 + 1e-9, c


# ── 12. all-in ────────────────────────────────────────────────────────
def test_12_allin_is_marked_and_capped_at_effective_stack():
    st = preflop_state()
    st["players"][-1]["stack"] = 30.0
    r = build_preflop_decision(st)
    allins = [c for c in r["candidates"] if c["allin"]]
    assert allins
    assert all(abs(c["amount_bb"] - 30.0) <= 1e-9 for c in allins)


def test_12_short_stack_raises_are_all_rejected_by_min_raise_to():
    """有效栈 2bb、min_raise_to 5bb → 所有加注都因低于最小加注被拒；
    只剩 fold/call（非法加注绝不生成）。"""
    st = preflop_state(to_call=1.5, pot=3.0)
    st["players"][-1]["stack"] = 2.0
    r = build_preflop_decision(st)
    ctx = r["preflop_context"]
    assert ctx["ready"] is True, ctx["incomplete"]
    assert {c["action"] for c in r["candidates"]} == {"fold", "call"}
    call = next(c for c in r["candidates"] if c["action"] == "call")
    assert abs(call["amount_bb"] - 1.5) <= 1e-9
    assert any(p.startswith("raise_below_min_raise_to") for p in r["problems"])


def test_12_call_never_exceeds_effective_stack_and_allin_flag_is_exact():
    """call 金额恒 ≤ 有效栈；等于有效栈时 allin 标记必须为真（构造性保证）。"""
    for stack, tc in ((100.0, 2.5), (50.0, 2.5), (10.0, 2.5), (2.5, 2.5)):
        st = preflop_state(to_call=tc, pot=3.0)
        st["players"][-1]["stack"] = stack
        r = build_preflop_decision(st)
        if not r["preflop_context"]["ready"]:
            continue                     # 门禁判定不可用 → 不产出候选（见下一测试）
        call = next(c for c in r["candidates"] if c["action"] == "call")
        eff_bb = r["preflop_context"]["effective_stack_bb"]
        assert call["amount_bb"] <= eff_bb + 1e-9
        assert call["allin"] is (abs(call["amount_bb"] - eff_bb) <= 1e-9)


def test_12_v3g1_flags_to_call_ge_stack_as_conflict_so_no_candidates():
    """**交互边界（如实登记，非本层决定）**：V3G-1 门禁把
    `to_call > hero_stack` 视为冲突 → `INCOMPLETE` → 本层 `ready=False`
    且不产出候选（含合法的 fold/call）。

    真实牌局中"跟注全下"是合法动作，因此这是**门禁的近似限制**（V3G-1 契约
    如此，本阶段不修改）。此处固定该交互，供后续阶段评估是否放宽，
    **不**在本层偷偷绕过。
    """
    st = preflop_state(to_call=2.5, pot=3.0)
    st["players"][-1]["stack"] = 2.0
    r = build_preflop_decision(st)
    integ = r["preflop_context"]["integrity"]
    assert any(c.startswith("to_call_exceeds_hero_stack")
               for c in integ["conflicts"])
    assert integ["level"] == "INCOMPLETE"
    assert r["preflop_context"]["ready"] is False
    assert r["candidates"] == []


def test_12_no_bet_short_stack_allin_bet_is_capped_and_flagged():
    """无下注 + 短栈：all-in 以 bet 形式给出并被有效栈封顶。"""
    st = preflop_state(to_call=0.0, pot=9.0)
    st["action_history"] = []
    st["players"][-1]["stack"] = 5.0
    r = build_preflop_decision(st)
    assert r["preflop_context"]["legal_actions"] == ["check", "bet", "allin"]
    bets = [c for c in r["candidates"] if c["action"] == "bet"]
    assert bets and all(c["amount_bb"] <= 5.0 + 1e-9 for c in bets)
    assert any(c["allin"] for c in bets)
    # 如实登记：合法集含字面 "allin"，但候选层一律以 bet/raise 表达全下
    assert not [c for c in r["candidates"] if c["action"] == "allin"]


# ── 13. call / check legality ─────────────────────────────────────────
def test_13_no_bet_facing_gives_check_and_bet_not_call_or_raise():
    st = preflop_state(to_call=0.0, pot=3.0)
    st["action_history"] = []
    r = build_preflop_decision(st)
    acts = {c["action"] for c in r["candidates"]}
    assert acts <= {"check", "bet"}
    assert "call" not in acts and "raise" not in acts
    assert "check" in acts
    assert next(c for c in r["candidates"] if c["action"] == "check")[
        "amount_bb"] in (0.0, None)


def test_13_every_candidate_passes_the_frozen_legality_gate():
    for n in (2, 3, 6):
        r = build_preflop_decision(preflop_state(n))
        legal = r["preflop_context"]["legal_actions"]
        for c in r["candidates"]:
            ok, why = validate_action(c["action"], legal)
            assert ok, (c["action"], why)


def test_13_call_amount_matches_to_call_not_more():
    st = preflop_state(to_call=2.5)
    r = build_preflop_decision(st)
    call = next(c for c in r["candidates"] if c["action"] == "call")
    assert abs(call["amount_bb"] - 2.5) <= 1e-9


# ── 14. 已 fold 玩家不得再行动 ────────────────────────────────────────
def test_14_folded_player_acting_later_is_flagged():
    st = preflop_state()
    st["players"][0]["status"] = "folded"
    st["action_history"] = [
        {"street": "preflop", "player_id": st["players"][0]["player_id"],
         "action": "fold", "amount": None},
        {"street": "preflop", "player_id": st["players"][0]["player_id"],
         "action": "call", "amount": 2.5},        # 已 fold 后又行动 → 违规
    ]
    r = build_preflop_decision(st)
    assert any(p.startswith("folded_player_acted") for p in r["problems"])


def test_14_folded_players_are_excluded_from_seats_in_hand():
    st = preflop_state()
    st["players"][0]["status"] = "folded"
    r = build_preflop_decision(st)
    seated = [s["player_id"] for s in r["preflop_context"]["seats_in_hand"]]
    assert st["players"][0]["player_id"] not in seated


# ── 15/16. CASH mode + MTT/PKO/Mystery 隔离（fail closed）─────────────
def test_15_cash_mode_context_present():
    r = build_preflop_decision(preflop_state())
    assert r["present"] is True
    assert r["preflop_context"]["mode"] == "cash"


@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin", None])
def test_16_non_cash_modes_fail_closed(mode):
    r = build_preflop_decision(preflop_state(game_type=mode))
    assert r["present"] is False
    assert "preflop_context" not in r and "candidates" not in r
    assert "mode_not_cash" in r["reason"] or "not_cash_mode" in r["reason"]


def test_16_no_icm_bounty_or_tournament_fields_leak_into_context():
    r = build_preflop_decision(preflop_state())
    blob = json.dumps(r)
    for banned in ("icm", "bounty", "satellite", "payout", "tournament",
                   "blind_pressure", "mystery"):
        assert banned not in blob.lower(), banned


@pytest.mark.parametrize("street", ["flop", "turn", "river"])
def test_16_non_preflop_street_fails_closed(street):
    r = build_preflop_decision(preflop_state(street=street, board=["Ah", "7c", "2c"]))
    assert r["present"] is False
    assert r["reason"].startswith("not_preflop")


# ── 17/18. V3G-1 gate + V3G-2 evidence ───────────────────────────────
def test_17_v3g1_gate_result_is_carried():
    r = build_preflop_decision(preflop_state())
    integ = r["preflop_context"]["integrity"]
    assert integ["level"] == "COMPLETE"
    assert integ["missing"] == [] and integ["conflicts"] == []
    assert integ["evidence_grade"] in ("SYNTHETIC", "REAL_OBSERVED",
                                       "HISTORICAL_REPLAY", "COUNTERFACTUAL",
                                       "MECHANISM_ONLY", "NOT_VALIDATABLE")


def test_17_gate_conflict_blocks_readiness():
    st = preflop_state(street="preflop", board=["Ah", "7c", "2c"])  # 冲突
    r = build_preflop_decision(st)
    # board/preflop 冲突 → 门禁 INCOMPLETE → 不 ready（不猜）
    assert r["preflop_context"]["integrity"]["level"] == "INCOMPLETE"
    assert r["preflop_context"]["ready"] is False
    assert r["candidates"] == []


def test_18_v3g2_evidence_context_is_carried():
    r = build_preflop_decision(preflop_state())
    ev = r["preflop_context"]["evidence"]
    assert ev["present"] is True and ev["mode"] == "cash"
    for k in ("state_integrity", "completeness", "missing", "conflicts",
              "unknown", "evidence_grade"):
        assert k in ev, k


# ── 19/20. 确定性 / 可复现 ────────────────────────────────────────────
def test_19_candidates_are_deterministic_and_json_safe():
    a = build_preflop_decision(preflop_state())
    b = build_preflop_decision(preflop_state())
    assert a == b
    json.dumps(a)


def test_20_repeated_runs_on_same_input_are_identical():
    st = preflop_state(6)
    first = build_preflop_decision(st)
    for _ in range(5):
        assert build_preflop_decision(st) == first


# ── 只读 / 分层 / 依赖审计 ────────────────────────────────────────────
def test_layer_does_not_mutate_input_state():
    st = preflop_state()
    before = copy.deepcopy(st)
    build_preflop_decision(st)
    assert st == before


def test_legality_and_preference_are_separated_and_no_preference_generated():
    """合法性先于偏好；本层**不产生**偏好（只能来自既有冻结参照）。"""
    r = build_preflop_decision(preflop_state())
    assert r["candidates"]
    for c in r["candidates"]:
        assert c["legality"]["ok"] is True
        assert c["preference"]["present"] is False
        assert c["preference"]["source"] == "frozen_reference"


def test_layer_does_not_compute_positions_or_ev():
    """不得重算 seat position、不得含 EV/equity（按 AST 检查可执行代码，
    不受 docstring 措辞影响）。"""
    src = (PKG / "cash_preflop_decision.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    called = set()
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            nm = (n.func.id if isinstance(n.func, ast.Name)
                  else getattr(n.func, "attr", ""))
            if nm:
                called.add(nm)
        if isinstance(n, ast.Name):
            names.add(n.id)
        if isinstance(n, ast.Attribute):
            names.add(n.attr)
    for banned in ("positions_for", "_positions_for", "compute_ev",
                   "evaluate_intel", "to_summary"):
        assert banned not in called, f"调用了 {banned}"
        assert banned not in names, f"引用了 {banned}"
    # 不得从 seat/button 反推位置：可执行代码中不出现 button_seat 取值
    assert "button_seat" not in {
        n.slice.value for n in ast.walk(tree)
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)}
    assert "equity" not in names and "pot_odds" not in names


def test_layer_imports_only_sanctioned_modules():
    src = (PKG / "cash_preflop_decision.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("research_unfreeze", "strategy_intel", "ice", "agents"):
        assert banned not in joined, banned
    assert set(imported) <= {"__future__", "dataclasses", "typing",
                             "cash_integrity_context", "decision_contract",
                             "domain.poker.strategy_core"}, imported
    # 门禁只经 V3G-2 适配层到达（本层不得直接 import 门禁）
    assert "cash_state_integrity" not in " ".join(imported)
