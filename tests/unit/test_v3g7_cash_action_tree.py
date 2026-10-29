"""V3G-7 — CASH Action Tree / Bet Sizing 测试（授权清单 1–25）。

**不修改**任何既有测试定义；**不触碰**冻结文件。
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.cash_action_tree import (
    ACTION_TREE_VERSION, SIZING_KINDS, TERMINAL_ALLIN, TERMINAL_FOLD,
    TERMINAL_RIVER, TERMINAL_SHOWDOWN, build_action_tree)
from poker_decision_ai.decision_contract import derive_legal_actions
from poker_decision_ai.domain.poker.strategy_core import (BET_FRACTIONS,
                                                          RAISE_TO_MULTS)

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"


def st(**over) -> dict:
    b = {
        "game_type": "cash", "street": "flop", "board": ["Ah", "7c", "2c"],
        "pot": 20.0, "to_call": 5.0, "bb": 1.0, "sb": 0.5,
        "min_raise_to": 10.0, "hero_id": "hero", "button_seat": 2,
        "table_size": 3, "synthetic_test_only": True,
        "players": [
            {"player_id": "v1", "seat": 1, "position": "BB", "status": "active",
             "stack": 100.0, "hole_cards": []},
            {"player_id": "hero", "seat": 2, "position": "BTN", "is_hero": True,
             "status": "active", "hole_cards": ["As", "Kd"], "stack": 100.0},
            {"player_id": "v2", "seat": 3, "position": "SB", "status": "active",
             "stack": 100.0, "hole_cards": []},
        ],
        "action_history": [{"street": "flop", "player_id": "v1",
                            "action": "bet", "amount": 5.0}],
    }
    b.update(over)
    return b


def _sizing(r, action=None, kind=None):
    out = r["sizing"]
    if action:
        out = [c for c in out if c["action"] == action]
    if kind:
        out = [c for c in out if c["sizing_kind"] == kind]
    return out


# ── 1. check / call / fold ────────────────────────────────────────────
def test_01_fold_and_call_when_facing_a_bet():
    r = build_action_tree(st())
    assert r["legal_actions"] == ["fold", "call", "raise"]
    fold = _sizing(r, "fold")[0]
    call = _sizing(r, "call")[0]
    assert fold["amount_bb"] == 0.0 and fold["sizing_kind"] == "fold_no_investment"
    assert call["amount_bb"] == 5.0 and call["sizing_kind"] == "call_match_to_call"
    assert "check" not in {c["action"] for c in r["sizing"]}


def test_01_check_when_no_bet_facing():
    r = build_action_tree(st(to_call=0.0, action_history=[]))
    assert r["legal_actions"] == ["check", "bet"]
    chk = _sizing(r, "check")[0]
    assert chk["amount_bb"] == 0.0 and chk["sizing_kind"] == "check_no_investment"
    assert not _sizing(r, "call") and not _sizing(r, "fold")


# ── 2. min-raise ──────────────────────────────────────────────────────
def test_02_min_raise_to_is_used_verbatim():
    r = build_action_tree(st(min_raise_to=11.0))
    mr = _sizing(r, "raise", "raise_min_raise_to")
    assert mr and mr[0]["amount_bb"] == 11.0
    assert mr[0]["source"] == "state.min_raise_to"
    assert mr[0]["raise_to_bb"] == 11.0


def test_02_min_raise_to_not_a_raise_is_rejected():
    """min_raise_to ≤ to_call 不构成加注 → 记入 rejected，不产出该尺寸。"""
    r = build_action_tree(st(to_call=5.0, min_raise_to=5.0))
    assert any(x.startswith("min_raise_to_not_a_raise") for x in r["rejected"])
    assert not _sizing(r, "raise", "raise_min_raise_to")


# ── 3. pot fraction ───────────────────────────────────────────────────
def test_03_bet_pot_fractions_come_from_frozen_constants():
    r = build_action_tree(st(to_call=0.0, action_history=[], pot=100.0))
    bets = _sizing(r, "bet", "bet_pot_fraction")
    assert len(bets) == len(BET_FRACTIONS)
    expected = sorted({round(min(100.0 * f, 100.0), 6) for f in BET_FRACTIONS})
    assert sorted(c["amount_bb"] for c in bets) == expected
    for c in bets:
        assert c["source"].startswith("frozen_BET_FRACTIONS:")
        assert c["evidence"]


def test_03_pot_fraction_capped_by_effective_stack():
    r = build_action_tree(st(to_call=0.0, action_history=[], pot=1000.0,
                             bb=1.0))
    for c in r["sizing"]:
        assert c["amount_bb"] <= r["effective_stack_bb"] + 1e-9


# ── 4. raise multiple ─────────────────────────────────────────────────
def test_04_raise_multiples_come_from_frozen_constants():
    r = build_action_tree(st(to_call=5.0, min_raise_to=10.0))
    mults = _sizing(r, "raise", "raise_to_multiple")
    assert len(mults) == len(RAISE_TO_MULTS)
    for c, m in zip(mults, sorted(RAISE_TO_MULTS)):
        assert c["source"] == f"frozen_RAISE_TO_MULTS:{m}"
        assert abs(c["raise_to_bb"] - 5.0 * m) < 1e-9


# ── 5. all-in ─────────────────────────────────────────────────────────
def test_05_allin_candidates_are_capped_and_flagged():
    r = build_action_tree(st())
    allins = [c for c in r["sizing"] if c["allin"]]
    assert allins
    for c in allins:
        assert abs(c["amount_bb"] - r["effective_stack_bb"]) < 1e-9


def test_05_short_effective_stack_gives_capped_bet_allin():
    r = build_action_tree(st(to_call=0.0, action_history=[], pot=1000.0,
                             players=[
                                 {"player_id": "v1", "seat": 1,
                                  "position": "BB", "status": "active",
                                  "stack": 100.0},
                                 {"player_id": "hero", "seat": 2,
                                  "position": "BTN", "is_hero": True,
                                  "status": "active", "stack": 30.0},
                                 {"player_id": "v2", "seat": 3,
                                  "position": "SB", "status": "active",
                                  "stack": 100.0}]))
    assert r["effective_stack_bb"] == 30.0
    bets = _sizing(r, "bet")
    assert bets and all(c["amount_bb"] <= 30.0 + 1e-9 for c in bets)
    assert any(c["allin"] for c in bets)


# ── 6. short-stack ────────────────────────────────────────────────────
def test_06_stack_below_to_call_has_no_raise_at_all():
    """有效栈(0.5bb) < to_call(5bb) → 既有 `derive_legal_actions` 只给
    fold/call（**没有** raise），因此本层不产出任何 raise，也不产生
    "拒绝"记录（因为根本没有 raise 进入候选流程）。
    call 被有效栈封顶并标 all-in。"""
    r = build_action_tree(st(players=[
        {"player_id": "v1", "seat": 1, "position": "BB", "status": "active",
         "stack": 100.0},
        {"player_id": "hero", "seat": 2, "position": "BTN", "is_hero": True,
         "status": "active", "stack": 0.5},
        {"player_id": "v2", "seat": 3, "position": "SB", "status": "active",
         "stack": 100.0}]))
    assert r["effective_stack_bb"] == 0.5
    assert r["legal_actions"] == ["fold", "call"]
    assert {c["action"] for c in r["sizing"]} == {"fold", "call"}
    call = _sizing(r, "call")[0]
    assert call["amount_bb"] == 0.5 and call["allin"] is True
    assert call["sizing_kind"] == "call_allin_capped"
    assert r["rejected"] == []


def test_06_min_raise_to_above_stack_rejects_allin_raise():
    """min_raise_to(10bb) > 有效栈(5bb) → 唯一可能的 raise（全下）低于最小
    加注 → **不构成合法加注**，拒绝并记录。"""
    r = build_action_tree(st(to_call=2.0, pot=20.0, min_raise_to=10.0,
                             players=[
                                 {"player_id": "v1", "seat": 1,
                                  "position": "BB", "status": "active",
                                  "stack": 100.0},
                                 {"player_id": "hero", "seat": 2,
                                  "position": "BTN", "is_hero": True,
                                  "status": "active", "stack": 5.0},
                                 {"player_id": "v2", "seat": 3,
                                  "position": "SB", "status": "active",
                                  "stack": 100.0}]))
    assert r["effective_stack_bb"] == 5.0
    assert not _sizing(r, "raise")
    assert any(x.startswith("raise_below_min_raise_to") for x in r["rejected"])


# ── 7. to_call ────────────────────────────────────────────────────────
def test_07_call_amount_tracks_to_call():
    for tc in (0.5, 5.0, 25.0):
        r = build_action_tree(st(to_call=tc, pot=max(tc, 20.0)))
        call = _sizing(r, "call")[0]
        assert abs(call["amount_bb"] - tc) < 1e-9, tc
        assert abs(call["bounds"]["to_call_bb"] - tc) < 1e-9


def test_07_call_capped_at_effective_stack_is_flagged():
    r = build_action_tree(st(to_call=50.0, pot=100.0, min_raise_to=100.0,
                             players=[
                                 {"player_id": "v1", "seat": 1,
                                  "position": "BB", "status": "active",
                                  "stack": 100.0},
                                 {"player_id": "hero", "seat": 2,
                                  "position": "BTN", "is_hero": True,
                                  "status": "active", "stack": 30.0},
                                 {"player_id": "v2", "seat": 3,
                                  "position": "SB", "status": "active",
                                  "stack": 100.0}]))
    call = _sizing(r, "call")[0]
    assert call["sizing_kind"] == "call_allin_capped"
    assert call["amount_bb"] == 30.0 and call["allin"] is True


# ── 8. min_raise_to 缺失 ──────────────────────────────────────────────
def test_08_missing_min_raise_to_yields_no_raise_candidates():
    s = st()
    s.pop("min_raise_to", None)
    r = build_action_tree(s)
    assert not _sizing(r, "raise")
    assert any(x.startswith("min_raise_to_unknown") for x in r["rejected"])
    assert "min_raise_to" in r["unknown"]
    assert r["min_raise_to"] is None


# ── 9. amount / amount_bb 单位 ────────────────────────────────────────
def test_09_amount_units_are_explicit_and_consistent():
    r = build_action_tree(st(bb=2.0, to_call=10.0, pot=40.0, min_raise_to=20.0))
    for c in r["sizing"]:
        assert c["unit"] == {"amount": "chips", "amount_bb": "bb"}
        assert abs(c["amount"] - c["amount_bb"] * 2.0) < 1e-6
    call = _sizing(r, "call")[0]
    assert call["amount_bb"] == 5.0 and call["amount"] == 10.0


# ── 10. raise-to 与增量 bet ───────────────────────────────────────────
def test_10_raise_to_and_increment_are_distinguished():
    r = build_action_tree(st(to_call=5.0, min_raise_to=10.0))
    for c in _sizing(r, "raise"):
        assert "raise_to_bb" in c and "increment_bb" in c
        assert abs(c["increment_bb"] - (c["raise_to_bb"] - 5.0)) < 1e-6
    # 非 raise 动作不得携带 raise_to_bb
    for c in r["sizing"]:
        if c["action"] != "raise":
            assert "raise_to_bb" not in c and "increment_bb" not in c


# ── 11. 非法 action ───────────────────────────────────────────────────
def test_11_illegal_actions_never_enter_the_tree():
    r = build_action_tree(st())
    legal = set(r["legal_actions"])
    for c in r["sizing"]:
        assert c["action"] in legal, c
    # 与既有唯一口径一致
    assert set(r["legal_actions"]) == set(derive_legal_actions(5.0, 100.0, 20.0))


def test_11_no_check_when_facing_bet_and_no_call_when_not():
    facing = build_action_tree(st(to_call=5.0, min_raise_to=10.0))
    assert "check" not in {c["action"] for c in facing["sizing"]}
    not_facing = build_action_tree(st(to_call=0.0, action_history=[]))
    assert "call" not in {c["action"] for c in not_facing["sizing"]}


# ── 12. 非法 amount ───────────────────────────────────────────────────
def test_12_rejected_amounts_are_recorded_with_reasons():
    r = build_action_tree(st(min_raise_to=200.0))     # 高于有效栈
    assert r["rejected"]
    assert all(isinstance(x, str) and x for x in r["rejected"])
    assert all(c["amount_bb"] >= 0 for c in r["sizing"])


# ── 13. 超 effective stack ────────────────────────────────────────────
def test_13_no_sizing_exceeds_effective_stack():
    for stack in (5.0, 20.0, 100.0):
        r = build_action_tree(st(players=[
            {"player_id": "v1", "seat": 1, "position": "BB",
             "status": "active", "stack": 200.0},
            {"player_id": "hero", "seat": 2, "position": "BTN",
             "is_hero": True, "status": "active", "stack": stack},
            {"player_id": "v2", "seat": 3, "position": "SB",
             "status": "active", "stack": 200.0}]))
        eff = r["effective_stack_bb"]
        for c in r["sizing"]:
            assert c["amount_bb"] <= eff + 1e-9, (stack, c)


# ── 14. UNKNOWN / incomplete ──────────────────────────────────────────
@pytest.mark.parametrize("missing", ["bb", "pot", "to_call"])
def test_14_missing_required_field_is_incomplete(missing):
    s = st()
    s.pop(missing, None)
    r = build_action_tree(s)
    assert r["present"] is True
    assert r["complete"] is False
    assert missing in r["unknown"]
    assert r["sizing"] == [] and r["transitions"] == []


def test_14_incomplete_never_fabricates_transitions():
    s = st()
    s.pop("bb", None)
    r = build_action_tree(s)
    assert r["transitions"] == []


def test_14_unknown_street_fails_closed():
    s = st()
    s["street"] = None
    r = build_action_tree(s)
    assert r["present"] is False
    assert "street" in r["unknown"]


def test_14_no_defaults_for_stack_or_table_size_or_position():
    """缺 stack → 不默认；缺 table_size/position 也不得被补。"""
    s = st()
    s["players"][1].pop("stack", None)
    s.pop("table_size", None)
    s["players"][1].pop("position", None)
    r = build_action_tree(s)
    assert r["complete"] is False
    assert "hero.stack" in r["unknown"]
    assert r["hero_stack"] is None and r["effective_stack_bb"] is None


def test_14_unknown_is_not_empty():
    """UNKNOWN ≠ EMPTY：缺失字段为 None/unknown，不得写成 0。"""
    s = st()
    s.pop("min_raise_to", None)
    r = build_action_tree(s)
    assert r["min_raise_to"] is None
    assert r["min_raise_to_bb"] is None
    assert "min_raise_to" in r["unknown"]


# ── 15. action history ────────────────────────────────────────────────
def test_15_action_history_is_carried_structurally():
    r = build_action_tree(st())
    assert r["action_history"] == [{"street": "flop", "actor": "v1",
                                    "action": "bet", "amount": 5.0}]


def test_15_missing_action_history_is_flagged_not_defaulted():
    s = st()
    s.pop("action_history", None)
    r = build_action_tree(s)
    assert r["action_history"] == []
    assert "action_history" in r["unknown"]
    assert r["current_actor"] is None


def test_15_current_actor_uses_seat_order_after_last_actor():
    r = build_action_tree(st())
    assert r["current_actor"] == "hero"


# ── 16. terminal ──────────────────────────────────────────────────────
def test_16_terminal_fold_when_everyone_folded():
    s = st()
    for p in s["players"]:
        p["status"] = "folded"
    r = build_action_tree(s)
    assert r["terminal"] is True and r["terminal_reason"] == TERMINAL_FOLD
    assert r["transitions"] == []


def test_16_terminal_allin_when_hero_allin():
    s = st()
    s["players"][1]["status"] = "allin"
    r = build_action_tree(s)
    assert r["terminal"] is True and r["terminal_reason"] == TERMINAL_ALLIN


def test_16_terminal_river_and_showdown():
    river_open = build_action_tree(st(street="river", to_call=0.0,
                                      board=["Ah", "7c", "2c", "2d"],
                                      action_history=[]))
    assert river_open["terminal_reason"] == TERMINAL_RIVER
    river_full = build_action_tree(st(street="river", to_call=0.0,
                                      board=["Ah", "7c", "2c", "2d", "3s"],
                                      action_history=[]))
    assert river_full["terminal_reason"] == TERMINAL_SHOWDOWN


def test_16_terminal_node_has_no_transitions():
    for s in (st(street="river", to_call=0.0,
                 board=["Ah", "7c", "2c", "2d", "3s"], action_history=[]),):
        r = build_action_tree(s)
        assert r["terminal"] is True and r["transitions"] == []


def test_16_transition_terminal_flags_for_fold_and_allin():
    r = build_action_tree(st())
    by = {t["action"] + ":" + t["sizing_kind"]: t for t in r["transitions"]}
    fold_t = by["fold:fold_no_investment"]
    assert fold_t["terminal"] is True and fold_t["terminal_reason"] == TERMINAL_FOLD
    allin_t = [t for t in r["transitions"] if t["becomes_allin"]]
    assert allin_t and all(t["terminal"] for t in allin_t)


# ── 17. CASH / non-CASH ───────────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin", None])
def test_17_non_cash_fails_closed(mode):
    r = build_action_tree(st(game_type=mode))
    assert r["present"] is False
    assert "sizing" not in r and "legal_actions" not in r


def test_17_non_dict_fails_closed():
    for bad in (None, [], "cash", 3):
        assert build_action_tree(bad)["present"] is False


def test_17_no_tournament_or_icm_fields():
    r = build_action_tree(st())
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("icm", "bounty", "satellite", "payout", "tournament",
                   "equity", "gto", "equilibrium"):
        assert banned not in blob, banned


# ── 18. 多玩家 / 多节点隔离 ───────────────────────────────────────────
def test_18_hero_only_and_players_isolated():
    st_a = st()
    st_b = st()
    st_b["players"][0]["stack"] = 55.0          # 对手栈变化
    a = build_action_tree(st_a)
    b = build_action_tree(st_b)
    # 对手栈不影响 hero 的尺寸候选（有效栈 = hero 栈，既有 v1 语义）
    assert a["sizing"] == b["sizing"]


def test_18_no_cross_player_action_attribution():
    r = build_action_tree(st())
    assert all(t["street"] == "flop" for t in r["transitions"])
    # 树只描述 hero 视角节点，不产出对手节点
    assert all("hero" not in t for t in r["transitions"])


# ── 19. 确定性 ────────────────────────────────────────────────────────
def test_19_repeated_runs_identical_and_json_safe():
    s = st()
    first = build_action_tree(s)
    for _ in range(5):
        assert build_action_tree(copy.deepcopy(s)) == first
    json.dumps(first)


def test_19_does_not_mutate_input():
    s = st()
    before = copy.deepcopy(s)
    build_action_tree(s)
    assert s == before


# ── 20. 不生成默认尺寸 ────────────────────────────────────────────────
def test_20_no_default_sizing_without_sources():
    """min_raise_to 缺失 → 不生成任何 raise；不得回退到 2×to_call 之类默认。"""
    s = st()
    s.pop("min_raise_to", None)
    r = build_action_tree(s)
    assert not _sizing(r, "raise")
    assert all(c["action"] in ("fold", "call", "check", "bet")
               for c in r["sizing"])


def test_20_sizing_kinds_are_from_declared_set():
    r = build_action_tree(st())
    r2 = build_action_tree(st(to_call=0.0, action_history=[]))
    for node in (r, r2):
        for c in node["sizing"]:
            assert c["sizing_kind"] in SIZING_KINDS, c


# ── 21. V3G-3/4/5/6 回归 ─────────────────────────────────────────────
def test_21_v3g_3_4_5_6_still_consistent():
    from poker_decision_ai.cash_multistreet import build_multistreet_context
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_preflop_decision import build_preflop_decision
    from poker_decision_ai.cash_range_model import build_range_model
    s = st(street="preflop", board=[], street_extra=None) if False else st(
        street="preflop", board=[], to_call=2.5, min_raise_to=5.0)
    pf = build_preflop_decision(s)
    ms = build_multistreet_context(s, preflop=pf)
    om = build_opponent_model(s, this_hand_actions=[
        {"player_id": "v1", "action": "raise"}])
    rm = build_range_model(s, opponent_model=om)
    at = build_action_tree(s)
    assert pf["present"] and ms["present"] and om["present"]
    assert rm["present"] and at["present"]
    assert at["street"] == "preflop" and at["next_street"] == "flop"


# ── 22. 旁路接线前后 Judge 逐位一致 ───────────────────────────────────
async def test_22_judge_output_bitwise_identical_with_and_without_tree():
    import uuid
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    cfg = load_config()
    logs = ROOT / ".tmp" / "v3g7_logs" / uuid.uuid4().hex
    logs.mkdir(parents=True, exist_ok=True)
    ctx = AgentContext(request_id="v3g7", session_id="v3g7", config=cfg,
                       providers=build_providers(cfg), repos=build_in_memory(),
                       logger=EventLogger(str(logs)),
                       metrics=MetricsCollector())
    s = st(pot=200.0, to_call=50.0, bb=2.0, min_raise_to=100.0)
    eq = {"hero_equity": 0.8, "outcome": {"win": 0.8, "tie": 0.0, "lose": 0.2}}
    inp = {"state": copy.deepcopy(s), "equity": copy.deepcopy(eq)}

    before = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output
    build_action_tree(s)                       # 跑 action tree
    after = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output

    assert set(before) == set(after)
    for k in sorted(before):
        assert before.get(k) == after.get(k), k
    # 明确覆盖授权点名的字段
    for k in ("action", "alternative_action", "recommended_size", "ev_core",
              "weights_used", "final_inputs_summary", "confidence",
              "reasoning", "warnings"):
        assert k in before and before[k] == after[k], k


# ── 23/24. random / future cards / runout ────────────────────────────
def test_23_no_random_import_or_sampling_calls():
    src = (PKG / "cash_action_tree.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name != "random" for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert (n.module or "") != "random"
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("sample_runouts", "remaining_deck", "sample_weighted",
                   "choice", "shuffle", "random"):
        assert banned not in called, banned
    assert "random" not in attrs


def test_24_no_future_cards_or_runout_fields():
    r = build_action_tree(st())
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("runout", "future_board", "future_cards", "sampled",
                   "turn_card", "river_card"):
        assert banned not in blob, banned
    # transitions 明确声明不生成未来牌
    for t in r["transitions"]:
        assert "未来牌" in t["note"] or "future" in t["note"].lower()


# ── 25. 不产生 ICM/bounty/payout/tournament 逻辑 ─────────────────────
def test_25_no_tournament_logic_in_module():
    src = (PKG / "cash_action_tree.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("research_unfreeze", "icm", "bounty", "tournament",
                   "vision2", "judge", "multistreet", "range_model"):
        assert banned not in joined, banned
    assert set(imported) <= {"__future__", "dataclasses", "typing",
                             "cash_integrity_context", "decision_contract",
                             "domain.poker.strategy_core"}, imported


# ── 附：未接线 / 只读审计 ─────────────────────────────────────────────
def test_layer_is_not_wired_into_production():
    importers = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name == "cash_action_tree.py":
            continue
        src = p.read_text(encoding="utf-8-sig", errors="ignore")
        if "cash_action_tree" in src:
            importers.append(str(p.relative_to(ROOT)))
    # BENCH-1 Task0.1（owner 授权基线迁移，8 个引用方）：
    # 2026-09 中旬 P30-P41 生产桥/策略链阶段已授权下列模块引用本层。
    # 按 test_v3g11 SANCTIONED_WIRING 惯例登记为合法基线（精确集合匹配——
    # 任何未登记的新引用方、或已登记引用方消失，都会被检出）。
    sanctioned_importers = {
        "poker_decision_ai/cash_architecture_closure_audit.py",
        "poker_decision_ai/cash_decision_environment.py",
        "poker_decision_ai/cash_loop_stability_verification.py",
        "poker_decision_ai/cash_strategy_action_policy.py",
        "poker_decision_ai/cash_strategy_belief_update.py",
        "poker_decision_ai/cash_strategy_consumption_gate.py",
        "poker_decision_ai/cash_strategy_multistreet_eval.py",
        "poker_decision_ai/cash_strategy_search_tree.py",
    }
    normalized = {i.replace(chr(92), '/') for i in importers}
    assert normalized == sanctioned_importers, (
        f"cash_action_tree 引用方与授权基线不符（新引用方需所有者授权登记）: "
        f"unregistered={sorted(normalized - sanctioned_importers)} "
        f"missing={sorted(sanctioned_importers - normalized)}")


def test_layer_does_not_call_compute_ev_or_positions_for():
    src = (PKG / "cash_action_tree.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("compute_ev", "evaluate_intel", "positions_for",
                   "_positions_for", "to_summary"):
        assert banned not in called and banned not in attrs, banned
    # 只读引用既有冻结常数（不新增常数）
    assert "BET_FRACTIONS" in src and "RAISE_TO_MULTS" in src


def test_v3g1_to_call_exceeding_stack_semantics_untouched():
    """V3G-7 **不修复** V3G-1 的 `to_call > hero_stack` 语义：只记录、保持原样。"""
    s = st(to_call=500.0, pot=100.0,
           players=[
               {"player_id": "v1", "seat": 1, "position": "BB",
                "status": "active", "stack": 200.0},
               {"player_id": "hero", "seat": 2, "position": "BTN",
                "is_hero": True, "status": "active", "stack": 30.0},
               {"player_id": "v2", "seat": 3, "position": "SB",
                "status": "active", "stack": 200.0}])
    r = build_action_tree(s)
    # 门禁既有判定原样保留（本层不改、不绕过，仅如实转述）
    assert any(x.startswith("to_call_exceeds_hero_stack")
               for x in r["integrity"]["conflicts"])
    # V3G-7 仍按既有合法集产出结构（不像 V3G-3 那样因门禁 INCOMPLETE 而拒绝）
    assert r["present"] is True
    call = _sizing(r, "call")[0]
    assert call["amount_bb"] == 30.0 and call["allin"] is True
