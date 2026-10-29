"""V3G-4 — CASH Multi-Street Planning Layer 测试（授权矩阵 1–21）。

覆盖：街道推进 4 例、四类终结、UNKNOWN street/board/hole cards/action history、
多人行动序列、空座、incomplete state、确定性转移、**不生成随机未来牌**、
line 节点完整性、CASH mode、MTT/PKO/Mystery fail closed、V3G-1/V3G-2/V3G-3
回归、同输入重复一致。

**不修改**任何既有测试定义；**不触碰**冻结文件。
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.cash_multistreet import (
    BOARD_SIZE, MULTISTREET_LAYER_VERSION, NEXT_STREET, STREET_ORDER,
    build_multistreet_context, next_street_transition, showdown_terminal)

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

_BOARD = {"preflop": (), "flop": ("Ah", "7c", "2c"),
          "turn": ("Ah", "7c", "2c", "2d"),
          "river": ("Ah", "7c", "2c", "2d", "3s")}


def ms_state(street: str = "flop", *, n: int = 3, **over) -> dict:
    players = []
    for i in range(n):
        is_hero = (i == 1)
        players.append({
            "player_id": "hero" if is_hero else f"v{i + 1}",
            "seat": i + 1,
            "position": ("BB", "BTN", "SB", "CO", "MP", "UTG")[i],
            "status": "active", "stack": 100.0, "is_hero": is_hero,
            "hole_cards": ["As", "Kd"] if is_hero else [],
        })
    st = {
        "game_type": "cash", "street": street, "board": list(_BOARD[street]),
        "pot": 10.0, "to_call": 0.0, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "table_size": n, "synthetic_test_only": True,
        "players": players,
        "action_history": [{"street": street, "player_id": "v1",
                            "action": "check", "amount": None}],
    }
    st.update(over)
    return st


# ── 1/2/3/4. 街道推进与 river terminal ────────────────────────────────
@pytest.mark.parametrize("street,nxt", [("preflop", "flop"), ("flop", "turn"),
                                        ("turn", "river")])
def test_01_to_03_deterministic_street_advance(street, nxt):
    t = next_street_transition(ms_state(street))
    assert t["present"] is True
    assert t["current_street"] == street and t["next_street"] == nxt
    assert t["terminal"] is False and t["transition_legal"] is True


def test_04_river_is_terminal_no_future_street():
    t = next_street_transition(ms_state("river"))
    assert t["terminal"] is True
    assert t["terminal_reason"] == "river_no_future_street"
    assert t["next_street"] is None and t["transition_legal"] is False


def test_street_order_matches_frozen_semantics():
    assert STREET_ORDER == ("preflop", "flop", "turn", "river")
    assert NEXT_STREET == {"preflop": "flop", "flop": "turn", "turn": "river"}
    assert BOARD_SIZE == {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


# ── 5/6/7. fold / all-in / showdown terminal ─────────────────────────
def test_05_fold_terminal_when_everyone_folded():
    st = ms_state("flop")
    for p in st["players"]:
        p["status"] = "folded"
    t = next_street_transition(st)
    assert t["terminal"] is True and t["terminal_reason"] == "fold"
    sd = showdown_terminal(st)
    assert sd["everyone_folded"] is True and sd["showdown"] is False


def test_06_all_in_terminal_no_future_hero_decision():
    st = ms_state("flop")
    st["players"][1]["status"] = "allin"
    t = next_street_transition(st)
    assert t["terminal"] is True
    assert t["terminal_reason"] == "all_in_no_future_decision"
    assert t["next_street"] is None


def test_07_showdown_only_when_board_complete_and_not_all_folded():
    sd_river = showdown_terminal(ms_state("river"))
    assert sd_river["board_complete"] is True and sd_river["showdown"] is True
    assert sd_river["hand_terminal"] is True
    sd_flop = showdown_terminal(ms_state("flop"))
    assert sd_flop["board_complete"] is False
    assert sd_flop["showdown"] is None            # 未知 → 不猜
    assert sd_flop["hand_terminal"] is False


# ── 8. UNKNOWN street ─────────────────────────────────────────────────
def test_08_unknown_street_is_not_treated_as_preflop():
    for bad in (None, "", "mystery_street", "FLOPX", 5):
        st = ms_state("flop")
        st["street"] = bad
        t = next_street_transition(st)
        assert t["present"] is False, bad
        assert t["current_street"] is None and t["next_street"] is None
        assert "street" in t["unknown"]
        assert t["transition_legal"] is False
        # 关键：**不得**退化成 preflop→flop
        assert t["terminal_reason"] == "unknown_street"


def test_08_unknown_street_context_fails_closed_without_line():
    st = ms_state("flop")
    st["street"] = "nonsense"
    r = build_multistreet_context(st)
    assert r["present"] is False
    assert "multistreet_context" not in r
    assert "unknown" in r and "street" in r["unknown"]


# ── 9. UNKNOWN board ──────────────────────────────────────────────────
def test_09_missing_board_is_unresolved_not_invented():
    st = ms_state("flop")
    st["board"] = []
    t = next_street_transition(st)
    assert t["board_state"] == "unresolved"
    assert t["board_size"] == 0 and t["expected_board_size"] == 3
    assert "board" in t["unknown"]
    r = build_multistreet_context(st)
    assert "board" in r["multistreet_context"]["unknown"]
    assert r["multistreet_context"]["current_board"] == []


def test_09_missing_board_key_is_unresolved():
    st = ms_state("turn")
    st.pop("board", None)
    t = next_street_transition(st)
    assert t["board_state"] == "unresolved" and "board" in t["unknown"]


def test_09_over_long_board_is_conflict_not_truncated():
    st = ms_state("flop", board=["Ah", "7c", "2c", "2d"])
    t = next_street_transition(st)
    assert t["board_state"] == "conflict"
    assert t["board_size"] == 4 and t["expected_board_size"] == 3


# ── 10. UNKNOWN hole cards ────────────────────────────────────────────
def test_10_unknown_hole_cards_keep_unknown_and_not_ready():
    st = ms_state("flop")
    st["players"][1]["hole_cards"] = []
    r = build_multistreet_context(st)
    c = r["multistreet_context"]
    assert c["ready"] is False
    assert "hero.hole_cards" in c["incomplete"]
    assert "hero.hole_cards" in c["unknown"]
    assert c["hero"]["hole_cards"] is None
    assert c["hero"]["hole_cards_unknown"] is True


def test_10_null_hole_card_element_is_unknown():
    st = ms_state("flop")
    st["players"][1]["hole_cards"] = ["As", None]
    r = build_multistreet_context(st)
    assert r["multistreet_context"]["hero"]["hole_cards_unknown"] is True


# ── 11. UNKNOWN action history ────────────────────────────────────────
def test_11_missing_action_history_is_flagged_not_fabricated():
    st = ms_state("flop")
    st.pop("action_history", None)
    r = build_multistreet_context(st)
    c = r["multistreet_context"]
    assert c["line"] == []
    assert "action_history" in c["incomplete"]
    assert "action_history" in c["unknown"]
    assert c["current_actor"] is None             # 不猜行动者


def test_11_partial_history_nodes_mark_unknown_fields():
    st = ms_state("flop")
    st["action_history"] = [
        {"street": "flop", "player_id": "v1", "action": "bet", "amount": 5.0},
        {"street": "flop", "player_id": "hero"},       # action/amount 缺失
    ]
    r = build_multistreet_context(st)
    line = r["multistreet_context"]["line"]
    assert len(line) == 2
    assert line[0]["evidence"]["status"] == "declared"
    assert line[1]["action"] is None and line[1]["amount"] is None
    assert set(line[1]["evidence"]["unknown"]) == {"action", "amount"}
    # 缺失字段不得被编造
    assert line[1]["amount_bb"] is None


# ── 12. 多人 action sequence ──────────────────────────────────────────
def test_12_multiplayer_sequence_preserves_order_and_actors():
    st = ms_state("flop", n=6)
    st["action_history"] = [
        {"street": "flop", "player_id": "v1", "action": "check", "amount": None},
        {"street": "flop", "player_id": "hero", "action": "bet", "amount": 5.0},
        {"street": "flop", "player_id": "v3", "action": "call", "amount": 5.0},
        {"street": "flop", "player_id": "v4", "action": "fold", "amount": None},
    ]
    r = build_multistreet_context(st)
    line = r["multistreet_context"]["line"]
    assert [n["sequence"] for n in line] == [0, 1, 2, 3]
    assert [n["actor"] for n in line] == ["v1", "hero", "v3", "v4"]
    assert [n["action"] for n in line] == ["check", "bet", "call", "fold"]
    assert line[1]["amount_bb"] == 5.0


def test_12_current_actor_follows_seat_order_after_last_actor():
    st = ms_state("flop", n=3)
    st["action_history"] = [{"street": "flop", "player_id": "v1",
                             "action": "check", "amount": None}]
    r = build_multistreet_context(st)
    # 座位序 v1(1) → hero(2) → v2(3)：v1 之后应为 hero
    assert r["multistreet_context"]["current_actor"] == "hero"


def test_12_current_actor_is_none_when_unreliable():
    st = ms_state("flop", n=3)
    st["action_history"] = [{"street": "turn", "player_id": "v1",
                             "action": "check", "amount": None}]
    r = build_multistreet_context(st)
    assert r["multistreet_context"]["current_actor"] is None


# ── 13. 空座 ──────────────────────────────────────────────────────────
def test_13_empty_seats_do_not_appear_in_players():
    st = ms_state("flop", n=6)
    st["players"] = [p for p in st["players"] if p["seat"] not in (3, 5)]
    st["players"][1]["is_hero"] = True
    r = build_multistreet_context(st)
    assert r["present"] is True
    assert len(r["multistreet_context"]["line"]) == 1


# ── 14. incomplete state ──────────────────────────────────────────────
def test_14_incomplete_state_reports_not_ready_without_inventing_line():
    st = ms_state("flop")
    st["players"][1]["stack"] = None
    st.pop("action_history", None)
    r = build_multistreet_context(st)
    c = r["multistreet_context"]
    assert c["ready"] is False
    assert "hero.stack" in c["incomplete"]
    assert "action_history" in c["incomplete"]
    assert c["effective_stack_bb"] is None        # 不默认 100
    assert c["line"] == []                        # 不制造线路


def test_14_gate_conflict_blocks_readiness():
    st = ms_state("preflop", board=["Ah", "7c", "2c"])   # board/preflop 冲突
    r = build_multistreet_context(st)
    c = r["multistreet_context"]
    assert c["integrity"]["level"] == "INCOMPLETE"
    assert c["ready"] is False
    assert "state_integrity_incomplete" in c["incomplete"]


# ── 15/16. deterministic transition + 不生成随机未来牌 ────────────────
def test_15_transition_is_deterministic():
    st = ms_state("flop")
    first = next_street_transition(st)
    for _ in range(5):
        assert next_street_transition(st) == first


def test_16_no_random_future_card_is_ever_generated():
    for street in STREET_ORDER:
        t = next_street_transition(ms_state(street))
        fb = t["future_board"]
        assert fb["state"] == "unresolved"
        assert fb["cards"] is None
        assert fb["must_not_invent"] is True
    r = build_multistreet_context(ms_state("flop"))
    blob = json.dumps(r, ensure_ascii=False)
    assert "random" not in blob.lower()
    # 未来牌不得出现在任何字段里
    assert r["multistreet_context"]["transition"]["future_board"]["cards"] is None


def test_16_module_does_not_sample_or_use_randomness():
    src = (PKG / "cash_multistreet.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    assert "random" not in " ".join(imported)
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for banned in ("sample_runouts", "remaining_deck", "compute_ev",
                   "evaluate_intel", "positions_for"):
        assert banned not in called, banned


# ── 17. line 节点完整性 ───────────────────────────────────────────────
def test_17_line_node_has_all_required_fields():
    st = ms_state("flop")
    st["action_history"] = [{"street": "flop", "player_id": "hero",
                             "action": "bet", "amount": 5.0}]
    r = build_multistreet_context(st)
    node = r["multistreet_context"]["line"][0]
    for k in ("actor", "street", "action", "amount", "pot_before",
              "stack_before", "resulting_state", "evidence"):
        assert k in node, k
    assert node["actor"] == "hero" and node["action"] == "bet"
    assert node["stack_before"] == 100.0
    # resulting_state 明确标注"不推断"（缺逐节点快照）
    assert node["resulting_state"]["derivable"] is False


# ── 18/19. CASH mode + 非 CASH fail closed ────────────────────────────
def test_18_cash_mode_present_and_no_tournament_fields():
    r = build_multistreet_context(ms_state("turn"))
    assert r["present"] is True
    assert r["multistreet_context"]["mode"] == "cash"
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("icm", "bounty", "satellite", "payout", "tournament",
                   "blind_pressure", "mystery"):
        assert banned not in blob, banned


@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin", None])
def test_19_non_cash_modes_fail_closed(mode):
    r = build_multistreet_context(ms_state("flop", game_type=mode))
    assert r["present"] is False
    assert "multistreet_context" not in r


def test_19_non_dict_state_fails_closed():
    for bad in (None, [], "cash", 7):
        assert build_multistreet_context(bad)["present"] is False


# ── 20. V3G-1/V3G-2/V3G-3 回归（本层视角）─────────────────────────────
def test_20_v3g1_and_v3g2_carried_and_v3g3_reusable():
    from poker_decision_ai.cash_preflop_decision import build_preflop_decision
    st = ms_state("preflop")
    st["button_seat"] = 2                      # 补全可选上下文（不影响必需字段）
    pf = build_preflop_decision(st)
    assert pf["present"] is True
    r = build_multistreet_context(st, preflop=pf)
    c = r["multistreet_context"]
    assert c["integrity"]["level"] == "COMPLETE"          # V3G-1
    assert c["evidence"]["present"] is True               # V3G-2
    assert r["preflop_reuse"]["reused"] is True           # V3G-3 只读转述
    assert r["preflop_reuse"]["candidate_count"] == len(pf["candidates"])


def test_20_missing_button_seat_is_carried_as_degraded_unknown():
    """如实记录 V3G-1 语义：缺 `button_seat`（可选字段）→ DEGRADED +
    `unknown=['button_seat']`，但仍是**就绪**状态（无 missing/冲突）。"""
    st = ms_state("preflop")
    r = build_multistreet_context(st)
    c = r["multistreet_context"]
    assert c["integrity"]["level"] == "DEGRADED"
    assert "button_seat" in c["integrity"]["unknown"]
    assert c["integrity"]["missing"] == [] and c["integrity"]["conflicts"] == []
    assert c["ready"] is True


def test_20_preflop_reuse_is_readonly_and_reports_unavailable():
    st = ms_state("preflop")
    r = build_multistreet_context(st, preflop={"present": False,
                                               "reason": "not_cash_mode"})
    assert r["preflop_reuse"]["reused"] is False
    r2 = build_multistreet_context(st)
    assert "preflop_reuse" not in r2


def test_20_to_call_exceeding_stack_semantics_untouched():
    """V3G-4 **不修复** V3G-1 的 `to_call > hero_stack` 语义：原样保留。"""
    st = ms_state("flop", to_call=500.0, pot=10.0)
    st["players"][1]["stack"] = 100.0
    r = build_multistreet_context(st)
    c = r["multistreet_context"]
    assert any(x.startswith("to_call_exceeds_hero_stack")
               for x in c["integrity"]["conflicts"])
    assert c["ready"] is False


# ── 21. 同输入重复运行结果一致 ────────────────────────────────────────
def test_21_repeated_runs_are_identical_and_json_safe():
    st = ms_state("flop", n=6)
    first = build_multistreet_context(st)
    for _ in range(5):
        assert build_multistreet_context(st) == first
    json.dumps(first)


# ── 只读 / 依赖审计 ───────────────────────────────────────────────────
def test_layer_does_not_mutate_input_state():
    st = ms_state("flop")
    before = copy.deepcopy(st)
    build_multistreet_context(st)
    next_street_transition(st)
    showdown_terminal(st)
    assert st == before


def test_layer_is_not_wired_into_production_paths():
    """本阶段作为独立模块交付；除自身外无生产模块 import 它。"""
    importers = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name == "cash_multistreet.py":
            continue
        src = p.read_text(encoding="utf-8-sig", errors="ignore")
        if "cash_multistreet" in src:
            importers.append(str(p.relative_to(ROOT)))
    # BENCH-1 Task0.1（owner 授权基线迁移，3 个引用方）：
    # 2026-09 中旬 P30-P41 生产桥/策略链阶段已授权下列模块引用本层。
    # 按 test_v3g11 SANCTIONED_WIRING 惯例登记为合法基线（精确集合匹配——
    # 任何未登记的新引用方、或已登记引用方消失，都会被检出）。
    sanctioned_importers = {
        "poker_decision_ai/cash_architecture_closure_audit.py",
        "poker_decision_ai/cash_strategy_consumption_gate.py",
        "poker_decision_ai/cash_strategy_search_tree.py",
    }
    normalized = {i.replace(chr(92), '/') for i in importers}
    assert normalized == sanctioned_importers, (
        f"cash_multistreet 引用方与授权基线不符（新引用方需所有者授权登记）: "
        f"unregistered={sorted(normalized - sanctioned_importers)} "
        f"missing={sorted(sanctioned_importers - normalized)}")


def test_layer_imports_only_sanctioned_modules():
    src = (PKG / "cash_multistreet.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("research_unfreeze", "strategy_core", "multistreet",
                   "vision2", "positions_for"):
        assert banned not in joined, banned
    assert set(imported) <= {"__future__", "dataclasses", "typing",
                             "cash_integrity_context",
                             "decision_contract"}, imported
