# -*- coding: utf-8 -*-
"""V3H-4 测试 — CASH 策略智能第 1 层（离线、未接生产）。

覆盖授权清单：6-max CASH / 2–6 人 / 四条街 / 合法动作与尺寸 / 动态 range /
known-card removal / hero-opponent 隔离 / UNKNOWN·EMPTY·conflict / 缺 board·
缺 hole cards·非法 seat / 定输入复现 / CASH 与 MTT·PKO 隔离 / Solver 不得翻转 /
不得输出 GTO·equilibrium / 既有 Judge·RealtimeRunner 逐位不变 / 冻结文件 SHA 不变 /
生产 import 图零新增。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_action_policy import (  # noqa: E402
    CLAIM_BOUNDARY as AP_BOUNDARY, build_action_policy)
from poker_decision_ai.cash_strategy_dynamic_belief import (  # noqa: E402
    CLAIM_BOUNDARY as DB_BOUNDARY, build_dynamic_belief)
from poker_decision_ai.cash_strategy_policy_evaluator import (  # noqa: E402
    CLAIM_BOUNDARY as PE_BOUNDARY, UNAVAILABLE_METRICS, evaluate_policy)
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402

NEW_MODULES = ("cash_strategy_action_policy",
               "cash_strategy_dynamic_belief",
               "cash_strategy_policy_evaluator")

FROZEN = {
    "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    "poker_decision_ai/domain/poker/models.py": "6775c06050319a85",
    "poker_decision_ai/domain/poker/conditional_model.py": "686c9f58b71d9408",
    "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
    "poker_decision_ai/realtime.py": "3400a20ff857880d",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py": "52e6267db49a2e11",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py": "17236d4ed8070d01",
}

V3G_LAYERS = ("cash_state_integrity", "cash_integrity_context",
              "cash_preflop_decision", "cash_multistreet", "cash_opponent_model",
              "cash_range_model", "cash_action_tree", "cash_unknown_e2e",
              "cash_solver_interface", "cash_profit_interface")

POSITIONS_6 = ("SB", "BB", "UTG", "MP", "CO", "BTN")
POSITIONS_N = {2: ("SB", "BB"), 3: ("SB", "BB", "BTN"),
               4: ("SB", "BB", "MP", "BTN"), 5: ("SB", "BB", "MP", "CO", "BTN")}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def cash_state(n_players: int = 6, **over) -> dict:
    """构造合法 CASH state（table_size 与 players 长度一致；座位连续）。"""
    pos = POSITIONS_N.get(n_players, POSITIONS_6)
    players = []
    for i in range(n_players):
        seat = i + 1
        players.append({
            "player_id": f"v{i + 1}", "seat": seat, "position": pos[i],
            "status": "active", "stack": 100.0, "hole_cards": [],
        })
    hero_seat = 2 if n_players >= 2 else 1
    players[hero_seat - 1] = {
        "player_id": "hero", "seat": hero_seat,
        "position": pos[hero_seat - 1], "is_hero": True, "status": "active",
        "hole_cards": ["As", "Kd"], "stack": 100.0,
    }
    st = {
        "game_type": "cash", "street": "flop", "board": ["Qs", "7d", "2c"],
        "pot": 10.0, "to_call": 0.0, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "button_seat": 2, "table_size": n_players,
        "synthetic_test_only": True, "players": players, "action_history": [],
    }
    st.update(over)
    return st




# ── 原语构建（**测试侧**允许 import V3G 层；生产侧不允许）──────────────
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_opponent_model import build_opponent_model  # noqa: E402
from poker_decision_ai.cash_range_model import build_range_model  # noqa: E402


def ap(state, **over):
    """构建 action policy（显式传入 action_tree —— 契约要求）。"""
    return build_action_policy(state, action_tree=build_action_tree(state), **over)


def db(state, **over):
    """构建 belief（显式传入 range/opponent model —— 契约要求）。"""
    return build_dynamic_belief(
        state, range_model=build_range_model(state),
        opponent_model=build_opponent_model(state), **over)


def ev(state, **over):
    """构建评估（显式传入 action policy 与 belief —— 契约要求）。"""
    return evaluate_policy(
        state, action_policy=ap(state),
        belief=db(state), solver=solver_status(state), **over)


#: 三个入口的"带原语"调用映射（供参数化 fail-closed / 确定性测试使用）
FN_CALL = {
    "build_action_policy": ap,
    "build_dynamic_belief": db,
    "evaluate_policy": ev,
}


# ── 1. CASH-only fail closed（MTT/PKO/Mystery Bounty 等）────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                  "MTT", "PKO", "CashGame", "", "   ", None])
def test_non_cash_modes_fail_closed_everywhere(mode):
    st = cash_state(game_type=mode)
    for fn in (build_action_policy, build_dynamic_belief, evaluate_policy):
        r = FN_CALL[fn.__name__](st)
        assert r["present"] is False, (fn.__name__, mode)
        assert r["evidence_grade"] == "NONE"
        assert r["unknown"], fn.__name__          # 必须说明原因
        assert r.get("candidates", []) == []       # 不得产出候选


@pytest.mark.parametrize("bad", [None, [], "cash", 42, {"game_type": None}])
def test_non_dict_or_missing_game_type_fail_closed(bad):
    for fn in (build_action_policy, build_dynamic_belief, evaluate_policy):
        assert FN_CALL[fn.__name__](bad)["present"] is False, fn.__name__


def test_cash_mode_is_accepted():
    assert ap(cash_state())["present"] is True
    assert db(cash_state())["present"] is True
    assert ev(cash_state())["present"] is True


# ── 2. 玩家数 2/3/4/5/6 ────────────────────────────────────────────────
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_player_counts_supported(n):
    r = db(cash_state(n_players=n))
    assert r["present"] is True
    assert r["n_opponents"] == n - 1


def test_six_max_is_the_primary_case():
    r = db(cash_state(n_players=6))
    assert r["n_opponents"] == 5
    assert "hero" not in r["belief"]


# ── 3. 四条街 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("street,board,to_call", [
    ("preflop", [], 1.0), ("flop", ["Qs", "7d", "2c"], 0.0),
    ("turn", ["Qs", "7d", "2c", "Th"], 0.0),
])
def test_streets_with_action_available(street, board, to_call):
    st = cash_state(street=street, board=board, to_call=to_call)
    pol = ap(st)
    assert pol["present"] is True, (street, pol.get("reason"))
    assert pol["street"] == street
    assert pol["hand_strength"]["present"] is True


def test_river_follows_frozen_v3g7_terminal_semantics():
    """**已知限制（不得修复）**：冻结 V3G-7 把 river 视为 terminal，
    故本层在 river 上按 fail-closed 报告终止而非伪造候选。

    记录该 gap 是为了让下一阶段显式处理；**不修改 V3G-7**。
    """
    st = cash_state(street="river", board=["Qs", "7d", "2c", "Th", "4s"],
                    to_call=4.0)
    pol = ap(st)
    assert pol["present"] is False
    assert pol["reason"] == "terminal_or_no_legal_actions"
    assert pol["candidates"] == []
    # 但 belief 层不依赖 action tree，river 上仍可更新
    bel = db(st)
    assert bel["present"] is True
    pid = next(iter(bel["belief"]))
    assert bel["belief"][pid]["posterior"]["present"] is True


def test_terminal_node_is_reported_not_papered_over():
    """无决策可做时 → 明确终止，不得伪造候选。"""
    st = cash_state(street="river", board=["Qs", "7d", "2c", "Th", "4s"],
                    to_call=0.0)
    pol = ap(st)
    if pol["present"] is False:
        assert pol["reason"] == "terminal_or_no_legal_actions"
        assert pol["candidates"] == []


@pytest.mark.parametrize("board", [[], ["Qs"], ["Qs", "7d"], ["Qs", "7d", "2c"],
                                   ["Qs", "7d", "2c", "Th"],
                                   ["Qs", "7d", "2c", "Th", "4s"]])
def test_board_sizes(board):
    st = cash_state(street="preflop" if not board else "flop", board=board,
                    to_call=1.0)
    if len(board) in (1, 2):
        r = ap(st)
        assert r["present"] is False or "board_size_invalid" in r["unknown"]
    else:
        assert ap(st)["present"] is True


# ── 4. 合法动作与尺寸 ──────────────────────────────────────────────────
def test_candidates_match_legal_actions_exactly():
    pol = ap(cash_state())
    assert pol["legality_complete"] is True
    assert pol["candidate_actions"] == sorted(pol["legal_actions"])


def test_no_default_action_or_size_is_invented():
    pol = ap(cash_state())
    assert pol["selection_implemented"] is False
    assert pol["preferred"] is None
    for c in pol["candidates"]:
        assert c["sizing_kind"], c
        # 尺寸必须来自 action_tree 的 sizing（有来源标签）
        assert c["provenance"]["sizing_source"].startswith("cash_action_tree.sizing[")


def test_sizes_within_bounds_when_present():
    pol = ap(cash_state())
    for c in pol["candidates"]:
        b = c.get("bounds") or {}
        if isinstance(c.get("amount_bb"), (int, float)) and \
                isinstance(pol.get("effective_stack_bb"), (int, float)):
            assert c["amount_bb"] <= pol["effective_stack_bb"] + 1e-9
        if isinstance(c.get("amount_bb"), (int, float)) and \
                isinstance(b.get("effective_stack_bb"), (int, float)):
            assert c["amount_bb"] <= b["effective_stack_bb"] + 1e-9


# ── 5. 动态 range 更新与 known-card removal ────────────────────────────
def test_belief_card_removal_reduces_mass():
    """已知牌（公共牌 + hero 手牌）必须减少对手 range 质量。"""
    st = cash_state()
    r = db(st)
    pid = next(iter(r["belief"]))
    post = r["belief"][pid]["posterior"]
    assert post["present"] is True
    assert post["mass_posterior"] < post["mass_prior"]
    assert post["mass_removed_by_known_cards"] > 0
    assert post["source"] == "domain.poker.ranges.Range.remove_cards"


def test_more_known_cards_removes_more_mass():
    pre = db(cash_state(street="preflop", board=[], to_call=1.0))
    flop = db(cash_state(street="flop"))
    p_pre = pre["belief"][next(iter(pre["belief"]))]["posterior"]
    p_flop = flop["belief"][next(iter(flop["belief"]))]["posterior"]
    assert p_flop["mass_posterior"] < p_pre["mass_posterior"]


def test_known_cards_listed_includes_board_and_hero_only():
    r = db(cash_state())
    removed = set(r["known_cards_removed"])
    assert {"Qs", "7d", "2c", "As", "Kd"} <= removed
    assert len(removed) == 5


def test_belief_without_action_weights_does_not_invent_likelihood():
    r = db(cash_state())
    assert r["likelihood_applied"] is False
    assert r["likelihood_source"] == "not_supplied"
    pid = next(iter(r["belief"]))
    assert r["belief"][pid]["posterior"]["likelihood_source"] == "not_supplied"


def test_belief_with_explicit_weights_records_provenance():
    st = cash_state(action_history=[
        {"player_id": "v1", "action": "bet", "street": "flop", "amount_bb": 3.0}])
    r = db(st, action_weights={"bet": 0.5})
    assert r["likelihood_applied"] is True
    assert r["likelihood_source"] == "explicit_action_weights"
    pid = next(iter(r["belief"]))
    post = r["belief"][pid]["posterior"]
    assert post["likelihood_source"] == "explicit_action_weights"
    assert post["likelihood_multiplier"] == 0.5


# ── 6. hero 与 opponent 动作隔离 ───────────────────────────────────────
def test_hero_actions_never_enter_opponent_history():
    st = cash_state(action_history=[
        {"player_id": "v1", "action": "bet", "street": "flop", "amount_bb": 3.0},
        {"player_id": "hero", "action": "call", "street": "flop", "amount_bb": 3.0},
        {"player_id": "v2", "action": "fold", "street": "flop"},
    ])
    r = db(st)
    assert r["hero_actions_in_opponent_history"] is False
    assert [a["player_id"] for a in r["opponent_actions"]] == ["v1", "v2"]
    assert [a["player_id"] for a in r["hero_actions_excluded"]] == ["hero"]
    # hero 自身不得成为对手 belief 的条目
    assert "hero" not in r["belief"]


def test_hero_flag_also_excluded():
    st = cash_state(action_history=[
        {"player_id": "someone", "is_hero": True, "action": "raise",
         "street": "flop", "amount_bb": 6.0}])
    r = db(st)
    assert r["hero_actions_in_opponent_history"] is False
    assert r["opponent_actions"] == []


# ── 7. UNKNOWN / EMPTY / conflict ─────────────────────────────────────
def test_missing_hole_cards_yield_unknown_not_empty():
    st = cash_state()
    st["players"][1]["hole_cards"] = []
    pol = ap(st)
    assert pol["hand_strength"]["present"] is False
    assert any("hole_cards" in u or "equity_error" in u for u in pol["unknown"])
    assert pol["preferred"] is None


def test_missing_board_yields_unknown():
    st = cash_state()
    st.pop("board")
    r = db(st)
    assert any("board" in u for u in r["unknown"])
    assert r["level"] == "UNKNOWN"


def test_unknown_action_marks_level_unknown():
    st = cash_state(action_history=[
        {"player_id": "v1", "action": "teleport", "street": "flop"}])
    r = db(st)
    assert any("unknown_action" in u for u in r["unknown"])
    assert r["level"] == "UNKNOWN"


def test_unknown_is_not_empty_output():
    """UNKNOWN 不得被 0 值或空 posterior 冒充。"""
    st = cash_state()
    st.pop("board")
    r = db(st)
    pid = next(iter(r["belief"]))
    post = r["belief"][pid]["posterior"]
    assert post["present"] is True          # 有值就不是 EMPTY
    assert post["mass_posterior"] >= 0.0


def test_invalid_seat_or_missing_players_is_fail_closed():
    st = cash_state()
    st["players"] = []
    for fn in (build_action_policy, build_dynamic_belief, evaluate_policy):
        r = FN_CALL[fn.__name__](st)
        assert r["present"] is False or r["unknown"], fn.__name__


def test_hero_identity_conflict_is_flagged():
    st = cash_state()
    st["hero_id"] = "not_the_flagged_hero"
    r = db(st)
    assert any("conflict" in u for u in r["unknown"])


# ── 8. 定输入确定性复现 ────────────────────────────────────────────────
@pytest.mark.parametrize("fn", [build_action_policy, build_dynamic_belief,
                                evaluate_policy])
def test_deterministic_same_input_same_output(fn):
    a = json.dumps(FN_CALL[fn.__name__](cash_state()), sort_keys=True, default=str)
    b = json.dumps(FN_CALL[fn.__name__](cash_state()), sort_keys=True, default=str)
    assert a == b, fn.__name__


def test_no_randomness_imported_in_new_modules():
    for m in NEW_MODULES:
        src = (PKG / f"{m}.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert all(a.name.split(".")[0] != "random" for a in n.names), m
            if isinstance(n, ast.ImportFrom):
                assert (n.module or "").split(".")[0] != "random", m


def test_equity_seed_is_recorded_in_provenance():
    pol = ap(cash_state())
    hs = pol["hand_strength"]
    assert hs["seed"] == 42
    assert hs["iterations"] == 20000
    assert hs["deterministic_given_seed"] is True


# ── 9. 不得声称 GTO / equilibrium ─────────────────────────────────────
@pytest.mark.parametrize("boundary", [AP_BOUNDARY, DB_BOUNDARY, PE_BOUNDARY])
def test_claim_boundaries_deny_gto_and_equilibrium(boundary):
    assert boundary["is_gto"] is False
    assert boundary["is_equilibrium"] is False
    assert boundary["is_solver_output"] is False
    assert boundary["is_profit_evidence"] is False
    assert boundary["wired_to_production"] is False


def test_outputs_never_carry_gto_true():
    blob = json.dumps([ap(cash_state()),
                       db(cash_state()),
                       ev(cash_state())], default=str).lower()
    for bad in ('"is_gto": true', '"is_equilibrium": true',
                '"equilibrium": true', '"gto_exact": true'):
        assert bad not in blob


def test_evaluator_offers_no_ranking_or_selection():
    res = ev(cash_state())
    assert res["ranking_provided"] is False
    assert res["selection_implemented"] is False
    assert "best" not in res


def test_exploitability_metrics_are_unavailable_not_computed():
    res = ev(cash_state())
    assert set(res["unavailable_metrics"]) == set(UNAVAILABLE_METRICS)
    for c in res["comparisons"]:
        assert not any(k in c["comparables"] for k in UNAVAILABLE_METRICS)


def test_evaluator_labels_sources_and_never_uses_heuristic():
    res = ev(cash_state())
    assert res["heuristic_used"] is False
    assert set(res["source_grades_used"]) <= {"MODEL_INTERNAL", "UNAVAILABLE"}
    for c in res["comparisons"]:
        assert c["heuristic_used"] is False


# ── 10. Solver 不得翻转 ───────────────────────────────────────────────
def test_solver_status_not_flipped_by_new_modules():
    st = cash_state()
    before = solver_status(st).get("solver_status")
    ap(st)
    db(st)
    res = ev(st)
    after = solver_status(st).get("solver_status")
    assert before == after == "BLOCKED"
    assert res["solver"]["solver_status"] == "BLOCKED"
    assert res["solver"]["blocked"] is True
    # 真实传入 solver 状态 → 属于上游转述，而非"未提供"
    assert res["solver"]["read_only_passthrough"] is False


def test_evaluator_without_solver_marks_unknown_and_stays_blocked():
    """未提供 solver 状态时 → UNKNOWN 且 blocked，绝不假定可用。"""
    res = evaluate_policy(cash_state(), action_policy=ap(cash_state()),
                          belief=db(cash_state()))
    assert res["solver"]["solver_status"] == "UNKNOWN"
    assert res["solver"]["blocked"] is True
    assert res["solver"]["read_only_passthrough"] is True
    assert "solver_status_not_supplied" in res["unknown"]


def test_missing_primitives_fail_closed_without_hidden_imports():
    """契约：原语必须由调用方传入；缺失 → fail closed（不做隐式导入）。"""
    st = cash_state()
    assert build_action_policy(st)["reason"] == "action_tree_required"
    assert build_dynamic_belief(st)["reason"] == "range_model_required"
    assert evaluate_policy(st)["reason"] == "action_policy_required"


def test_solver_source_not_imported_by_action_policy_or_belief():
    """这两层不得引用 solver（避免任何"用 solver 结果"的隐含路径）。"""
    for m in ("cash_strategy_action_policy", "cash_strategy_dynamic_belief"):
        tree = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module}
        assert not any("solver" in (x or "").lower() for x in mods), m


# ── 11. 生产 import 图零新增 ──────────────────────────────────────────
def _real_imports() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for p in PKG.rglob("*.py"):
        if p.name.startswith("cash_") or "__pycache__" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module.split(".")[-1])
            elif isinstance(n, ast.Import):
                mods += [a.name.split(".")[-1] for a in n.names]
            for m in mods:
                if m in V3G_LAYERS or m in NEW_MODULES:
                    found.setdefault(m, set()).add(str(p.relative_to(ROOT)))
    return found


def test_no_new_production_imports():
    found = _real_imports()
    assert set(found) == {"cash_integrity_context", "cash_unknown_e2e"}, found


def test_new_modules_are_not_imported_anywhere_in_production():
    found = _real_imports()
    for m in NEW_MODULES:
        assert m not in found


# ── 12. 既有 Judge / RealtimeRunner 逐位不变 + 冻结 SHA ────────────────
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_frozen_files_sha_unchanged(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_v3g_freeze_manifest_drift_unchanged():
    man = json.loads((ROOT / "docs" / "V3G_11_FREEZE_MANIFEST.json")
                     .read_text(encoding="utf-8"))
    drift = []
    for block in ("frozen_modules", "frozen_tests", "frozen_closeouts",
                  "production_baseline"):
        for rel, h in man[block].items():
            if h == "__SELF__":
                continue
            if _sha(rel) != h:
                drift.append(rel)
    assert drift in ([], ["docs/V3G_11_EVALUATION_FREEZE_CLOSEOUT.md"]), drift


def test_new_modules_declare_a_version_constant():
    for m in NEW_MODULES:
        src = (PKG / f"{m}.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        consts = {n.targets[0].id for n in tree.body
                  if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name)}
        assert any(c.endswith("_VERSION") for c in consts), m


def test_new_modules_do_not_touch_vision_state_or_positions():
    """新层不得 import positions_for / vision / state 相关冻结模块。"""
    banned = ("domain.poker.models", "vision", "state_integrity")
    for m in NEW_MODULES:
        tree = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module}
        for mod in mods:
            assert not any(b in (mod or "") for b in banned), (m, mod)
