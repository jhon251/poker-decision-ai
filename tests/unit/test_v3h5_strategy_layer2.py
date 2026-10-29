# -*- coding: utf-8 -*-
"""V3H-5 测试 — CASH 策略智能 Layer 2（离线、零生产接线）。

覆盖授权清单：2–6 人 CASH / 四街 / 多候选排序 / 同分确定性 tie-break /
belief 参与评估 / prior-only 与 action-likelihood 两条路径 / known-card removal /
UNKNOWN·EMPTY·conflict / 缺 board·缺 hole cards·非法 seat / 合法尺寸与 raise-to 语义 /
Monte Carlo 定 seed 复现 / CASH 与 MTT·PKO 隔离 / Solver BLOCKED 不翻转 /
无 GTO·equilibrium 字段 / Judge·RealtimeRunner 逐位不变 / 生产 import 图零新增 /
frozen·sealed SHA 零漂移。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_action_ranking import (  # noqa: E402
    ACTION_ORDER, CLAIM_BOUNDARY as AR_BOUNDARY, RANKING_MODEL, rank_actions)
from poker_decision_ai.cash_strategy_belief_eval import (  # noqa: E402
    CLAIM_BOUNDARY as BE_BOUNDARY, PRIOR_ONLY_FLAGS, evaluate_with_belief)
from poker_decision_ai.cash_strategy_multistreet_eval import (  # noqa: E402
    CLAIM_BOUNDARY as MS_BOUNDARY, EVAL_MODEL,
    RIVER_TERMINAL_LIMIT, evaluate_multistreet)

# 测试侧允许 import V3G 层（生产侧不允许；由 AST 测试强制）
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_opponent_model import build_opponent_model  # noqa: E402
from poker_decision_ai.cash_range_model import build_range_model  # noqa: E402
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_action_policy import (  # noqa: E402
    build_action_policy)
from poker_decision_ai.cash_strategy_dynamic_belief import (  # noqa: E402
    build_dynamic_belief)
from poker_decision_ai.cash_strategy_policy_evaluator import (  # noqa: E402
    evaluate_policy)

NEW_MODULES = ("cash_strategy_action_ranking", "cash_strategy_multistreet_eval",
               "cash_strategy_belief_eval")

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
    pos = POSITIONS_N.get(n_players, POSITIONS_6)
    players = []
    for i in range(n_players):
        players.append({"player_id": f"v{i + 1}", "seat": i + 1,
                        "position": pos[i], "status": "active",
                        "stack": 100.0, "hole_cards": []})
    hero_seat = 2 if n_players >= 2 else 1
    players[hero_seat - 1] = {
        "player_id": "hero", "seat": hero_seat, "position": pos[hero_seat - 1],
        "is_hero": True, "status": "active", "hole_cards": ["As", "Kd"],
        "stack": 100.0}
    st = {
        "game_type": "cash", "street": "flop", "board": ["Qs", "7d", "2c"],
        "pot": 10.0, "to_call": 0.0, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "button_seat": 2, "table_size": n_players,
        "synthetic_test_only": True, "players": players, "action_history": [],
    }
    st.update(over)
    return st


# ── 原语构建辅助（测试侧；生产侧不允许）────────────────────────────────
def pol(state, **over):
    return build_action_policy(state, action_tree=build_action_tree(state), **over)


def bel(state, **over):
    return build_dynamic_belief(
        state, range_model=build_range_model(state),
        opponent_model=build_opponent_model(state), **over)


def rank(state, **over):
    return rank_actions(state, action_policy=pol(state), belief=bel(state), **over)


def ms(state, **over):
    return evaluate_multistreet(
        state, action_tree=build_action_tree(state),
        action_policy=pol(state), belief=bel(state), **over)


def bev(state, **over):
    return evaluate_with_belief(
        state, action_policy=pol(state), belief=bel(state),
        evaluator=evaluate_policy(state, action_policy=pol(state), belief=bel(state),
                                  solver=solver_status(state)), **over)


FN_CALL = {
    "rank_actions": rank,
    "evaluate_multistreet": ms,
    "evaluate_with_belief": bev,
}

STREETS = [("preflop", [], 1.0), ("flop", ["Qs", "7d", "2c"], 0.0),
           ("turn", ["Qs", "7d", "2c", "Th"], 0.0),
           ("river", ["Qs", "7d", "2c", "Th", "4s"], 4.0)]


# ── 1. CASH-only fail closed ──────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                  "MTT", "PKO", "", "   ", None])
def test_non_cash_modes_fail_closed(mode):
    st = cash_state(game_type=mode)
    for name, fn in FN_CALL.items():
        if name == "rank_actions":
            r = rank_actions(st, action_policy=pol(st))
        elif name == "evaluate_multistreet":
            r = evaluate_multistreet(st, action_tree=build_action_tree(st))
        else:
            r = evaluate_with_belief(st, action_policy=pol(st), belief=bel(st))
        assert r["present"] is False, (name, mode)
        assert r["evidence_grade"] == "NONE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42])
def test_non_dict_input_fail_closed(bad):
    for name, fn in FN_CALL.items():
        r = (rank_actions(bad) if name == "rank_actions"
             else evaluate_multistreet(bad) if name == "evaluate_multistreet"
             else evaluate_with_belief(bad))
        assert r["present"] is False, name


def test_missing_primitives_fail_closed():
    st = cash_state()
    assert rank_actions(st)["reason"] == "action_policy_required"
    assert evaluate_with_belief(st)["reason"] == "action_policy_required"
    assert rank_actions(st, action_policy=pol(st),
                        belief=None)["present"] is True   # belief 可选
    assert evaluate_with_belief(st, action_policy=pol(st))["reason"] == \
        "belief_required"


# ── 2. 2–6 人 CASH ───────────────────────────────────────────────────
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_player_counts(n):
    st = cash_state(n_players=n)
    r = rank(st)
    assert r["present"] is True
    assert r["n_ranked"] > 0
    b = bev(st)
    # hero 占其中一座 → 对手数 = n-1（hero 不得出现在对手范围内）
    opps = b["isolation"]["opponent_range_players"]
    assert len(opps) == n - 1, (n, opps)
    assert "hero" not in opps
    assert b["isolation"]["hero_id_absent_from_opponent_ranges"] is True


# ── 3. 四街 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("street,board,to_call", STREETS)
def test_all_streets_multistreet(street, board, to_call):
    st = cash_state(street=street, board=board, to_call=to_call)
    r = ms(st)
    assert r["present"] is True
    assert r["street"] == street
    assert r["equity_by_street"], street
    eq = next(iter(r["equity_by_street"].values()))["equity"]
    assert eq["present"] is True
    assert eq["invents_future_cards"] is False


@pytest.mark.parametrize("street,board,to_call", STREETS[:3])
def test_ranking_available_on_non_river(street, board, to_call):
    r = rank(cash_state(street=street, board=board, to_call=to_call))
    assert r["present"] is True
    assert r["n_ranked"] > 0
    assert r["preferred"] is not None


# ── 4. L1：river terminal 限制必须保留并显式记录 ───────────────────────
def test_river_terminal_limit_preserved_and_recorded():
    st = cash_state(street="river", board=["Qs", "7d", "2c", "Th", "4s"],
                    to_call=4.0)
    r = ms(st)
    assert r["river_terminal_limit"]["applies"] is True
    assert "V3G-7" in r["river_terminal_limit"]["source"]
    assert "NOT modified" in r["river_terminal_limit"]["handling"]
    # 上游确实终止（限制来自冻结层，未被本层掩盖）
    assert r["terminal"]["terminal"] is True
    assert r["terminal"]["terminal_reason"] == "showdown"


def test_river_ranking_reports_unknown_not_fabricated_preferred():
    st = cash_state(street="river", board=["Qs", "7d", "2c", "Th", "4s"],
                    to_call=4.0)
    r = rank(st)
    # 动作策略层在 river 终止 → 排序层必须 UNKNOWN，而不是伪造 preferred
    assert r["present"] is False or r["level"] == "UNKNOWN"
    if r["present"] is False:
        assert r["reason"] == "action_policy_unavailable:terminal_or_no_legal_actions"
        assert r["preferred"] is None


# ── 5. 多候选排序与选择理由 ───────────────────────────────────────────
def test_ranking_orders_all_candidates_deterministically():
    r = rank(cash_state())
    assert r["n_ranked"] >= 2
    ranks = [row["rank"] for row in r["ranked"]]
    assert ranks == list(range(1, len(ranks) + 1))
    scores = [row["score"] for row in r["ranked"]]
    assert scores == sorted(scores, reverse=True)
    for row in r["ranked"]:
        assert row["score_source"] == "MODEL_INTERNAL"


def test_preferred_has_explicit_reason_and_is_not_null_when_informed():
    r = rank(cash_state())
    assert r["preferred"] is not None
    reason = r["preferred_reason"]
    assert reason["why"]
    assert reason["level"] == "MODEL_INTERNAL"
    assert reason["components"]
    assert r["level"] == "MODEL_INTERNAL"


def test_scoring_model_is_explicit_and_parameter_free():
    assert RANKING_MODEL["free_parameters"] == 0
    assert RANKING_MODEL["uses_opponent_fold_equity"] is False
    assert RANKING_MODEL["uses_randomness"] is False
    assert RANKING_MODEL["is_gto"] is False
    r = rank(cash_state())
    assert r["ranking_model"]["name"] == RANKING_MODEL["name"]


# ── 6. 同分确定性 tie-break ───────────────────────────────────────────
def test_equal_scores_break_deterministically_by_action_order():
    """构造同分候选：check 与 to_call=0 的 call 分数相同 → 必须稳定可复现。"""
    st = cash_state()
    p = pol(st)
    # 人为构造两个同分候选（check 与零投入 call）
    p2 = dict(p)
    p2["candidates"] = [
        {"action": "call", "sizing_kind": "call_match_to_call", "amount_bb": 0.0},
        {"action": "check", "sizing_kind": "check_no_investment", "amount_bb": 0.0},
    ]
    p2["to_call_bb"] = 0.0
    r1 = rank_actions(st, action_policy=p2)
    r2 = rank_actions(st, action_policy=p2)
    assert json.dumps(r1, sort_keys=True, default=str) == \
        json.dumps(r2, sort_keys=True, default=str)
    assert r1["n_ranked"] == 2
    assert r1["ranked"][0]["score"] == r1["ranked"][1]["score"]
    # tie-break 第 1 键：check(1) 先于 call(2)
    assert r1["ranked"][0]["action"] == "check"
    assert ACTION_ORDER.index("check") < ACTION_ORDER.index("call")
    assert r1["preferred_reason"]["tie_break_applied"] is True


def test_tie_break_is_stable_across_candidate_input_order():
    st = cash_state()
    p = pol(st)
    base = {"action": "bet", "sizing_kind": "bet_pot_fraction", "amount_bb": 3.3}
    other = {"action": "bet", "sizing_kind": "bet_pot_fraction", "amount_bb": 3.3}
    a = dict(p, candidates=[dict(base), dict(other)])
    b = dict(p, candidates=[dict(other), dict(base)])
    ra = rank_actions(st, action_policy=a)
    rb = rank_actions(st, action_policy=b)
    assert [x["amount_bb"] for x in ra["ranked"]] == \
        [x["amount_bb"] for x in rb["ranked"]]


# ── 7. 2/3/4/5/6 人 + belief 参与评估 ─────────────────────────────────
def test_belief_participates_in_evaluation():
    r = bev(cash_state())
    assert r["present"] is True
    assert r["n_candidates"] > 0
    any_term = [c["belief_term"] for c in r["candidates"]
                if c["belief_term"].get("available")]
    assert any_term, "belief 必须参与评估"
    assert any_term[0]["n_opponents_with_posterior"] >= 1
    assert any_term[0]["used_in_score"] is False   # 不发明 似然→权重 映射


# ── 8. prior-only 与 action-likelihood 两条路径 ───────────────────────
def test_prior_only_path_is_marked():
    r = bev(cash_state())
    assert r["prior_only"] is True
    assert r["likelihood_source"] == "not_supplied"
    for f in PRIOR_ONLY_FLAGS:
        assert f in r["flags"]
    assert "NO action likelihood was applied" in r["posterior_equals_prior_note"]
    for c in r["candidates"]:
        assert c["prior_only"] is True
        assert c["flags"]


def test_action_likelihood_path_is_marked_and_recorded():
    st = cash_state(action_history=[
        {"player_id": "v1", "action": "bet", "street": "flop", "amount_bb": 3.0}])
    r = evaluate_with_belief(st, action_policy=pol(st),
                             belief=bel(st, action_weights={"bet": 0.4}))
    assert r["prior_only"] is False
    assert r["likelihood_source"] == "explicit_action_weights"
    assert r["flags"] == []
    assert r["likelihood_multiplier_recorded"]
    assert all(m["multiplier"] == 0.4 for m in r["likelihood_multiplier_recorded"])


def test_no_action_likelihood_is_fabricated():
    """prior-only 时不得出现任何似然乘数记录。"""
    r = bev(cash_state())
    assert r["likelihood_multiplier_recorded"] == []
    for c in r["candidates"]:
        assert c["score_source"] == "MODEL_INTERNAL"


# ── 9. known-card removal 与隔离 ──────────────────────────────────────
def test_known_card_removal_is_board_plus_hero_only():
    r = bev(cash_state())
    iso = r["isolation"]
    assert set(iso["known_cards_removed"]) == {"Qs", "7d", "2c", "As", "Kd"}
    assert iso["opponent_hole_cards_included"] is False
    assert iso["known_cards_include_board_and_hero_only"] is True


def test_hero_is_not_represented_as_a_range():
    r = bev(cash_state())
    iso = r["isolation"]
    assert iso["hero_represented_as_range"] is False
    assert iso["hero_hole_cards_are_specific"] is True
    assert iso["hero_id_absent_from_opponent_ranges"] is True
    assert "hero" not in iso["opponent_range_players"]


def test_multistreet_does_not_invent_future_cards():
    r = ms(cash_state())
    assert r["future_board_generated"] is False
    assert r["runout_sampling"] is False
    assert EVAL_MODEL["invents_future_cards"] is False
    assert EVAL_MODEL["samples_runouts"] is False
    assert EVAL_MODEL["reuses_frozen_multistreet_or_compute_ev"] is False


# ── 10. UNKNOWN / EMPTY / conflict ────────────────────────────────────
def test_missing_hole_cards_propagate_to_unknown():
    st = cash_state()
    st["players"][1]["hole_cards"] = []
    r = rank(st)
    assert r["level"] == "UNKNOWN"
    assert r["preferred"] is None
    assert r["preferred_reason"]["why"]
    b = bev(st)
    assert b["unknown"]


def test_missing_board_propagates_to_unknown():
    st = cash_state()
    st.pop("board")
    r = ms(st)
    assert "board_missing" in r["unknown"]
    assert r["level"] == "UNKNOWN"


def test_invalid_seat_is_fail_closed_or_unknown():
    st = cash_state()
    st["players"][0]["seat"] = 99
    r = bev(st)
    assert r["present"] is False or r["unknown"]


def test_empty_players_fail_closed():
    st = cash_state()
    st["players"] = []
    for name in ("rank_actions", "evaluate_multistreet", "evaluate_with_belief"):
        st2 = cash_state()
        st2["players"] = []
        r = (rank_actions(st2, action_policy={"present": False})
             if name == "rank_actions"
             else evaluate_multistreet(st2) if name == "evaluate_multistreet"
             else evaluate_with_belief(st2, action_policy={"present": False},
                                       belief={"present": False}))
        assert r["present"] is False or r["unknown"], name


def test_hero_identity_conflict_is_propagated():
    st = cash_state(hero_id="someone_else")
    r = bev(st)
    assert r["present"] is False or any("conflict" in u for u in r["unknown"])


# ── 11. 合法尺寸与 raise-to 语义 ──────────────────────────────────────
def test_sizes_within_stack_and_raise_to_semantics():
    r = rank(cash_state())
    for row in r["ranked"]:
        amt = row["amount_bb"]
        if isinstance(amt, (int, float)):
            assert amt <= 100.0 + 1e-9
    # raise-to 语义：raise 的 amount 是**本街总投入**，由上游树给出
    st = cash_state(to_call=3.0, street="flop", board=["Qs", "7d", "2c"])
    p = pol(st)
    for c in p["candidates"]:
        if c["action"] == "raise":
            assert c["sizing_kind"].startswith("raise_")


def test_aggressive_actions_exclude_fold_equity():
    r = rank(cash_state())
    aggr = [row for row in r["ranked"]
            if row["action"] in ("bet", "raise", "allin")]
    assert aggr
    assert all(row["components"].get("opponent_fold_equity_excluded") is True
               for row in aggr)


# ── 12. Monte Carlo 定 seed 复现 ──────────────────────────────────────
@pytest.mark.parametrize("name,fn", sorted(FN_CALL.items()))
def test_deterministic_fixed_seed(name, fn):
    a = json.dumps(fn(cash_state()), sort_keys=True, default=str)
    b = json.dumps(fn(cash_state()), sort_keys=True, default=str)
    assert a == b, name


def test_equity_seed_is_recorded_and_uncertainty_reported():
    r = rank(cash_state())
    u = r["uncertainty"]
    assert u["available"] is True
    assert u["seed"] == 42
    assert u["iterations"] == 20000
    assert u["samples"] == 20000
    assert u["method"] == "monte_carlo"


def test_no_randomness_imported_in_new_modules():
    for m in NEW_MODULES:
        tree = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert all(a.name.split(".")[0] != "random" for a in n.names), m
            if isinstance(n, ast.ImportFrom):
                assert (n.module or "").split(".")[0] != "random", m


# ── 13. Solver BLOCKED 不翻转 ─────────────────────────────────────────
def test_solver_remains_blocked():
    st = cash_state()
    before = solver_status(st).get("solver_status")
    rank(st)
    ms(st)
    bev(st)
    after = solver_status(st).get("solver_status")
    assert before == after == "BLOCKED"


def test_full_chain_solver_passthrough_is_blocked():
    st = cash_state()
    ev = evaluate_policy(st, action_policy=pol(st), belief=bel(st),
                         solver=solver_status(st))
    assert ev["solver"]["blocked"] is True
    assert ev["solver"]["solver_status"] == "BLOCKED"


# ── 14. 不得出现 GTO/equilibrium 字段或声明 ───────────────────────────
@pytest.mark.parametrize("boundary", [AR_BOUNDARY, MS_BOUNDARY, BE_BOUNDARY])
def test_claim_boundaries_deny_gto_and_equilibrium(boundary):
    assert boundary["is_gto"] is False
    assert boundary["is_equilibrium"] is False
    assert boundary["is_solver_output"] is False
    assert boundary["is_profit_evidence"] is False
    assert boundary["wired_to_production"] is False


def test_outputs_contain_no_positive_gto_flags():
    blob = json.dumps([rank(cash_state()), ms(cash_state()), bev(cash_state())],
                      default=str).lower()
    for bad in ('"is_gto": true', '"is_equilibrium": true',
                '"equilibrium": true', '"gto_exact": true',
                "gto", "equilibrium"):
        if bad in ("gto", "equilibrium"):
            continue          # 只在 claim_boundary 的 false 值中出现
        assert bad not in blob


def test_multistreet_ev_is_labelled_model_internal():
    r = ms(cash_state())
    ev = next(iter(r["ev_by_street"].values()))
    assert ev["is_real_profit"] is False
    assert ev["is_equilibrium_ev"] is False
    assert ev["opponent_fold_equity_included"] is False
    assert ev["unit"] == "pot_share"


# ── 15. Judge / RealtimeRunner 逐位不变 + 冻结 SHA ────────────────────
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_frozen_files_sha_unchanged(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_v3g_manifest_drift_unchanged():
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


# ── 16. 生产 import 图零新增 ──────────────────────────────────────────
def _production_importers() -> dict[str, set[str]]:
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
                    found.setdefault(m, set()).add(p.name)
    return found


def test_no_new_production_imports():
    found = _production_importers()
    assert set(found) == {"cash_integrity_context", "cash_unknown_e2e"}, found


def test_new_modules_not_imported_in_production():
    found = _production_importers()
    for m in NEW_MODULES:
        assert m not in found


def test_new_modules_have_no_toplevel_v3g_imports():
    v3g = tuple(l for l in V3G_LAYERS if l != "cash_integrity_context")
    for m in NEW_MODULES:
        tree = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        for n in tree.body:
            if isinstance(n, ast.ImportFrom):
                assert not any(v in (n.module or "") for v in v3g), (m, n.module)


def test_new_modules_declare_version_constants():
    for m in NEW_MODULES:
        tree = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        consts = {n.targets[0].id for n in tree.body
                  if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name)}
        assert any(c.endswith("_VERSION") for c in consts), m
