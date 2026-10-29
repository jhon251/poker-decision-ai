# -*- coding: utf-8 -*-
"""V3H-7 测试 — CASH Multistreet Search Layer（离线、零生产接线）。

覆盖授权清单：2–6 人 / 四街 / 单街·两街·三街 / 已知 river terminal /
未知未来 board / sampled 与 unavailable 两种路径 / 固定 seed 复现 /
搜索预算边界 / leaf coverage / UNKNOWN leaf / dynamic belief /
prior-only 与 likelihood-conditioned / 合法动作与尺寸 / tie 与排序输入无关 /
CASH 与 MTT·PKO 隔离 / Solver BLOCKED 不翻转 / 无 GTO·equilibrium 结论 /
Judge·RealtimeRunner 逐位不变 / 生产 import 图零新增 / frozen·sealed SHA 零漂移。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_likelihood_model import (  # noqa: E402
    action_likelihood_model)
from poker_decision_ai.cash_strategy_search_eval import (  # noqa: E402
    CLAIM_BOUNDARY as SE_BOUNDARY, SEARCH_SCORE_MODEL, evaluate_search)
from poker_decision_ai.cash_strategy_search_tree import (  # noqa: E402
    CLAIM_BOUNDARY as ST_BOUNDARY, DEFAULT_BUDGET, FUTURE_BOARD_STATES,
    NODE_FIELDS, build_search_tree)

# 测试侧允许 import V3G 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_range_model import build_range_model  # noqa: E402
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_action_policy import (  # noqa: E402
    build_action_policy)
from poker_decision_ai.cash_strategy_belief_update import (  # noqa: E402
    update_belief_with_likelihood)
from poker_decision_ai.cash_strategy_dynamic_belief import (  # noqa: E402
    build_dynamic_belief)

NEW_MODULES = ("cash_strategy_search_tree", "cash_strategy_search_eval")

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

#: 小预算（保持测试快速，同时仍触达 enumerated / listed / unavailable）
SMALL = {"max_street_depth": 1, "max_response_depth": 1,
         "max_candidate_actions": 2, "max_leaves": 40,
         "max_runout_children": 2}
DEEP = {**SMALL, "max_street_depth": 2, "max_runout_children": 2}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def cash_state(n_players: int = 3, street: str = "flop", board=None,
               history=None, **over) -> dict:
    pos = POSITIONS_N.get(n_players, POSITIONS_6)
    hero_seat = n_players
    players = []
    vi = 1
    for seat in range(1, n_players + 1):
        if seat == hero_seat:
            players.append({"player_id": "hero", "seat": seat,
                            "position": pos[seat - 1], "is_hero": True,
                            "status": "active", "hole_cards": ["As", "Kd"],
                            "stack": 100.0})
        else:
            players.append({"player_id": f"v{vi}", "seat": seat,
                            "position": pos[seat - 1], "status": "active",
                            "stack": 100.0, "hole_cards": []})
            vi += 1
    st = {
        "game_type": "cash", "street": street,
        "board": ["Qs", "7d", "2c"] if board is None else board,
        "pot": 10.0, "to_call": 0.0, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "button_seat": 2, "table_size": n_players,
        "synthetic_test_only": True, "players": players,
        "action_history": history or [],
    }
    st.update(over)
    return st


def pol(state, **over):
    return build_action_policy(state, action_tree=build_action_tree(state), **over)


def bel(state, **over):
    return build_dynamic_belief(state, range_model=build_range_model(state), **over)


def tree(state, budget=None, cap=2, **over):
    return build_search_tree(state, action_policy=pol(state), belief=bel(state),
                             budget=budget or SMALL, max_root_candidates=cap,
                             **over)


def seval(state, budget=None, cap=2, **over):
    tr = tree(state, budget, cap)
    return evaluate_search(state, search_tree=tr, **over)


# ── 1. CASH-only fail closed ──────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed(mode):
    st = cash_state(game_type=mode)
    assert build_search_tree(st, action_policy=pol(st))["present"] is False
    assert evaluate_search(st, search_tree={"present": False})["present"] is False


@pytest.mark.parametrize("bad", [None, [], "cash", 42])
def test_non_dict_fail_closed(bad):
    assert build_search_tree(bad)["present"] is False
    assert evaluate_search(bad)["present"] is False


def test_missing_inputs_fail_closed():
    st = cash_state()
    assert build_search_tree(st)["reason"] == "action_policy_required"
    assert evaluate_search(st)["reason"] == "search_tree_required"


# ── 2. 2–6 人 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_player_counts(n):
    st = cash_state(n_players=n)
    r = seval(st)
    assert r["present"] is True
    assert r["n_candidates"] >= 1
    assert r["belief_provenance"]["n_opponents"] == n - 1


# ── 3. 四街 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("street,board,to_call", [
    ("preflop", [], 1.0), ("flop", ["Qs", "7d", "2c"], 0.0),
    ("turn", ["Qs", "7d", "2c", "Th"], 0.0),
    ("river", ["Qs", "7d", "2c", "Th", "4s"], 4.0)])
def test_all_streets(street, board, to_call):
    st = cash_state(street=street, board=board, to_call=to_call)
    tr = tree(st)
    if street == "river":
        # 冻结 V3G-7：river 为 terminal → 上游 action_policy 无候选 → 本层 fail closed
        assert tr["present"] is False
        assert "action_policy_unavailable" in tr["reason"]
        assert "terminal_or_no_legal_actions" in tr["reason"]
    else:
        assert tr["present"] is True, (street, tr.get("reason"))
        assert tr["street"] == street


# ── 4. 单街 / 两街 / 三街 ─────────────────────────────────────────────
def test_single_street_search():
    tr = tree(cash_state(), {**SMALL, "max_street_depth": 0})
    assert tr["present"] is True
    assert tr["coverage"]["n_leaves"] >= 1
    # 只到起始街：没有 enumerated 叶子
    assert all(l["board_size"] <= 3 for l in tr["leaves"])


def test_two_street_search_produces_enumerated_leaves():
    tr = tree(cash_state(), {**SMALL, "max_street_depth": 2})
    assert "enumerated" in tr["future_board_states_used"]
    assert any(l["future_board_state"] == "enumerated" for l in tr["leaves"])
    # enumerated 叶子板张数 > 起始街
    assert any(l["board_size"] > 3 for l in tr["leaves"]
               if l["future_board_state"] == "enumerated")


def test_three_street_search_reaches_river_board():
    tr = tree(cash_state(), {**SMALL, "max_street_depth": 3,
                             "max_runout_children": 1})
    assert any(l["board_size"] == 5 for l in tr["leaves"]), \
        [(l["board_size"], l["future_board_state"]) for l in tr["leaves"]]


def test_known_river_terminal_is_evaluated_not_searched():
    """已知 river：不展开搜索；但 leaf 评估函数本身可用。"""
    from poker_decision_ai.cash_strategy_multistreet_eval import (
        conditional_absolute_equity)
    res = conditional_absolute_equity(["As", "Kd"], ["Qs", "7d", "2c", "Th", "4s"])
    assert res["present"] is True
    assert res["runouts_remaining"] == 0


# ── 5. 未知未来 board：不伪造、不预测 ─────────────────────────────────
def test_unknown_future_board_is_enumerated_not_predicted():
    tr = tree(cash_state(), {**SMALL, "max_street_depth": 2})
    fc = [n for n in _nodes(tr["root"])
          if n["actor"] == "future_board"]
    assert fc, "应有 future_board 节点"
    for n in fc:
        assert n["future_board_state"] == "enumerated"
        assert n["provenance"]["is_known_fact"] is False
        assert n["provenance"]["source"] == "deck_minus_known_enumerated"
        assert "不" in n["provenance"]["note"] or "not" in n["provenance"]["note"]
    assert tr["runout_prediction_made"] is False
    assert tr["sampling_used"] is False


def test_no_sampled_path_used_and_seed_none():
    tr = tree(cash_state(), DEEP)
    assert tr["sampling_used"] is False
    assert tr["seed"] is None
    assert "sampled" not in tr["future_board_states_used"]


#: V3H-7 原始闭集（**历史口径，登记保留**）：未来牌路径只允许
#: "列出 / 枚举 / 采样 / 不可得"，不允许任何"预测具体未来牌"的路径。
FUTURE_BOARD_STATES_V3H7_HISTORICAL = ("listed", "enumerated", "sampled",
                                       "unavailable")

#: V3I-G7-1 新增的**唯一**成员（只新增，未删除任何历史成员）：
#: 到达下一条街需补**多张**牌时（如 preflop→flop 需 +3），逐个补牌会产生
#: "1 张牌"的非法街态，故按契约**不发明、不采样**任何具体未来牌，改以
#: **当前合法牌面**上的条件绝对权益作为该"到达"的期望叶。
#: 它是"多张补牌的聚合期望"，**不是**对任何单条 runout 的预测。
FUTURE_BOARD_STATES_V3I_G7_1_ADDED = ("aggregated_expected",)


def test_future_board_states_enum_is_closed():
    # 闭集约束**不得放宽**：必须与"历史成员 ∪ G7-1 新增成员"**逐成员精确相等**，
    # 而不是"包含于某个更大的集合"（后者会让未登记成员静默进入）。
    assert set(FUTURE_BOARD_STATES) == (
        set(FUTURE_BOARD_STATES_V3H7_HISTORICAL)
        | set(FUTURE_BOARD_STATES_V3I_G7_1_ADDED))
    # 历史成员一个都不能少（G7-1 只做新增）
    assert set(FUTURE_BOARD_STATES_V3H7_HISTORICAL) <= set(FUTURE_BOARD_STATES)
    # 不允许重复成员（去重后长度必须不变）
    assert len(set(FUTURE_BOARD_STATES)) == len(FUTURE_BOARD_STATES)
    tr = tree(cash_state(), DEEP)
    assert set(tr["future_board_states_used"]) <= set(FUTURE_BOARD_STATES)


def test_aggregated_expected_state_is_aggregate_not_prediction():
    """V3I-G7-1 合法语义：`aggregated_expected` 只用于"多张补牌的聚合期望"。

    闭集新增成员**不得**退化为"点名 / 采样未来牌"的路径：preflop 起步
    （到 flop 需补 3 张）时，到达节点必须是聚合期望节点 ——
    `cards_required > 1`、`candidate_card is None`、`candidates_expanded == 0`、
    `is_known_fact is False`；对应到达叶的权益在**当前合法牌面**（0 张）上计算，
    且树级 `runout_prediction_made=False`、`sampling_used=False`。
    """
    tr = tree(cash_state(street="preflop", board=[]), SMALL)
    assert tr["present"] is True
    assert "aggregated_expected" in tr["future_board_states_used"]
    agg_nodes = [n for n in _nodes(tr["root"])
                 if n["future_board_state"] == "aggregated_expected"]
    assert agg_nodes, "preflop→flop(+3) 必须发出聚合期望节点"
    for n in agg_nodes:
        p = n["provenance"]
        assert p["source"] == "street_faithful_aggregate_expectation"
        assert p["cards_required"] > 1
        assert p["candidate_card"] is None
        assert p["candidates_expanded"] == 0
        assert p["is_known_fact"] is False
    agg_leaves = [l for l in tr["leaves"]
                  if l["future_board_state"] == "aggregated_expected"]
    assert agg_leaves, "聚合期望节点必须落到聚合期望叶"
    for l in agg_leaves:
        # 权益在当前合法牌面（preflop: 0 张）上计算 —— 未补任何未来牌
        assert l["board_size"] == 0
        assert (l.get("path") or {}).get("street_arrival") == \
            "aggregate_expected"
    assert tr["runout_prediction_made"] is False
    assert tr["sampling_used"] is False


def test_unavailable_path_recorded_with_reason():
    tr = tree(cash_state(), {**SMALL, "max_street_depth": 0})
    unav = [l for l in tr["leaves"] if l["future_board_state"] == "unavailable"]
    assert unav
    assert all(l["terminal_reason"] for l in unav)


# ── 6. 节点 schema / provenance / depth / uncertainty ─────────────────
def test_every_node_has_required_fields():
    tr = tree(cash_state(), DEEP)
    for n in _nodes(tr["root"]):
        for f in NODE_FIELDS:
            assert f in n, (n.get("node_id"), f)
        assert isinstance(n["provenance"], dict) and n["provenance"]
        assert isinstance(n["uncertainty"], dict)
        assert n["depth"] >= 0


def test_leaves_have_terminal_kind_and_equity_provenance():
    tr = tree(cash_state(), DEEP)
    for l in tr["leaves"]:
        assert l["terminal_kind"] in ("fold", "showdown", "unknown")
        assert "equity_source" in l
        assert l["is_gto"] is False and l["is_equilibrium"] is False
    kinds = {l["terminal_kind"] for l in tr["leaves"]}
    assert "fold" in kinds


def test_opponent_response_nodes_use_action_syntax_not_a_model():
    tr = tree(cash_state(), DEEP)
    opp = [n for n in _nodes(tr["root"]) if n["actor"] == "opponent"]
    assert opp
    for n in opp:
        assert n["provenance"]["source"] == "known_action_syntax"
        assert n["uncertainty"]["opponent_model"] == "not_used"
        assert n["uncertainty"]["response_weighting"] == "none"


# ── 7. 预算：显式、可复现、不得偷偷扩大 ───────────────────────────────
def test_default_budget_is_declared():
    assert set(DEFAULT_BUDGET) == {"max_street_depth", "max_response_depth",
                                   "max_candidate_actions", "max_leaves",
                                   "max_runout_children"}
    assert all(isinstance(v, int) and v >= 0 for v in DEFAULT_BUDGET.values())


def test_budget_echoed_and_hits_recorded():
    tr = tree(cash_state(), {"max_candidate_actions": 1, "max_street_depth": 1,
                             "max_response_depth": 1, "max_leaves": 40,
                             "max_runout_children": 2}, cap=6)
    assert tr["budget"]["max_candidate_actions"] == 1
    assert "max_candidate_actions" in tr["budget_hits"]
    assert tr["budget_exceeded"] is True
    assert tr["budget_used"]["root_candidates"] == 1


def test_unknown_budget_keys_ignored_and_recorded():
    tr = tree(cash_state(), {"max_street_depth": 1, "bogus_key": 5})
    assert "unknown_budget_keys_ignored" in tr["budget_hits"]
    assert "bogus_key" not in tr["budget"]


def test_max_runout_children_bounds_enumeration():
    tr = tree(cash_state(), {**SMALL, "max_street_depth": 2,
                             "max_runout_children": 1})
    fc = [n for n in _nodes(tr["root"]) if n["actor"] == "future_board"]
    assert fc
    for n in fc:
        assert n["provenance"]["candidates_expanded"] == 1
        assert n["provenance"]["candidates_total"] > 1
    assert "max_runout_children" in tr["budget_hits"]


def test_budget_not_exceeded_when_generous():
    """调用方的 max_root_candidates 是**调用方上限**，不是预算触顶。"""
    tr = tree(cash_state(), {"max_street_depth": 1, "max_response_depth": 1,
                             "max_candidate_actions": 6, "max_leaves": 500,
                             "max_runout_children": 60}, cap=2)
    assert "max_candidate_actions" not in tr["budget_hits"]
    assert "max_root_candidates" in tr["caller_caps_applied"]


# ── 8. leaf coverage 与 UNKNOWN leaf ──────────────────────────────────
def test_coverage_stats_present_and_consistent():
    tr = tree(cash_state(), DEEP)
    cov = tr["coverage"]
    assert cov["n_leaves"] == len(tr["leaves"])
    assert cov["n_leaves_with_equity"] + cov["n_leaves_unknown"] == cov["n_leaves"]
    assert 0.0 <= cov["leaf_coverage"] <= 1.0


def test_unknown_leaf_when_hole_cards_missing():
    st = cash_state()
    st["players"][-1]["hole_cards"] = []
    tr = tree(st, SMALL)
    assert tr["present"] is True
    unknown_leaves = [l for l in tr["leaves"] if not l["equity_present"]]
    assert unknown_leaves
    assert tr["coverage"]["n_leaves_unknown"] >= 1
    # UNKNOWN 不得转 EMPTY：叶子必须带 unknown 说明
    assert all(l["unknown"] for l in unknown_leaves)


def test_unknown_leaf_not_counted_as_evaluable_in_eval():
    st = cash_state()
    st["players"][-1]["hole_cards"] = []
    tr = tree(st, SMALL)
    r = evaluate_search(st, search_tree=tr)
    for c in r["candidates"]:
        assert c["n_leaves_evaluable"] + c["n_leaves_unknown"] + \
            c["n_leaves_unavailable"] == c["n_leaves_total"]


def test_leaf_coverage_reported_per_candidate():
    r = seval(cash_state(), DEEP)
    for c in r["candidates"]:
        assert 0.0 <= c["leaf_coverage"] <= 1.0
        assert c["n_leaves_total"] >= 0


# ── 9. search-aware 评估：聚合、独立评分、不越权 ──────────────────────
def test_search_score_model_is_independent_and_explicit():
    assert SEARCH_SCORE_MODEL["independent_of_v3h5_ranking_formula"] is True
    assert SEARCH_SCORE_MODEL["free_parameters"] == 0
    assert SEARCH_SCORE_MODEL["uses_opponent_fold_equity"] is False
    r = seval(cash_state(), DEEP)
    assert r["search_score_model"]["name"] == SEARCH_SCORE_MODEL["name"]
    for c in r["candidates"]:
        assert c["score_source"] == "MODEL_INTERNAL"


def test_eval_records_budget_coverage_and_belief_provenance():
    r = seval(cash_state(), DEEP)
    assert r["search_budget"] is not None
    assert r["search_budget_used"] is not None
    assert r["search_coverage"] is not None
    assert r["belief_provenance"] is not None


def test_eval_suppresses_preferred_when_no_evaluable_leaves():
    st = cash_state()
    st["players"][-1]["hole_cards"] = []
    r = seval(st, SMALL)
    if r["n_candidates_scored"] == 0:
        assert r["preferred"] is None
        assert r["preferred_note"]


def test_eval_ranks_deterministically_and_input_order_independent():
    a = seval(cash_state(), DEEP)
    b = seval(cash_state(), DEEP)
    assert json.dumps(a, sort_keys=True, default=str) == \
        json.dumps(b, sort_keys=True, default=str)
    ranks = [c["search_rank"] for c in a["ranked"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_eval_never_makes_final_decision():
    r = seval(cash_state(), DEEP)
    assert r["final_decision_made"] is False
    assert SE_BOUNDARY["modifies_v3h5_ranking"] is False


# ── 10. belief 复用：prior-only 与 likelihood-conditioned ─────────────
def test_prior_only_path_propagates_into_search():
    st = cash_state(history=[{"player_id": "v1", "action": "bet",
                              "street": "flop", "amount_bb": 3.0}])
    r = seval(st, DEEP)
    assert r["belief_provenance"]["path"] in (None, "prior_only")


def test_likelihood_conditioned_path_propagates_into_search():
    st = cash_state(history=[{"player_id": "v1", "action": "bet",
                              "street": "flop", "amount_bb": 3.0}])
    lm = action_likelihood_model([{
        "action": "bet", "street": "flop", "size_bucket": "small",
        "multiplier": 1.4, "source": "declared_demo",
        "evidence_level": "DECLARED_MODEL", "provenance": {},
        "uncertainty": {}}])
    bu = update_belief_with_likelihood(st, belief=bel(st),
                                       likelihood_model=lm, pot_bb=10.0)
    tr = build_search_tree(st, action_policy=pol(st), belief=bel(st),
                           belief_update=bu, budget=DEEP, max_root_candidates=2)
    assert tr["belief"]["path"] == "likelihood_conditioned"
    assert tr["belief"]["prior_only"] is False
    r = evaluate_search(st, search_tree=tr)
    assert r["present"] is True


def test_search_does_not_reimplement_range_or_likelihood():
    src = (PKG / "cash_strategy_search_tree.py").read_text(encoding="utf-8")
    for banned in ("parse_range(", "ACTION_WEIGHT", "remove_cards(",
                   "build_range_model(", "action_weights"):
        assert banned not in src, banned


# ── 11. 合法动作与尺寸 ────────────────────────────────────────────────
def test_root_candidates_come_from_action_policy():
    st = cash_state()
    ap = pol(st)
    tr = tree(st, {**SMALL, "max_candidate_actions": 99}, cap=99)
    root_actions = [n["action"] for n in tr["root"]["children"]]
    assert root_actions == [c["action"] for c in ap["candidates"]]
    for n in tr["root"]["children"]:
        assert n["provenance"]["legal_action"] is True


def test_size_buckets_derived_from_geometry():
    tr = tree(cash_state(), {**SMALL, "max_candidate_actions": 99}, cap=99)
    seen = [n["size_bucket"] for n in tr["root"]["children"]]
    assert any(s in ("no_investment", "small", "medium", "large", "allin")
               for s in seen)


# ── 12. 定输入确定性复现 ──────────────────────────────────────────────
@pytest.mark.parametrize("budget", [SMALL, DEEP])
def test_deterministic_reproducibility(budget):
    st = cash_state()
    a = json.dumps(tree(st, budget), sort_keys=True, default=str)
    b = json.dumps(tree(st, budget), sort_keys=True, default=str)
    assert a == b


def test_no_randomness_imported_in_new_modules():
    for m in NEW_MODULES:
        tree_src = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        for n in ast.walk(tree_src):
            if isinstance(n, ast.Import):
                assert all(a.name.split(".")[0] != "random" for a in n.names), m
            if isinstance(n, ast.ImportFrom):
                assert (n.module or "").split(".")[0] != "random", m


def test_no_frozen_multistreet_sample_runouts_reuse():
    """不得**调用/导入**冻结的 sample_runouts 或旧 leaf-EV 逻辑。

    只在**代码层面**检查（docstring 里的"不复用"说明不算违规）。
    """
    for m in NEW_MODULES:
        src = (PKG / f"{m}.py").read_text(encoding="utf-8")
        tree_src = ast.parse(src)
        for n in ast.walk(tree_src):
            if isinstance(n, ast.ImportFrom):
                assert "cash_multistreet" not in (n.module or ""), m
            if isinstance(n, ast.Import):
                assert all("cash_multistreet" not in a.name for a in n.names), m
            if isinstance(n, ast.Attribute):
                assert n.attr != "sample_runouts", m
            if isinstance(n, ast.Name):
                assert n.id != "sample_runouts", m


# ── 13. Solver BLOCKED 不翻转 ─────────────────────────────────────────
def test_solver_remains_blocked():
    st = cash_state()
    before = solver_status(st).get("solver_status")
    r = seval(st, DEEP, solver=solver_status(st))
    after = solver_status(st).get("solver_status")
    assert before == after == "BLOCKED"
    assert r["solver"]["blocked"] is True
    assert r["solver"]["solver_status"] == "BLOCKED"


def test_solver_without_input_is_unknown_but_blocked():
    r = seval(cash_state(), SMALL)
    assert r["solver"]["solver_status"] == "UNKNOWN"
    assert r["solver"]["blocked"] is True
    assert r["solver"]["read_only_passthrough"] is True


# ── 14. 不得出现 GTO/equilibrium 结论 ─────────────────────────────────
@pytest.mark.parametrize("boundary", [ST_BOUNDARY, SE_BOUNDARY])
def test_claim_boundaries(boundary):
    assert boundary["is_gto"] is False
    assert boundary["is_equilibrium"] is False
    assert boundary["is_solver_output"] is False
    assert boundary["is_profit_evidence"] is False
    assert boundary["wired_to_production"] is False


def test_no_positive_gto_flags_in_output():
    blob = json.dumps([tree(cash_state(), SMALL), seval(cash_state(), SMALL)],
                      default=str).lower()
    for bad in ('"is_gto": true', '"is_equilibrium": true',
                '"equilibrium": true', '"gto_exact": true'):
        assert bad not in blob


def test_search_invents_no_future_cards_flag():
    assert ST_BOUNDARY["invents_future_cards"] is False
    assert ST_BOUNDARY["samples_runouts"] is False
    tr = tree(cash_state(), SMALL)
    assert tr["claim_boundary"]["invents_future_cards"] is False
    assert tr["claim_boundary"]["samples_runouts"] is False


# ── 15. Judge / RealtimeRunner 逐位不变 + 冻结 SHA ────────────────────
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_frozen_files_unchanged(rel, prefix):
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
            tree_src = ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for n in ast.walk(tree_src):
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
    for m in NEW_MODULES:
        tree_src = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        for n in tree_src.body:
            if isinstance(n, ast.ImportFrom):
                assert not any(v in (n.module or "") for v in V3G_LAYERS), \
                    (m, n.module)


def test_new_modules_declare_version_constants():
    for m in NEW_MODULES:
        tree_src = ast.parse((PKG / f"{m}.py").read_text(encoding="utf-8"))
        consts = {n.targets[0].id for n in tree_src.body
                  if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name)}
        assert any(c.endswith("_VERSION") for c in consts), m


def _nodes(node: dict):
    yield node
    for c in node.get("children") or []:
        if isinstance(c, dict):
            yield from _nodes(c)
