"""V3I-P6 — P4 bridge 在 multistreet/search 上下文中的实际决策作用（测试）。

覆盖授权要求的测试项：同一状态不同 population；不同 opponent history；
多街状态；deterministic repeat；UNKNOWN/conflict；ranking 改变与 unchanged；
assembly ABSTAIN；**不允许非法适应**；以及最关键的**接口阻塞定位**。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_search_context_verification import (
    BRIDGE_OUTPUTS,
    CASH_P6_SEARCH_CONTEXT_VERSION,
    CLAIM_BOUNDARY,
    MUST_NOT_CHANGE,
    SEARCH_EVAL_INPUTS,
    SEARCH_TREE_INPUTS,
    check_multistreet,
    compare_bridge_effect,
    p6_contract,
    probe_search_interface,
    search_sensitivity_probe,
    stable_fingerprint,
    trace_loop_layers,
)

POP_PARAMS = {
    "DECLARED_TIGHT_PASSIVE": {"modeled_preflop_continue_multiplier": 0.7,
                               "modeled_postflop_aggression_multiplier": 0.6},
    "DECLARED_BALANCED": {"modeled_preflop_continue_multiplier": 1.0,
                          "modeled_postflop_aggression_multiplier": 1.0},
    "DECLARED_LOOSE_PASSIVE": {"modeled_preflop_continue_multiplier": 1.4,
                               "modeled_postflop_aggression_multiplier": 0.5},
    "DECLARED_AGGRESSIVE": {"modeled_preflop_continue_multiplier": 1.0,
                            "modeled_postflop_aggression_multiplier": 1.8},
    "DECLARED_MANIAC": {"modeled_preflop_continue_multiplier": 1.8,
                        "modeled_postflop_aggression_multiplier": 2.6},
}

P6_MODULE = pathlib.Path(
    "poker_decision_ai/cash_search_context_verification.py")


# ---------------------------------------------------------------------------
# T1. 契约与边界（要求 2、6、7、8、10）
# ---------------------------------------------------------------------------
def test_t1_contract_and_claim_boundary():
    c = p6_contract()
    assert c["version"] == CASH_P6_SEARCH_CONTEXT_VERSION
    assert c["adds_strategy_capability"] is False
    assert c["conditioned_evaluation_has_a_slot"] is False
    assert c["interface_block_reported"] is True
    assert tuple(c["search_tree_inputs"]) == SEARCH_TREE_INPUTS
    assert tuple(c["search_eval_inputs"]) == SEARCH_EVAL_INPUTS
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output", "is_ev",
                 "wired_to_production", "adds_strategy_capability",
                 "modifies_frozen_layers", "bypasses_frozen_contract"):
        assert flag in CLAIM_BOUNDARY


def test_t1_must_not_change_list_is_explicit():
    for item in ("search_depth", "leaf_budget", "river_terminal", "tier_rules",
                 "action_policy_candidate_order", "v3h4_dynamic_belief",
                 "v3h6_likelihood_model", "v3h8_decision_assembly"):
        assert item in MUST_NOT_CHANGE


def test_t1_module_imports_only_stdlib():
    src = P6_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)
    assert "import random" not in src


def test_t1_no_ev_or_external_claim_terms():
    src = P6_MODULE.read_text(encoding="utf-8")
    # `is_ev` / `is_gto` 等键**刻意存在**，作为"否认该宣称"的声明（恒为 False）；
    # 因此只禁止它们被设为 True，以及禁止出现 EV 这类正向标签。
    for t in ("expected_value", '"is_ev": True', '"is_gto": True',
              '"is_equilibrium": True', '"is_solver_output": True',
              '"is_profit_evidence": True', '"is_external_validation": True'):
        assert t not in src, t
    assert 'EVIDENCE_GRADE = "MODEL_INTERNAL"' in src


# ---------------------------------------------------------------------------
# T2. 接口能力（要求 2、10）—— 本阶段的核心发现
# ---------------------------------------------------------------------------
def test_t2_search_tree_interface_has_no_conditioned_slot():
    r = probe_search_interface()
    assert r["present"] is True
    assert r["conditioned_evaluation_has_a_slot"] is False
    assert r["bridge_outputs_with_a_slot"] == []
    for opt in BRIDGE_OUTPUTS:
        assert opt not in SEARCH_TREE_INPUTS
        assert opt not in SEARCH_EVAL_INPUTS


def test_t2_interface_matches_frozen_signatures():
    """与冻结签名逐一核对（防止本层声明与实现脱节）。"""
    import inspect
    from poker_decision_ai.cash_strategy_search_tree import build_search_tree
    from poker_decision_ai.cash_strategy_search_eval import evaluate_search
    got_tree = set(inspect.signature(build_search_tree).parameters)
    assert got_tree == set(SEARCH_TREE_INPUTS), got_tree
    got_eval = set(inspect.signature(evaluate_search).parameters)
    assert got_eval == set(SEARCH_EVAL_INPUTS), got_eval


def test_t2_probe_reports_rejected_inputs():
    r = probe_search_interface(passed_inputs=list(SEARCH_TREE_INPUTS)
                               + ["opponent_conditioned", "bridge"])
    assert "opponent_conditioned" in r["rejected_inputs"]
    assert "bridge" in r["rejected_inputs"]
    assert r["conditioned_evaluation_has_a_slot"] is False


def test_t2_root_expansion_is_frozen_order_slice():
    """实测：root 展开是 ap['candidates'][:cap]，不按任何优先级分数重排。"""
    src = pathlib.Path(
        "poker_decision_ai/cash_strategy_search_tree.py").read_text(
            encoding="utf-8")
    assert "candidates = [c for c in (ap.get(\"candidates\") or [])" in src
    assert "expanded_root = candidates[:cap]" in src
    # 不得存在按 bridge/conditioned 分数排序的逻辑
    for bad in ("opponent_conditioned", "bridge_score", "conditioned_score"):
        assert bad not in src


# ---------------------------------------------------------------------------
# T3. 链路分离（要求 5）
# ---------------------------------------------------------------------------
def _loop(pref_rank, pref_search, abstained, status="ABSTAINED",
          bc_present=False, bc_pref=None):
    return {
        "present": True, "loop_status": status,
        "stage_outputs": {
            "ranking": {"present": True, "preferred": pref_rank},
            "belief_conditioned": ({"present": True, "preferred": bc_pref}
                                   if bc_present else None),
            "search_eval": {"present": True, "preferred": pref_search,
                            "n_candidates": 3},
        },
        "decision_record": {"preferred": None if abstained else pref_search,
                            "abstained": abstained,
                            "disagreements": ([{"tier_a": "search",
                                                "tier_b": "ranking"}]
                                              if abstained else []),
                            "selected_by_tier": None},
    }


def test_t3_trace_separates_the_four_layers():
    r = trace_loop_layers(loop_out=_loop({"action": "bet", "amount_bb": 100.0},
                                         {"action": "bet", "amount_bb": 0.75},
                                         abstained=True),
                          bridge_out={"present": True,
                                      "preferred": {"action": "bet",
                                                    "amount_bb": 0.5},
                                      "ranked": [{"score": 1.0}]})
    assert r["present"] is True
    L = r["layers"]
    assert L["bridge_independent"]["enters_search"] is False
    assert L["frozen_search"]["accepts_conditioned_eval"] is False
    assert L["frozen_assembly"]["abstained"] is True
    assert r["assembly_abstained"] is True
    assert "abstain_unknown" in r["abstain_reason"]


def test_t3_trace_fail_closed_on_bad_input():
    assert trace_loop_layers(loop_out=None)["present"] is False


def test_t3_compare_attributes_difference_to_bridge_layer_only():
    a = trace_loop_layers(loop_out=_loop({"action": "bet", "amount_bb": 1.0},
                                         {"action": "bet", "amount_bb": 0.75},
                                         abstained=True),
                          bridge_out={"present": True,
                                      "preferred": {"action": "bet",
                                                    "amount_bb": 0.5},
                                      "ranked": [{"score": 1.0}]})
    b = trace_loop_layers(loop_out=_loop({"action": "bet", "amount_bb": 1.0},
                                         {"action": "bet", "amount_bb": 0.75},
                                         abstained=True),
                          bridge_out={"present": True,
                                      "preferred": {"action": "bet",
                                                    "amount_bb": 2.0},
                                      "ranked": [{"score": 2.0}]})
    r = compare_bridge_effect(traces=[a, b])
    assert r["bridge_layer_changed"] is True
    assert r["search_layer_changed"] is False
    assert r["decision_changed"] is False
    assert r["difference_is_bridge_only"] is True


def test_t3_compare_needs_two_traces():
    a = trace_loop_layers(loop_out=_loop(None, None, True))
    assert compare_bridge_effect(traces=[a])["present"] is False


# ---------------------------------------------------------------------------
# T4. 敏感性（要求 4、5）
# ---------------------------------------------------------------------------
def test_t4_search_insensitive_to_opponent_inputs():
    loops = {
        "aggressive": _loop({"action": "bet", "amount_bb": 100.0},
                            {"action": "bet", "amount_bb": 0.75}, True),
        "tight": _loop({"action": "bet", "amount_bb": 100.0},
                       {"action": "bet", "amount_bb": 0.75}, True),
    }
    r = search_sensitivity_probe(loops_by_input=loops)
    assert r["present"] is True
    assert r["search_sensitive"] is False
    assert r["assembly_sensitive"] is False
    assert r["search_accepts_conditioned_eval"] is False


def test_t4_detects_when_ranking_layer_does_change():
    loops = {
        "a": _loop({"action": "bet", "amount_bb": 1.0},
                   {"action": "bet", "amount_bb": 0.75}, True),
        "b": _loop({"action": "check", "amount_bb": 0.0},
                   {"action": "bet", "amount_bb": 0.75}, True),
    }
    r = search_sensitivity_probe(loops_by_input=loops)
    assert r["ranking_sensitive"] is True
    assert r["search_sensitive"] is False


def test_t4_sensitivity_needs_two_inputs():
    assert search_sensitivity_probe(
        loops_by_input={"only": _loop(None, None, True)})["present"] is False


# ---------------------------------------------------------------------------
# T5. multistreet 前置（要求 3、6）
# ---------------------------------------------------------------------------
def _transition(**kw):
    base = {"present": True, "current_street": "flop", "next_street": "turn",
            "transition_legal": True, "terminal": False, "terminal_reason": None,
            "future_board": {"state": "unresolved", "cards": None,
                             "must_not_invent": True}}
    base.update(kw)
    return base


def test_t5_legal_transition_passes():
    r = check_multistreet(transition=_transition(),
                          state={"street": "flop"})
    assert r["valid"] is True, r["violations"]
    assert r["invents_future_cards"] is False
    assert r["current_street"] == "flop" and r["next_street"] == "turn"


def test_t5_rejects_invented_future_cards():
    t = _transition(future_board={"state": "known", "cards": ["2c"],
                                  "must_not_invent": True})
    r = check_multistreet(transition=t, state={"street": "flop"})
    assert r["valid"] is False
    assert any("future_board" in v for v in r["violations"])


def test_t5_rejects_illegal_transition():
    r = check_multistreet(transition=_transition(transition_legal=False),
                          state={"street": "flop"})
    assert r["valid"] is False
    assert "transition_illegal" in r["violations"]


def test_t5_river_must_stay_terminal():
    t = _transition(current_street="river", next_street=None, terminal=False)
    r = check_multistreet(transition=t, state={"street": "river"})
    assert r["valid"] is False
    assert "river_must_be_terminal" in r["violations"]


def test_t5_multistreet_fail_closed():
    assert check_multistreet(transition=None, state={})["present"] is False


# ---------------------------------------------------------------------------
# T6. 真实引擎集成（要求 1、3、4、5）
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _real():
    if "out" in _CACHE:
        return _CACHE["out"]
    from poker_decision_ai.cash_decision_environment import (
        build_scenario, POPULATION_PARAMS)
    from poker_decision_ai.cash_decision_loop import run_decision_loop
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_multistreet import next_street_transition
    from poker_decision_ai.cash_solver_interface import solver_status
    from poker_decision_ai.cash_strategy_action_policy import build_action_policy
    from poker_decision_ai.cash_strategy_dynamic_belief import (
        build_dynamic_belief)
    from poker_decision_ai.cash_strategy_belief_update import (
        update_belief_with_likelihood)
    from poker_decision_ai.cash_strategy_action_ranking import rank_actions
    from poker_decision_ai.cash_strategy_belief_conditioned_eval import (
        evaluate_belief_conditioned)
    from poker_decision_ai.cash_strategy_search_tree import (
        build_search_tree, DEFAULT_BUDGET)
    from poker_decision_ai.cash_strategy_search_eval import evaluate_search
    from poker_decision_ai.cash_strategy_decision_assembly import (
        assemble_decision)
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_opponent_conditioned)
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import parse_range

    ITER = 200
    calc = EquityCalculator()
    pipe = {"build_action_policy": build_action_policy,
            "build_dynamic_belief": build_dynamic_belief,
            "update_belief": update_belief_with_likelihood,
            "rank_actions": rank_actions,
            "evaluate_belief_conditioned": evaluate_belief_conditioned,
            "build_search_tree": build_search_tree,
            "evaluate_search": evaluate_search,
            "assemble_decision": assemble_decision}

    def equity_fn(hero, board, classes, *, iterations=ITER, seed=42):
        rng = parse_range(",".join(classes), name="p6_test_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        return {"present": True, "hero_equity": float(res.hero_equity),
                "samples": int(res.samples), "method": res.method}

    def one(street="flop", weights=None):
        st = build_scenario(seed=11, street=street)["state"]
        om = build_opponent_model(st)
        rm = build_range_model(st, opponent_model=om)
        tr = next_street_transition(st)
        aw = weights or {"_last_action": "raise", "raise": 1.0}
        out = run_decision_loop(
            st, pipeline=pipe, integrity_context=cash_integrity_context(st),
            opponent_model=om, range_model=rm,
            action_tree=build_action_tree(st), transition=tr, belief=None,
            likelihood_model=None, action_weights=aw,
            solver=solver_status(st), budget=DEFAULT_BUDGET,
            max_root_candidates=3, equity_seed=42, equity_iterations=20000,
            policy=None)
        ap = build_action_policy(st, action_tree=build_action_tree(st),
                                 range_model=rm,
                                 integrity_context=cash_integrity_context(st),
                                 equity_seed=42, equity_iterations=20000)
        bel = build_dynamic_belief(
            st, range_model=rm, opponent_model=om, action_weights=aw,
            integrity_context=cash_integrity_context(st))
        br = evaluate_opponent_conditioned(
            state=st, belief=bel, range_model=rm, candidates=ap["candidates"],
            population="DECLARED_AGGRESSIVE",
            population_params=POPULATION_PARAMS, required_equity=0.0,
            equity_fn=equity_fn, player_id="opp2", iterations=ITER, seed=42)
        return st, tr, out, br

    _CACHE["out"] = {"one": one, "pipe": pipe, "params": POPULATION_PARAMS,
                     "equity_fn": equity_fn, "ITER": ITER}
    return _CACHE["out"]


def test_t6_loop_runs_and_assembly_abstains():
    d = _real()
    st, tr, out, br = d["one"]("flop")
    assert out["present"] is True
    dec = out["decision_record"]
    assert dec["abstained"] is True
    assert dec["preferred"] is None
    assert out["loop_status"] == "ABSTAINED"
    assert dec["disagreements"]


def test_t6_multistreet_transition_is_legal_and_non_inventing():
    d = _real()
    for street in ("flop", "turn"):
        st, tr, out, br = d["one"](street)
        r = check_multistreet(transition=tr, state=st)
        assert r["valid"] is True, (street, r["violations"])
        assert r["invents_future_cards"] is False


def test_t6_bridge_output_does_not_enter_search():
    """要求 4/5：bridge 有输出，但 search 层的 preferred 与其无关。"""
    d = _real()
    st, tr, out, br = d["one"]("flop")
    assert br["present"] is True
    tr_ = trace_loop_layers(loop_out=out, bridge_out=br)
    L = tr_["layers"]
    assert L["bridge_independent"]["present"] is True
    assert L["bridge_independent"]["enters_search"] is False
    assert L["frozen_search"]["accepts_conditioned_eval"] is False
    # search 层的指纹与 bridge 指纹不同（没有把 bridge 结果当成 search 结果）
    assert L["frozen_search"]["fingerprint"] != \
        L["bridge_independent"]["fingerprint"]


def test_t6_search_layer_identical_across_opponent_inputs():
    """要求 5：不同 opponent 输入下 search / assembly 层完全不变。"""
    d = _real()
    loops = {}
    for name, aw in (("maniac", {"_last_action": "raise", "raise": 2.6}),
                     ("tight", {"_last_action": "raise", "raise": 0.6})):
        st, tr, out, br = d["one"]("flop", aw)
        loops[name] = out
    r = search_sensitivity_probe(loops_by_input=loops)
    assert r["search_sensitive"] is False
    assert r["assembly_sensitive"] is False
    assert r["search_accepts_conditioned_eval"] is False


def test_t6_bridge_layer_is_sensitive_where_loop_layers_are_not():
    """把 bridge 的敏感性与 loop 层的惰性并排报告（区分谁产生差异）。"""
    d = _real()
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_opponent_conditioned)
    from poker_decision_ai.cash_decision_environment import POPULATION_PARAMS
    st, tr, out, br_a = d["one"]("flop")
    ap_bridge = {}
    loops = {}
    for pop in ("DECLARED_TIGHT_PASSIVE", "DECLARED_AGGRESSIVE"):
        st2, tr2, o, _ = d["one"]("flop")
        loops[pop] = o
        from poker_decision_ai.cash_strategy_dynamic_belief import (
            build_dynamic_belief)
        from poker_decision_ai.cash_opponent_model import build_opponent_model
        from poker_decision_ai.cash_range_model import build_range_model
        from poker_decision_ai.cash_integrity_context import (
            cash_integrity_context)
        from poker_decision_ai.cash_strategy_action_policy import (
            build_action_policy)
        from poker_decision_ai.cash_action_tree import build_action_tree
        om = build_opponent_model(st2)
        rm = build_range_model(st2, opponent_model=om)
        bel = build_dynamic_belief(
            st2, range_model=rm, opponent_model=om,
            action_weights={"_last_action": "raise", "raise": 1.0},
            integrity_context=cash_integrity_context(st2))
        ap2 = build_action_policy(st2, action_tree=build_action_tree(st2),
                                  range_model=rm,
                                  integrity_context=cash_integrity_context(st2),
                                  equity_seed=42, equity_iterations=20000)
        ap_bridge[pop] = evaluate_opponent_conditioned(
            state=st2, belief=bel, range_model=rm, candidates=ap2["candidates"],
            population=pop, population_params=POPULATION_PARAMS,
            required_equity=0.0, equity_fn=d["equity_fn"], player_id="opp2",
            iterations=d["ITER"], seed=42)
    fps = {p: stable_fingerprint(b["ranked"]) for p, b in ap_bridge.items()}
    assert len(set(fps.values())) > 1, "bridge 必须对 population 敏感"
    r = search_sensitivity_probe(loops_by_input=loops)
    assert r["search_sensitive"] is False, "search 层不敏感（接口无槽位）"


def test_t6_deterministic_repeat_across_layers():
    d = _real()
    a = d["one"]("flop")[2]
    b = d["one"]("flop")[2]
    assert stable_fingerprint(a["stage_outputs"].get("search_eval")) == \
        stable_fingerprint(b["stage_outputs"].get("search_eval"))
    assert stable_fingerprint(a["decision_record"]) == \
        stable_fingerprint(b["decision_record"])


def test_t6_provenance_and_no_external_claims():
    d = _real()
    st, tr, out, br = d["one"]("flop")
    assert out["evidence_level"] == "MODEL_INTERNAL"
    for e in (br,):
        assert e["evidence_grade"] == "MODEL_INTERNAL"
        assert all(v is False for v in e["claim_boundary"].values())
    se = out["stage_outputs"]["search_eval"]
    # frozen search_eval 用 evidence_grade + claim_boundary 表达证据等级
    assert se["evidence_grade"] == "MODEL_INTERNAL"
    assert all(v is False for v in se["claim_boundary"].values())
    assert tr is not None


# ---------------------------------------------------------------------------
# T7. UNKNOWN / fail-closed 保持（要求 9）
# ---------------------------------------------------------------------------
def test_t7_unknown_population_and_modes_still_fail_closed():
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_opponent_conditioned)
    d = _real()
    st, tr, out, br = d["one"]("flop")
    from poker_decision_ai.cash_strategy_dynamic_belief import (
        build_dynamic_belief)
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_strategy_action_policy import build_action_policy
    from poker_decision_ai.cash_action_tree import build_action_tree
    om = build_opponent_model(st)
    rm = build_range_model(st, opponent_model=om)
    bel = build_dynamic_belief(
        st, range_model=rm, opponent_model=om,
        action_weights={"_last_action": "raise", "raise": 1.0},
        integrity_context=cash_integrity_context(st))
    ap = build_action_policy(st, action_tree=build_action_tree(st),
                             range_model=rm,
                             integrity_context=cash_integrity_context(st),
                             equity_seed=42, equity_iterations=20000)
    bad = evaluate_opponent_conditioned(
        state=st, belief=bel, range_model=rm, candidates=ap["candidates"],
        population="NOT_A_POP", population_params=d["params"],
        required_equity=0.0, equity_fn=d["equity_fn"], player_id="opp2",
        iterations=d["ITER"], seed=42)
    assert bad["present"] is False
    mtt = dict(st, game_type="mtt")
    bad2 = evaluate_opponent_conditioned(
        state=mtt, belief=bel, range_model=rm, candidates=ap["candidates"],
        population="DECLARED_BALANCED", population_params=d["params"],
        required_equity=0.0, equity_fn=d["equity_fn"], player_id="opp2",
        iterations=d["ITER"], seed=42)
    assert bad2["present"] is False
    assert bad2["reason"] == "not_cash_mode_fail_closed"


def test_t7_interface_probe_reports_block_not_bypass():
    r = probe_search_interface()
    assert r["conditioned_evaluation_has_a_slot"] is False
    assert len(r["resolution"]) > 20
    assert CLAIM_BOUNDARY["bypasses_frozen_contract"] is False
    for item in ("action_policy_candidate_order", "river_terminal",
                 "tier_rules", "leaf_budget"):
        assert item in r["must_not_change"]
