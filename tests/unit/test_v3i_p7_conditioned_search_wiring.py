"""V3I-P7 — opponent-conditioned evaluation → search 接线（测试）。

覆盖授权要求的测试项；核心结论是**契约级阻塞**（要求 13）：
"仅新增 adapter" 不可行，必须修改冻结 `cash_strategy_search_eval._summarize_leaves`。

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

from poker_decision_ai.cash_conditioned_search_wiring import (
    CASH_P7_SEARCH_WIRING_VERSION,
    CLAIM_BOUNDARY,
    MINIMAL_CONTRACT_CHANGE,
    REJECTED_WORKAROUNDS,
    adapter_contract,
    conditioned_evaluation_input,
    evaluate_search_with_conditioning,
    materialize_conditioned_tree,
    probe_conditioning_reaches_per_candidate_score,
    probe_conditioning_reaches_score,
    probe_leaf_pool_is_global,
    stable_fingerprint,
)

M = pathlib.Path("poker_decision_ai/cash_conditioned_search_wiring.py")
SEARCH_EVAL = pathlib.Path("poker_decision_ai/cash_strategy_search_eval.py")
SEARCH_TREE = pathlib.Path("poker_decision_ai/cash_strategy_search_tree.py")


# ---------------------------------------------------------------------------
# T1. 契约与边界（要求 3、5、11、13）
# ---------------------------------------------------------------------------
def test_t1_contract_declares_block_and_explicit_input():
    c = adapter_contract()
    assert c["version"] == CASH_P7_SEARCH_WIRING_VERSION
    assert c["explicit_input_name"] == "conditioned_input"
    assert c["per_candidate"] is True
    assert c["top1_only"] is False
    assert c["uses_global_state"] is False
    assert c["uses_monkeypatch"] is False
    assert c["hardcodes_population_to_action"] is False
    assert c["modifies_frozen_layers"] is False
    assert c["minimal_contract_change_required"] is True
    assert c["contract_change_authorized"] is False


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output", "is_ev",
                 "wired_to_production", "adds_strategy_capability",
                 "modifies_frozen_layers", "breaks_frozen_contract",
                 "uses_global_state_or_monkeypatch"):
        assert flag in CLAIM_BOUNDARY


def test_t1_module_imports_only_stdlib_and_no_monkeypatch():
    src = M.read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)
    # 只禁止**代码用法**；文档与"被拒做法"列表中出现该词是合规的
    for bad in ("setattr(", "globals()[", "import random", "eval(", "exec("):
        assert bad not in src, bad
    # 无法在运行期替换冻结函数：本模块**不 import 任何项目内模块**
    # （见上方 mods 断言），因此拿不到任何冻结函数引用去改写。
    assert "import" in src
    # 不得出现属性写入/替换冻结函数的调用形态
    for bad in ("setattr(", "globals()[", "delattr("):
        assert bad not in src, bad


def test_t1_minimal_contract_change_is_precise():
    cc = MINIMAL_CONTRACT_CHANGE
    assert cc["file"].endswith("cash_strategy_search_eval.py")
    assert cc["function"] == "_summarize_leaves"
    assert "root_action" in cc["required_change_1"]
    assert "per-candidate" in cc["required_change_2"]
    assert cc["contract_change_authorized_in_this_stage"] is False


def test_t1_rejected_workarounds_are_recorded():
    names = {w["name"] for w in REJECTED_WORKAROUNDS}
    assert "reverse_solve_leaf_values" in names
    assert "monkeypatch_summarize_leaves" in names
    assert "reorder_action_policy_candidates" in names
    for w in REJECTED_WORKAROUNDS:
        assert w["why_rejected"]


def test_t1_frozen_search_files_untouched_by_p7():
    """P7 不得修改冻结 search 文件：核验其关键行仍在。"""
    se = SEARCH_EVAL.read_text(encoding="utf-8")
    # V3I-P8 已按 P7 定位的最小契约给 _summarize_leaves 增加候选身份过滤；
    # 此处只断言**结构仍在**（评分公式与划分逻辑未变）。
    assert "def _summarize_leaves(" in se
    assert "for leaf in tree.get(\"leaves\") or []:" in se
    assert "def _leaf_root_action(" in se
    stt = SEARCH_TREE.read_text(encoding="utf-8")
    assert "expanded_root = candidates[:cap]" in stt
    assert 'DEFAULT_BUDGET' in stt


# ---------------------------------------------------------------------------
# T2. 显式输入形状（要求 5、6）
# ---------------------------------------------------------------------------
def _bridge_out(n=3):
    ranked = []
    for i in range(n):
        ranked.append({"action": "bet" if i else "check",
                       "sizing_kind": "bet_pot_fraction" if i else
                       "check_no_investment",
                       "amount_bb": float(i) * 0.5,
                       "hero_equity_vs_continuing": 0.5 + 0.1 * i,
                       "p_continue": 0.2 * i,
                       "n_continuation_classes": 3,
                       "score": 1.0 - 0.1 * i})
    return {"present": True, "population": "P", "player_id": "opp1",
            "ranked": ranked}


def test_t2_conditioned_input_is_per_candidate_not_top1():
    ci = conditioned_evaluation_input(bridge_out=_bridge_out(4))
    assert ci["present"] is True
    assert ci["n_candidates"] == 4
    assert ci["provides_per_candidate_not_top1"] is True
    assert ci["explicit_input"] is True
    assert ci["uses_global_state"] is False
    assert ci["uses_monkeypatch"] is False
    for v in ci["per_candidate"].values():
        assert "conditioned_score" in v
        assert "action" in v


def test_t2_conditioned_input_fail_closed():
    assert conditioned_evaluation_input(bridge_out=None)["present"] is False
    assert conditioned_evaluation_input(
        bridge_out={"present": False})["present"] is False
    assert conditioned_evaluation_input(
        bridge_out={"present": True, "ranked": []})["present"] is False


def test_t2_materialize_preserves_structure():
    tree = {"present": True, "root": {"children": [
        {"action": "check", "amount_bb": 0.0},
        {"action": "bet", "amount_bb": 0.5}]},
        "leaves": [
            {"path": {"root_action": "check"}, "equity": {"hero_equity": 0.9}},
            {"path": {"root_action": "bet"}, "equity": {"hero_equity": 0.9}}]}
    ci = conditioned_evaluation_input(bridge_out=_bridge_out(2))
    r = materialize_conditioned_tree(search_tree=tree, conditioned_input=ci)
    assert r["present"] is True
    assert r["leaf_count_preserved"] is True
    assert r["n_leaves_in"] == r["n_leaves_out"] == 2
    # 原树未被就地修改
    assert tree["leaves"][0]["equity"]["hero_equity"] == 0.9
    assert "conditioned" not in tree["leaves"][0]["equity"]


def test_t2_materialize_fail_closed():
    assert materialize_conditioned_tree(
        search_tree=None, conditioned_input={})["present"] is False
    assert materialize_conditioned_tree(
        search_tree={"present": True},
        conditioned_input={"present": False})["present"] is False


def test_t2_passthrough_when_no_bridge_input():
    """要求：无 bridge 输入时保持原有 search 行为。"""
    calls = []

    def fake_eval(state, *, search_tree=None, solver=None):
        calls.append(search_tree)
        return {"present": True, "candidates": [], "preferred": None}

    tree = {"present": True, "leaves": []}
    r = evaluate_search_with_conditioning(state={}, search_tree=tree,
                                         conditioned_input=None,
                                         evaluate_search_fn=fake_eval)
    assert r["conditioning_applied"] is False
    assert r["path"] == "passthrough_no_bridge_input"
    assert r["frozen_behavior_preserved"] is True
    assert calls == [tree], "必须把**原树**原样传给冻结函数"


def test_t2_conditioned_path_passes_materialized_tree():
    seen = []

    def fake_eval(state, *, search_tree=None, solver=None):
        seen.append(search_tree)
        return {"present": True, "candidates": [], "preferred": None}

    tree = {"present": True, "root": {"children": [
        {"action": "check", "amount_bb": 0.0}]},
        "leaves": [{"path": {"root_action": "check"},
                    "equity": {"hero_equity": 0.9}}]}
    ci = conditioned_evaluation_input(bridge_out=_bridge_out(1))
    r = evaluate_search_with_conditioning(state={}, search_tree=tree,
                                         conditioned_input=ci,
                                         evaluate_search_fn=fake_eval)
    assert r["conditioning_applied"] is True
    assert r["path"] == "explicit_conditioned_tree_to_frozen_evaluator"
    assert seen and seen[0] is not tree, "必须传物化后的新树，不就地改原树"


# ---------------------------------------------------------------------------
# T3. 池化证明（要求 6、9、13）
# ---------------------------------------------------------------------------
def test_t3_leaf_pool_is_global():
    tree = {"present": True, "leaves": [
        {"path": {"root_action": "check"}, "equity_present": True,
         "equity": {"hero_equity": 0.5}},
        {"path": {"root_action": "bet"}, "equity_present": True,
         "equity": {"hero_equity": 0.5}}]}
    ev = {"present": True, "candidates": [
        {"action": "check", "amount_bb": 0.0, "n_leaves_evaluable": 2,
         "future_equity_mean": 0.5},
        {"action": "bet", "amount_bb": 0.5, "n_leaves_evaluable": 2,
         "future_equity_mean": 0.5}]}
    r = probe_leaf_pool_is_global(search_eval=ev, search_tree=tree)
    assert r["every_candidate_reports_the_global_pool"] is True
    assert r["future_equity_identical_across_candidates"] is True
    assert r["root_action_parameter_used_by_frozen_code"] is False
    assert r["true_leaves_by_root_action"] == {"check": 1, "bet": 1}


def test_t3_p8_probe_detects_real_per_candidate_channel():
    """**V3I-P8 契约修正后**：新探针必须报告"条件化已真正生效"。

    （P7 阶段的 pooled 断层断言已由 P8 的目标取代；此处改为验证 P8 探针。
    真实引擎上的逐候选验证见 test_v3i_p8_minimal_contract_fix.py。）
    """
    tree = {"present": True,
            "root": {"children": [{"action": "check", "amount_bb": 0.0},
                                  {"action": "bet", "amount_bb": 0.5}]},
            "leaves": ([{"path": {"root_action": "check",
                                  "root_candidate_amount_bb": 0.0},
                         "equity_present": True,
                         "equity": {"hero_equity": 0.9}}]
                       + [{"path": {"root_action": "bet",
                                    "root_candidate_amount_bb": 0.5},
                           "equity_present": True,
                           "equity": {"hero_equity": 0.3}}] * 3)}

    def fake_eval(state, *, search_tree=None, solver=None):
        from poker_decision_ai.cash_strategy_search_eval import (
            _summarize_leaves)
        cands = []
        for ch in search_tree["root"]["children"]:
            inv = 0.0 if ch["action"] == "check" else float(ch["amount_bb"])
            sm = _summarize_leaves(search_tree, str(ch["action"]), inv)
            ev = sm["evaluable_equities"]
            m = sum(ev) / len(ev) if ev else None
            s = (m * (1.5 + 2 * inv) / 1.5 - inv / 1.5) if m is not None else None
            cands.append({"action": ch["action"], "amount_bb": ch["amount_bb"],
                          "future_equity_mean": m,
                          "search_score": round(s, 6) if s is not None else None})
        return {"present": True, "candidates": cands,
                "preferred": max(cands, key=lambda c: c["search_score"] or -9)}

    base = {"present": True, "per_candidate": {
        "check|check_no_investment|0.0": {"action": "check", "amount_bb": 0.0,
                                          "hero_equity_vs_continuing": 0.9},
        "bet|bet_pot_fraction|0.5": {"action": "bet", "amount_bb": 0.5,
                                     "hero_equity_vs_continuing": 0.3}}}
    r = probe_conditioning_reaches_per_candidate_score(
        search_tree=tree, conditioned_input=base,
        evaluate_search_fn=fake_eval, state={})
    assert r["present"] is True
    assert r["future_equity_is_per_candidate"] is True
    assert r["per_candidate_scores_respond_to_own_conditioned_equity"] is True
    assert r["n_rows_changed"] > 0


def test_t3_probe_fail_closed():
    assert probe_leaf_pool_is_global(search_eval=None,
                                     search_tree={})["present"] is False
    assert probe_conditioning_reaches_per_candidate_score(
        search_tree=None, conditioned_input={}, evaluate_search_fn=None,
        state={})["present"] is False


def test_t3_reaches_score_probe_detects_identical():
    plain = {"present": True, "candidates": [
        {"action": "bet", "amount_bb": 0.5, "future_equity_mean": 0.5,
         "search_score": 1.0}]}
    cond = {"present": True, "candidates": [
        {"action": "bet", "amount_bb": 0.5, "future_equity_mean": 0.5,
         "search_score": 1.0}]}
    r = probe_conditioning_reaches_score(search_eval_plain=plain,
                                        search_eval_conditioned=cond)
    assert r["identical"] is True
    assert r["conditioning_changed_any_score"] is False


def test_t3_reaches_score_probe_detects_change():
    plain = {"present": True, "candidates": [
        {"action": "bet", "amount_bb": 0.5, "future_equity_mean": 0.5,
         "search_score": 1.0}]}
    cond = {"present": True, "candidates": [
        {"action": "bet", "amount_bb": 0.5, "future_equity_mean": 0.6,
         "search_score": 1.2}]}
    r = probe_conditioning_reaches_score(search_eval_plain=plain,
                                        search_eval_conditioned=cond)
    assert r["identical"] is False
    assert r["conditioning_changed_any_score"] is True


# ---------------------------------------------------------------------------
# T4. 真实引擎集成（要求 7、9、10）
# ---------------------------------------------------------------------------
_CACHE: dict = {}


def _real():
    if "out" in _CACHE:
        return _CACHE["out"]
    from poker_decision_ai.cash_decision_environment import (
        build_scenario, POPULATION_PARAMS)
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_multistreet import next_street_transition
    from poker_decision_ai.cash_strategy_action_policy import (
        build_action_policy)
    from poker_decision_ai.cash_strategy_dynamic_belief import (
        build_dynamic_belief)
    from poker_decision_ai.cash_strategy_search_tree import (
        build_search_tree, DEFAULT_BUDGET)
    from poker_decision_ai.cash_strategy_search_eval import evaluate_search
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_opponent_conditioned)
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import parse_range

    ITER = 200
    calc = EquityCalculator()

    def equity_fn(hero, board, classes, *, iterations=ITER, seed=42):
        rng = parse_range(",".join(classes), name="p7_test_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        return {"present": True, "hero_equity": float(res.hero_equity),
                "samples": int(res.samples), "method": res.method}

    def setup(street="flop"):
        st = build_scenario(seed=11, street=street)["state"]
        om = build_opponent_model(st)
        rm = build_range_model(st, opponent_model=om)
        ap = build_action_policy(
            st, action_tree=build_action_tree(st), range_model=rm,
            integrity_context=cash_integrity_context(st), equity_seed=42,
            equity_iterations=20000)
        bel = build_dynamic_belief(
            st, range_model=rm, opponent_model=om,
            action_weights={"_last_action": "raise", "raise": 1.0},
            integrity_context=cash_integrity_context(st))
        tree = build_search_tree(
            st, action_policy=ap, belief=bel, belief_update=None,
            transition=next_street_transition(st), budget=DEFAULT_BUDGET,
            max_root_candidates=3)
        return st, rm, ap, bel, tree

    _CACHE["out"] = {"setup": setup, "equity_fn": equity_fn, "ITER": ITER,
                     "params": POPULATION_PARAMS,
                     "evaluate_search": evaluate_search,
                     "bridge": evaluate_opponent_conditioned}
    return _CACHE["out"]


def _bridge_for(st, rm, ap, bel, pop):
    d = _real()
    return d["bridge"](state=st, belief=bel, range_model=rm,
                       candidates=ap["candidates"], population=pop,
                       population_params=d["params"], required_equity=0.0,
                       equity_fn=d["equity_fn"], player_id="opp2",
                       iterations=d["ITER"], seed=42)


def test_t4_adapter_accepts_real_bridge_and_preserves_leaf_count():
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    br = _bridge_for(st, rm, ap, bel, "DECLARED_BALANCED")
    ci = conditioned_evaluation_input(bridge_out=br)
    mat = materialize_conditioned_tree(search_tree=tree, conditioned_input=ci)
    assert mat["leaf_count_preserved"] is True
    assert mat["n_leaves_rewritten"] == len(tree["leaves"])
    # 候选上限与顺序保持不变
    before = [c["amount_bb"] for c in tree["root"]["children"]]
    after = [c["amount_bb"] for c in mat["tree"]["root"]["children"]]
    assert before == after


def test_t4_three_populations_produce_different_conditioned_input():
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    vecs = {}
    for pop in ("DECLARED_TIGHT_PASSIVE", "DECLARED_BALANCED",
                "DECLARED_AGGRESSIVE"):
        br = _bridge_for(st, rm, ap, bel, pop)
        ci = conditioned_evaluation_input(bridge_out=br)
        vecs[pop] = json.dumps({k: v["hero_equity_vs_continuing"]
                                for k, v in ci["per_candidate"].items()},
                               sort_keys=True)
    assert len(set(vecs.values())) == 3, "三种 population 必须给出不同条件化权益"


def test_t4_scores_now_respond_to_per_candidate_equity_real_engine():
    """真实引擎：**V3I-P8 契约修正后**，分数确实随候选自身条件化权益变化。

    （P7 阶段本测试断言的是"池均值固定 ⇒ 评分字段不变"；P8 修正
    `_summarize_leaves` 的 `root_action` 归属后该断言**不再成立**，
    这正是 P8 的目标，故在此更新为断言**修正后**的行为。）
    """
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    br = _bridge_for(st, rm, ap, bel, "DECLARED_BALANCED")
    ci = conditioned_evaluation_input(bridge_out=br)
    r = probe_conditioning_reaches_per_candidate_score(
        search_tree=tree, conditioned_input=ci,
        evaluate_search_fn=d["evaluate_search"], state=st)
    assert r["present"] is True
    assert r["future_equity_is_per_candidate"] is True
    assert r["per_candidate_scores_respond_to_own_conditioned_equity"] is True
    assert r["n_rows_changed"] > 0


def test_t4_real_leaf_attribution_is_per_candidate():
    """V3I-P8 后：每个候选只统计**自己**的叶子（不再是全局池）。"""
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    plain = d["evaluate_search"](st, search_tree=tree, solver=None)
    r = probe_leaf_pool_is_global(search_eval=plain, search_tree=tree)
    # 归属已修正：报告的叶子数等于该 root 动作自身的叶子数
    assert r["every_candidate_reports_the_global_pool"] is False
    assert r["total_leaves"] > 0
    total_by_root = r["true_leaves_by_root_action"]
    assert len(total_by_root) >= 2
    for key, n in r["reported_n_leaves_evaluable"].items():
        ra = key.split("|")[0]
        assert n == total_by_root[ra], (key, n, total_by_root)


def test_t4_passthrough_matches_direct_frozen_call_on_real_engine():
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    direct = d["evaluate_search"](st, search_tree=tree, solver=None)
    via = evaluate_search_with_conditioning(
        state=st, search_tree=tree, conditioned_input=None,
        evaluate_search_fn=d["evaluate_search"])
    assert stable_fingerprint(via["search_eval"]) == stable_fingerprint(direct)


def test_t4_multistreet_conditioning_effective_on_both_streets():
    """V3I-P8 后：flop 与 turn 上条件化都真正进入逐候选评分。"""
    d = _real()
    for street in ("flop", "turn"):
        st, rm, ap, bel, tree = d["setup"](street)
        br = _bridge_for(st, rm, ap, bel, "DECLARED_AGGRESSIVE")
        ci = conditioned_evaluation_input(bridge_out=br)
        r = probe_conditioning_reaches_per_candidate_score(
            search_tree=tree, conditioned_input=ci,
            evaluate_search_fn=d["evaluate_search"], state=st)
        assert r["future_equity_is_per_candidate"] is True, street
        assert r["n_rows_changed"] > 0, street


def test_t4_deterministic_repeat():
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    a = _bridge_for(st, rm, ap, bel, "DECLARED_MANIAC")
    b = _bridge_for(st, rm, ap, bel, "DECLARED_MANIAC")
    assert stable_fingerprint(a) == stable_fingerprint(b)
    c1 = conditioned_evaluation_input(bridge_out=a)
    c2 = conditioned_evaluation_input(bridge_out=b)
    assert stable_fingerprint(c1) == stable_fingerprint(c2)


def test_t4_no_external_claims_in_conditioned_input():
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    br = _bridge_for(st, rm, ap, bel, "DECLARED_BALANCED")
    ci = conditioned_evaluation_input(bridge_out=br)
    assert ci["evidence_grade"] == "MODEL_INTERNAL"
    assert all(v is False for v in ci["claim_boundary"].values())


# ---------------------------------------------------------------------------
# T5. UNKNOWN / fail-closed（要求 10）
# ---------------------------------------------------------------------------
def test_t5_unknown_population_fails_closed():
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    bad = d["bridge"](state=st, belief=bel, range_model=rm,
                      candidates=ap["candidates"], population="NOT_A_POP",
                      population_params=d["params"], required_equity=0.0,
                      equity_fn=d["equity_fn"], player_id="opp2",
                      iterations=d["ITER"], seed=42)
    assert bad["present"] is False
    assert conditioned_evaluation_input(bridge_out=bad)["present"] is False


def test_t5_no_evidence_does_not_produce_conditioned_input():
    d = _real()
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_prior_only)
    st, rm, ap, bel, tree = d["setup"]("flop")
    po = evaluate_prior_only(state=st, candidates=ap["candidates"])
    assert po["adaptation_applied"] is False
    assert po["equity_calls_used"] == 0
    # prior-only 形状相似但**无对手证据** → 必须 fail-closed
    r = conditioned_evaluation_input(bridge_out=po)
    assert r["present"] is False
    assert r["reason"] == "prior_only_has_no_opponent_evidence"


def test_t5_adaptation_flag_false_is_rejected():
    """显式 adaptation_applied=False 的输出不得被当作条件化输入。"""
    fake = {"present": True, "adaptation_applied": False,
            "ranked": [{"action": "bet", "sizing_kind": "bet_pot_fraction",
                        "amount_bb": 0.5, "score": 1.0}]}
    r = conditioned_evaluation_input(bridge_out=fake)
    assert r["present"] is False
    assert r["reason"] == "adaptation_not_applied"


def test_t5_real_bridge_output_is_accepted_while_prior_only_is_not():
    """并排对照：真实 bridge 输出被接受；prior-only 被拒绝。"""
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_prior_only)
    d = _real()
    st, rm, ap, bel, tree = d["setup"]("flop")
    good = conditioned_evaluation_input(
        bridge_out=_bridge_for(st, rm, ap, bel, "DECLARED_BALANCED"))
    bad = conditioned_evaluation_input(
        bridge_out=evaluate_prior_only(state=st, candidates=ap["candidates"]))
    assert good["present"] is True and good["n_candidates"] >= 1
    assert bad["present"] is False
