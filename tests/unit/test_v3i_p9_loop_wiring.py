"""V3I-P9 — 把 opponent-conditioned search bridge 最小接入 V3I-P1 决策链（测试）。

覆盖授权要求的测试项：P1 → P8 真实数据流、三种 population、candidate-local
evaluation、ranking 变化、无 conditioning 回归、UNKNOWN/conflict、multistreet、
deterministic repeat、frozen drift、production wiring。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_opponent_conditioned_loop import (
    CASH_P9_LOOP_WIRING_VERSION,
    CLAIM_BOUNDARY,
    EXPECTED_REQUIRED_STAGES,
    EXTRA_PIPELINE_KEYS,
    LOOP_EVALUATE_SEARCH_INPUTS,
    audit_loop_wiring_contract,
    compare_loop_runs,
    make_conditioned_evaluate_search,
    p9_contract,
    prepare_conditioned_search,
    run_loop_with_conditioning,
    stable_fingerprint,
)

P9_MODULE = pathlib.Path("poker_decision_ai/cash_opponent_conditioned_loop.py")
LOOP_MODULE = pathlib.Path("poker_decision_ai/cash_decision_loop.py")


# ---------------------------------------------------------------------------
# T1. 契约与边界（要求 1、2、4、11、12）
# ---------------------------------------------------------------------------
def test_t1_contract_no_loop_modification_no_widening():
    c = p9_contract()
    assert c["version"] == CASH_P9_LOOP_WIRING_VERSION
    assert c["needs_loop_modification"] is False
    assert c["needs_frozen_contract_widening"] is False
    assert c["adds_strategy_capability"] is False
    assert "closure" in c["approach"]
    assert tuple(c["extra_pipeline_keys"]) == EXTRA_PIPELINE_KEYS
    assert tuple(c["loop_evaluate_search_inputs"]) == LOOP_EVALUATE_SEARCH_INPUTS


def test_t1_claim_boundary_all_false():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_gto", "is_equilibrium", "is_solver_output", "is_ev",
                 "wired_to_production", "rewrites_p1_loop",
                 "modifies_frozen_layers", "widens_frozen_contract",
                 "hardcodes_population_to_action"):
        assert flag in CLAIM_BOUNDARY


def test_t1_module_imports_only_stdlib_no_monkeypatch():
    src = P9_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib", "inspect"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)
    for bad in ("setattr(", "globals()[", "delattr(", "import random",
                "eval(", "exec("):
        assert bad not in src, bad


def test_t1_p1_loop_source_untouched():
    """要求 2、12：P1 loop 未被本阶段修改（关键结构仍在）。"""
    src = LOOP_MODULE.read_text(encoding="utf-8")
    for key in ("def run_decision_loop(", "REQUIRED_PIPELINE_STAGES",
                "def _call(", "_call(\"evaluate_search\", search_tree=stree,"
                " solver=solver)", "STAGE_ORDER"):
        assert key in src, key


def test_t1_audit_reports_extra_keys_allowed():
    from poker_decision_ai.cash_decision_loop import run_decision_loop
    a = audit_loop_wiring_contract(run_decision_loop_fn=run_decision_loop)
    assert a["present"] is True
    assert a["has_state"] is True and a["has_pipeline"] is True
    assert a["extra_pipeline_keys_allowed"] is True
    assert a["needs_loop_modification"] is False
    assert a["needs_frozen_contract_widening"] is False


def test_t1_audit_fail_closed_on_non_callable():
    assert audit_loop_wiring_contract(run_decision_loop_fn=None)["present"] is False
    assert audit_loop_wiring_contract(run_decision_loop_fn=42)["present"] is False


def test_t1_expected_stages_match_p1_constant():
    from poker_decision_ai.cash_decision_loop import REQUIRED_PIPELINE_STAGES
    assert tuple(REQUIRED_PIPELINE_STAGES) == EXPECTED_REQUIRED_STAGES


# ---------------------------------------------------------------------------
# T2. 闭包：无 conditioning 时保持原行为（要求 6、5）
# ---------------------------------------------------------------------------
def test_t2_passthrough_when_prepared_is_none():
    seen = []

    def frozen(state, *, search_tree=None, solver=None):
        seen.append(search_tree)
        return {"present": True, "candidates": [], "preferred": None}

    built = make_conditioned_evaluate_search(prepared=None,
                                            frozen_evaluate_search=frozen)
    assert built["present"] is True
    assert built["will_route"] is False
    tree = {"present": True, "leaves": []}
    out = built["callable"]({}, search_tree=tree, solver=None)
    assert seen == [tree], "必须把原树原样传给冻结函数"
    assert out["p9_routing"]["routed"] is False
    assert out["p9_routing"]["path"] == "passthrough_no_conditioning"


def test_t2_passthrough_when_prepared_unavailable():
    def frozen(state, *, search_tree=None, solver=None):
        return {"present": True, "candidates": []}

    built = make_conditioned_evaluate_search(
        prepared={"present": False, "reason": "bridge_unavailable:x"},
        frozen_evaluate_search=frozen)
    assert built["will_route"] is False
    out = built["callable"]({}, search_tree={"present": True}, solver=None)
    assert out["p9_routing"]["routed"] is False
    assert out["p9_routing"]["reason"] == "bridge_unavailable:x"


def test_t2_routes_when_prepared_present():
    calls = []

    def frozen(state, *, search_tree=None, solver=None):
        calls.append(search_tree)
        return {"present": True, "candidates": []}

    def materialize(*, search_tree=None, conditioned_input=None):
        return {"present": True, "tree": {"materialized": True},
                "n_leaves_rewritten": 5}

    built = make_conditioned_evaluate_search(
        prepared={"present": True, "population": "P", "n_candidates": 3,
                  "materialize_fn": materialize,
                  "conditioned_input": {"present": True}},
        frozen_evaluate_search=frozen)
    assert built["will_route"] is True
    out = built["callable"]({}, search_tree={"orig": True}, solver=None)
    assert calls == [{"materialized": True}], "必须传物化后的树"
    assert out["p9_routing"]["routed"] is True
    assert out["p9_routing"]["population"] == "P"


def test_t2_passthrough_when_materialize_fails():
    def frozen(state, *, search_tree=None, solver=None):
        return {"present": True}

    def materialize(*, search_tree=None, conditioned_input=None):
        return {"present": False, "reason": "boom"}

    built = make_conditioned_evaluate_search(
        prepared={"present": True, "materialize_fn": materialize,
                  "conditioned_input": {}},
        frozen_evaluate_search=frozen)
    out = built["callable"]({}, search_tree={"orig": True}, solver=None)
    assert out["p9_routing"]["routed"] is False
    assert out["p9_routing"]["path"] == "passthrough_materialize_failed"


def test_t2_closure_signature_matches_loop_call():
    """闭包必须能被 loop 以 (state, search_tree=, solver=) 调用。"""
    import inspect

    def frozen(state, *, search_tree=None, solver=None):
        return {}

    built = make_conditioned_evaluate_search(prepared=None,
                                            frozen_evaluate_search=frozen)
    got = set(inspect.signature(built["callable"]).parameters)
    assert got == set(LOOP_EVALUATE_SEARCH_INPUTS), got


def test_t2_closure_fail_closed_on_bad_frozen():
    assert make_conditioned_evaluate_search(
        prepared=None, frozen_evaluate_search=None)["present"] is False


# ---------------------------------------------------------------------------
# T3. prepare_conditioned_search fail-closed（要求 5）
# ---------------------------------------------------------------------------
def test_t3_prepare_fail_closed_on_missing_injections():
    base = dict(state={}, belief={}, range_model={}, candidates=[],
                population="P", population_params={}, required_equity=0.0,
                equity_fn=lambda *a, **k: {})
    for missing in ("bridge_fn", "input_fn", "materialize_fn"):
        kw = dict(base)
        kw["bridge_fn"] = lambda **k: {"present": False}
        kw["input_fn"] = lambda **k: {"present": False}
        kw["materialize_fn"] = lambda **k: {"present": False}
        kw[missing] = None
        r = prepare_conditioned_search(**kw)
        assert r["present"] is False
        assert r["reason"] == f"{missing}_required"


def test_t3_prepare_fail_closed_when_bridge_unavailable():
    r = prepare_conditioned_search(
        state={}, belief={}, range_model={}, candidates=[], population="P",
        population_params={}, required_equity=0.0,
        equity_fn=lambda *a, **k: {},
        bridge_fn=lambda **k: {"present": False, "reason": "not_cash_mode_fail_closed"},
        input_fn=lambda **k: {"present": True},
        materialize_fn=lambda **k: {"present": True})
    assert r["present"] is False
    assert "bridge_unavailable" in r["reason"]
    assert "not_cash_mode_fail_closed" in r["reason"]


def test_t3_prepare_fail_closed_on_prior_only():
    """prior-only（无对手证据）不得产生条件化。"""
    r = prepare_conditioned_search(
        state={}, belief={}, range_model={}, candidates=[], population="P",
        population_params={}, required_equity=0.0,
        equity_fn=lambda *a, **k: {},
        bridge_fn=lambda **k: {"present": True, "prior_only": True,
                               "adaptation_applied": False,
                               "ranked": [{"score": 1.0}]},
        input_fn=lambda **k: {"present": False,
                              "reason": "prior_only_has_no_opponent_evidence"},
        materialize_fn=lambda **k: {"present": True})
    assert r["present"] is False
    assert "conditioned_input_unavailable" in r["reason"]


def test_t3_prepare_fail_closed_when_not_per_candidate():
    """只有一个候选 → 不满足"每候选都有条件化" → 拒绝。"""
    r = prepare_conditioned_search(
        state={}, belief={}, range_model={}, candidates=[], population="P",
        population_params={}, required_equity=0.0,
        equity_fn=lambda *a, **k: {},
        bridge_fn=lambda **k: {"present": True, "preferred": {}},
        input_fn=lambda **k: {"present": True,
                              "per_candidate": {"only": {}}},
        materialize_fn=lambda *a, **k: {"present": True})
    assert r["present"] is False
    assert r["reason"] == "conditioned_input_not_per_candidate"


# ---------------------------------------------------------------------------
# T4. 真实循环集成（要求 3、7、8）
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
    from poker_decision_ai.cash_strategy_action_policy import (
        build_action_policy)
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
        evaluate_opponent_conditioned, evaluate_prior_only)
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input, materialize_conditioned_tree)
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import parse_range

    ITER = 200
    calc = EquityCalculator()
    pipeline = {
        "build_action_policy": build_action_policy,
        "build_dynamic_belief": build_dynamic_belief,
        "update_belief": update_belief_with_likelihood,
        "rank_actions": rank_actions,
        "evaluate_belief_conditioned": evaluate_belief_conditioned,
        "build_search_tree": build_search_tree,
        "evaluate_search": evaluate_search,
        "assemble_decision": assemble_decision,
    }

    def equity_fn(hero, board, classes, *, iterations=ITER, seed=42):
        rng = parse_range(",".join(classes), name="p9_test_range")
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
        loop_kwargs = dict(
            integrity_context=cash_integrity_context(st), opponent_model=om,
            range_model=rm, action_tree=build_action_tree(st),
            transition=next_street_transition(st), belief=None,
            likelihood_model=None,
            action_weights={"_last_action": "raise", "raise": 1.0},
            solver=solver_status(st), budget=DEFAULT_BUDGET,
            max_root_candidates=3, equity_seed=42, equity_iterations=20000,
            policy=None)
        return st, rm, ap, bel, loop_kwargs

    def prepare(st, rm, ap, bel, pop):
        return prepare_conditioned_search(
            state=st, belief=bel, range_model=rm, candidates=ap["candidates"],
            population=pop, population_params=POPULATION_PARAMS,
            required_equity=0.0, equity_fn=equity_fn, player_id="opp2",
            iterations=ITER, seed=42,
            bridge_fn=evaluate_opponent_conditioned,
            input_fn=conditioned_evaluation_input,
            materialize_fn=materialize_conditioned_tree)

    _CACHE["out"] = {"pipeline": pipeline, "setup": setup, "prepare": prepare,
                     "run_decision_loop": run_decision_loop,
                     "evaluate_search": evaluate_search,
                     "evaluate_prior_only": evaluate_prior_only,
                     "bridge": evaluate_opponent_conditioned}
    return _CACHE["out"]


def _run(st, loop_kwargs, pipeline, prepared,
         run_decision_loop_fn, evaluate_search_fn):
    return run_loop_with_conditioning(
        state=st, pipeline=pipeline, run_decision_loop_fn=run_decision_loop_fn,
        prepared=prepared, frozen_evaluate_search=evaluate_search_fn,
        loop_kwargs=loop_kwargs)


def test_t4_conditioning_reaches_real_loop():
    """要求 3：条件化进入**真实** loop（非测试专用路径）。"""
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("flop")
    prep = d["prepare"](st, rm, ap, bel, "DECLARED_BALANCED")
    r = _run(st, lk, d["pipeline"], prep, d["run_decision_loop"],
             d["evaluate_search"])
    assert r["present"] is True
    assert r["routed_through_p9"] is True
    assert r["conditioned_routing"]["path"] == \
        "explicit_conditioned_tree_to_frozen_evaluator"
    se = r["loop_out"]["stage_outputs"]["search_eval"]
    assert se["p9_routing"]["routed"] is True
    # 确实由真实 loop 产出
    assert r["loop_out"]["loop_status"] in ("ABSTAINED", "COMPLETE", "UNKNOWN")
    assert r["loop_out"]["stage_trace"]


def test_t4_baseline_loop_is_passthrough():
    """要求 6：无 conditioning 输入时 loop 走原路径。"""
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("flop")
    r = _run(st, lk, d["pipeline"], None, d["run_decision_loop"],
             d["evaluate_search"])
    assert r["routed_through_p9"] is False
    assert r["conditioned_routing"]["path"] == "passthrough_no_conditioning"


def test_t4_baseline_identical_to_direct_frozen_loop():
    """要求 6：passsthrough 的 search 层与直接跑冻结 loop 逐字节一致。"""
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("flop")
    pip = dict(d["pipeline"])
    direct = d["run_decision_loop"](st, pipeline=pip, **lk)
    via = _run(st, lk, d["pipeline"], None, d["run_decision_loop"],
               d["evaluate_search"])
    dse = direct["stage_outputs"]["search_eval"]
    vse = dict(via["loop_out"]["stage_outputs"]["search_eval"])
    vse.pop("p9_routing", None)
    assert stable_fingerprint(vse) == stable_fingerprint(dse)


def test_t4_three_populations_differ_in_candidate_evaluation():
    """要求 7：三种 population 经真实 loop 产生不同 candidate 评估。"""
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("flop")
    runs = {}
    for pop in ("DECLARED_TIGHT_PASSIVE", "DECLARED_BALANCED",
                "DECLARED_AGGRESSIVE"):
        prep = d["prepare"](st, rm, ap, bel, pop)
        runs[pop] = _run(st, lk, d["pipeline"], prep, d["run_decision_loop"],
                         d["evaluate_search"])
    cmp = compare_loop_runs(runs_by_population=runs)
    assert cmp["present"] is True
    assert cmp["n_candidates_with_difference"] >= 1, cmp["per_population"]
    assert cmp["search_layer_changed"] is True
    assert cmp["preferred_changed"] is True
    assert cmp["all_conditioned_runs_routed"] is True
    assert cmp["n_conditioned_runs"] == 3


def test_t4_candidate_local_evaluation_is_not_pooled():
    """要求 7：同一条件下各候选的 future_equity_mean 必须**不同**。"""
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("flop")
    prep = d["prepare"](st, rm, ap, bel, "DECLARED_AGGRESSIVE")
    r = _run(st, lk, d["pipeline"], prep, d["run_decision_loop"],
             d["evaluate_search"])
    se = r["loop_out"]["stage_outputs"]["search_eval"]
    means = {c["future_equity_mean"] for c in se["candidates"]}
    assert len(means) > 1, means


def test_t4_population_label_is_not_the_cause():
    """要求 8：同标签换参数 → 结果相同；异标签同参数 → 结果相同。

    即：因果来源是**参数数值**（经 bridge → conditioned equity），
    不是 population 这个字符串标签。
    """
    from poker_decision_ai.cash_decision_environment import (
        POPULATION_PARAMS as PP)
    from poker_decision_ai.cash_opponent_conditioned_bridge import (
        evaluate_opponent_conditioned as bridge_fn)
    from poker_decision_ai.cash_conditioned_search_wiring import (
        conditioned_evaluation_input as input_fn,
        materialize_conditioned_tree as materialize_fn)

    d = _real()
    st, rm, ap, bel, lk = d["setup"]("flop")
    eq = _equity_fn()

    def prep(pop, params):
        return prepare_conditioned_search(
            state=st, belief=bel, range_model=rm, candidates=ap["candidates"],
            population=pop, population_params=params, required_equity=0.0,
            equity_fn=eq, player_id="opp2", iterations=200, seed=42,
            bridge_fn=bridge_fn, input_fn=input_fn,
            materialize_fn=materialize_fn)

    balanced = prep("DECLARED_BALANCED", PP)

    # (a) AGGRESSIVE 这个**标签** + BALANCED 的**参数** → 与 BALANCED 相同
    params_a = dict(PP)
    params_a["DECLARED_AGGRESSIVE"] = dict(PP["DECLARED_BALANCED"])
    agg_as_bal = prep("DECLARED_AGGRESSIVE", params_a)
    assert json.dumps(balanced["conditioned_input"]["per_candidate"],
                      sort_keys=True, default=str) == \
        json.dumps(agg_as_bal["conditioned_input"]["per_candidate"],
                   sort_keys=True, default=str)

    # (b) 真 AGGRESSIVE 参数 → 必须不同（差异来自参数数值）
    real_agg = prep("DECLARED_AGGRESSIVE", PP)
    assert json.dumps(balanced["conditioned_input"]["per_candidate"],
                      sort_keys=True, default=str) != \
        json.dumps(real_agg["conditioned_input"]["per_candidate"],
                   sort_keys=True, default=str)


def _equity_fn():
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import parse_range
    calc = EquityCalculator()

    def fn(hero, board, classes, *, iterations=200, seed=42):
        rng = parse_range(",".join(classes), name="p9_test_range")
        res = calc.hand_vs_range(list(hero), list(board), rng,
                                 iterations=int(iterations), seed=int(seed))
        return {"present": True, "hero_equity": float(res.hero_equity),
                "samples": int(res.samples), "method": res.method}
    return fn


def test_t4_deterministic_repeat():
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("flop")
    outs = []
    for _ in range(2):
        prep = d["prepare"](st, rm, ap, bel, "DECLARED_MANIAC")
        r = _run(st, lk, d["pipeline"], prep, d["run_decision_loop"],
                 d["evaluate_search"])
        outs.append(r["search_eval_fingerprint"])
    assert outs[0] == outs[1], outs


# ---------------------------------------------------------------------------
# T5. multistreet（要求：multistreet）
# ---------------------------------------------------------------------------
def test_t5_conditioning_effective_on_turn():
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("turn")
    prep = d["prepare"](st, rm, ap, bel, "DECLARED_BALANCED")
    r = _run(st, lk, d["pipeline"], prep, d["run_decision_loop"],
             d["evaluate_search"])
    assert r["routed_through_p9"] is True
    se = r["loop_out"]["stage_outputs"]["search_eval"]
    means = {c["future_equity_mean"] for c in se["candidates"]}
    assert len(means) > 1, means


def test_t5_baseline_preserved_on_turn():
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("turn")
    pip = dict(d["pipeline"])
    direct = d["run_decision_loop"](st, pipeline=pip, **lk)
    via = _run(st, lk, d["pipeline"], None, d["run_decision_loop"],
               d["evaluate_search"])
    vse = dict(via["loop_out"]["stage_outputs"]["search_eval"])
    vse.pop("p9_routing", None)
    assert stable_fingerprint(vse) == \
        stable_fingerprint(direct["stage_outputs"]["search_eval"])


# ---------------------------------------------------------------------------
# T6. assembly 未被修改（要求 4）
# ---------------------------------------------------------------------------
def test_t6_assembly_layer_unchanged_and_abstain_preserved():
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("flop")
    prep = d["prepare"](st, rm, ap, bel, "DECLARED_BALANCED")
    r = _run(st, lk, d["pipeline"], prep, d["run_decision_loop"],
             d["evaluate_search"])
    dec = r["loop_out"]["decision_record"]
    # 冻结 assembly 未改：仍按 tier 分歧弃权，本层不干预
    assert r["assembly"]["abstained"] == dec["abstained"]
    assert r["loop_status"] == r["loop_out"]["loop_status"]
    if dec["abstained"]:
        assert dec["preferred"] is None
        assert dec["disagreements"]


def test_t6_frozen_hashes_unchanged():
    """要求 4、11：冻结层与 acceptance/manifest/strategy_core 未改。"""
    frozen = {
        "poker_decision_ai/cash_decision_loop.py": "7c4b0a11c1e73a7f",
        "poker_decision_ai/cash_decision_environment.py": "ccbf7f0048eee525",
        "poker_decision_ai/cash_strategy_dynamic_belief.py": "01549df2a3385d01",
        "poker_decision_ai/cash_strategy_likelihood_model.py": "df207954ade71fa6",
  # V3I-P13 经授权在 cash_strategy_decision_assembly 实施 P12 设计的
  # feasible_keys 最小契约（**仅新增可选入参**）；装配语义未变，
  # 故此处更新为**授权的**新哈希。
        "poker_decision_ai/cash_strategy_decision_assembly.py": "3732314ab0182063",
        "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
        "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
        "poker_decision_ai/realtime.py": "3400a20ff857880d",
    }
    for path, want in frozen.items():
        got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
        assert got == want, (path, got, want)


def test_t6_manifest_unchanged():
    import json as _json
    man = _json.loads(pathlib.Path(
        "docs/V3G_11_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    assert man["counts"] == {"modules": 10, "tests": 11, "closeouts": 13,
                             "production_baseline": 7, "total": 41}


# ---------------------------------------------------------------------------
# T7. production import / wiring（要求：production wiring）
# ---------------------------------------------------------------------------
def test_t7_no_production_wiring_added():
    """本阶段不新增生产接线：生产导入图仍只有两条受许路径。"""
    def imports_of(p):
        tree = ast.parse(pathlib.Path(p).read_text(encoding="utf-8"))
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                if n.module:
                    mods.add(n.module.split(".")[-1])
                for a in n.names:
                    mods.add(a.name.split(".")[-1])
            elif isinstance(n, ast.Import):
                for a in n.names:
                    mods.add(a.name.split(".")[-1])
        return mods

    found = {}
    for f in ("poker_decision_ai/realtime.py",
              "poker_decision_ai/agents/judge/agent.py"):
        for m in sorted(imports_of(f)):
            if m.startswith("cash_"):
                found.setdefault(m, []).append(pathlib.Path(f).name)
    assert {k: sorted(set(v)) for k, v in found.items()} == {
        "cash_integrity_context": ["agent.py", "realtime.py"],
        "cash_unknown_e2e": ["realtime.py"],
    }, found
    for mod in ("cash_opponent_conditioned_loop",
                "cash_opponent_conditioned_bridge",
                "cash_conditioned_search_wiring"):
        assert mod not in found


def test_t7_search_budget_and_cap_unchanged():
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    assert DEFAULT_BUDGET == {"max_street_depth": 2, "max_response_depth": 1,
                              "max_candidate_actions": 6, "max_leaves": 200,
                              "max_runout_children": 8}


def test_t7_extra_pipeline_keys_do_not_disturb_loop():
    """额外键必须被 loop 忽略（不得导致失败）。"""
    d = _real()
    st, rm, ap, bel, lk = d["setup"]("flop")
    prep = d["prepare"](st, rm, ap, bel, "DECLARED_BALANCED")
    pip = dict(d["pipeline"])
    pip["some_unrelated_key"] = lambda *a, **k: {}
    pip["opponent_conditioned_search"] = {"present": True}
    pip["p9_conditioning_status"] = "conditioned"
    r = _run(st, lk, pip, prep, d["run_decision_loop"], d["evaluate_search"])
    assert r["present"] is True
    assert r["loop_out"]["present"] is True
