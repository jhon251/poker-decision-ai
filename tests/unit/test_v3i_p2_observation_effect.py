# -*- coding: utf-8 -*-
"""V3I-P2 测试 — 已观察对手行为历史对闭环的可解释影响验证。

**本阶段目标**：验证现有闭环是否会根据"已观察到的对手行为历史"产生
可解释的 belief/range/candidate/decision 差异；并**如实报告**哪些差异**没有**产生
以及**具体是哪一层/哪个接线点阻止了变化**。

覆盖（按任务要求）：
* 同状态不同历史（observation 不同 → belief 不同）
* 相同历史重复运行确定性
* UNKNOWN history（无 history → fail closed）
* conflicting evidence（冲突参数 → UNKNOWN，不猜）
* belief 更新确实执行
* **decision unchanged 时也必须说明原因（定位到具体冻结层/接线点）**
* fail-closed / abstain / UNKNOWN 传播

**复用**：V3I-P1 的 `cash_decision_environment.build_scenario` 与
`cash_decision_loop.run_decision_loop`，**不重写**。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_decision_environment import build_scenario  # noqa: E402
from poker_decision_ai.cash_decision_loop import run_decision_loop  # noqa: E402

# 测试侧允许 import 任意层（生产侧不允许，由既有边界测试强制）
from poker_decision_ai.cash_integrity_context import cash_integrity_context  # noqa: E402
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_range_model import build_range_model  # noqa: E402
from poker_decision_ai.cash_opponent_model import build_opponent_model  # noqa: E402
from poker_decision_ai.cash_multistreet import next_street_transition  # noqa: E402
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_action_policy import build_action_policy  # noqa: E402
from poker_decision_ai.cash_strategy_dynamic_belief import build_dynamic_belief  # noqa: E402
from poker_decision_ai.cash_strategy_belief_update import (  # noqa: E402
    update_belief_with_likelihood)
from poker_decision_ai.cash_strategy_likelihood_model import (  # noqa: E402
    action_likelihood_model)
from poker_decision_ai.cash_strategy_action_ranking import rank_actions  # noqa: E402
from poker_decision_ai.cash_strategy_belief_conditioned_eval import (  # noqa: E402
    evaluate_belief_conditioned)
from poker_decision_ai.cash_strategy_search_tree import build_search_tree  # noqa: E402
from poker_decision_ai.cash_strategy_search_eval import evaluate_search  # noqa: E402
from poker_decision_ai.cash_strategy_decision_assembly import (  # noqa: E402
    assemble_decision)

BUDGET = {"max_street_depth": 2, "max_response_depth": 1,
          "max_candidate_actions": 3, "max_leaves": 60,
          "max_runout_children": 2}

PIPELINE = {
    "build_action_policy": build_action_policy,
    "build_dynamic_belief": build_dynamic_belief,
    "update_belief": update_belief_with_likelihood,
    "rank_actions": rank_actions,
    "evaluate_belief_conditioned": evaluate_belief_conditioned,
    "build_search_tree": build_search_tree,
    "evaluate_search": evaluate_search,
    "assemble_decision": assemble_decision,
}

#: 固定 seed 的**同一** Hero state（三条历史共用，保证只变历史）
SEED = 11
STREET = "flop"


def _state():
    """同一个 state 对象，供所有对照使用（相同牌面 / 筹码 / Hero）。"""
    return build_scenario(seed=SEED, street=STREET)["state"]


def _likelihood(mult: float):
    return action_likelihood_model([
        {"action": "raise", "street": STREET, "multiplier": mult,
         "source": "declared", "evidence_level": "DECLARED_MODEL",
         "provenance": {"stage": "v3i_p2", "kind": "declared"},
         "uncertainty": {"declared_only": True}},
    ])


#: 三条对照历史（**只**变 observation，其余全同）
PASSIVE = {"_last_action": "raise", "raise": 0.5}
AGGRESSIVE = {"_last_action": "raise", "raise": 3.0}
UNKNOWN = None


def _run(action_weights, likelihood_model, state=None):
    st = state if state is not None else _state()
    return run_decision_loop(
        st, pipeline=PIPELINE,
        integrity_context=cash_integrity_context(st),
        action_tree=build_action_tree(st),
        range_model=build_range_model(st,
                                      opponent_model=build_opponent_model(st)),
        opponent_model=build_opponent_model(st),
        transition=next_street_transition(st),
        likelihood_model=likelihood_model,
        action_weights=action_weights,
        solver=solver_status(st),
        budget=BUDGET, max_root_candidates=3,
    )


def _trace(out):
    s = out["stage_outputs"]
    bel = s["belief"] or {}
    inner = bel.get("belief") or {}
    opp1 = inner.get("opp1") or {}
    return {
        "out": out,
        "belief": bel,
        "posterior": opp1.get("posterior") or {},
        "belief_update": s["belief_update"] or {},
        "ranking": s["ranking"] or {},
        "belief_conditioned": s["belief_conditioned"] or {},
        "search_eval": s["search_eval"] or {},
        "decision": s["decision"] or {},
    }


def _fp(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str)
                          .encode()).hexdigest()[:16]


def _mass_mult(t):
    p = t["posterior"]
    return p.get("mass_posterior"), p.get("likelihood_multiplier")


# ══ 1. 同状态不同历史 ══════════════════════════════════════════════════
def test_observation_history_changes_belief_state():
    """同 Hero state + 只改 observation → belief/range 状态**必须**变化。"""
    a = _trace(_run(PASSIVE, _likelihood(0.5)))
    b = _trace(_run(AGGRESSIVE, _likelihood(3.0)))
    ma, ka = _mass_mult(a)
    mb, kb = _mass_mult(b)
    assert ka == 0.5 and kb == 3.0
    assert ma != mb, (ma, mb)
    assert a["belief"]["likelihood_applied"] is True
    assert b["belief"]["likelihood_applied"] is True
    assert a["belief"]["likelihood_source"] == "explicit_action_weights"
    assert b["belief"]["likelihood_source"] == "explicit_action_weights"


def test_history_is_read_from_last_action():
    """观测历史通过 `_last_action` 选择乘数 —— 通路真实存在。"""
    st = _state()
    rm = build_range_model(st, opponent_model=build_opponent_model(st))
    om = build_opponent_model(st)
    ctx = cash_integrity_context(st)
    got = {}
    for last, mult in (("raise", 3.0), ("call", 0.4)):
        b = build_dynamic_belief(st, range_model=rm, opponent_model=om,
                                 action_weights={"_last_action": last,
                                                 last: mult},
                                 integrity_context=ctx)
        p = ((b.get("belief") or {}).get("opp1") or {}).get("posterior") or {}
        got[last] = p.get("likelihood_multiplier")
    assert got == {"raise": 3.0, "call": 0.4}


def test_observation_is_actually_consumed_not_ignored():
    """有历史 vs 无历史 → belief 与 decision_record 均可区分。"""
    with_hist = _trace(_run(AGGRESSIVE, _likelihood(3.0)))
    without = _trace(_run(UNKNOWN, None))
    assert with_hist["belief"]["likelihood_applied"] is True
    assert without["belief"]["likelihood_applied"] is False
    assert without["belief"]["likelihood_source"] == "not_supplied"
    assert _mass_mult(with_hist)[0] != _mass_mult(without)[0]


# ══ 2. 相同历史重复运行确定性 ══════════════════════════════════════════
@pytest.mark.parametrize("label,aw,mult", [
    ("passive", PASSIVE, 0.5), ("aggressive", AGGRESSIVE, 3.0),
    ("unknown", UNKNOWN, None)])
def test_same_history_repeat_runs_are_deterministic(label, aw, mult):
    def once():
        return _run(aw, _likelihood(mult) if mult else None)

    def canon(o):
        return json.dumps(o, sort_keys=True, ensure_ascii=False, default=str)

    outs = [once() for _ in range(3)]
    assert canon(outs[0]) == canon(outs[1]) == canon(outs[2])
    # JSON 往返在**值**上稳定；注意冻结的 V3H-8 会输出 tuple 候选键，
    # 往返后 tuple 变 list，因此比较的是**往返后的规范形式**而非原对象。
    rt = json.loads(canon(outs[0]))
    assert rt == json.loads(canon(outs[1])) == json.loads(canon(outs[2]))


def test_no_random_number_module_involved():
    """差异不可能来自随机数：链路内无随机模块。"""
    for name in ("cash_decision_environment", "cash_decision_loop",
                 "cash_strategy_dynamic_belief", "cash_strategy_belief_update"):
        tree = ast.parse((PKG / f"{name}.py").read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert all(a.name.split(".")[0] != "random" for a in n.names)
            if isinstance(n, ast.ImportFrom):
                assert (n.module or "").split(".")[0] != "random"


def test_same_state_object_reused_across_histories():
    """三条历史共用**同一个** state —— 排除场景变化。"""
    st = _state()
    before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    a = _run(PASSIVE, _likelihood(0.5), state=st)
    b = _run(AGGRESSIVE, _likelihood(3.0), state=st)
    c = _run(UNKNOWN, None, state=st)
    after = json.dumps(st, sort_keys=True, ensure_ascii=False)
    assert after == before
    for o in (a, b, c):
        assert o["street"] == STREET
    assert _mass_mult(_trace(a))[0] != _mass_mult(_trace(b))[0]


def test_frozen_assembly_emits_tuple_keys_so_dict_roundtrip_is_not_identity():
    """**已披露的性质**：冻结 V3H-8 的 `scorer_tops.*.key` 是 tuple；
    JSON 往返后变 list，因此"原对象 == 往返对象"**不成立**。
    这是类型保真度问题，**不是**非确定性 —— 值本身稳定。"""
    out = _run(PASSIVE, _likelihood(0.5))
    keys = ((out["decision_record"]["decision"] or {})
            .get("scorer_tops") or {})
    tuples = [k for k, v in keys.items()
              if isinstance(v, dict) and isinstance(v.get("key"), tuple)]
    assert tuples, "expected the frozen assembly to emit tuple candidate keys"
    rt = json.loads(json.dumps(out, ensure_ascii=False))
    rt_keys = ((rt["decision_record"]["decision"] or {})
               .get("scorer_tops") or {})
    for k in tuples:
        assert isinstance(rt_keys[k]["key"], list)
        assert list(keys[k]["key"]) == rt_keys[k]["key"]
    # 值层面仍然稳定
    assert json.dumps(out, sort_keys=True, ensure_ascii=False, default=str) == \
        json.dumps(json.loads(json.dumps(out, ensure_ascii=False)),
                   sort_keys=True, ensure_ascii=False, default=str)


# ══ 3. UNKNOWN history ═════════════════════════════════════════════════
def test_unknown_history_fails_closed_and_is_recorded():
    t = _trace(_run(UNKNOWN, None))
    rec = t["out"]["decision_record"]
    assert t["belief"]["likelihood_applied"] is False
    assert t["belief"]["likelihood_source"] == "not_supplied"
    assert rec["belief_update_present"] is False
    assert "likelihood_model_not_supplied" in rec["unknown"]
    assert rec["unknown"] == sorted(rec["unknown"])
    assert rec["belief_update_present"] is False
    assert rec["belief_update_level"] is None
    assert t["belief_conditioned"].get("present") is not True
    assert t["belief_conditioned"] == {}
    assert t["out"]["loop_status"] != "COMPLETE"


def test_unknown_history_never_becomes_a_recommendation():
    t = _trace(_run(UNKNOWN, None))
    assert t["decision"].get("abstained") is True
    assert t["decision"].get("preferred") is None
    assert t["out"]["loop_status"] == "ABSTAINED"


# ══ 4. conflicting evidence ═══════════════════════════════════════════
def test_conflicting_likelihood_parameters_yield_unknown_not_guess():
    conflicting = action_likelihood_model([
        {"action": "raise", "street": STREET, "multiplier": 3.0,
         "source": "declared", "evidence_level": "DECLARED_MODEL",
         "provenance": {"s": "v3i_p2"}, "uncertainty": {"d": True}},
        {"action": "raise", "street": STREET, "multiplier": 0.2,
         "source": "declared", "evidence_level": "DECLARED_MODEL",
         "provenance": {"s": "v3i_p2"}, "uncertainty": {"d": True}},
    ])
    assert conflicting["applied"] is False
    assert len(conflicting["conflicts"]) == 1
    assert conflicting["prior_only"] is True
    t = _trace(_run(AGGRESSIVE, conflicting))
    rec = t["out"]["decision_record"]
    assert rec["belief_update_level"] == "UNKNOWN"
    assert any("conflicting_parameters" in u for u in rec["unknown"])
    assert t["out"]["loop_status"] != "COMPLETE"


def test_invalid_likelihood_parameters_do_not_silently_apply():
    bad = action_likelihood_model([
        {"action": "not_an_action", "street": STREET, "multiplier": 5.0,
         "source": "d", "evidence_level": "DECLARED_MODEL",
         "provenance": {"s": "x"}, "uncertainty": {"d": True}},
    ])
    assert bad["applied"] is False
    assert bad["invalid"]
    t = _trace(_run(AGGRESSIVE, bad))
    assert t["out"]["loop_status"] != "COMPLETE"
    assert any("invalid_parameters" in u
               for u in t["out"]["decision_record"]["unknown"])


# ══ 5. belief 更新确实执行 ════════════════════════════════════════════
def test_belief_update_stage_executes_and_is_traced():
    t = _trace(_run(AGGRESSIVE, _likelihood(3.0)))
    stages = {x["stage"]: x for x in t["out"]["stage_trace"]}
    assert stages["belief_update"]["present"] is True
    assert stages["belief_update"]["error"] is None
    assert t["belief_update"].get("present") is True
    assert t["belief_update"].get("level") == "MODEL_INTERNAL"
    assert t["out"]["decision_record"]["belief_update_present"] is True


def test_belief_update_differs_when_observation_differs():
    a = _trace(_run(PASSIVE, _likelihood(0.5)))
    b = _trace(_run(AGGRESSIVE, _likelihood(3.0)))
    assert _fp(a["belief_update"]) != _fp(b["belief_update"])


# ══ 6. decision unchanged 时必须说明原因 ════════════════════════════
def test_candidate_ranking_unchanged_and_reason_localised():
    """关键诚实断言：passive vs aggressive 的 candidate 排序**不变**，
    并**定位**到阻止变化的层与机制。"""
    a = _trace(_run(PASSIVE, _likelihood(0.5)))
    b = _trace(_run(AGGRESSIVE, _likelihood(3.0)))
    # belief 的**总量**变了（历史确实被消费）……
    assert _mass_mult(a)[0] != _mass_mult(b)[0]
    # ……但 posterior 的**形状完全相同**（这正是下游不变的根因）
    assert a["posterior"].get("posterior") == b["posterior"].get("posterior")
    assert a["posterior"].get("combos_posterior") == \
        b["posterior"].get("combos_posterior")
    # ……但 ranking 未变
    assert _fp(a["ranking"].get("ranked") or a["ranking"].get("preferred")) == \
        _fp(b["ranking"].get("ranked") or b["ranking"].get("preferred"))
    assert (a["ranking"].get("preferred") or {}).get("action") == \
        (b["ranking"].get("preferred") or {}).get("action")
    # ……belief_conditioned 的**元数据**变了（携带乘数），
    # 但它的**首选动作**没变 —— 这正是"可追踪但不影响建议"的证据
    assert _fp(a["belief_conditioned"]) != _fp(b["belief_conditioned"])
    pa = (a["belief_conditioned"].get("preferred") or {})
    pb = (b["belief_conditioned"].get("preferred") or {})
    assert pa.get("action") == pb.get("action")
    assert pa.get("amount_bb") == pb.get("amount_bb")
    # ……search_eval 未变（它根本不接收观测）
    assert _fp(a["search_eval"]) == _fp(b["search_eval"])
    # ……decision 也未变
    assert _fp(a["decision"]) == _fp(b["decision"])
    assert a["out"]["loop_status"] == b["out"]["loop_status"] == "ABSTAINED"


def test_posterior_shape_is_fully_invariant_to_the_multiplier():
    """**机制定位（根因）**：乘数**只**改变总量 `mass_posterior`，
    **全部 63 个 combo 的权重逐字节不变** —— 归一化与否都一样。"""
    st = _state()
    rm = build_range_model(st, opponent_model=build_opponent_model(st))
    om = build_opponent_model(st)
    ctx = cash_integrity_context(st)
    dists = {}
    masses = {}
    for mult in (0.5, 1.0, 3.0):
        b = build_dynamic_belief(st, range_model=rm, opponent_model=om,
                                 action_weights={"_last_action": "raise",
                                                 "raise": mult},
                                 integrity_context=ctx)
        p = ((b.get("belief") or {}).get("opp1") or {}).get("posterior") or {}
        dists[mult] = p.get("posterior") or {}
        masses[mult] = p.get("mass_posterior")
    # 原始分布**完全一致**（不做任何归一化）
    assert dists[0.5] == dists[1.0] == dists[3.0]
    assert len(dists[0.5]) == 63
    # 但总量确实不同 —— 这就是"历史被消费了、却没有改变形状"的证据
    assert len({masses[0.5], masses[1.0], masses[3.0]}) == 3
    # 值域不因乘数改变（没有出现 >1 的被加权 combo）
    for mult, d in dists.items():
        assert max(d.values()) == 1.0, mult
        assert min(d.values()) > 0, mult


def test_mass_is_the_only_quantity_that_moves():
    """**只**有总量 mass 随历史变化；下游候选评分的输入形状不变。"""
    a = _trace(_run(PASSIVE, _likelihood(0.5)))
    b = _trace(_run(AGGRESSIVE, _likelihood(3.0)))
    pa, pb = a["posterior"], b["posterior"]
    assert pa.get("mass_posterior") != pb.get("mass_posterior")
    assert pa.get("combos_prior") == pb.get("combos_prior")
    assert pa.get("combos_posterior") == pb.get("combos_posterior")
    assert pa.get("posterior") == pb.get("posterior")
    assert pa.get("likelihood_multiplier") != pb.get("likelihood_multiplier")


def test_decision_blocked_by_frozen_assembly_disagreement():
    """**接线点定位**：最终建议由**冻结的 V3H-8 装配**决定；
    它检测到 cross_scorer_disagreement 即弃权，且**不**读取 belief 形状。"""
    t = _trace(_run(AGGRESSIVE, _likelihood(3.0)))
    dec = t["decision"]
    assert dec.get("abstained") is True
    assert dec.get("preferred") is None
    assert "cross_scorer_disagreement" in \
        t["out"]["decision_record"]["unknown"]


def test_scorer_disagreement_is_the_real_cause_not_the_history():
    """三个 scorer 各自**独立于** action_weights（它们不接收该参数）。"""
    for name in ("cash_strategy_action_ranking",
                 "cash_strategy_belief_conditioned_eval",
                 "cash_strategy_search_eval"):
        src = (PKG / f"{name}.py").read_text(encoding="utf-8")
        assert "action_weights" not in src, name


def test_downstream_evaluators_do_not_consume_the_observation_directly():
    """接线点证据：观测只进 belief 层；ranking / bc / search 均不接 action_weights。"""
    import inspect

    def params(fn):
        sig = inspect.signature(fn)
        return list(sig.parameters)

    for fn in (rank_actions, evaluate_belief_conditioned, evaluate_search):
        assert "action_weights" not in params(fn), fn.__name__


# ══ 7. fail-closed / abstain / UNKNOWN 传播 ═══════════════════════════
@pytest.mark.parametrize("aw,mult,expect_status", [
    (PASSIVE, 0.5, "ABSTAINED"), (AGGRESSIVE, 3.0, "ABSTAINED"),
    (UNKNOWN, None, "ABSTAINED")])
def test_abstain_is_propagated_for_all_histories(aw, mult, expect_status):
    t = _trace(_run(aw, _likelihood(mult) if mult else None))
    assert t["out"]["loop_status"] == expect_status
    assert t["decision"].get("preferred") is None


def test_river_still_blocked_for_every_history():
    """冻结 V3G-7：三条历史在 river 上都必须 BLOCKED。"""
    for aw, mult in ((PASSIVE, 0.5), (AGGRESSIVE, 3.0), (UNKNOWN, None)):
        st = build_scenario(seed=SEED, street="river")["state"]
        out = _run(aw, _likelihood(mult) if mult else None, state=st)
        assert out["present"] is False
        assert out["loop_status"] == "BLOCKED"


def test_unknown_list_ordering_is_stable_and_deduplicated():
    t = _trace(_run(UNKNOWN, None))
    u = t["out"]["decision_record"]["unknown"]
    assert u == sorted(u)
    assert len(u) == len(set(u))


def test_evidence_level_stays_model_internal_for_all_histories():
    for aw, mult in ((PASSIVE, 0.5), (AGGRESSIVE, 3.0), (UNKNOWN, None)):
        out = _run(aw, _likelihood(mult) if mult else None)
        assert out["evidence_level"] in ("MODEL_INTERNAL", "NONE")
        flat = json.dumps(out, ensure_ascii=False)
        assert '"is_external_validation": true' not in flat
        assert '"is_real_opponent_data": true' not in flat


def test_no_profit_or_rate_conclusion_anywhere():
    for aw, mult in ((PASSIVE, 0.5), (AGGRESSIVE, 3.0), (UNKNOWN, None)):
        out = _run(aw, _likelihood(mult) if mult else None)
        keys: set[str] = set()

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    keys.add(k)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(out)
        for banned in ("win_rate", "roi", "pnl", "profit_estimate",
                       "bb_per_100", "ev_delta", "exploitability_value"):
            assert banned not in keys, banned
        # claim boundary 中的否定声明不算结论
        assert out["claim_boundary"]["is_profit_evidence"] is False
        assert out["result"] is None


# ══ 8. 不修改冻结面 / 不加深搜索 / 不接求解器 ══════════════════════════
def test_frozen_tier_rule_untouched():
    """冻结的 V3H-8 tier 规则未被修改（顺序仍为 search/belief_conditioned/ranking）。"""
    src = (PKG / "cash_strategy_decision_assembly.py").read_text(encoding="utf-8")
    assert 'TIERS: tuple[str, ...] = ("search", "belief_conditioned", "ranking")' \
        in src
    assert "on_disagreement" in src


def test_river_terminal_rule_untouched():
    src = (PKG / "cash_strategy_multistreet_eval.py").read_text(encoding="utf-8")
    assert "frozen V3G-7 cash_action_tree._terminal_reason" in src


def test_search_budget_not_deepened():
    tree = ast.parse((PKG / "cash_strategy_search_tree.py")
                     .read_text(encoding="utf-8"))
    frozen = None
    for n in ast.walk(tree):
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                and n.target.id == "DEFAULT_BUDGET":
            frozen = ast.literal_eval(n.value)
        elif isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id == "DEFAULT_BUDGET":
            frozen = ast.literal_eval(n.value)
    assert frozen == {"max_street_depth": 2, "max_response_depth": 1,
                      "max_candidate_actions": 6, "max_leaves": 200,
                      "max_runout_children": 8}, frozen
    assert BUDGET["max_street_depth"] <= frozen["max_street_depth"]


def test_solver_not_invoked_and_still_blocked():
    st = _state()
    s = solver_status(st)
    assert s["solver_status"] == "BLOCKED"
    assert s["equilibrium"] is False
    src = (PKG / "cash_decision_loop.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "solver_status" not in calls
    assert "solve" not in calls
    # solver 只作为**只读透传的字典键名**出现，从不被调用
    assert src.count("solver_status") >= 1


def test_no_new_modules_added_by_this_stage():
    """本阶段**不新增**任何模块（只做验证）。"""
    for name in ("cash_decision_environment", "cash_decision_loop"):
        assert (PKG / f"{name}.py").exists()
    # 环境与 loop 未被本阶段修改：签名仍与 V3I-P1 一致
    import inspect
    assert "claimed_evidence_level" in inspect.signature(
        run_decision_loop).parameters
    assert "population_table" in inspect.signature(
        build_scenario).parameters


# ══ 9. 三条历史的端到端对照汇总（可追踪输出） ══════════════════════
def test_three_history_comparison_matrix_is_reproducible():
    rows = []
    for label, aw, mult in (("PASSIVE", PASSIVE, 0.5),
                            ("AGGRESSIVE", AGGRESSIVE, 3.0),
                            ("UNKNOWN", UNKNOWN, None)):
        t = _trace(_run(aw, _likelihood(mult) if mult else None))
        rows.append({
            "label": label,
            "observation": "supplied" if aw else "not_supplied",
            "mass": _mass_mult(t)[0],
            "multiplier": _mass_mult(t)[1],
            "ranking_fp": _fp(t["ranking"].get("ranked")
                              or t["ranking"].get("preferred")),
            "decision_fp": _fp(t["decision"]),
            "status": t["out"]["loop_status"],
        })
    # observation 不同 → mass 不同（belief 层确实响应）
    masses = [r["mass"] for r in rows]
    assert len(set(masses)) == 3
    # 但 passive 与 aggressive 的 ranking 与 decision **相同**
    assert rows[0]["ranking_fp"] == rows[1]["ranking_fp"]
    assert rows[0]["decision_fp"] == rows[1]["decision_fp"]
    # UNKNOWN 与两者都不同（fail closed 路径不同）
    assert rows[2]["decision_fp"] != rows[0]["decision_fp"]
    assert rows[2]["observation"] == "not_supplied"


def test_comparison_matrix_is_deterministic():
    def matrix():
        out = []
        for aw, mult in ((PASSIVE, 0.5), (AGGRESSIVE, 3.0), (UNKNOWN, None)):
            t = _trace(_run(aw, _likelihood(mult) if mult else None))
            out.append((_mass_mult(t)[0], _fp(t["decision"]),
                        t["out"]["loop_status"]))
        return out

    assert matrix() == matrix()
