# -*- coding: utf-8 -*-
"""V3I-G1 Phase 1 — Action-Sensitive Search 测试（黑盒 + 反事实 + 阴性对照）。

覆盖：D1–D7 / C0–C7 中**适用于 Phase 1** 的部分，以及**强制阴性对照**
（恒定 equity、fold-only 污染、随机打乱 equity）。

纪律（授权要求 11）
------------------
* 本文件**不调用**被测 value aggregation 实现；动作价值一律经**公开入口**
  `evaluate_search(..., action_value_input=...)` /
  `evaluate_search_with_action_value(...)` 取得（见末条自检测试）；
* 反事实/阴性对照只通过**构造输入**实现，不改实现；
* 探测器（`detect_degeneracy`）是**检查器**，可被测试直接调用。

环境约束：本沙箱禁止 `%TEMP%` 写入，`tmp_path`/`ctx`/`runtime` fixture 会抛
`PermissionError [WinError 5]`，故本文件不使用需要临时目录的 fixture。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.cash_action_value_contract import (  # noqa: E402
    CLAIM_BOUNDARY as AV_CLAIM_BOUNDARY, CONTRACT as AV_CONTRACT,
    basis_fingerprint, candidate_set_scope, contract as av_contract,
    detect_degeneracy)
from poker_decision_ai.cash_conditioned_search_wiring import (  # noqa: E402
    action_value_input_from_bridge, evaluate_search_with_action_value,
    conditioned_evaluation_input, materialize_conditioned_tree)
from poker_decision_ai.cash_strategy_search_eval import (  # noqa: E402
    SEARCH_SCORE_MODEL)
from poker_decision_ai.cash_strategy_search_tree import (  # noqa: E402
    DEFAULT_BUDGET, build_search_tree, _walk_nodes)
from poker_decision_ai.cash_multistreet import next_street_transition  # noqa: E402
from poker_decision_ai.cash_strategy_action_ranking import rank_actions  # noqa: E402
from poker_decision_ai.cash_strategy_decision_assembly import (  # noqa: E402
    ASSEMBLY_POLICY, TIERS, assemble_decision)
from poker_decision_ai.cash_feasible_domain_wiring import (  # noqa: E402
    feasible_keys_from_search_tree)

#: 黑盒纪律自检用的"禁止调用名"（拼接构造，避免本文件自身包含该字面量）
_FORBIDDEN = ("aggregate" + "_action_values",
              "candidate_action_value" + "(")

BIG_BUDGET = {**DEFAULT_BUDGET, "max_leaves": 2000}

#: v3（SCORE-HYGIENE-1）legacy golden —— 必须逐位保持
LEGACY_SCORES = {"flop": [0.767877, 0.944676, 1.035754],
                 "turn": [0.880387, 1.131442, 1.260774]}

_M: dict = {}


def _p8():
    if "p8" not in _M:
        spec = importlib.util.spec_from_file_location(
            "_p8_mod",
            ROOT / "tests" / "unit" / "test_v3i_p8_minimal_contract_fix.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules["_p8_mod"] = mod
        spec.loader.exec_module(mod)
        _M["p8"] = mod
    return _M["p8"]


def _base(street: str) -> dict:
    key = f"base::{street}"
    if key not in _M:
        d = _p8()._real()
        st, rm, ap, bel, tree_default = d["setup"](street)
        _M[key] = {"d": d, "st": st, "rm": rm, "ap": ap, "bel": bel,
                   "tree_default": tree_default}
    return _M[key]


def _tree(street: str, cap: int, max_leaves: int = 2000) -> dict:
    key = f"tree::{street}::{cap}::{max_leaves}"
    if key not in _M:
        b = _base(street)
        _M[key] = build_search_tree(
            b["st"], action_policy=b["ap"], belief=b["bel"], belief_update=None,
            transition=next_street_transition(b["st"]),
            budget={**BIG_BUDGET, "max_leaves": max_leaves},
            max_root_candidates=cap)
    return _M[key]


def _av_input(street: str, population: str) -> dict:
    key = f"av::{street}::{population}"
    if key not in _M:
        b = _base(street)
        br = _p8()._bridge_for(b["st"], b["rm"], b["ap"], b["bel"], population)
        _M[key] = action_value_input_from_bridge(bridge_out=br)
    return _M[key]


def _ev(street: str, cap: int, population: str = "DECLARED_BALANCED",
        with_av: bool = True, tree: dict | None = None) -> dict:
    """公开入口（黑盒）。`tree=None` → 大预算树（可计价）；否则用给定树。"""
    cacheable = with_av and tree is None
    key = f"ev::{street}::{cap}::{population}::{with_av}"
    if cacheable and key in _M:
        return _M[key]
    b = _base(street)
    kwargs = {}
    if with_av:
        kwargs["action_value_input"] = _av_input(street, population)
    out = b["d"]["evaluate_search"](
        b["st"], search_tree=tree if tree is not None else _tree(street, cap),
        solver=None, **kwargs)
    if cacheable:
        _M[key] = out
    return out


def _prov(ev: dict) -> dict:
    return {f"{c['action']}|{c['amount_bb']}": (c["action_value_provenance"] or {})
            for c in ev["candidates"]}


def _records(ev: dict, *, pop: str = "VARIANT", eq_of=None, w_of=None,
             drop_eq_for=None) -> list:
    """按 artifact 复原/改写记录；**保留真实 threshold/类别数**（使 L1/L2 有意义）。"""
    prov = _prov(ev)
    rows = {r["key"]: r for r in ev["ranked_by_action_value"]}
    out = []
    for key, p in prov.items():
        r = rows.get(key)
        if r is None:
            continue
        eq = (eq_of(r) if callable(eq_of) else r["equity_vs_continuing"])
        p_c, p_f = (w_of(r) if callable(w_of)
                    else (r["p_continue"], r["p_fold"]))
        rec = {"action": r["action"], "amount_bb": r["amount_bb"],
               "equity_vs_continuing": eq, "p_continue": p_c, "p_fold": p_f,
               "threshold": (p.get("threshold") if p.get("threshold") is not None
                             else 0.0),
               "n_continuation_classes": p.get("n_continuation_classes") or 1,
               "population": pop,
               "equity_basis": "equity_vs_continuing_measured"}
        if drop_eq_for is not None and key == drop_eq_for:
            rec.pop("equity_vs_continuing")
        rec["basis_fingerprint"] = basis_fingerprint(rec)
        out.append(rec)
    return out


def _ev_with(street: str, cap: int, records: list) -> dict:
    b = _base(street)
    return b["d"]["evaluate_search"](
        b["st"], search_tree=_tree(street, cap), solver=None,
        action_value_input={"present": True, "source": "test_variant",
                            "population": "VARIANT", "records": records})


def _values(ev: dict) -> dict:
    return {r["key"]: r["value"] for r in ev["ranked_by_action_value"]}


def _argmax(ev: dict):
    ranked = ev["ranked_by_action_value"]
    return ranked[0]["key"] if ranked else None


# ══ 1. legacy 口径保留（要求 8）════════════════════════════════════════
@pytest.mark.parametrize("street", ["flop", "turn"])
def test_legacy_calibre_preserved_and_dual_reported(street):
    ev = _ev(street, 3, with_av=False)
    assert [c["search_score"] for c in ev["candidates"]] == LEGACY_SCORES[street]
    assert ev["legacy_search_score_preserved"] is True
    assert ev["dual_report"]["legacy_field"] == "search_score"
    assert ev["dual_report"]["legacy_model_unchanged"] is True
    assert ev["dual_report"]["legacy_model"]["name"] == "search_runout_pot_share_v1"
    assert ev["dual_report"]["action_value_model"]["name"] == \
        "action_sensitive_net_pot_share_v1"
    assert ev["action_value_status"] == "UNKNOWN"
    assert ev["action_value_reason"] == "response_weights_not_supplied"
    assert ev["preferred_by_action_value"] is None
    assert all(c["search_score"] is not None for c in ev["candidates"])
    ev2 = _ev(street, 3)
    assert [c["search_score"] for c in ev2["candidates"]] == LEGACY_SCORES[street]


def test_search_score_model_identity_unchanged():
    assert SEARCH_SCORE_MODEL["name"] == "search_runout_pot_share_v1"
    assert SEARCH_SCORE_MODEL["unit"] == "pot_share"
    assert SEARCH_SCORE_MODEL["input_policy"] == "real_equity_leaves_only"
    assert SEARCH_SCORE_MODEL["free_parameters"] == 0
    assert SEARCH_SCORE_MODEL["uses_opponent_fold_equity"] is False


# ══ 2. D1 ═════════════════════════════════════════════════════════════
def test_d1_shared_single_equity_cannot_reconstruct_action_value():
    ev = _ev("flop", 6)
    assert ev["action_value_status"] == "OK"
    diff = ev["action_value"]["differentiation"]
    assert diff["n_distinct_equity_inputs"] == 6
    assert diff["source"] == "equity_vs_continuing+fold_continue_weights"

    tree = _tree("flop", 6)
    pot = float(tree["root"]["pot_bb"])
    shared = None
    for c in ev["candidates"]:
        if c["future_equity_mean_real_equity_only"] is not None:
            shared = c["future_equity_mean_real_equity_only"]
            break
    assert shared is not None
    for c in ev["candidates"]:                      # legacy 可被单一共享 equity 重建
        inv = 0.0 if str(c["action"]) == "check" else float(c["amount_bb"] or 0.0)
        pred = shared * (pot + 2.0 * inv) / pot - inv / pot
        assert abs(pred - float(c["search_score"])) < 1e-05

    ev_shared = _ev_with("flop", 6, _records(ev, pop="SHARED_INPUT",
                                            eq_of=lambda r: shared))
    assert ev_shared["action_value_status"] == "OK"
    real, sh = _values(ev), _values(ev_shared)
    deltas = {k: abs(real[k] - sh[k]) for k in real}
    assert max(deltas.values()) > 0.5, deltas


# ══ 3. D2 ═════════════════════════════════════════════════════════════
def test_d2_branch_weights_measured_and_enter_value():
    ev = _ev("flop", 6)
    ladder = sorted([c for c in ev["candidates"]
                     if float(c["amount_bb"] or 0.0) > 0],
                    key=lambda c: float(c["amount_bb"]))
    pf, eq = [], []
    for c in ladder:
        assert c["action_value_status"] == "OK"
        t = c["action_value_terms"]
        assert t["terminal_values_distinct"] == 2
        assert 0.0 < t["fold_probability"] < 1.0
        assert 0.0 < t["showdown_probability"] < 1.0
        assert t["fold_term"] > 0.0            # 旧口径完全没有弃牌项
        pf.append(t["fold_probability"])
        eq.append(c["action_value_measured"]["equity_vs_continuing"])
    assert pf == sorted(pf) and len(set(pf)) == len(pf)
    assert eq == sorted(eq, reverse=True) and len(set(eq)) == len(eq)
    check = [c for c in ev["candidates"] if str(c["action"]) == "check"][0]
    assert check["action_value_terms"]["fold_applicable"] is False
    assert check["action_value_terms"]["fold_term"] == 0.0


# ══ 4. D3 / C3 ════════════════════════════════════════════════════════
def test_d3_interior_optimum_and_legacy_ordering_reversal():
    ev = _ev("flop", 6)
    av_vals = _values(ev)
    av_order = [r["key"] for r in ev["ranked_by_action_value"]]
    legacy_order = [f"{c['action']}|{c['amount_bb']}" for c in ev["ranked"]]
    assert legacy_order[0] == "bet|100.0"      # legacy：单调假象 ⇒ all-in 第一
    assert av_order[0] == "bet|0.75"           # 动作价值：内部最优
    assert av_order[-1] == "bet|100.0"         # all-in 掉到最后
    assert av_vals["bet|100.0"] < av_vals["check|0.0"] - 0.2
    assert float([c for c in ev["candidates"]
                  if c["amount_bb"] == 100.0][0]["search_score"]) > 30.0
    ladder = sorted([c for c in ev["candidates"]
                     if float(c["amount_bb"] or 0.0) > 0],
                    key=lambda c: float(c["amount_bb"]))
    seq = [av_vals[f"{c['action']}|{c['amount_bb']}"] for c in ladder]
    assert any(b < a for a, b in zip(seq, seq[1:])), seq
    d = ev["action_value"]["differentiation"]
    assert d["n_distinct_values"] == 6 and d["spread"] > 0.39
    assert ev["detector"]["verdict"] == "CONSISTENT"


# ══ 5. D4 ═════════════════════════════════════════════════════════════
def test_d4_state_accounting_per_action_and_legacy_fields_intact():
    tree = _tree("flop", 6)
    pot = float(tree["root"]["pot_bb"])
    stack = float(tree["root"]["stack_bb"])
    pots = set()
    for n in tree["root"]["children"]:
        inv = float(n["hero_investment_bb"])
        assert abs(float(n["pot_after_hero_bb"]) - (pot + inv)) < 1e-09
        assert abs(float(n["stack_after_hero_bb"]) - (stack - inv)) < 1e-09
        assert float(n["pot_bb"]) == pot and float(n["stack_bb"]) == stack
        pots.add(float(n["pot_after_hero_bb"]))
    assert len(pots) == len(tree["root"]["children"])
    nodes = list(_walk_nodes(tree["root"]))
    assert {float(n["pot_bb"]) for n in nodes} == {pot}
    assert {float(n["stack_bb"]) for n in nodes} == {stack}
    for l in tree["leaves"]:
        a = float(l["hero_investment_bb"])
        assert l["admission_index"] >= 1
        if l["terminal_kind"] == "fold":
            assert abs(float(l["pot_at_terminal_bb"]) - (pot + a)) < 1e-09
            assert l["showdown_pot_bb"] is None
            assert l["terminal_value_net_pot_share"] == 1.0
            expect = ("not_applicable_no_investment" if a == 0.0
                      else "opponent_folds_to_aggression")
            assert l["terminal_semantics"] == expect
        else:
            assert abs(float(l["showdown_pot_bb"]) - (pot + 2.0 * a)) < 1e-09
            assert l["pot_at_terminal_bb"] == l["showdown_pot_bb"]
    assert tree["action_accounting"]["free_parameters"] == 0
    assert tree["action_accounting"]["weights_source"].startswith(
        "not_synthesised_in_tree")


# ══ 6. D5 / C4 ════════════════════════════════════════════════════════
def test_d5_action_value_population_sensitive_while_legacy_is_not():
    tight = _ev("turn", 3, "DECLARED_TIGHT_PASSIVE")
    bal = _ev("turn", 3, "DECLARED_BALANCED")
    aggr = _ev("turn", 3, "DECLARED_AGGRESSIVE")
    legacy = {tuple(sorted((c["action"], c["amount_bb"], c["search_score"])
                           for c in e["candidates"]))
              for e in (tight, bal, aggr)}
    assert len(legacy) == 1, "legacy search_score 不得随 population 变化"
    assert _argmax(tight) == "bet|0.495"
    assert _argmax(bal) == "bet|0.75"
    assert _argmax(aggr) == "bet|0.495"
    vt, vb, va = _values(tight), _values(bal), _values(aggr)
    assert vt["check|0.0"] != vb["check|0.0"] != va["check|0.0"]
    assert abs(vt["check|0.0"] - va["check|0.0"]) > 0.04


def test_c4_population_delta_decomposes_into_measured_channels():
    bal = _ev("turn", 3, "DECLARED_BALANCED")
    tight = _ev("turn", 3, "DECLARED_TIGHT_PASSIVE")
    for c_bal, c_tight in zip(bal["candidates"], tight["candidates"]):
        tb, tt = c_bal["action_value_terms"], c_tight["action_value_terms"]
        for t, c in ((tb, c_bal), (tt, c_tight)):
            assert abs((t["fold_term"] + t["showdown_term"])
                       - float(c["action_value_pot_share"])) < 1e-05
        dv = float(c_bal["action_value_pot_share"]) - \
            float(c_tight["action_value_pot_share"])
        dfold = tb["fold_term"] - tt["fold_term"]
        dshow = tb["showdown_term"] - tt["showdown_term"]
        assert abs(dv - (dfold + dshow)) < 1e-05


# ══ 7. D6 ═════════════════════════════════════════════════════════════
def test_d6_irrelevant_tree_field_does_not_change_value():
    base = _ev("flop", 6)
    tree = json.loads(json.dumps(_tree("flop", 6), default=str))
    tree["irrelevant_field"] = {"x": 1}
    b = _base("flop")
    again = b["d"]["evaluate_search"](
        b["st"], search_tree=tree, solver=None,
        action_value_input=_av_input("flop", "DECLARED_BALANCED"))
    assert _values(again) == _values(base)


def test_d6_zero_investment_ignores_weights_definitionally():
    ev = _ev("turn", 3)
    base_check = _values(ev)["check|0.0"]
    recs = _records(ev, pop="ZERO_INV_W",
                    w_of=lambda r: ((0.7, 0.3) if r["action"] == "check"
                                    else (r["p_continue"], r["p_fold"])))
    out = _ev_with("turn", 3, recs)
    assert out["action_value_status"] == "OK"
    assert _values(out)["check|0.0"] == base_check
    assert out["detector"]["flags"]["same_threshold_group_inconsistent"] is False


# ══ 8. D7 ═════════════════════════════════════════════════════════════
def test_d7_candidate_set_mismatch_reported_separately():
    ev3 = _ev("flop", 3)
    search_keys = [f"{c['action']}|{c['amount_bb']}" for c in ev3["candidates"]]
    b = _base("flop")
    ranking = rank_actions(b["st"], action_policy=b["ap"], belief=b["bel"])
    ranking_keys = [f"{r['action']}|{r['amount_bb']}" for r in ranking["ranked"]]
    # 候选集不一致：单独报告为**范围**问题，不计入证据 disagreement
    scope = candidate_set_scope(tiers={"search": search_keys,
                                       "ranking": ranking_keys})
    assert scope["candidate_set_mismatch"] is True
    assert scope["only_in_tier"]["ranking"], scope
    assert scope["counts_as_evidence_disagreement"] is False
    # 无 feasible_keys ⇒ 出现（截断假象导致的）disagreement（既有行为不变）
    dec_plain = assemble_decision(state=b["st"], ranking=ranking,
                                  belief_conditioned=None, search_eval=ev3)
    assert dec_plain["disagreements"], dec_plain
    assert dec_plain["abstained"] is True
    # 用**既有** feasible_keys 机制收敛到共同可行域 ⇒ 截断假象消失
    fk = feasible_keys_from_search_tree(search_tree=_tree("flop", 3))
    dec_feas = assemble_decision(state=b["st"], ranking=ranking,
                                 belief_conditioned=None, search_eval=ev3,
                                 feasible_keys=fk["feasible_keys"])
    assert dec_feas["disagreements"] == [], dec_feas["disagreements"]


#: Phase 1 时点：ranking **未被修改**（历史 pin，保留对照，不删除）
RANKING_HASH_PHASE1_UNTOUCHED = "2c1830e77157baaa"
#: Phase 2（v5，经授权）：统一动作价值 ⇒ ranking 当前值
RANKING_HASH_PHASE2_UNIFIED = "53dc7aa44245eb84"


def test_assembly_gate_tiers_override_unchanged_and_ranking_pin_migrated():
    assert float(ASSEMBLY_POLICY["min_leaf_coverage"]) == 0.8
    assert ASSEMBLY_POLICY["override_default"] is False
    assert ASSEMBLY_POLICY["on_disagreement"] == "abstain_unknown"
    assert TIERS == ("search", "belief_conditioned", "ranking")
    assert RANKING_HASH_PHASE1_UNTOUCHED != RANKING_HASH_PHASE2_UNIFIED
    for rel, want in (("poker_decision_ai/cash_strategy_action_ranking.py",
                       RANKING_HASH_PHASE2_UNIFIED),):
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]
        assert got == want, (rel, got, want)
    # Phase 2 未改动的受保护模块仍是 Phase 1（v4）值
    for rel, want in (
            ("poker_decision_ai/cash_strategy_search_tree.py",
             "5f5b29be063a6f5a"),
            ("poker_decision_ai/cash_strategy_decision_assembly.py",
             "3732314ab0182063")):
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]
        assert got == want, (rel, got, want)


# ══ 9. C0 ═════════════════════════════════════════════════════════════
@pytest.mark.parametrize("street", ["flop", "turn"])
def test_c0_empty_counterfactual_reproduces_legacy_exactly(street):
    ev = _ev(street, 3)
    recs = _records(ev, pop="C0_COUNTERFACTUAL", w_of=lambda r: (1.0, 0.0),
                    eq_of=lambda r: None)
    # eq 必须取候选自身的 legacy real mean（= 该树的口径），故单独构造
    real_means = {f"{c['action']}|{c['amount_bb']}":
                  c["future_equity_mean_real_equity_only"]
                  for c in ev["candidates"]}
    for rec in recs:
        key = f"{rec['action']}|{rec['amount_bb']}"
        rec["equity_vs_continuing"] = real_means[key]
        rec["basis_fingerprint"] = basis_fingerprint(rec)
    out = _ev_with(street, 3, recs)
    assert out["action_value_status"] == "OK"
    for c in out["candidates"]:
        assert c["action_value_pot_share"] == c["search_score"], \
            (street, c["action"], c["amount_bb"])


# ══ 10. C1 ════════════════════════════════════════════════════════════
def test_c1_constant_equity_negative_control_detected():
    ev = _ev("flop", 6)
    shared = ev["ranked_by_action_value"][0]["equity_vs_continuing"]
    out = _ev_with("flop", 6, _records(ev, pop="CONST_EQ",
                                       eq_of=lambda r: shared))
    assert out["action_value_status"] == "OK"
    flags = out["detector"]["flags"]
    assert flags["constant_equity_input"] is True
    assert flags["no_action_differentiation"] is False
    assert out["action_value"]["differentiation"]["source"] == \
        "fold_continue_weights"
    assert out["action_value"]["differentiation"]["n_distinct_values"] > 1


# ══ 11. C2 ════════════════════════════════════════════════════════════
def test_c2_weight_permutation_changes_ordering_not_equity():
    ev = _ev("flop", 6)
    a, bkey = "bet|0.495", "bet|100.0"
    rows = {r["key"]: r for r in ev["ranked_by_action_value"]}

    def w(r):
        if r["key"] == a:
            return rows[bkey]["p_continue"], rows[bkey]["p_fold"]
        if r["key"] == bkey:
            return rows[a]["p_continue"], rows[a]["p_fold"]
        return r["p_continue"], r["p_fold"]

    out = _ev_with("flop", 6, _records(ev, pop="SWAP", w_of=w))
    assert out["action_value_status"] == "OK"
    assert _argmax(out) == a, _values(out)
    out_rows = {r["key"]: r for r in out["ranked_by_action_value"]}
    for k in (a, bkey):
        assert out_rows[k]["equity_vs_continuing"] == \
            rows[k]["equity_vs_continuing"]
    assert _values(out) != _values(ev)


# ══ 12. C5 ════════════════════════════════════════════════════════════
def test_c3_size_ladder_channels_monotone_value_not():
    """C3：注额阶梯上两个已测通道单调，而**价值非单调**（内部最优）。"""
    ev = _ev("flop", 6)
    ladder = sorted([c for c in ev["candidates"]
                     if float(c["amount_bb"] or 0.0) > 0],
                    key=lambda c: float(c["amount_bb"]))
    pf = [c["action_value_terms"]["fold_probability"] for c in ladder]
    eq = [c["action_value_measured"]["equity_vs_continuing"] for c in ladder]
    nc = [c["action_value_provenance"]["n_continuation_classes"] for c in ladder]
    vals = [float(c["action_value_pot_share"]) for c in ladder]
    assert pf == sorted(pf) and len(set(pf)) == len(pf)          # 弃牌率单调增
    assert eq == sorted(eq, reverse=True)                        # 续跟权益单调降
    assert nc == sorted(nc, reverse=True)                        # 续跟类别单调减
    assert any(b < a for a, b in zip(vals, vals[1:])), vals      # 价值非单调
    assert max(vals) != vals[-1], "最大注额不得是价值最优"


def test_c5_deterministic_repeat_identical():
    b = _base("turn")
    av = _av_input("turn", "DECLARED_BALANCED")
    outs = [b["d"]["evaluate_search"](b["st"], search_tree=_tree("turn", 3),
                                      solver=None, action_value_input=av)
            for _ in range(2)]
    assert [c["search_score"] for c in outs[0]["candidates"]] == \
        [c["search_score"] for c in outs[1]["candidates"]]
    assert json.dumps(outs[0]["action_value"], sort_keys=True, default=str) == \
        json.dumps(outs[1]["action_value"], sort_keys=True, default=str)


# ══ 13. C6 ════════════════════════════════════════════════════════════
def test_c6_missing_inputs_fail_closed_without_defaults():
    b = _base("turn")
    tree = _tree("turn", 3)
    f = b["d"]["evaluate_search"]
    no_kw = f(b["st"], search_tree=tree, solver=None)
    assert no_kw["action_value_reason"] == "response_weights_not_supplied"
    assert all(c["action_value_pot_share"] is None for c in no_kw["candidates"])
    empty = f(b["st"], search_tree=tree, solver=None,
              action_value_input={"present": True, "records": []})
    assert empty["action_value_status"] == "UNKNOWN"
    assert empty["action_value_rejections"]
    ev = _ev("turn", 3)
    out = _ev_with("turn", 3, _records(ev, pop="DROP_EQ",
                                       drop_eq_for="bet|0.495"))
    bad = [c for c in out["candidates"] if c["amount_bb"] == 0.495][0]
    assert bad["action_value_status"] == "UNKNOWN"
    assert any("equity_vs_continuing" in u for u in bad["action_value_unknown"])
    assert out["action_value"]["n_valued"] == len(out["candidates"]) - 1
    out2 = _ev_with("turn", 3, _records(ev, pop="BAD_SUM",
                                        w_of=lambda r: (0.5, 0.6)))
    assert out2["action_value"]["n_valued"] == 0
    assert any("weights_do_not_sum_to_one" in str(x)
               for x in out2["action_value_rejections"])


def test_c6_fold_only_basis_negative_control_detected():
    ev = _ev("turn", 3)
    recs = _records(ev, pop="FOLD_ONLY")
    for rec in recs:
        rec["equity_basis"] = "legacy_all_leaves_including_fold_only"
        rec["basis_fingerprint"] = basis_fingerprint(rec)
    out = _ev_with("turn", 3, recs)
    assert out["action_value"]["n_valued"] == 0
    assert out["detector"]["flags"]["fold_only_equity_basis_rejected"] is True
    assert out["detector"]["verdict"] == "DEGENERATE"


def test_c6_shuffled_equity_negative_control_detected():
    ev = _ev("turn", 3)
    rows = ev["ranked_by_action_value"]
    assert len(rows) >= 2
    recs = []
    for i, r in enumerate(rows):
        other = rows[(i + 1) % len(rows)]
        rec = [x for x in _records(ev, pop="SHUFFLED")
               if f"{x['action']}|{x['amount_bb']}" == r["key"]][0]
        rec["equity_vs_continuing"] = other["equity_vs_continuing"]
        # **故意不更新** fingerprint ⇒ L3 必须发现
        recs.append(rec)
    out = _ev_with("turn", 3, recs)
    assert out["detector"]["flags"]["basis_fingerprint_mismatch"] is True
    assert out["detector"]["verdict"] == "DEGENERATE"


def test_detector_l2_identical_threshold_group_law():
    """检查器单测：同一门槛必须给出同一续跟集合（否则 L2 报警）。"""
    def arte(rows):
        return {"action_value": {"values": rows, "unknown": [],
                                 "differentiation": {"source": "x"}},
                "action_value_rejections": [],
                "action_value_budget_violation": False,
                "candidate_set_scope": {"candidate_set_mismatch": False}}
    base = {"status": "OK", "value": 0.5, "amount_bb": 0.5,
            "measured": {"equity_vs_continuing": 0.6, "p_continue": 0.2,
                         "p_fold": 0.8},
            "provenance": {"threshold": 0.4, "n_continuation_classes": 10},
            "terms": {"terminal_values_distinct": 2}}
    ok = {"a": json.loads(json.dumps(base)), "b": json.loads(json.dumps(base))}
    assert detect_degeneracy(artifact=arte(ok))["flags"][
        "same_threshold_group_inconsistent"] is False
    bad = json.loads(json.dumps(ok))
    bad["b"]["measured"]["equity_vs_continuing"] = 0.61
    assert detect_degeneracy(artifact=arte(bad))["flags"][
        "same_threshold_group_inconsistent"] is True
    l1 = {"a": json.loads(json.dumps(base))}
    c = json.loads(json.dumps(base))
    c["amount_bb"] = 1.0
    c["provenance"]["threshold"] = 0.8
    c["measured"]["p_continue"] = 0.5
    c["measured"]["p_fold"] = 0.5
    l1["b"] = c
    assert detect_degeneracy(artifact=arte(l1))["flags"][
        "weights_not_monotone_in_threshold"] is True


# ══ 14. C7 ════════════════════════════════════════════════════════════
def test_c7_max_leaves_overrun_registered_with_provenance_and_fail_closed():
    tree = _base("flop")["tree_default"]
    lb = tree["leaf_budget"]
    assert tree["max_leaves_exceeded"] is True
    assert lb["declared_max_leaves"] == 200
    assert lb["observed_leaves"] == len(tree["leaves"]) == 219
    assert lb["overrun"] == 19 and lb["n_overrun_leaves"] == 19
    assert lb["overrun_has_provenance"] is True
    assert lb["registered_in_budget_hits"] is True
    assert "max_leaves" in tree["budget_hits"]
    assert lb["truncation_applied"] is False
    by_id = {l["leaf_id"]: l["admission_index"] for l in tree["leaves"]}
    assert len(by_id) == len(tree["leaves"])
    assert all(by_id[i] > 200 for i in lb["overrun_leaf_ids"])
    assert [i for i, idx in by_id.items() if idx > 200] == lb["overrun_leaf_ids"]
    ev = _ev("flop", 3, tree=tree)                     # 超限树 ⇒ fail closed
    assert ev["action_value_status"] == "UNKNOWN"
    assert ev["action_value_reason"] == "leaf_budget_violated"
    assert ev["action_value_budget_violation"] is True
    assert ev["detector"]["flags"]["budget_violated"] is True
    assert all(c["search_score"] is not None for c in ev["candidates"])
    ev_turn = _ev("turn", 3)
    assert ev_turn["max_leaves_exceeded"] is False
    assert ev_turn["action_value_status"] == "OK"


# ══ 15. 编排 / 物化 ════════════════════════════════════════════════════
def test_orchestrator_and_materialisation_channel():
    b = _base("turn")
    br = _p8()._bridge_for(b["st"], b["rm"], b["ap"], b["bel"],
                           "DECLARED_BALANCED")
    orch = evaluate_search_with_action_value(
        state=b["st"], search_tree=_tree("turn", 3), bridge_out=br,
        evaluate_search_fn=b["d"]["evaluate_search"])
    assert orch["present"] is True and orch["action_value_applied"] is True
    assert orch["search_eval"]["action_value_status"] == "OK"
    assert orch["n_records"] >= 1
    ci = conditioned_evaluation_input(bridge_out=br)
    mat = materialize_conditioned_tree(search_tree=b["tree_default"],
                                       conditioned_input=ci)
    resp = [l["response"] for l in mat["tree"]["leaves"] if l.get("response")]
    assert resp, "物化必须写入 leaf.response"
    assert resp[0]["equity_basis"] == "equity_vs_continuing_measured"
    assert resp[0]["free_parameters"] == 0
    assert mat["n_response_written"] == len(mat["tree"]["leaves"])
    assert mat["n_leaves_rewritten"] == 27           # 旧计数语义不变


# ══ 16. 契约自述 / 零自由参数 / 非声明 ════════════════════════════════
def test_contract_zero_free_parameters_and_claim_boundary():
    assert AV_CONTRACT["free_parameters"] == 0
    assert AV_CONTRACT["uses_randomness"] is False
    assert all(v is False for v in AV_CLAIM_BOUNDARY.values())
    c = av_contract()
    assert c["contract"]["free_parameters"] == 0
    assert set(c["detector_laws"]) == {"L1", "L2", "L3"}
    ev = _ev("flop", 6)
    assert ev["action_value_model"]["free_parameters"] == 0
    for cand in ev["candidates"]:
        assert (cand["action_value_provenance"] or {}).get(
            "free_parameters", 0) == 0
    assert ev["action_value_unit"] == "pot_share_of_initial_pot"


# ══ 17. 黑盒纪律自检（要求 11）════════════════════════════════════════
def test_wiring_fingerprint_matches_contract_definition():
    """生产侧指纹必须满足契约的定义（wiring 只 import 标准库 ⇒ 就地实现）。"""
    for street, pop in (("flop", "DECLARED_BALANCED"),
                        ("turn", "DECLARED_TIGHT_PASSIVE")):
        inp = _av_input(street, pop)
        assert inp["present"] is True and inp["records"]
        for rec in inp["records"]:
            assert rec["basis_fingerprint"] == basis_fingerprint(rec), \
                (street, rec.get("action"), rec.get("amount_bb"))


def test_blackbox_discipline_no_aggregation_implementation_called():
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    for bad in _FORBIDDEN:
        assert bad not in src, f"黑盒测试不得调用被测聚合实现：{bad}"
    assert "evaluate_search" in src
