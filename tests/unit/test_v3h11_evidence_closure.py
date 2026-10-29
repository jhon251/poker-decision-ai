# -*- coding: utf-8 -*-
"""V3H-11 测试 — CASH Shadow 证据闭合审计（描述性计数 + 缺口清单）。

覆盖：CASH-only fail closed / 非 dict / 缺 records fail closed /
min_records 非法 / 记录不得从 state 隐式读取（口径必须显式）/ 
分歧计数封闭枚举（越界 → unavailable 不猜）/ V3H-10 stub 原因计数 /
unknown 原因归一为 token / 弃权·分歧分别计数 / **零比率零百分比**（输出与源码双证）/
`rate`/`pct` 恒为 None 且拒绝原因在场 / 已观测范围只给 min·max /
版本不符逐条拒绝 / 非 dict 元素计入 malformed / 禁止性 claim 标志被检出 /
executed=true 被检出 / environment 仅计数且只标 missing / eligibility 仅计数 /
closed 判据恒 False（Solver BLOCKED 不翻转）/ closure_reasons 必含 solver 与既有 blocker /
would_close_if 完整 / solv·gto·equilibrium·exploitability 恒 None / 
定输入复现 / state 不变异 / AST·import 隔离（零 V3G、零任何层、仅减法加法）/
生产 import 图零新增 / frozen·sealed drift / 真实全栈端到端。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_evidence_closure import (  # noqa: E402
    CASH_STRATEGY_EVIDENCE_CLOSURE_VERSION, CLAIM_BOUNDARY,
    DEFAULT_MIN_RECORDS, DIVERGENCE_KEYS, ESTABLISHED_GAPS, GAP_DIMENSIONS,
    SHADOW_RESULT_KEYS, audit_evidence_closure)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
from poker_decision_ai.cash_profit_interface import evaluate_entry_gate  # noqa: E402
from poker_decision_ai.cash_range_model import build_range_model  # noqa: E402
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_action_policy import (  # noqa: E402
    build_action_policy)
from poker_decision_ai.cash_strategy_action_ranking import rank_actions  # noqa: E402
from poker_decision_ai.cash_strategy_belief_conditioned_eval import (  # noqa: E402
    evaluate_belief_conditioned)
from poker_decision_ai.cash_strategy_decision_assembly import (  # noqa: E402
    assemble_decision)
from poker_decision_ai.cash_strategy_dynamic_belief import (  # noqa: E402
    build_dynamic_belief)
from poker_decision_ai.cash_strategy_experiment_eligibility import (  # noqa: E402
    ENVIRONMENT_FACT_KEYS, audit_experiment_eligibility)
from poker_decision_ai.cash_strategy_multistreet_eval import (  # noqa: E402
    conditional_absolute_equity)
from poker_decision_ai.cash_strategy_search_eval import evaluate_search  # noqa: E402
from poker_decision_ai.cash_strategy_search_tree import build_search_tree  # noqa: E402
from poker_decision_ai.cash_strategy_shadow_evaluation import (  # noqa: E402
    CASH_STRATEGY_SHADOW_EVALUATION_VERSION, evaluate_shadow_decision,
    make_shadow_observation)

NEW_MODULE = "cash_strategy_evidence_closure"
MODULE_PATH = PKG / f"{NEW_MODULE}.py"

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
V3H_LAYERS = tuple(f.stem for f in PKG.glob("cash_strategy_*.py"))

BUDGET = {"max_street_depth": 2, "max_response_depth": 1,
          "max_candidate_actions": 3, "max_leaves": 60, "max_runout_children": 2}
ALL_FACTS = {k: True for k in ENVIRONMENT_FACT_KEYS}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def cash_state(n_players: int = 3, street: str = "flop", board=None, **over) -> dict:
    pos = {2: ("SB", "BB"), 3: ("SB", "BB", "BTN"),
           4: ("SB", "BB", "MP", "BTN"),
           5: ("SB", "BB", "MP", "CO", "BTN"),
           6: ("SB", "BB", "UTG", "MP", "CO", "BTN")}[n_players]
    hero_seat = n_players
    players, vi = [], 1
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
        "synthetic_test_only": True, "players": players, "action_history": [],
    }
    st.update(over)
    return st


def shadow(present: bool = True, divergence: str = "aligned", **over) -> dict:
    """合成一条 V3H-10 输出（隔离本层聚合逻辑）。"""
    if not present:
        d = {"present": False, "version": CASH_STRATEGY_SHADOW_EVALUATION_VERSION,
             "reason": over.pop("reason", "shadow_absent"),
             "evidence_grade": "NONE", "unknown": over.pop("unknown", []),
             "divergence": "unavailable", "record": None, "executed": False}
        d.update(over)
        return d
    rec = {
        "recommended": {"action": "check", "amount_bb": 0.0,
                        "selected_by_tier": "search",
                        "score_source": "MODEL_INTERNAL"},
        "observed": {"action": "check", "amount_bb": 0.0, "source": "log"},
        "divergence": divergence,
        "divergence_is_descriptive_only": True,
        "preference_judgement_made": False,
        "candidate_sizes_from_action_policy": {},
        "legal_actions": [],
        "amount_tolerance_bb": 1e-6,
        "shadow_snapshot": {"present": True, "method": "synthetic_mc",
                            "samples": 20000, "adds_new_formula": False},
        "eligibility_trace": None,
        "evidence_grade": "MODEL_INTERNAL",
        "executed": False,
        "is_gto": False, "is_equilibrium": False, "is_solver_output": False,
        "is_profit_evidence": False, "is_causal_claim": False,
    }
    rec_over = over.pop("record", "KEEP")
    if rec_over is None or not isinstance(rec_over, dict) \
            and rec_over != "KEEP":
        rec = rec_over
    elif rec_over != "KEEP":
        rec = dict(rec)
        rec.update(rec_over)
    d = {"present": True, "version": CASH_STRATEGY_SHADOW_EVALUATION_VERSION,
         "mode": "cash", "street": "flop", "divergence": divergence,
         "divergence_classes": list(DIVERGENCE_KEYS), "record": rec,
         "available": True, "level": "MODEL_INTERNAL",
         "evidence_grade": "MODEL_INTERNAL",
         "uncertainty": {"snapshot_present": True, "snapshot_method":
                         "synthetic_mc", "snapshot_samples": 20000,
                         "amount_tolerance_bb": 1e-6},
         "unknown": [], "executed": False,
         "claim_boundary": dict(CLAIM_BOUNDARY), "note": "x"}
    d.update(over)
    return d


def real_shadow(state, observations) -> list:
    """真实 V3H-10 输出序列（给定每个决策点的已观测动作）。"""
    ap = build_action_policy(state, action_tree=build_action_tree(state))
    bl = build_dynamic_belief(state, range_model=build_range_model(state))
    rk = rank_actions(state, action_policy=ap, belief=bl)
    bc = evaluate_belief_conditioned(state, action_policy=ap, ranking=rk)
    tr = build_search_tree(state, action_policy=ap, belief=bl, budget=BUDGET,
                           max_root_candidates=3)
    se = evaluate_search(state, search_tree=tr)
    dec = assemble_decision(state, ranking=rk, belief_conditioned=bc,
                            search_eval=se, solver=solver_status(state))
    return [evaluate_shadow_decision(
        state, decision=dec, action_policy=ap, observed=obs,
        equity_snapshot=conditional_absolute_equity) for obs in observations]


# ── 1. CASH-only fail closed ─────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed(mode):
    r = audit_evidence_closure(cash_state(game_type=mode),
                               shadow_records=[shadow()])
    assert r["present"] is False
    assert r["reason"] == "not_cash_mode_fail_closed"
    assert r["n_records"] == 0
    assert r["gaps"] == []
    assert r["closure_test"] is None
    assert r["closure_possible"] is False
    assert r["executed"] is False
    assert r["evidence_grade"] == "NONE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42, 3.5, True])
def test_non_dict_state_fail_closed(bad):
    assert audit_evidence_closure(bad, shadow_records=[shadow()])["present"] \
        is False


@pytest.mark.parametrize("bad", [None, "records", 42, {"a": 1}])
def test_missing_records_fail_closed(bad):
    r = audit_evidence_closure(cash_state(), shadow_records=bad)
    assert r["present"] is False
    assert r["reason"] == "shadow_records_required"
    assert r["closure_possible"] is False
    assert r["executed"] is False


@pytest.mark.parametrize("bad", [0, -1, "1200", 1.5, True, None])
def test_invalid_min_records_fail_closed(bad):
    r = audit_evidence_closure(cash_state(), shadow_records=[shadow()],
                               min_records=bad)
    assert r["present"] is False
    assert r["reason"] == "min_records_invalid"


def test_records_never_taken_implicitly_from_state():
    st = cash_state(shadow_records=[shadow(), shadow()])
    r = audit_evidence_closure(st, shadow_records=None)
    assert r["present"] is False
    assert r["reason"] == "shadow_records_required"
    r2 = audit_evidence_closure(st, shadow_records=[shadow()])
    assert r2["present"] is False
    assert r2["reason"] == "state_shadow_records_not_accepted_use_explicit_arg"


def test_claim_boundary_all_false_and_13_flags():
    assert len(CLAIM_BOUNDARY) == 13
    assert all(v is False for v in CLAIM_BOUNDARY.values())
    assert CLAIM_BOUNDARY["computes_rates_or_percentages"] is False
    assert CLAIM_BOUNDARY["claims_experiment_ready"] is False
    r = audit_evidence_closure(cash_state(), shadow_records=[shadow()])
    assert all(v is False for v in r["claim_boundary"].values())


# ── 2. 分歧计数（封闭枚举，越界不猜）────────────────────────────────
def test_divergence_counts_fixed_keys_and_unrecognised_maps_to_unavailable():
    recs = [shadow(divergence="aligned"), shadow(divergence="different"),
            shadow(divergence="unavailable"), shadow(divergence="ALIGNED"),
            shadow(divergence="nonsense"), shadow(divergence=None), shadow(divergence=7)]
    r = audit_evidence_closure(cash_state(), shadow_records=recs)
    assert r["present"] is True
    c = r["counts"]["by_divergence"]
    assert set(c) == set(DIVERGENCE_KEYS)
    assert c["aligned"] == 2                         # 含大写归一
    assert c["different"] == 1
    assert c["unavailable"] == 4                     # 未识别一律计入，不新增键
    assert r["n_records"] == 7
    assert r["n_shadow_present"] == 7
    assert r["n_shadow_stub"] == 0


def test_divergence_keys_are_the_frozen_v3h10_set():
    assert DIVERGENCE_KEYS == ("aligned", "different", "unavailable")


def test_stub_records_counted_separately_with_reasons():
    recs = [shadow(present=False, reason="not_cash_mode_fail_closed"),
            shadow(present=False, reason="decision_required"),
            shadow(present=False, reason="decision_required"),
            shadow()]
    r = audit_evidence_closure(cash_state(), shadow_records=recs)
    assert r["n_shadow_present"] == 1
    assert r["n_shadow_stub"] == 3
    assert r["counts"]["by_divergence"]["unavailable"] == 3
    assert r["counts"]["shadow_stub_reasons"] == {
        "not_cash_mode_fail_closed": 1, "decision_required": 2}


def test_non_dict_elements_counted_as_malformed():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow(), None, "x", 42, []])
    assert r["n_records"] == 5
    assert r["n_shadow_present"] == 1
    assert r["counts"]["shadow_stub_reasons"]["shadow_result_not_a_dict"] == 4
    assert r["counts"]["by_divergence"]["unavailable"] == 4


def test_version_mismatch_rejects_records_one_by_one():
    good = shadow()
    bad = shadow(version="v3h10-unknown-v9")
    r = audit_evidence_closure(cash_state(), shadow_records=[good, bad],
                               shadow_version=CASH_STRATEGY_SHADOW_EVALUATION_VERSION)
    assert r["n_shadow_present"] == 1
    assert r["counts"]["shadow_stub_reasons"]["shadow_version_mismatch"] == 1
    assert r["present"] is True


def test_version_check_skipped_when_not_supplied():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow(version="whatever")])
    assert r["n_shadow_present"] == 1
    assert "shadow_version_mismatch" not in r["counts"]["shadow_stub_reasons"]


# ── 3. unknown / 弃权 / 分歧 归因计数 ───────────────────────────────
def test_unknown_reasons_grouped_by_token():
    recs = [shadow(unknown=["decision_abstained", "decision_abstained"]),
            shadow(unknown=["decision_disagreements:2"]),
            shadow(unknown=["eligibility_not_supplied"]),
            shadow(unknown=["eligibility_not_supplied"])]
    r = audit_evidence_closure(cash_state(), shadow_records=recs)
    assert r["counts"]["unknown_reasons"] == {
        "decision_abstained": 2, "decision_disagreements": 1,
        "eligibility_not_supplied": 2}
    ct = r["closure_test"]
    assert ct["n_abstained"] == 2
    assert ct["n_disagreement"] == 1
    assert ct["n_unknown_reasons"] == 3


def test_blank_unknown_reason_normalised_not_dropped():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow(unknown=["", "   ", None])])
    assert r["counts"]["unknown_reasons"] == {"<empty>": 3}


def test_stub_unknown_list_also_counted():
    r = audit_evidence_closure(
        cash_state(),
        shadow_records=[shadow(present=False, unknown=["a:1", "b"])])
    assert r["counts"]["unknown_reasons"] == {"a": 1, "b": 1}


def test_record_missing_is_reported_in_unknown():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow(record=None)])
    assert "record_missing" in r["unknown"]
    assert r["closure_test"]["n_unreconcilable"] == 1


def test_non_dict_record_is_not_reconcilable():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow(record="bad")])
    assert "record_missing" in r["unknown"]
    assert r["closure_test"]["n_unreconcilable"] == 1


# ── 4. 零比率 / 零百分比（输出与源码双证）───────────────────────────
def test_rate_and_pct_are_none_and_refusal_reason_present():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow() for _ in range(4)])
    assert r["rate"] is None
    assert r["pct"] is None
    assert r["rate_computed"] is False
    assert "比率" in r["rate_refusal_reason"]
    assert r["counts_only"] is True
    assert r["rates_forbidden"] is True


def test_no_rate_like_output_key_exists():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow() for _ in range(3)])
    flat = json.dumps(r, ensure_ascii=False)
    for banned in ("alignment_rate", "win_rate", "equity_delta", "ev_delta",
                   "expected_value", "roi", "pnl"):
        assert banned not in flat, banned
    # exploitability / gto / equilibrium 只允许以 **None + 显式拒绝** 出现
    assert r["uncertainty"]["exploitability"] is None
    assert r["uncertainty"]["gto"] is None
    assert r["uncertainty"]["equilibrium"] is None
    # 扁平键名扫描（允许 *_count / n_* 计数）
    keys = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(r)
    for k in keys:
        assert not k.endswith("_rate"), k
        assert not k.endswith("_pct"), k
        assert not k.endswith("_ratio"), k


def test_source_has_no_multiplication_or_division():
    """零比率保证的**源码级**证据：模块内不得出现乘 / 除 / 幂运算。"""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    ops = {type(n.op).__name__ for n in ast.walk(tree)
           if isinstance(n, ast.BinOp)}
    assert "Mult" not in ops
    assert "Div" not in ops
    assert "Pow" not in ops
    assert "FloorDiv" not in ops
    assert "Mod" not in ops
    assert ops <= {"Sub", "Add", "BitOr"}, ops


def test_observed_range_gives_only_min_max_no_central_tendency():
    recs = [shadow(record={"recommended": {"action": "bet", "amount_bb": a}})
            for a in (2.0, 4.0, 6.0)]
    r = audit_evidence_closure(cash_state(), shadow_records=recs)
    rng = r["observed_range"]["recommended_amount_bb"]
    assert rng == {"n": 3, "min": 2.0, "max": 6.0}
    flat = json.dumps(r, ensure_ascii=False)
    for banned in ("mean", "median", "average", "stdev", "variance", "percentile"):
        assert banned not in flat, banned


def test_observed_range_none_when_no_amounts():
    recs = [shadow(record={"recommended": {"action": "check", "amount_bb": None}})]
    r = audit_evidence_closure(cash_state(), shadow_records=recs)
    assert r["observed_range"]["recommended_amount_bb"] == {
        "n": 0, "min": None, "max": None}


# ── 5. 动作 / 街道 / 证据等级计数 ──────────────────────────────────
def test_action_and_street_counts():
    recs = [shadow(street="flop", record={"recommended": {"action": "bet"},
                                         "observed": {"action": "call"}}),
            shadow(street="flop", record={"recommended": {"action": "bet"},
                                         "observed": {"action": "call"}}),
            shadow(street="turn", record={"recommended": {"action": "check"},
                                         "observed": {"action": "check"}})]
    r = audit_evidence_closure(cash_state(), shadow_records=recs)
    assert r["counts"]["by_street"] == {"flop": 2, "turn": 1}
    assert r["counts"]["recommended_actions"] == {"bet": 2, "check": 1}
    assert r["counts"]["observed_actions"] == {"call": 2, "check": 1}


def test_missing_action_counted_as_none_not_dropped():
    recs = [shadow(record={"recommended": None, "observed": None}),
            shadow(record={"recommended": "bad", "observed": 7})]
    r = audit_evidence_closure(cash_state(), shadow_records=recs)
    assert r["counts"]["recommended_actions"] == {"<none>": 1, "<non-dict>": 1}
    assert r["counts"]["observed_actions"] == {"<none>": 1, "<non-dict>": 1}


def test_evidence_grade_and_level_counts_preserved():
    recs = [shadow(), shadow(evidence_grade="NONE", level="UNKNOWN",
                             divergence="unavailable")]
    r = audit_evidence_closure(cash_state(), shadow_records=recs)
    assert r["counts"]["by_evidence_grade"] == {"MODEL_INTERNAL": 1, "NONE": 1}
    assert r["counts"]["by_levels"] == {"MODEL_INTERNAL": 1, "UNKNOWN": 1}
    assert r["evidence_grade"] == "MODEL_INTERNAL"


def test_snapshot_counts_and_uncertainty_ranges():
    recs = [shadow(),
            shadow(record={"shadow_snapshot": {"present": True,
                                               "method": "synthetic_mc",
                                               "samples": 5000}}),
            shadow(record={"shadow_snapshot": {"present": False,
                                               "unknown": ["x"]}})]
    r = audit_evidence_closure(cash_state(), shadow_records=recs)
    c = r["counts"]
    assert c["snapshot_present"] == 2
    assert c["snapshot_absent"] == 1
    assert c["snapshot_methods"] == {"synthetic_mc": 2}
    u = r["uncertainty"]
    assert u["snapshot_samples_count"] == 2
    assert u["snapshot_samples_min"] == 5000
    assert u["snapshot_samples_max"] == 20000
    assert u["monte_carlo_inherited_from_v3h5"] is True


def test_uncertainty_never_invents_solver_facts():
    r = audit_evidence_closure(cash_state(), shadow_records=[shadow()])
    u = r["uncertainty"]
    assert u["exploitability"] is None
    assert u["gto"] is None
    assert u["equilibrium"] is None
    assert u["solver"]["solver_status"] == "BLOCKED"
    assert u["solver"]["available"] is False
    assert u["solver"]["equilibrium"] is None
    assert u["solver"]["discovery_state"] == "UNAVAILABLE"


# ── 6. 禁止性标志与 executed 检出 ──────────────────────────────────
def test_executed_true_is_detected_and_blocks_closure():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow(record={"executed": True})])
    assert r["counts"]["n_records_claiming_executed"] == 1
    assert "an_upstream_record_claims_executed" in \
        r["closure_test"]["closure_reasons"]
    assert r["closure_possible"] is False


@pytest.mark.parametrize("flag", ["is_gto", "is_equilibrium", "is_solver_output",
                                  "is_profit_evidence", "is_causal_claim"])
def test_forbidden_claim_flags_detected(flag):
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow(record={flag: True})])
    assert r["counts"]["n_forbidden_claim_flags_true"] == 1
    assert "a_forbidden_claim_flag_is_true" in \
        r["closure_test"]["closure_reasons"]


# ── 7. environment / eligibility 仅计数 ────────────────────────────
def test_environment_only_counted_never_gates():
    r_ok = audit_evidence_closure(cash_state(), shadow_records=[shadow()],
                                  environment=ALL_FACTS)
    assert r_ok["closure_test"]["environment_missing"] == []
    r_bad = audit_evidence_closure(cash_state(), shadow_records=[shadow()],
                                   environment={"a": True, "b": False, "c": None})
    assert r_bad["closure_test"]["environment_missing"] == ["b", "c"]
    # 无论 environment 如何，solver 仍阻塞 → closure_possible 恒 False
    assert r_ok["closure_possible"] is False
    assert r_bad["closure_possible"] is False


def test_environment_absent_is_recorded_as_missing():
    r = audit_evidence_closure(cash_state(), shadow_records=[shadow()])
    assert "environment_facts_not_supplied" in r["closure_test"]["closure_reasons"]
    with_env = audit_evidence_closure(cash_state(), shadow_records=[shadow()],
                                      environment=ALL_FACTS)
    assert "environment_facts_not_supplied" not in \
        with_env["closure_test"]["closure_reasons"]


def test_eligibility_only_counted_not_recomputed():
    st = cash_state()
    el = audit_experiment_eligibility(st, decision={"present": True,
                                                    "abstained": False,
                                                    "unknown": [],
                                                    "disagreements": [],
                                                    "evidence_grade":
                                                    "MODEL_INTERNAL",
                                                    "preferred":
                                                    {"action": "check"}},
                                      solver=solver_status(st))
    assert el["present"] is True
    r = audit_evidence_closure(cash_state(), shadow_records=[shadow()],
                               eligibility=[el])
    assert "eligibility_blocked" in r["counts"]["unknown_reasons"]
    assert r["closure_test"]["n_eligibility_not_eligible"] == 1
    assert "eligibility_not_eligible:1" in \
        r["closure_test"]["closure_reasons"]
    # 资格判定本身不被复制进输出
    assert "eligible" not in r
    assert "status" not in r


def test_eligibility_eligible_true_is_not_counted_as_gap():
    r = audit_evidence_closure(
        cash_state(), shadow_records=[shadow()],
        eligibility=[{"present": True, "eligible": True, "status": "OK"}])
    assert r["closure_test"]["n_eligibility_not_eligible"] == 0
    assert "eligibility_blocked" not in r["counts"]["unknown_reasons"]


def test_eligibility_non_list_ignored():
    r = audit_evidence_closure(cash_state(), shadow_records=[shadow()],
                               eligibility={"nope": 1})
    assert r["counts"]["unknown_reasons"] == {}


# ── 8. 关闭判定：恒 False，条件与既有 blocker 完整 ─────────────────
def test_closure_possible_always_false_and_solver_blocked():
    for env in (None, ALL_FACTS):
        r = audit_evidence_closure(
            cash_state(),
            shadow_records=[shadow() for _ in range(DEFAULT_MIN_RECORDS)],
            environment=env)
        assert r["closure_possible"] is False
        reasons = r["closure_test"]["closure_reasons"]
        assert "solver_blocked_and_no_equilibrium_reference" in reasons


def test_established_gaps_registered_and_not_measured_here():
    assert set(ESTABLISHED_GAPS) == {"B1", "B3", "B4", "B5", "solver"}
    for v in ESTABLISHED_GAPS.values():
        assert v["status"] == "BLOCKED"
        assert v["measured_by_this_layer"] is False
    assert ESTABLISHED_GAPS["B1"]["measured_by_this_layer"] is False
    r = audit_evidence_closure(cash_state(), shadow_records=[shadow()])
    ct = r["closure_test"]
    assert ct["established_blockers"] == ["B1", "B3", "B4", "B5", "solver"]
    assert ct["established_blockers_measured_by_this_layer"] is False


def test_records_short_by_computed_against_explicit_minimum():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow() for _ in range(5)],
                               min_records=1200)
    ct = r["closure_test"]
    assert ct["records_supplied"] == 5
    assert ct["records_expected_min"] == 1200
    assert ct["records_short_by"] == 1195
    assert "records_below_minimum:1195" in ct["closure_reasons"]
    assert r["closure_test"]["gap_counts"]["evidence_count"] == 1195


def test_records_short_by_zero_when_enough():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow() for _ in range(7)],
                               min_records=7)
    assert r["closure_test"]["records_short_by"] == 0
    assert not any("records_below_minimum" in x
                   for x in r["closure_test"]["closure_reasons"])


def test_all_zero_record_set_still_not_closure_possible():
    """全部 aligned 且数量达标且环境齐全 → 计数无缺口，
    但**仍**因独立 blocker 与 Solver BLOCKED 而不可关闭（不猜、不翻转）。"""
    r = audit_evidence_closure(
        cash_state(),
        shadow_records=[shadow() for _ in range(DEFAULT_MIN_RECORDS)],
        environment=ALL_FACTS)
    ct = r["closure_test"]
    assert ct["records_short_by"] == 0
    assert ct["n_abstained"] == 0
    assert ct["n_disagreement"] == 0
    assert ct["n_unreconcilable"] == 0
    assert ct["n_unknown_reasons"] == 0
    assert ct["gap_counts"]["divergence"] == 0
    assert ct["gap_counts"]["solver"] == 1
    assert any("established_blockers_unresolved" in x for x in ct["closure_reasons"])
    assert r["closure_possible"] is False
    assert r["is_complete"] is False          # closure_reasons 非空
    assert r["claim_boundary"]["claims_experiment_ready"] is False


def test_would_close_if_list_is_complete_and_present():
    r = audit_evidence_closure(cash_state(), shadow_records=[shadow()])
    wc = r["closure_test"]["would_close_if"]
    assert isinstance(wc, list) and len(wc) >= 9
    joined = " ".join(wc)
    for must in ("records_supplied", "different", "unavailable", "n_abstained",
                 "n_disagreement", "n_unknown_reasons", "n_unreconcilable",
                 "environment_missing", "no_independent_blocker_remains"):
        assert must in joined, must


def test_gap_dimensions_enumerated_with_counts():
    r = audit_evidence_closure(
        cash_state(),
        shadow_records=[shadow(divergence="different",
                               unknown=["decision_abstained",
                                        "decision_disagreements:1"]),
                        shadow(present=False)],
        min_records=10)
    dims = {g["dimension"]: g["count_or_flag"] for g in r["gaps"]}
    assert set(dims) == set(GAP_DIMENSIONS)
    assert dims["divergence"] == 2
    assert dims["abstention"] == 1
    assert dims["disagreement"] == 1
    assert dims["evidence_count"] == 8
    assert dims["solver"] == 1
    assert dims["unknown_reason"] >= 1


# ── 9. 定输入复现 / 不变异 ─────────────────────────────────────────
def test_deterministic_same_input_same_output():
    recs = [shadow(), shadow(present=False, reason="decision_required"),
            shadow(divergence="different", unknown=["z:1", "a", "z:2"]), None]
    outs = [audit_evidence_closure(cash_state(), shadow_records=recs,
                                   environment=ALL_FACTS) for _ in range(5)]
    assert all(o == outs[0] for o in outs)


def test_reason_ordering_is_sorted_for_reproducibility():
    r = audit_evidence_closure(cash_state(),
                               shadow_records=[shadow(unknown=["zeta", "alpha",
                                                               "mu"])])
    assert list(r["counts"]["unknown_reasons"]) == ["alpha", "mu", "zeta"]


def test_state_and_records_not_mutated():
    st = cash_state()
    recs = [shadow(), shadow(unknown=["a"])]
    st_before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    recs_before = json.dumps(recs, sort_keys=True, ensure_ascii=False)
    audit_evidence_closure(st, shadow_records=recs, environment=ALL_FACTS)
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == st_before
    assert json.dumps(recs, sort_keys=True, ensure_ascii=False) == recs_before


def test_output_is_json_serialisable():
    r = audit_evidence_closure(cash_state(), shadow_records=[shadow()],
                               environment=ALL_FACTS)
    assert json.loads(json.dumps(r, ensure_ascii=False)) == r


# ── 10. AST · import 隔离 ─────────────────────────────────────────
def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
    assert CASH_STRATEGY_EVIDENCE_CLOSURE_VERSION.startswith("v3h11-")


def test_top_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], mods


def test_zero_v3g_imports_anywhere_including_function_bodies():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            assert not any(v in (n.module or "") for v in V3G_LAYERS), n.module
        if isinstance(n, ast.Import):
            for a in n.names:
                assert not any(v in a.name for v in V3G_LAYERS), a.name


def test_zero_layer_imports_anywhere():
    """V3G-11 边界语义：本层不 import **任何** V3G/V3H 层（全部参数注入）。"""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom):
            mods.append(n.module or "")
        elif isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        for m in mods:
            assert not any(v in m for v in V3G_LAYERS + V3H_LAYERS), m


def test_no_random_import():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name.split(".")[0] != "random" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "random"


def test_no_frozen_multistreet_or_sample_runouts():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all("cash_multistreet" not in a.name for a in n.names)
        if isinstance(n, ast.Name):
            assert n.id != "sample_runouts"


def test_no_execution_or_betting_path_anywhere():
    src = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("click", "pyautogui", "keyboard.", "subprocess", "socket",
                   "requests", "urllib", "open(", "input(", "screenshot"):
        assert banned not in src, banned


def test_shadow_result_keys_pinned_to_v3h10_surface():
    assert len(SHADOW_RESULT_KEYS) == 15
    for k in ("present", "version", "divergence", "divergence_classes",
              "record", "available", "level", "evidence_grade", "uncertainty",
              "unknown", "executed", "claim_boundary"):
        assert k in SHADOW_RESULT_KEYS
    src = MODULE_PATH.read_text(encoding="utf-8")
    assert "SHADOW_RESULT_KEYS" in src


# ── 11. 生产 import 图零新增 ──────────────────────────────────────
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
                if m in V3G_LAYERS or m in V3H_LAYERS:
                    found.setdefault(m, set()).add(p.name)
    return found


def test_no_new_production_imports():
    found = _production_importers()
    assert set(found) == {"cash_integrity_context", "cash_unknown_e2e"}, found


def test_new_module_not_imported_in_production():
    assert NEW_MODULE not in _production_importers()


def test_all_v3h_layers_remain_unwired():
    found = _production_importers()
    for layer in V3H_LAYERS:
        assert layer not in found, layer
    # 集合**包含式**断言：后续阶段新增层不会使本测试失效，
    # 但任一 V3H-11 时已存在的层消失仍会失败。
    # 实质断言是上面的"零接线"，此处只防止 glob 静默漏文件。
    required = {
        "cash_strategy_action_policy",
        "cash_strategy_action_ranking",
        "cash_strategy_belief_conditioned_eval",
        "cash_strategy_belief_eval",
        "cash_strategy_belief_update",
        "cash_strategy_decision_assembly",
        "cash_strategy_dynamic_belief",
        "cash_strategy_evidence_closure",
        "cash_strategy_experiment_eligibility",
        "cash_strategy_likelihood_model",
        "cash_strategy_multistreet_eval",
        "cash_strategy_policy_evaluator",
        "cash_strategy_search_eval",
        "cash_strategy_search_tree",
        "cash_strategy_shadow_evaluation",
    }
    assert required <= set(V3H_LAYERS), sorted(required - set(V3H_LAYERS))


# ── 12. frozen / sealed drift · Solver ────────────────────────────
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


def test_solver_still_blocked_not_flipped():
    s = solver_status(cash_state())
    assert s["solver_status"] == "BLOCKED"
    assert s["available"] is False
    assert s["equilibrium"] is False
    assert s["gto_exact"] is False
    assert s["executed"] is False
    assert s["discovery_state"] == "UNAVAILABLE"


def test_default_min_records_is_the_frozen_b1_threshold():
    assert DEFAULT_MIN_RECORDS == 1200


def test_non_cash_states_never_produce_counts():
    for mode in ("mtt", "pko", "ko", "mystery", "spin"):
        r = audit_evidence_closure(
            cash_state(game_type=mode),
            shadow_records=[shadow() for _ in range(3)],
            environment=ALL_FACTS)
        assert r["present"] is False
        assert r["n_records"] == 0
        assert r["closure_possible"] is False


# ── 13. 真实全栈端到端 ───────────────────────────────────────────
def test_end_to_end_real_stack_counts_and_gaps():
    st = cash_state()
    recs = real_shadow(st, [make_shadow_observation("check", 0.0),
                            make_shadow_observation("fold", None)])
    r = audit_evidence_closure(st, shadow_records=recs,
                               shadow_version=CASH_STRATEGY_SHADOW_EVALUATION_VERSION,
                               environment=ALL_FACTS)
    assert r["present"] is True
    assert r["n_records"] == 2
    assert r["n_shadow_present"] == 2
    c = r["counts"]["by_divergence"]
    assert c["aligned"] == 1
    assert c["different"] == 1
    assert c["unavailable"] == 0
    assert r["counts"]["snapshot_present"] == 2
    assert r["counts"]["snapshot_methods"] != {}
    assert r["closure_test"]["n_abstained"] == 0
    assert r["closure_test"]["n_disagreement"] == 0
    assert r["closure_possible"] is False
    assert r["executed"] is False
    assert r["uncertainty"]["gto"] is None
    assert r["uncertainty"]["equilibrium"] is None
    assert r["uncertainty"]["exploitability"] is None
    assert r["rate"] is None and r["pct"] is None
    assert r["evidence_grade"] == "MODEL_INTERNAL"


def test_end_to_end_river_terminal_is_counted_as_unavailable():
    """冻结语义（V3G-7）：river 为终局 → V3H-10 记 unavailable；
    本层**如实计数**，不伪造建议、不绕过。"""
    st = cash_state(street="river", board=["Qs", "7d", "2c", "8h", "3s"])
    recs = real_shadow(st, [make_shadow_observation("check", 0.0)])
    assert recs[0]["present"] is True
    assert recs[0]["divergence"] == "unavailable"
    r = audit_evidence_closure(st, shadow_records=recs, environment=ALL_FACTS)
    assert r["counts"]["by_divergence"]["unavailable"] == 1
    assert r["counts"]["by_divergence"]["aligned"] == 0
    assert r["closure_test"]["gap_counts"]["divergence"] == 1
    assert any("unreconciled_or_divergent_records" in x
               for x in r["closure_test"]["closure_reasons"])
    assert "decision_abstained" in r["counts"]["unknown_reasons"]
    assert r["closure_possible"] is False


def test_end_to_end_mixed_cash_states_aggregate_deterministically():
    flop = cash_state()
    pre = cash_state(street="preflop", board=[])
    recs = (real_shadow(flop, [make_shadow_observation("check", 0.0)]) +
            real_shadow(pre, [make_shadow_observation("raise", 6.0)]))
    outs = [audit_evidence_closure(flop, shadow_records=recs,
                                   environment=ALL_FACTS) for _ in range(3)]
    assert all(o == outs[0] for o in outs)
    assert outs[0]["counts"]["by_street"] == {"flop": 1, "preflop": 1}


def test_end_to_end_eligibility_results_accepted_and_counted():
    st = cash_state()
    ap = build_action_policy(st, action_tree=build_action_tree(st))
    bl = build_dynamic_belief(st, range_model=build_range_model(st))
    rk = rank_actions(st, action_policy=ap, belief=bl)
    bc = evaluate_belief_conditioned(st, action_policy=ap, ranking=rk)
    tr = build_search_tree(st, action_policy=ap, belief=bl, budget=BUDGET,
                           max_root_candidates=3)
    se = evaluate_search(st, search_tree=tr)
    dec = assemble_decision(st, ranking=rk, belief_conditioned=bc,
                            search_eval=se, solver=solver_status(st))
    el = audit_experiment_eligibility(st, decision=dec, gate=evaluate_entry_gate(
        environment=ALL_FACTS), solver=solver_status(st))
    recs = [evaluate_shadow_decision(
        st, decision=dec, action_policy=ap,
        observed=make_shadow_observation(dec["preferred"]["action"],
                                        dec["preferred"].get("amount_bb")),
        equity_snapshot=conditional_absolute_equity)]
    r = audit_evidence_closure(st, shadow_records=recs, eligibility=[el],
                               environment=ALL_FACTS)
    assert r["present"] is True
    assert r["n_records"] == 1
    assert r["counts"]["by_divergence"]["aligned"] == 1
    # 资格结果被**计数**（真实闸门 + 真实决策：solver BLOCKED 使其不合格）
    assert r["closure_test"]["n_unreconcilable"] == 0
    assert r["closure_test"]["n_eligibility_not_eligible"] == \
        (0 if el["eligible"] else 1)
    assert "eligible" not in r
    assert r["closure_possible"] is False
