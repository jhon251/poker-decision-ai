# -*- coding: utf-8 -*-
"""V3H-14 测试 — CASH 外部就绪度审计（验证前检查 + 外部依赖清单）。

覆盖：CASH-only fail closed / 非 dict / 缺 closure·preregistration /
**verdict 诚实返回 `BLOCKED`**（5 项外部依赖全未满足）/
外部依赖清单逐项标注"由谁解决 / 能否内部满足 / 被禁止的内部替代品" /
**禁止内部替代品**（11 个标记，注入即 fail closed）/
SHA-256 内容指纹**不得**被当作真实性/认证/来源证明 /
外部信任锚必须带外且本层不签发、不代填、不自行判定 /
Solver 各量恒 None 且未被调用 / UNKNOWN 原样传播 /
`internally_implementable_next_capability = None`（**不制造内部替代品**）/
零结果（result/executed/analysis_executed 恒 None·False）/
零比率零统计（AST：无 Mult/Div/Pow/FloorDiv/Mod）/
定输入复现 / state 不变异 / AST·import 隔离（零 V3G、零任何层、无 IO）/
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

from poker_decision_ai.cash_strategy_external_readiness import (  # noqa: E402
    CASH_STRATEGY_EXTERNAL_READINESS_VERSION, CLAIM_BOUNDARY,
    EVIDENCE_LEVELS, EXTERNAL_REQUIREMENTS, FORBIDDEN_SUBSTITUTIONS,
    INTERNAL_READY_CAPABILITIES, PIPELINE_STAGES, READINESS_VERDICTS,
    REQUIRED_INPUTS, RESOLUTION_CLASSES, audit_external_readiness,
    external_dependency_manifest)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_action_tree import build_action_tree  # noqa: E402
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
from poker_decision_ai.cash_strategy_evidence_closure import (  # noqa: E402
    audit_evidence_closure)
from poker_decision_ai.cash_strategy_execution_intake import (  # noqa: E402
    intake_execution_records)
from poker_decision_ai.cash_strategy_experiment_eligibility import (  # noqa: E402
    ENVIRONMENT_FACT_KEYS)
from poker_decision_ai.cash_strategy_multistreet_eval import (  # noqa: E402
    conditional_absolute_equity)
from poker_decision_ai.cash_strategy_preregistration import (  # noqa: E402
    build_preregistration)
from poker_decision_ai.cash_strategy_search_eval import evaluate_search  # noqa: E402
from poker_decision_ai.cash_strategy_search_tree import build_search_tree  # noqa: E402
from poker_decision_ai.cash_strategy_shadow_evaluation import (  # noqa: E402
    evaluate_shadow_decision, make_shadow_observation)

NEW_MODULE = "cash_strategy_external_readiness"
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


def good_closure(**over) -> dict:
    c = {
        "present": True, "version": "v3h11-cash-strategy-evidence-closure-v1",
        "n_records": 5, "closure_possible": False,
        "counts": {"unknown_reasons": {}},
        "closure_test": {"records_short_by": 1195,
                         "established_blockers": ["B1", "B3", "B4", "B5",
                                                  "solver"]},
    }
    c.update(over)
    return c


def good_prereg(**over) -> dict:
    p = {
        "present": True,
        "version": "v3h12-cash-strategy-preregistration-v1",
        "analysis_ready": False, "result_generation_refused": True,
        "blockers_from_upstream": ["B1", "B3", "B4", "B5", "solver"],
        "blockers": {"M3": {"status": "BLOCKED"}, "B1": {"status": "BLOCKED"},
                     "B3": {"status": "BLOCKED"}, "B4": {"status": "BLOCKED"},
                     "B5": {"status": "BLOCKED"},
                     "solver": {"status": "BLOCKED"}},
        "result": None,
    }
    p.update(over)
    return p


def good_intake(**over) -> dict:
    i = {
        "present": True, "version": "v3h13-cash-strategy-execution-intake-v1",
        "intake_permitted": False, "admitted": 0, "n_admitted": 0,
        "n_refused": 0, "n_duplicate": 0, "result": None,
    }
    i.update(over)
    return i


def call(state=None, **over):
    kw = {"closure": good_closure(), "preregistration": good_prereg(),
          "intake": good_intake(), "solver": solver_status(cash_state()),
          "environment": ALL_FACTS}
    kw.update(over)
    return audit_external_readiness(state if state is not None else cash_state(),
                                    **kw)


def real_chain(state, observations):
    ap = build_action_policy(state, action_tree=build_action_tree(state))
    bl = build_dynamic_belief(state, range_model=build_range_model(state))
    rk = rank_actions(state, action_policy=ap, belief=bl)
    bc = evaluate_belief_conditioned(state, action_policy=ap, ranking=rk)
    tr = build_search_tree(state, action_policy=ap, belief=bl, budget=BUDGET,
                           max_root_candidates=3)
    se = evaluate_search(state, search_tree=tr)
    dec = assemble_decision(state, ranking=rk, belief_conditioned=bc,
                            search_eval=se, solver=solver_status(state))
    recs = [evaluate_shadow_decision(
        state, decision=dec, action_policy=ap, observed=obs,
        equity_snapshot=conditional_absolute_equity) for obs in observations]
    clo = audit_evidence_closure(state, shadow_records=recs,
                                 environment=ALL_FACTS)
    pre = build_preregistration(state, closure=clo, solver=solver_status(state),
                                environment=ALL_FACTS)
    itk = intake_execution_records(
        state, records=None, preregistration=pre,
        solver=solver_status(state), environment=ALL_FACTS,
        trust_anchor_verified=False, min_records=1)
    return dec, recs, clo, pre, itk


# ── 1. CASH-only fail closed ────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed(mode):
    r = call(cash_state(game_type=mode))
    assert r["present"] is False
    assert r["reason"] == "not_cash_mode_fail_closed"
    assert r["verdict"] == "BLOCKED"
    assert r["ready"] is False
    assert r["n_blocked_external"] == 0
    assert r["requirements"] == []
    assert r["result"] is None
    assert r["executed"] is False
    assert r["evidence_level"] == "UNAVAILABLE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42, 3.5, True])
def test_non_dict_state_fail_closed(bad):
    assert audit_external_readiness(bad)["present"] is False


@pytest.mark.parametrize("bad", [None, [], "x", 42, True])
def test_missing_closure_or_prereg_fail_closed(bad):
    r = call(closure=bad)
    assert r["present"] is False
    assert r["reason"] == "closure_and_preregistration_required"
    assert r["verdict"] == "BLOCKED"
    r2 = call(preregistration=bad)
    assert r2["present"] is False
    assert r2["reason"] == "closure_and_preregistration_required"


def test_optional_inputs_missing_are_recorded_not_guessed():
    r = audit_external_readiness(cash_state(), closure=good_closure(),
                                 preregistration=good_prereg())
    assert r["present"] is True
    for k in ("intake", "solver", "environment"):
        assert f"{k}_not_supplied" in r["unknown"]
    assert r["uncertainty"]["unresolved_inputs"] == ["intake", "solver",
                                                     "environment"]
    assert r["verdict"] == "BLOCKED"


def test_required_inputs_enum_is_complete():
    assert REQUIRED_INPUTS == ("closure", "preregistration", "intake", "solver",
                               "environment")


# ── 2. 诚实返回 BLOCKED ────────────────────────────────────────────
def test_verdict_is_blocked_with_five_unsatisfied_external_requirements():
    r = call()
    assert r["verdict"] == "BLOCKED"
    assert r["ready"] is False
    assert r["stops_internally"] is True
    assert r["n_blocked_external"] == 5
    assert r["n_satisfied"] == 0
    assert r["unsatisfied_external_requirement_ids"] == ["R1", "R2", "R3",
                                                         "R4", "R5"]
    assert r["next_step_depends_on_external_resources"] is True
    assert "外部资源" in r["honest_conclusion"]
    assert r["requirement_counts_by_class"]["BLOCKED_EXTERNAL_RESOURCE"] == 5


def test_verdict_enums_are_closed():
    assert READINESS_VERDICTS == ("READY", "BLOCKED", "PARTIAL")
    assert RESOLUTION_CLASSES == ("SATISFIED", "BLOCKED_INTERNAL",
                                 "BLOCKED_EXTERNAL_RESOURCE", "NOT_APPLICABLE")


def test_ready_is_unreachable_while_frozen_facts_hold():
    """即使把所有**允许**的输入都设为最优，仍不可能 READY（外部资源未满足）。"""
    r = call(closure=good_closure(closure_possible=True),
             preregistration=good_prereg(analysis_ready=True),
             intake=good_intake(intake_permitted=True, n_admitted=5000))
    assert r["verdict"] == "BLOCKED"
    assert r["ready"] is False
    assert r["n_blocked_external"] == 5


def test_partial_only_when_no_external_remains():
    r = call(available_external_resources={
        "solver": {"satisfied": True},
        "real_execution_data": {"satisfied": True},
        "independent_decision_points": {"satisfied": True},
        "real_profit_evidence": {"satisfied": True},
        "trust_anchor": {"satisfied": True}})
    # 5 项外部满足，但 M3 仍为内部未解 → PARTIAL
    assert r["verdict"] == "PARTIAL"
    assert r["n_blocked_external"] == 0
    assert r["n_blocked_internal"] == 1
    assert r["ready"] is False


def test_ready_only_when_every_requirement_satisfied():
    r = call(available_external_resources={
        "solver": {"satisfied": True},
        "real_execution_data": {"satisfied": True},
        "independent_decision_points": {"satisfied": True},
        "real_profit_evidence": {"satisfied": True},
        "trust_anchor": {"satisfied": True},
        "m3": {"satisfied": True}})
    assert r["verdict"] == "READY"
    assert r["ready"] is True
    assert r["n_satisfied"] == 6
    # 但**仍然**不产生结果、不执行
    assert r["result"] is None
    assert r["executed"] is False
    assert r["analysis_executed"] is False


# ── 3. 外部依赖清单 ────────────────────────────────────────────────
def test_manifest_lists_five_external_requirements():
    m = external_dependency_manifest()
    assert m["success"] is True
    assert m["n_requirements"] == 6
    assert m["n_external"] == 5
    assert m["external_requirement_ids"] == ["R1", "R2", "R3", "R4", "R5"]
    assert m["all_unsatisfied"] is True
    assert m["internally_satisfiable"] == []
    assert m["no_internal_substitute"] is True
    assert m["result"] is None
    assert m["executed"] is False


@pytest.mark.parametrize("rid", ["R1", "R2", "R3", "R4", "R5"])
def test_each_external_requirement_is_fully_annotated(rid):
    req = next(r for r in EXTERNAL_REQUIREMENTS if r["id"] == rid)
    assert req["kind"] == "external_resource"
    assert req["resolution_class"] == "BLOCKED_EXTERNAL_RESOURCE"
    assert req["satisfiable_by_this_project_internally"] is False
    assert req["no_internal_substitute"] is True
    assert req["internal_substitutes_forbidden"], rid
    assert req["required_input"]
    assert req["observable_check"]
    assert req["current_state"]
    assert req["measured_by_this_layer"] is False
    assert req["evidence_level"] == "EXTERNAL_RESOURCE_FACT"


def test_requirements_cover_the_named_blockers():
    blockers = {r["blocker"] for r in EXTERNAL_REQUIREMENTS}
    for b in ("solver", "real_execution_data", "B1", "B3", "trust_anchor", "M3"):
        assert b in blockers, b


def test_m3_is_the_only_non_external_requirement():
    non_ext = [r for r in EXTERNAL_REQUIREMENTS
               if r["kind"] != "external_resource"]
    assert [r["id"] for r in non_ext] == ["R6"]
    assert non_ext[0]["blocker"] == "M3"
    assert non_ext[0]["resolution_class"] == "BLOCKED_INTERNAL"


def test_manifest_forbids_solver_and_profit_substitutes():
    by_id = {r["id"]: r for r in EXTERNAL_REQUIREMENTS}
    r1 = " ".join(by_id["R1"]["internal_substitutes_forbidden"]).lower()
    assert "equilibrium" in r1 or "gto" in r1
    r4 = " ".join(by_id["R4"]["internal_substitutes_forbidden"]).lower()
    assert "profit" in r4 or "pnl" in r4
    r5 = " ".join(by_id["R5"]["internal_substitutes_forbidden"]).lower()
    assert "sha-256" in r5 or "digest" in r5


def test_manifest_never_declares_anything_ready():
    m = external_dependency_manifest()
    assert m["internally_satisfiable"] == []
    flat = json.dumps(m, ensure_ascii=False).lower()
    for banned in ("is_ready", "profitable", "validated", "certified"):
        assert banned not in flat, banned
    # claims_experiment_ready 只允许以 **false** 出现
    assert m["claim_boundary"]["claims_experiment_ready"] is False
    assert flat.count("claims_experiment_ready") == 1


# ── 4. 禁止内部替代品 ──────────────────────────────────────────────
@pytest.mark.parametrize("marker", list(FORBIDDEN_SUBSTITUTIONS))
def test_injected_internal_substitutes_are_refused(marker):
    r = call(available_external_resources={marker: {"satisfied": True}})
    assert r["present"] is False
    assert r["reason"].startswith("forbidden_internal_substitution_detected")
    assert marker in r["reason"]
    assert r["verdict"] == "BLOCKED"
    assert r["executed"] is False


def test_solver_fact_is_not_flipped_from_a_forbidden_substitute():
    """不得用 mock solver / 近似均衡 / 声明式可利用度"解决"求解器缺失。"""
    for marker in ("mock_solver", "approximated_equilibrium",
                   "declared_exploitability"):
        assert marker in FORBIDDEN_SUBSTITUTIONS
        r = call(available_external_resources={marker: True})
        assert r["present"] is False
        assert r["verdict"] == "BLOCKED"


def test_digest_must_not_be_treated_as_proof():
    assert "digest_as_proof" in FORBIDDEN_SUBSTITUTIONS
    r = call(available_external_resources={"digest_as_proof": True})
    assert r["present"] is False
    assert CLAIM_BOUNDARY["digest_treated_as_provenance_proof"] is False
    assert CLAIM_BOUNDARY["is_authenticity_proof"] is False
    assert CLAIM_BOUNDARY["is_certification"] is False


def test_self_issued_trust_anchor_is_forbidden():
    assert "self_issued_trust_anchor" in FORBIDDEN_SUBSTITUTIONS
    r = call(available_external_resources={"self_issued_trust_anchor": True})
    assert r["present"] is False
    r2 = call()
    assert r2["trust_anchor_policy"]["issued_by_this_project"] is False
    assert r2["trust_anchor_policy"]["verified_by_this_layer"] is False
    assert r2["trust_anchor_policy"]["must_be_out_of_band"] is True
    assert r2["trust_anchor_policy"]["digest_is_not_proof"] is True


def test_forbidden_substitution_list_is_closed_and_documented():
    assert len(FORBIDDEN_SUBSTITUTIONS) == 11
    for x in ("internal_substitute", "synthetic_execution",
              "simulated_execution", "model_internal_as_external",
              "generated_profit", "inferred_realised_outcome"):
        assert x in FORBIDDEN_SUBSTITUTIONS
    r = call()
    assert r["substitution_policy"]["substitution_forbidden"] is True
    assert r["substitution_policy"]["internal_substitutes_offered"] == []
    assert r["substitution_policy"]["refused_substitutions_detected"] == []


def test_no_internal_substitute_is_ever_offered():
    r = call()
    assert r["internally_implementable_next_capability"] is None
    assert r["substitution_policy"]["internal_substitutes_offered"] == []
    assert r["honest_conclusion"].startswith("下一步依赖")


# ── 5. 零结果 / 零执行 ────────────────────────────────────────────
def test_never_produces_results_or_executes():
    r = call()
    assert r["result"] is None
    assert r["results_available"] is False
    assert r["analysis_executed"] is False
    assert r["executed"] is False
    p = r["provenance"]
    assert p["external_resources_obtained_by_this_layer"] is False
    assert p["external_resources_simulated_or_fabricated"] is False
    assert p["digest_used_as_proof"] is False
    assert p["wrote_any_file"] is False
    assert p["opened_any_client"] is False
    assert p["real_money_moved"] is False


def test_flat_key_scan_has_no_result_or_statistic_field():
    r = call()
    keys: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(r)
    for banned in ("p_value", "effect_size", "confidence_interval", "win_rate",
                   "roi", "pnl", "profit_estimate", "alignment_rate",
                   "success_rate", "exploitability_value"):
        assert banned not in keys, banned


def test_claim_boundary_all_false_19_flags():
    assert len(CLAIM_BOUNDARY) == 19
    assert all(v is False for v in CLAIM_BOUNDARY.values())
    for k in ("fabricates_trust_anchor",
              "supplies_internal_substitute_for_external_resource",
              "generates_execution_or_profit_or_statistical_results",
              "digest_treated_as_provenance_proof",
              "claims_experiment_ready"):
        assert CLAIM_BOUNDARY[k] is False, k
    r = call()
    assert all(v is False for v in r["claim_boundary"].values())


def test_evidence_levels_preserved():
    assert EVIDENCE_LEVELS == ("EXTERNAL_RESOURCE_FACT",
                              "INTERNAL_DECLARATION", "UNAVAILABLE")
    r = call()
    assert r["evidence_level"] == "INTERNAL_DECLARATION"
    assert r["evidence_grade"] == "NONE"
    assert r["provenance"]["evidence_levels"] == list(EVIDENCE_LEVELS)
    assert r["uncertainty"]["measured"] is False


# ── 6. Solver ─────────────────────────────────────────────────────
def test_solver_quantities_are_always_none_and_never_invoked():
    for s in (None, solver_status(cash_state()), {"solver_status": "BLOCKED"}):
        v = call(solver=s)["solver"]
        assert v["equilibrium"] is None
        assert v["gto"] is None
        assert v["exploitability"] is None
        assert v["blocked"] is True
        assert v["solver_invoked"] is False


def test_solver_still_blocked_not_flipped():
    s = solver_status(cash_state())
    assert s["solver_status"] == "BLOCKED"
    assert s["available"] is False
    assert s["equilibrium"] is False
    assert s["gto_exact"] is False
    assert s["discovery_state"] == "UNAVAILABLE"


def test_caller_optimistic_solver_does_not_change_the_audit_verdict():
    r = call(solver={"solver_status": "AVAILABLE", "available": True,
                     "equilibrium": True, "discovery_state": "AVAILABLE"})
    # 求解器"可用"仍**不**使整体就绪：其余外部依赖仍未满足
    assert r["verdict"] == "BLOCKED"
    assert r["solver"]["blocked"] is False
    assert r["solver"]["equilibrium"] is None      # 仍不猜数值
    assert r["executed"] is False


# ── 7. 上游事实转述（不重算）──────────────────────────────────────
def test_upstream_facts_are_transcribed_not_recomputed():
    r = call()
    u = r["upstream_facts"]
    assert u["closure_closure_possible"] is False
    assert u["closure_established_blockers"] == ["B1", "B3", "B4", "B5",
                                                 "solver"]
    assert u["preregistration_analysis_ready"] is False
    assert u["preregistration_result_generation_refused"] is True
    assert u["intake_permitted"] is False
    assert u["intake_admitted"] == 0
    assert "转述" in u["note"]


def test_pipeline_stages_declared_statically():
    assert len(PIPELINE_STAGES) == 10
    stages = [s["stage"] for s in PIPELINE_STAGES]
    assert stages == [f"V3H-{n}" for n in range(4, 14)]
    r = call()
    assert [s["stage"] for s in r["pipeline_stages"]] == stages
    assert len(r["internal_ready_capabilities"]) == 6
    assert {c["capability"] for c in INTERNAL_READY_CAPABILITIES} == {
        "decision assembly", "experiment eligibility", "shadow evaluation",
        "evidence closure", "pre-registration / result gate",
        "execution intake / integrity digest"}


def test_blockers_not_measured_by_this_layer():
    r = call()
    assert r["blockers_measured_by_this_layer"] is False
    for req in r["requirements"]:
        assert req["measured_by_this_layer"] is False


# ── 8. 定输入复现 / 不变异 ─────────────────────────────────────────
def test_deterministic_same_input_same_output():
    outs = [call() for _ in range(5)]
    assert all(o == outs[0] for o in outs)
    outs2 = [external_dependency_manifest() for _ in range(3)]
    assert all(o == outs2[0] for o in outs2)


def test_missing_evidence_is_sorted_for_reproducibility():
    r = call()
    assert r["missing_evidence"] == sorted(r["missing_evidence"])
    assert r["unknown"] == sorted(r["unknown"])


def test_state_not_mutated():
    st = cash_state()
    before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    call(st)
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == before


def test_outputs_json_serialisable():
    for r in (call(), external_dependency_manifest()):
        assert json.loads(json.dumps(r, ensure_ascii=False)) == r


# ── 9. 零统计量（源码级）─────────────────────────────────────────
def test_source_has_no_multiplication_division_or_power():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    ops = {type(n.op).__name__ for n in ast.walk(tree)
           if isinstance(n, ast.BinOp)}
    for banned in ("Mult", "Div", "FloorDiv", "Pow", "Mod"):
        assert banned not in ops, banned


def test_source_has_no_statistics_network_or_io():
    src = MODULE_PATH.read_text(encoding="utf-8")
    lower = src.lower()
    for banned in ("scipy", "numpy", "statsmodels", "p_value", "effect_size",
                   "socket", "requests", "urllib", "http", "subprocess",
                   "open(", "pathlib", "sqlite", "csv."):
        assert banned not in lower, banned


# ── 10. AST · import 隔离 ────────────────────────────────────────
def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
    assert CASH_STRATEGY_EXTERNAL_READINESS_VERSION.startswith("v3h14-")


def test_top_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], mods
    plain = [a.name for n in tree.body if isinstance(n, ast.Import)
             for a in n.names]
    assert plain == [], plain


def test_zero_v3g_imports_anywhere_including_function_bodies():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            assert not any(v in (n.module or "") for v in V3G_LAYERS), n.module
        if isinstance(n, ast.Import):
            for a in n.names:
                assert not any(v in a.name for v in V3G_LAYERS), a.name


def test_zero_layer_imports_anywhere():
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


def test_no_frozen_multistreet_or_sample_runouts_name():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all("cash_multistreet" not in a.name for a in n.names)
        if isinstance(n, ast.Name):
            assert n.id != "sample_runouts"


def test_no_client_automation_path_anywhere():
    src = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("click", "pyautogui", "keyboard.", "screenshot", "input(",
                   "advapi32", "win32"):
        assert banned not in src, banned


# ── 11. 生产 import 图零新增 ─────────────────────────────────────
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
    required = {
        "cash_strategy_action_policy", "cash_strategy_action_ranking",
        "cash_strategy_belief_conditioned_eval", "cash_strategy_belief_eval",
        "cash_strategy_belief_update", "cash_strategy_decision_assembly",
        "cash_strategy_dynamic_belief", "cash_strategy_evidence_closure",
        "cash_strategy_execution_intake", "cash_strategy_experiment_eligibility",
        "cash_strategy_external_readiness", "cash_strategy_likelihood_model",
        "cash_strategy_multistreet_eval", "cash_strategy_policy_evaluator",
        "cash_strategy_preregistration", "cash_strategy_search_eval",
        "cash_strategy_search_tree", "cash_strategy_shadow_evaluation",
    }
    assert required <= set(V3H_LAYERS), sorted(required - set(V3H_LAYERS))


# ── 12. frozen / sealed drift ────────────────────────────────────
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


# ── 13. 真实全栈端到端 ───────────────────────────────────────────
def test_end_to_end_real_chain_yields_blocked_and_no_result():
    st = cash_state()
    dec, recs, clo, pre, itk = real_chain(
        st, [make_shadow_observation("check", 0.0),
             make_shadow_observation("fold", None)])
    r = audit_external_readiness(st, closure=clo, preregistration=pre,
                                 intake=itk, solver=solver_status(st),
                                 environment=ALL_FACTS)
    assert r["present"] is True
    assert r["verdict"] == "BLOCKED"
    assert r["ready"] is False
    assert r["n_blocked_external"] == 5
    assert r["internally_implementable_next_capability"] is None
    assert r["upstream_facts"]["closure_established_blockers"] == \
        ["B1", "B3", "B4", "B5", "solver"]
    assert r["upstream_facts"]["intake_permitted"] is False
    assert r["result"] is None
    assert r["executed"] is False
    assert r["provenance"]["evidence_level"] == "INTERNAL_DECLARATION"
    assert r["solver"]["equilibrium"] is None


def test_end_to_end_river_terminal_chain_still_blocked():
    st = cash_state(street="river", board=["Qs", "7d", "2c", "8h", "3s"])
    dec, recs, clo, pre, itk = real_chain(
        st, [make_shadow_observation("check", 0.0)])
    r = audit_external_readiness(st, closure=clo, preregistration=pre,
                                 intake=itk, solver=solver_status(st),
                                 environment=ALL_FACTS)
    assert r["verdict"] == "BLOCKED"
    assert r["ready"] is False
    assert r["result"] is None


def test_end_to_end_audit_does_not_change_any_frozen_fact():
    st = cash_state()
    before = solver_status(st)["solver_status"]
    _dec, _recs, clo, pre, itk = real_chain(
        st, [make_shadow_observation("check", 0.0)])
    audit_external_readiness(st, closure=clo, preregistration=pre, intake=itk,
                             solver=solver_status(st), environment=ALL_FACTS)
    assert solver_status(st)["solver_status"] == before == "BLOCKED"
    assert clo["closure_possible"] is False
    assert pre["analysis_ready"] is False
    assert itk["intake_permitted"] is False
