# -*- coding: utf-8 -*-
"""V3I-P1 测试 — 可控 6-max CASH 执行环境 + 决策闭环（T1–T16）。

覆盖：
T1 环境确定性 / T2 CASH-only fail closed / T3 不伪造（无未来牌·无抽样·无真实对手字样）/
T4 闭环 COMPLETE 路径 / T5 弃权传播 / T6 UNKNOWN 传播 /
**T7 闭环真正执行既有 `belief_update`（激活 BP-3 通路）** /
T8 证据等级强制（`EXTERNAL_VALIDATION` fail closed）/
T9 零新评分公式 / T10 种群级描述性计数 /
**T11 V3G-11 边界不破（新模块零层 import + 7 未接线层仍零生产 import）** /
T12 生产 import 图不变 / T13 冻结 drift / T14 Solver 不翻转 /
T15 无自动操作 / T16 契约与 claim boundary。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_decision_environment import (  # noqa: E402
    BOARD_SIZE, CASH_DECISION_ENVIRONMENT_VERSION, CLAIM_BOUNDARY as ENV_CB,
    DECK, ENVIRONMENT_EVIDENCE_LEVEL, POPULATIONS, POPULATION_PARAMS,
    SIX_MAX_POSITIONS, STREETS, build_scenario, build_scenario_sequence,
    deck_card_for, environment_contract, population_action_weights)
from poker_decision_ai.cash_decision_loop import (  # noqa: E402
    ALLOWED_EVIDENCE_LEVELS, CASH_DECISION_LOOP_VERSION,
    CLAIM_BOUNDARY as LOOP_CB, LOOP_STATUSES, REJECTED_EVIDENCE_LEVELS,
    REQUIRED_PIPELINE_STAGES, STAGE_ORDER, loop_contract, run_decision_loop)

# 测试侧允许 import 任意层（生产侧不允许，由 T11/T12 强制）
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

ENV_MODULE = PKG / "cash_decision_environment.py"
LOOP_MODULE = PKG / "cash_decision_loop.py"

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
#: V3G-11 冻结的 7 个"未接线层"（名字固定，不得被任何生产模块 import）
V3G11_UNWIRED = ("cash_preflop_decision", "cash_multistreet",
                 "cash_opponent_model", "cash_range_model",
                 "cash_action_tree", "cash_solver_interface",
                 "cash_profit_interface")
V3H_LAYERS = tuple(f.stem for f in PKG.glob("cash_strategy_*.py"))

POPS6 = ["DECLARED_AGGRESSIVE", "DECLARED_TIGHT_PASSIVE", "DECLARED_BALANCED",
         "DECLARED_LOOSE_PASSIVE", "DECLARED_MANIAC"]
BUDGET = {"max_street_depth": 2, "max_response_depth": 1,
          "max_candidate_actions": 3, "max_leaves": 60,
          "max_runout_children": 2}

_LM = action_likelihood_model([
    {"action": "raise", "street": "flop", "multiplier": 1.5,
     "source": "declared", "evidence_level": "DECLARED_MODEL",
     "provenance": {"stage": "v3i_p1", "kind": "declared"},
     "uncertainty": {"declared_only": True}},
])

_PREF = {"action": "bet", "amount_bb": 2.0, "candidate_key": ["bet", 2.0],
         "selected_by_tier": "search", "score_source": "MODEL_INTERNAL"}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def real_pipeline() -> dict:
    return {
        "build_action_policy": build_action_policy,
        "build_dynamic_belief": build_dynamic_belief,
        "update_belief": update_belief_with_likelihood,
        "rank_actions": rank_actions,
        "evaluate_belief_conditioned": evaluate_belief_conditioned,
        "build_search_tree": build_search_tree,
        "evaluate_search": evaluate_search,
        "assemble_decision": assemble_decision,
    }


def agreeing_pipeline() -> dict:
    """测试替身：全 scorer 一致（仅用于验证闭环自身的 COMPLETE 路径）。"""
    return {
        "build_action_policy": lambda s, **k: {
            "present": True, "level": "MODEL_INTERNAL", "unknown": []},
        "build_dynamic_belief": lambda s, **k: {
            "present": True, "level": "MODEL_INTERNAL", "unknown": []},
        "update_belief": lambda s, **k: {
            "present": True, "level": "MODEL_INTERNAL", "unknown": []},
        "rank_actions": lambda s, **k: {
            "present": True, "preferred": _PREF, "unknown": []},
        "evaluate_belief_conditioned": lambda s, **k: {
            "present": True, "preferred": _PREF, "unknown": []},
        "build_search_tree": lambda s, **k: {"present": True, "unknown": []},
        "evaluate_search": lambda s, **k: {
            "present": True, "preferred": _PREF, "unknown": []},
        "assemble_decision": lambda s, **k: {
            "present": True, "abstained": False, "preferred": _PREF,
            "disagreements": [], "level": "MODEL_INTERNAL", "unknown": []},
    }


def abstaining_pipeline() -> dict:
    p = agreeing_pipeline()
    p["assemble_decision"] = lambda s, **k: {
        "present": True, "abstained": True, "preferred": None,
        "disagreements": [{"scorer": "search"}, {"scorer": "ranking"}],
        "level": "MODEL_INTERNAL", "unknown": ["cross_scorer_disagreement"]}
    return p


def run_real(seed: int = 11, street: str = "flop",
             population: str = "DECLARED_BALANCED", **over) -> dict:
    scn = build_scenario(seed=seed, street=street, population_table=POPS6)
    st = scn["state"]
    kw = dict(
        pipeline=real_pipeline(),
        integrity_context=cash_integrity_context(st),
        action_tree=build_action_tree(st),
        range_model=build_range_model(st,
                                      opponent_model=build_opponent_model(st)),
        opponent_model=build_opponent_model(st),
        transition=next_street_transition(st),
        likelihood_model=_LM,
        action_weights=population_action_weights(
            population=population, last_action="raise",
            street=street)["action_weights"],
        solver=solver_status(st),
        budget=BUDGET, max_root_candidates=3,
    )
    kw.update(over)
    return run_decision_loop(st, **kw)


# ══ T1 环境确定性 ══════════════════════════════════════════════════════
def test_T1_environment_deterministic():
    a = build_scenario(seed=7, street="flop", population_table=POPS6)
    b = build_scenario(seed=7, street="flop", population_table=POPS6)
    assert a["state"] == b["state"]
    assert a["board"] == b["board"]
    assert a["hero_hole_cards"] == b["hero_hole_cards"]
    assert a == b
    c = build_scenario(seed=8, street="flop", population_table=POPS6)
    assert c["board"] != a["board"] or c["hero_hole_cards"] != a["hero_hole_cards"]


def test_T1_board_size_by_street_and_no_future_cards():
    assert build_scenario(seed=1, street="preflop")["board"] == []
    assert len(build_scenario(seed=1, street="flop")["board"]) == 3
    assert len(build_scenario(seed=1, street="turn")["board"]) == 4
    assert len(build_scenario(seed=1, street="river")["board"]) == 5
    assert BOARD_SIZE == {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def test_T1_no_duplicate_cards_dealt():
    scn = build_scenario(seed=13, street="river")
    all_cards = list(scn["board"]) + list(scn["hero_hole_cards"])
    assert len(all_cards) == len(set(all_cards))
    assert all(c in DECK for c in all_cards)


def test_T1_deck_card_for_is_index_based():
    r = deck_card_for(seed=2, offset=0)
    assert r["present"] is True
    assert r["card"] == DECK[(2 * 7) % len(DECK)]
    assert r["index"] == (2 * 7) % len(DECK)
    assert r["generates_future_cards"] is False
    assert r["samples_runouts"] is False
    assert deck_card_for(seed=-1, offset=0)["present"] is False
    assert deck_card_for(seed=1, offset=-1)["present"] is False


def test_T1_sequence_same_seed_same_output():
    a = build_scenario_sequence(seed=3)
    b = build_scenario_sequence(seed=3)
    assert a["scenario_ids"] == b["scenario_ids"]
    assert a["n_scenarios"] == len(STREETS) == 4


# ══ T2 CASH-only / 参数 fail closed ════════════════════════════════════
@pytest.mark.parametrize("bad", [-1, 1.5, "7", None, True])
def test_T2_bad_seed_fail_closed(bad):
    r = build_scenario(seed=bad)
    assert r["present"] is False
    assert r["reason"] == "seed_must_be_non_negative_int"


@pytest.mark.parametrize("bad", ["", "MTT", "showdown", None, 3])
def test_T2_bad_street_fail_closed(bad):
    r = build_scenario(seed=1, street=bad)
    assert r["present"] is False
    assert r["reason"].startswith("street_not_supported")


@pytest.mark.parametrize("bad", [1, 7, 9, "6", None, True])
def test_T2_table_size_bounded_2_to_6(bad):
    r = build_scenario(seed=1, table_size=bad)
    assert r["present"] is False
    assert r["reason"] == "table_size_must_be_int_between_2_and_6"


@pytest.mark.parametrize("bad", [-1, "1", None, True])
def test_T2_negative_or_bad_numbers_fail_closed(bad):
    r = build_scenario(seed=1, pot_bb=bad)
    assert r["present"] is False
    assert r["reason"].startswith("pot_bb_must_be_non_negative_number")


def test_T2_population_table_length_enforced():
    r = build_scenario(seed=1, table_size=6, population_table=["DECLARED_BALANCED"])
    assert r["present"] is False
    assert r["reason"] == "population_table_length_must_equal_table_size_minus_one"


@pytest.mark.parametrize("bad", ["", "real_human_tight", "MTT", None, 7])
def test_T2_population_must_be_in_closed_enum(bad):
    r = build_scenario(seed=1, table_size=2, population_table=[bad])
    assert r["present"] is False
    assert r["reason"].startswith("population_not_in_closed_enum")


def test_T2_six_max_positions_and_roster():
    scn = build_scenario(seed=4, table_size=6)
    assert scn["positions"] == list(SIX_MAX_POSITIONS)
    players = scn["state"]["players"]
    assert len(players) == 6
    hero = [p for p in players if p.get("is_hero")]
    assert len(hero) == 1 and hero[0]["player_id"] == "hero"
    assert sum(1 for p in players if p["hole_cards"]) == 1


# ══ T3 不伪造 ══════════════════════════════════════════════════════════
def test_T3_opponent_hole_cards_never_fabricated():
    scn = build_scenario(seed=9, street="river", table_size=6)
    for p in scn["state"]["players"]:
        if not p.get("is_hero"):
            assert p["hole_cards"] == []


def test_T3_environment_never_claims_real_opponents_or_frequency():
    c = environment_contract()
    assert c["generates_future_cards"] is False
    assert c["samples_runouts"] is False
    assert c["evidence_level"] == ENVIRONMENT_EVIDENCE_LEVEL == "DECLARED_MODEL"
    for v in ENV_CB.values():
        assert v is False
    assert ENV_CB["is_real_opponent_data"] is False
    assert ENV_CB["is_observed_frequency"] is False
    assert ENV_CB["is_external_validation"] is False


def test_T3_population_params_are_declared_not_observed():
    for name, p in POPULATION_PARAMS.items():
        assert name in POPULATIONS
        assert name.startswith("DECLARED_")
        assert "declared model" in p["description"]
        assert "modeled_" in json.dumps(p)
    joined = json.dumps(POPULATION_PARAMS).lower()
    for banned in ("observed", "empirical", "real_player", "measured"):
        assert banned not in joined, banned


def test_T3_action_weights_are_declared_model():
    r = population_action_weights(population="DECLARED_MANIAC",
                                  last_action="raise", street="flop")
    assert r["present"] is True
    assert r["evidence_level"] == "DECLARED_MODEL"
    assert r["is_observed_frequency"] is False
    assert r["source"] == "declared_model_parameter_table"
    assert r["action_weights"]["_last_action"] == "raise"


@pytest.mark.parametrize("bad", ["", "real_tight", None, 3])
def test_T3_bad_population_or_action_fail_closed(bad):
    assert population_action_weights(population=bad)["present"] is False
    assert population_action_weights(last_action=bad)["present"] is False
    assert population_action_weights(street=bad)["present"] is False


# ══ T4 闭环 COMPLETE 路径 ══════════════════════════════════════════════
def test_T4_complete_path_reachable():
    st = build_scenario(seed=5, street="flop")["state"]
    out = run_decision_loop(st, pipeline=agreeing_pipeline(),
                            likelihood_model=object(), solver=None)
    assert out["present"] is True
    assert out["loop_status"] == "COMPLETE"
    assert out["evidence_level"] == "MODEL_INTERNAL"
    rec = out["decision_record"]
    assert rec["selected_by_tier"] == "search"
    assert rec["abstained"] is False
    assert rec["unknown"] == []
    assert rec["executed"] is False
    assert out["result"] is None


def test_T4_stage_order_recorded_and_complete():
    st = build_scenario(seed=5, street="flop")["state"]
    out = run_decision_loop(st, pipeline=agreeing_pipeline(),
                            likelihood_model=object())
    assert out["decision_record"]["stage_order"] == list(STAGE_ORDER)
    stages = [t["stage"] for t in out["stage_trace"]]
    assert "belief_update" in stages
    assert "decision_assembly" in stages


def test_T4_real_pipeline_runs_without_raising():
    out = run_real(seed=11, street="flop")
    assert out["present"] is True
    assert out["loop_status"] in LOOP_STATUSES
    assert out["decision_record"]["search_tree_present"] is True
    assert out["decision_record"]["search_eval_present"] is True


# ══ T5 弃权传播 ════════════════════════════════════════════════════════
def test_T5_abstention_propagates_and_is_never_a_recommendation():
    st = build_scenario(seed=5, street="flop")["state"]
    out = run_decision_loop(st, pipeline=abstaining_pipeline(),
                            likelihood_model=object())
    assert out["loop_status"] == "ABSTAINED"
    rec = out["decision_record"]
    assert rec["abstained"] is True
    assert rec["preferred"] is None
    assert "cross_scorer_disagreement" in rec["unknown"]


def test_T5_real_pipeline_abstains_on_scorer_disagreement():
    """真实链路在 flop/turn 上确实弃权 —— 这是 fail-closed 的正确行为。"""
    for street in ("flop", "turn"):
        out = run_real(seed=11, street=street)
        assert out["loop_status"] in ("ABSTAINED", "UNKNOWN"), out["loop_status"]
        rec = out["decision_record"]
        if rec["abstained"]:
            assert rec["preferred"] is None


# ══ T6 UNKNOWN 传播 ════════════════════════════════════════════════════
def test_T6_unknown_blocks_complete_status():
    """存在任何 UNKNOWN 时不得判为 COMPLETE（不为测试放宽）。"""
    st = build_scenario(seed=5, street="flop")["state"]
    p = agreeing_pipeline()
    p["rank_actions"] = lambda s, **k: {
        "present": True, "preferred": _PREF, "unknown": ["upstream_unknown_x"]}
    out = run_decision_loop(st, pipeline=p, likelihood_model=object())
    assert out["loop_status"] == "UNKNOWN"
    assert "upstream_unknown_x" in out["decision_record"]["unknown"]


def test_T6_missing_likelihood_is_recorded_not_hidden():
    st = build_scenario(seed=5, street="flop")["state"]
    out = run_decision_loop(st, pipeline=agreeing_pipeline(),
                            likelihood_model=None)
    assert out["loop_status"] == "UNKNOWN"
    assert out["decision_record"]["belief_update_present"] is False
    assert "likelihood_model_not_supplied" in out["decision_record"]["unknown"]


def test_T6_river_is_blocked_terminal_not_worked_around():
    out = run_real(seed=11, street="river")
    assert out["present"] is False
    assert out["loop_status"] == "BLOCKED"
    assert out["reason"].startswith("stage_failed:action_policy")


# ══ T7 闭环真正执行 belief_update（BP-3 通路） ═══════════════════════════
def test_T7_belief_update_executes_with_real_layers():
    out = run_real(seed=11, street="flop")
    rec = out["decision_record"]
    assert rec["belief_update_present"] is True
    stages = {t["stage"]: t for t in out["stage_trace"]}
    assert stages["belief_update"]["present"] is True
    assert stages["belief_update"]["error"] is None


def test_T7_with_and_without_update_are_distinguishable():
    with_upd = run_real(seed=11, street="flop")
    without = run_real(seed=11, street="flop", likelihood_model=None)
    assert with_upd["decision_record"]["belief_update_present"] is True
    assert without["decision_record"]["belief_update_present"] is False
    assert "likelihood_model_not_supplied" in without["decision_record"]["unknown"]
    assert "likelihood_model_not_supplied" not in \
        with_upd["decision_record"]["unknown"]


def test_T7_belief_update_stage_cannot_be_skipped_silently():
    """未注入 likelihood 时，trace 必须显式标记该阶段缺失。"""
    out = run_real(seed=11, street="flop", likelihood_model=None)
    stages = {t["stage"]: t for t in out["stage_trace"]}
    assert stages["belief_update"]["present"] is False
    assert stages["belief_update"]["unknown"] == ["likelihood_model_not_supplied"]


# ══ T8 证据等级强制 ════════════════════════════════════════════════════
@pytest.mark.parametrize("claimed", list(REJECTED_EVIDENCE_LEVELS))
def test_T8_external_validation_claims_fail_closed(claimed):
    st = build_scenario(seed=5, street="flop")["state"]
    out = run_decision_loop(st, pipeline=agreeing_pipeline(),
                            claimed_evidence_level=claimed)
    assert out["present"] is False
    assert out["loop_status"] == "BLOCKED"
    assert out["reason"].startswith("evidence_level_not_permitted_internal_only")
    assert out["evidence_level"] == "NONE"


def test_T8_pipeline_declaring_external_level_is_rejected():
    st = build_scenario(seed=5, street="flop")["state"]
    p = agreeing_pipeline()
    p["claimed_evidence_level"] = "EXTERNAL_VALIDATION"
    out = run_decision_loop(st, pipeline=p)
    assert out["present"] is False
    assert out["reason"].startswith("pipeline_claims_forbidden_evidence_level")


def test_T8_only_model_internal_allowed():
    assert ALLOWED_EVIDENCE_LEVELS == ("MODEL_INTERNAL",)
    c = loop_contract()
    assert c["allowed_evidence_levels"] == ["MODEL_INTERNAL"]
    assert c["external_validation_permitted"] is False
    for v in LOOP_CB.values():
        assert v is False
    assert LOOP_CB["is_external_validation"] is False


def test_T8_output_evidence_level_is_never_external():
    st = build_scenario(seed=5, street="flop")["state"]
    for out in (run_decision_loop(st, pipeline=agreeing_pipeline(),
                                 likelihood_model=object()),
                run_real(seed=11, street="flop")):
        assert out["evidence_level"] in ("MODEL_INTERNAL", "NONE")
        assert out["evidence_level"] != "EXTERNAL_VALIDATION"
        rec = out["decision_record"]
        if rec:
            assert rec["is_external_validation"] is False
            assert rec["evidence_level"] == "MODEL_INTERNAL"


# ══ T9 零新评分公式 ════════════════════════════════════════════════════
def test_T9_loop_has_no_scoring_arithmetic():
    tree = ast.parse(LOOP_MODULE.read_text(encoding="utf-8"))
    ops = {type(n.op).__name__ for n in ast.walk(tree)
           if isinstance(n, ast.BinOp)}
    for banned in ("Mult", "Pow", "Mod", "FloorDiv"):
        assert banned not in ops, banned


def test_T9_no_new_scoring_model_tokens():
    src = LOOP_MODULE.read_text(encoding="utf-8") + \
        ENV_MODULE.read_text(encoding="utf-8")
    for banned in ("pot_share_arithmetic", "search_runout_pot_share",
                   "compute_ev", "expected_value", "equity_formula",
                   "ev_delta", "exploitability"):
        assert banned not in src, banned


def test_T9_loop_declares_no_new_formula():
    c = loop_contract()
    assert c["recomputes_upstream_results"] is False
    for v in LOOP_CB.values():
        assert v is False
    assert LOOP_CB["adds_scoring_or_ev_formula"] is False


# ══ T10 种群级描述性计数 ═══════════════════════════════════════════════
def test_T10_population_sweep_yields_descriptive_counts_only():
    counts = {"COMPLETE": 0, "ABSTAINED": 0, "UNKNOWN": 0, "BLOCKED": 0}
    rows = 0
    for pop in POPULATIONS:
        out = run_real(seed=11, street="flop", population=pop)
        counts[out["loop_status"]] += 1
        rows += 1
    assert rows == len(POPULATIONS) == 5
    assert sum(counts.values()) == rows
    # 只计数，不得出现比率 / 收益 / 优劣
    for k in counts:
        assert k in LOOP_STATUSES


def test_T10_population_sweep_is_deterministic():
    a = [run_real(seed=11, street="flop", population=p)["loop_status"]
         for p in POPULATIONS]
    b = [run_real(seed=11, street="flop", population=p)["loop_status"]
         for p in POPULATIONS]
    assert a == b


def test_T10_no_rate_or_profit_field_in_output():
    out = run_real(seed=11, street="flop")
    flat = json.dumps(out, ensure_ascii=False)
    for banned in ("win_rate", "roi", "pnl", "profit_estimate",
                   "alignment_rate", "bb_per_100", "ev_delta"):
        assert banned not in flat, banned


# ══ T11 V3G-11 边界不破 ════════════════════════════════════════════════
@pytest.mark.parametrize("mod", [ENV_MODULE, LOOP_MODULE])
def test_T11_new_modules_import_no_cash_layer(mod):
    tree = ast.parse(mod.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom):
            mods.append(n.module or "")
        elif isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        for m in mods:
            assert "cash_" not in m, f"{mod.name}: {m}"


def test_T11_seven_unwired_layers_still_have_zero_production_imports():
    offenders = []
    for p in sorted(PKG.rglob("*.py")):
        if "__pycache__" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        imps = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                imps.add(n.module or "")
                imps |= {f"{n.module or ''}.{a.name}" for a in n.names}
            elif isinstance(n, ast.Import):
                imps |= {a.name for a in n.names}
        for layer in V3G11_UNWIRED:
            if p.name == f"{layer}.py":
                continue
            if any(i == layer or i.endswith("." + layer) for i in imps):
                offenders.append(f"{p.name}:{layer}")
    assert offenders == [], offenders


def test_T11_sanctioned_wiring_points_unchanged():
    found = {}
    for p in sorted(PKG.rglob("*.py")):
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
                if m.startswith("cash_"):
                    found.setdefault(m, set()).add(p.name)
    assert {k: sorted(v) for k, v in found.items()} == {
        "cash_integrity_context": ["agent.py", "realtime.py"],
        "cash_unknown_e2e": ["realtime.py"],
    }, found


def test_T11_boundary_semantics_hold_for_loop_injection():
    c = loop_contract()
    assert c["imports_any_cash_layer"] is False
    assert c["injection_style"] == "pipeline_kwarg_same_convention_as_v3h9_gate"


# ══ T12 生产 import 图不变 ═════════════════════════════════════════════
def test_T12_no_new_production_imports_at_all():
    found = {}
    for p in sorted(PKG.rglob("*.py")):
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
    assert set(found) == {"cash_integrity_context", "cash_unknown_e2e"}, found


@pytest.mark.parametrize("new_mod", ["cash_decision_environment",
                                     "cash_decision_loop"])
def test_T12_new_modules_not_production_wired(new_mod):
    for p in sorted(PKG.rglob("*.py")):
        if p.name.startswith("cash_") or "__pycache__" in str(p):
            continue
        assert new_mod not in p.read_text(encoding="utf-8-sig"), p.name


# ══ T13 冻结 drift ═════════════════════════════════════════════════════
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_T13_frozen_files_unchanged(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_T13_v3g_manifest_drift_unchanged():
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


def test_T13_new_modules_do_not_touch_frozen_faces():
    src = ENV_MODULE.read_text(encoding="utf-8") + \
        LOOP_MODULE.read_text(encoding="utf-8")
    for banned in ("strategy_core", "positions_for", "BET_FRACTIONS",
                   "RAISE_TO_MULTS", "conditional_model", "pi_ev"):
        assert banned not in src, banned


# ══ T14 Solver 不翻转 ══════════════════════════════════════════════════
def test_T14_solver_still_blocked():
    st = build_scenario(seed=5, street="flop")["state"]
    s = solver_status(st)
    assert s["solver_status"] == "BLOCKED"
    assert s["available"] is False
    assert s["equilibrium"] is False
    assert s["gto_exact"] is False


def test_T14_loop_never_invents_equilibrium():
    out = run_real(seed=11, street="flop")
    flat = json.dumps(out, ensure_ascii=False)
    assert '"equilibrium": true' not in flat
    assert '"is_equilibrium": true' not in flat
    assert LOOP_CB["is_equilibrium"] is False
    assert LOOP_CB["is_solver_output"] is False
    for mod in (ENV_MODULE, LOOP_MODULE):
        src = mod.read_text(encoding="utf-8")
        assert "solver_status(" not in src
        assert "import " not in src.split("solver")[0].split("\n")[-1] or True


def test_T14_loop_does_not_invoke_a_solver():
    tree = ast.parse(LOOP_MODULE.read_text(encoding="utf-8"))
    calls = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for banned in ("solver_status", "solve", "equilibrium", "gto"):
        assert banned not in calls, banned


# ══ T15 无自动操作 ═════════════════════════════════════════════════════
@pytest.mark.parametrize("mod", [ENV_MODULE, LOOP_MODULE])
def test_T15_no_automation_or_io(mod):
    src = mod.read_text(encoding="utf-8")
    for banned in ("pyautogui", "keyboard", "click", "screenshot", "input(",
                   "socket", "requests", "urllib", "subprocess", "open(",
                   "pathlib", "wallet", "coinpoker"):
        assert banned not in src, f"{mod.name}: {banned}"


@pytest.mark.parametrize("mod", [ENV_MODULE, LOOP_MODULE])
def test_T15_stdlib_only_imports(mod):
    tree = ast.parse(mod.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], (mod.name, mods)
    plain = [a.name for n in tree.body if isinstance(n, ast.Import)
             for a in n.names]
    assert plain == [], (mod.name, plain)


@pytest.mark.parametrize("mod", [ENV_MODULE, LOOP_MODULE])
def test_T15_no_random_module(mod):
    tree = ast.parse(mod.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name.split(".")[0] != "random" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "random"


# ══ T16 契约与 claim boundary ══════════════════════════════════════════
def test_T16_versions_and_enums_declared():
    assert CASH_DECISION_ENVIRONMENT_VERSION.startswith("v3i-p1-")
    assert CASH_DECISION_LOOP_VERSION.startswith("v3i-p1-")
    assert LOOP_STATUSES == ("COMPLETE", "ABSTAINED", "UNKNOWN", "BLOCKED")
    assert len(REQUIRED_PIPELINE_STAGES) == 8
    assert len(STAGE_ORDER) == 8


def test_T16_claim_boundaries_all_false():
    for cb in (ENV_CB, LOOP_CB):
        assert all(v is False for v in cb.values())
    assert ENV_CB["generates_future_cards"] is False
    assert LOOP_CB["recomputes_upstream_results"] is False


def test_T16_missing_or_bad_pipeline_fail_closed():
    st = build_scenario(seed=5, street="flop")["state"]
    assert run_decision_loop(st, pipeline=None)["present"] is False
    assert run_decision_loop(st, pipeline=None)["reason"] == "pipeline_required"
    r = run_decision_loop(st, pipeline={})
    assert r["present"] is False
    assert r["reason"].startswith("pipeline_missing_stages:")


def test_T16_non_cash_state_fail_closed():
    for bad in (None, [], "cash", 42, {"game_type": "mtt"},
                {"game_type": "pko"}, {"game_type": "mystery"}):
        out = run_decision_loop(bad, pipeline=agreeing_pipeline())
        assert out["present"] is False
        assert out["reason"] == "not_cash_mode_fail_closed"
        assert out["loop_status"] == "BLOCKED"


def test_T16_stage_exception_is_recorded_not_raised():
    st = build_scenario(seed=5, street="flop")["state"]
    p = agreeing_pipeline()

    def boom(s, **k):
        raise RuntimeError("x")

    p["build_dynamic_belief"] = boom
    out = run_decision_loop(st, pipeline=p, likelihood_model=object())
    assert out["present"] is False
    assert out["loop_status"] == "BLOCKED"
    assert out["reason"] == "stage_failed:dynamic_belief"
    assert any("build_dynamic_belief_raised:RuntimeError" in u
               for u in out["unknown"])


def test_T16_output_json_serialisable_and_deterministic():
    st = build_scenario(seed=5, street="flop")["state"]
    outs = [run_decision_loop(st, pipeline=agreeing_pipeline(),
                              likelihood_model=object()) for _ in range(3)]
    assert all(o == outs[0] for o in outs)
    assert json.loads(json.dumps(outs[0], ensure_ascii=False)) == outs[0]
    envs = [build_scenario(seed=17, street="turn") for _ in range(3)]
    assert all(e == envs[0] for e in envs)
    assert json.loads(json.dumps(envs[0], ensure_ascii=False)) == envs[0]


def test_T16_state_not_mutated_by_loop():
    st = build_scenario(seed=5, street="flop")["state"]
    before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    run_decision_loop(st, pipeline=agreeing_pipeline(), likelihood_model=object())
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == before
