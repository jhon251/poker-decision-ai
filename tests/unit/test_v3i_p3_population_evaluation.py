"""V3I-P3 — 多群体模型内评估能力测试。

覆盖授权要求的六项：
1. 三个群体（保守 / 激进 / 平衡）；
2. 同群体重复运行确定性；
3. 群体差异是否进入 belief / range / candidate；
4. 决策未改变时的阻塞原因定位；
5. UNKNOWN / 冲突（fail-closed）；
6. 全链路 provenance。

环境约束：本沙箱**禁止** `%TEMP%` 写入，`tmp_path` / `ctx` / `runtime`
等 fixture 会抛 `PermissionError [WinError 5]`，因此本文件**不使用**任何
需要临时目录的 fixture —— 这是环境限制，**不是**为了绕过失败而改测试。
"""
from __future__ import annotations

import copy
import json
import sys

import pytest

sys.path.insert(0, ".")

from poker_decision_ai.cash_population_evaluation import (
    CASH_POPULATION_EVALUATION_VERSION,
    CHAIN_ARTIFACTS,
    CLAIM_BOUNDARY,
    DECLARED_ACTIONS,
    DECLARED_POPULATIONS,
    DECISION_LEVEL_ARTIFACTS,
    FROZEN_ATTRIBUTION,
    MINIMUM_REQUIRED_POPULATIONS,
    compare_populations,
    determinism_probe,
    evaluate_population,
    likelihood_conflict_probe,
    population_evaluation_contract,
    population_observation_prior,
    population_propensity,
    profile_free_parameters,
)

SEED = 11
STREET = "flop"
LAST = "raise"
P3_POPULATIONS = ("DECLARED_TIGHT_PASSIVE", "DECLARED_BALANCED",
                  "DECLARED_AGGRESSIVE")

#: V3I-P1 的声明式群体参数表（**照抄**，本测试不 import 该模块以保持纯单元性）
POP_PARAMS = {
    "DECLARED_TIGHT_PASSIVE": {
        "modeled_preflop_continue_multiplier": 0.7,
        "modeled_postflop_aggression_multiplier": 0.6},
    "DECLARED_BALANCED": {
        "modeled_preflop_continue_multiplier": 1.0,
        "modeled_postflop_aggression_multiplier": 1.0},
    "DECLARED_LOOSE_PASSIVE": {
        "modeled_preflop_continue_multiplier": 1.4,
        "modeled_postflop_aggression_multiplier": 0.5},
    "DECLARED_AGGRESSIVE": {
        "modeled_preflop_continue_multiplier": 1.0,
        "modeled_postflop_aggression_multiplier": 1.8},
    "DECLARED_MANIAC": {
        "modeled_preflop_continue_multiplier": 1.8,
        "modeled_postflop_aggression_multiplier": 2.6},
}


def _prior(pop, street=STREET, last=LAST, obs=1.0):
    return population_observation_prior(
        population=pop, street=street, last_action=last,
        population_params=POP_PARAMS, observation_factor=obs)


# ---------------------------------------------------------------------------
# 合成链路：只用于 compare_populations 的**纯逻辑**测试（不跑真实管线）
# ---------------------------------------------------------------------------
def _synthetic(entries):
    """entries: list of (population, {artifact: fingerprint})"""
    out = []
    for pop, fps in entries:
        chain = {a: {"present": True,
                     "fingerprint": fps.get(a, "same")} for a in CHAIN_ARTIFACTS}
        out.append({"present": True, "population": pop, "chain": chain})
    return out


def _real_pipeline():
    """跑一次真实管线（3 群体），结果缓存供多个断言复用。

    单次群体评估约 20–80 秒，故**只跑一次**；链路比较逻辑另用合成链路覆盖。
    """
    from poker_decision_ai.cash_decision_environment import (
        build_scenario, POPULATION_PARAMS)
    from poker_decision_ai.cash_decision_loop import run_decision_loop
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_multistreet import next_street_transition
    from poker_decision_ai.cash_solver_interface import solver_status
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    from poker_decision_ai.cash_strategy_action_policy import build_action_policy
    from poker_decision_ai.cash_strategy_dynamic_belief import build_dynamic_belief
    from poker_decision_ai.cash_strategy_belief_update import (
        update_belief_with_likelihood)
    from poker_decision_ai.cash_strategy_action_ranking import rank_actions
    from poker_decision_ai.cash_strategy_belief_conditioned_eval import (
        evaluate_belief_conditioned)
    from poker_decision_ai.cash_strategy_search_tree import build_search_tree
    from poker_decision_ai.cash_strategy_search_eval import evaluate_search
    from poker_decision_ai.cash_strategy_decision_assembly import assemble_decision
    from poker_decision_ai.cash_strategy_likelihood_model import (
        action_likelihood_model, validate_parameters)

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
    scenario = build_scenario(seed=SEED, street=STREET)
    state = scenario["state"]
    ctx = cash_integrity_context(state)
    om = build_opponent_model(state)
    rm = build_range_model(state, opponent_model=om)
    at = build_action_tree(state)
    tr = next_street_transition(state)
    sv = solver_status(state)

    outs = []
    for pop in P3_POPULATIONS:
        outs.append(evaluate_population(
            scenario=scenario, population=pop, street=STREET, last_action=LAST,
            population_params=POPULATION_PARAMS, pipeline=pipeline,
            integrity_context=ctx, opponent_model=om, range_model=rm,
            action_tree=at, transition=tr, solver=sv, budget=DEFAULT_BUDGET,
            policy=None, run_decision_loop=run_decision_loop,
            action_likelihood_model=action_likelihood_model,
            validate_parameters=validate_parameters))
    return outs


_CACHE: dict = {}


def _cached():
    if "outs" not in _CACHE:
        _CACHE["outs"] = _real_pipeline()
        _CACHE["cmp"] = compare_populations(evaluations=_CACHE["outs"])
    return _CACHE["outs"], _CACHE["cmp"]


# ===========================================================================
# T1. 契约 / 边界
# ===========================================================================
def test_t1_contract_identifiers_and_claim_boundary():
    c = population_evaluation_contract()
    assert c["version"] == CASH_POPULATION_EVALUATION_VERSION
    assert tuple(c["declared_populations"]) == DECLARED_POPULATIONS
    assert set(MINIMUM_REQUIRED_POPULATIONS) <= set(DECLARED_POPULATIONS)
    assert c["per_combo_likelihood"] is False
    assert c["modifies_frozen_layers"] is False
    assert c["wired_to_production"] is False
    assert c["evidence_level"] == "DECLARED_MODEL"


def test_t1_claim_boundary_all_false_no_external_claims():
    for k, v in CLAIM_BOUNDARY.items():
        assert v is False, k
    for flag in ("is_real_human_validation", "is_gto", "is_solver_output",
                 "is_profit_evidence", "is_external_validation",
                 "wired_to_production", "modifies_frozen_layers"):
        assert flag in CLAIM_BOUNDARY


def test_t1_chain_covers_required_layers():
    for layer in ("propensity", "likelihood_multipliers", "dynamic_belief",
                  "belief_update", "range_model", "ranking",
                  "belief_conditioned", "search_eval", "assembly"):
        assert layer in CHAIN_ARTIFACTS
    assert set(DECISION_LEVEL_ARTIFACTS) <= set(CHAIN_ARTIFACTS)


def test_t1_module_imports_no_cash_layers():
    """本层不得 import 任何 poker_decision_ai 内部层（含 cash_integrity_context）。"""
    import ast
    import pathlib
    src = pathlib.Path("poker_decision_ai/cash_population_evaluation.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert mods <= {"__future__", "typing", "json", "hashlib"}, mods
    assert not any(m.startswith("poker_decision_ai") for m in mods)


def test_t1_no_random_module_on_path():
    import pathlib
    src = pathlib.Path("poker_decision_ai/cash_population_evaluation.py").read_text(
        encoding="utf-8")
    assert "import random" not in src
    assert "random." not in src


# ===========================================================================
# T2. 三个群体：倾向被区分且可解释
# ===========================================================================
def test_t2_three_populations_have_distinct_propensity_sets():
    sets = {}
    for pop in P3_POPULATIONS:
        p = population_propensity(population=pop, street=STREET,
                                  population_params=POP_PARAMS)
        assert p["present"] is True
        sets[pop] = tuple(sorted(p["propensity"].items()))
    assert len(set(sets.values())) == 3, sets


def test_t2_balanced_is_neutral_baseline():
    p = population_propensity(population="DECLARED_BALANCED", street=STREET,
                              population_params=POP_PARAMS)
    assert set(p["propensity"].values()) == {1.0}
    assert p["distinguishes_actions"] is False
    pr = _prior("DECLARED_BALANCED")
    assert set(pr["likelihood_multiplier_by_action"].values()) == {1.0}


def test_t2_propensity_distinguishes_actions_not_one_scalar():
    """P3 新增点：动作**被区分**（V3I-P1 helper 给所有非 fold/check 同一个数）。"""
    p = population_propensity(population="DECLARED_LOOSE_PASSIVE", street=STREET,
                              population_params=POP_PARAMS)
    assert p["distinguishes_actions"] is True
    assert p["propensity"]["call"] == pytest.approx(1.4)
    assert p["propensity"]["bet"] == pytest.approx(0.5)
    assert p["propensity"]["fold"] == pytest.approx(1.0)


def test_t2_street_selects_declared_dimension():
    flop = population_propensity(population="DECLARED_MANIAC", street="flop",
                                 population_params=POP_PARAMS)
    pre = population_propensity(population="DECLARED_MANIAC", street="preflop",
                                population_params=POP_PARAMS)
    assert flop["applied_aggression_source"] == "postflop_aggression_multiplier"
    assert pre["applied_aggression_source"] == "preflop_continue_multiplier"
    assert flop["propensity"]["bet"] == pytest.approx(2.6)
    assert pre["propensity"]["bet"] == pytest.approx(1.8)


def test_t2_all_declared_populations_schema_valid():
    from poker_decision_ai.cash_strategy_likelihood_model import (
        action_likelihood_model)
    for pop in DECLARED_POPULATIONS:
        pr = _prior(pop)
        lm = action_likelihood_model(pr["likelihood_parameters"])
        assert lm["applied"] is True, pop
        assert lm["invalid"] == [] and lm["conflicts"] == [], pop
        assert len(lm["parameters"]) == len(DECLARED_ACTIONS), pop
        assert all(p["evidence_level"] == "DECLARED_MODEL"
                   for p in lm["parameters"])
        assert all(p["prior_only"] is False for p in lm["parameters"])


def test_t2_maniac_scales_above_tight_passive_on_non_baseline_actions():
    """fold/check 是**基线动作**（两群体都为 1.0），只有其余动作随群体变化。"""
    m = _prior("DECLARED_MANIAC")["likelihood_multiplier_by_action"]
    t = _prior("DECLARED_TIGHT_PASSIVE")["likelihood_multiplier_by_action"]
    for a in ("fold", "check"):
        assert m[a] == t[a] == pytest.approx(1.0), a
    for a in ("call", "bet", "raise", "all_in"):
        assert m[a] > t[a], a


# ===========================================================================
# T3. 同群体重复运行确定性
# ===========================================================================
def test_t3_determinism_probe_synthetic_identical():
    runs = [_synthetic([("P", {"propensity": "x", "assembly": "y"})])
            for _ in range(3)]
    # 每个 run 是一个 list[eval]；展平为 3 个相同 eval
    flat = [r[0] for r in runs]
    d = determinism_probe(evaluations_by_repeat={"P": flat})
    assert d["all_populations_deterministic"] is True
    assert d["per_population"]["P"]["n_repeats"] == 3
    assert d["per_population"]["P"]["changed_artifacts"] == []


def test_t3_determinism_probe_detects_instability():
    """同名重复运行必须**累积**比较，不得按名覆盖（覆盖会使其空转）。"""
    a = _synthetic([("P", {"assembly": "A"})])[0]
    b = _synthetic([("P", {"assembly": "B"})])[0]
    d = determinism_probe(evaluations_by_repeat={"P": [a, b]})
    assert d["all_populations_deterministic"] is False
    assert d["per_population"]["P"]["changed_artifacts"] == ["assembly"]
    assert d["per_population"]["P"]["n_repeats"] == 2


def test_t3_compare_accumulates_repeats_of_same_population():
    a = _synthetic([("P", {"assembly": "A"})])[0]
    b = _synthetic([("P", {"assembly": "B"})])[0]
    c = _synthetic([("P", {"assembly": "A"})])[0]
    cmp = compare_populations(evaluations=[a, b, c])
    assert cmp["n_populations"] == 1
    assert cmp["n_evaluations"] == 3
    assert cmp["repeats_per_population"] == {"P": 3}
    assert cmp["per_artifact"]["assembly"]["n_distinct"] == 2
    assert cmp["per_artifact"]["assembly"]["changed"] is True


def test_t3_same_population_repeat_identical_on_real_pipeline():
    """真实管线：同群体重复运行 → 全链条指纹**逐位相同**。"""
    outs, _ = _cached()
    from poker_decision_ai.cash_decision_environment import (
        build_scenario, POPULATION_PARAMS)
    from poker_decision_ai.cash_decision_loop import run_decision_loop
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_multistreet import next_street_transition
    from poker_decision_ai.cash_solver_interface import solver_status
    from poker_decision_ai.cash_strategy_search_tree import DEFAULT_BUDGET
    from poker_decision_ai.cash_strategy_likelihood_model import (
        action_likelihood_model, validate_parameters)
    first = outs[0]
    scenario = build_scenario(seed=SEED, street=STREET)
    state = scenario["state"]
    again = evaluate_population(
        scenario=scenario, population=first["population"], street=STREET,
        last_action=LAST, population_params=POPULATION_PARAMS,
        pipeline=_PIPE_FOR_REPEAT, integrity_context=cash_integrity_context(state),
        opponent_model=build_opponent_model(state),
        range_model=build_range_model(state,
                                      opponent_model=build_opponent_model(state)),
        action_tree=build_action_tree(state),
        transition=next_street_transition(state), solver=solver_status(state),
        budget=DEFAULT_BUDGET, policy=None, run_decision_loop=run_decision_loop,
        action_likelihood_model=action_likelihood_model,
        validate_parameters=validate_parameters)
    assert again["present"] is True
    for art in CHAIN_ARTIFACTS:
        assert again["chain"][art]["fingerprint"] == \
            first["chain"][art]["fingerprint"], art
    assert again["loop_status"] == first["loop_status"]


# ===========================================================================
# T4. 群体差异是否进入 belief / range / candidate（真实管线）
# ===========================================================================
def test_t4_observation_prior_differs_across_populations():
    outs, cmp = _cached()
    assert cmp["observation_prior_changed"] is True
    assert cmp["per_artifact"]["propensity"]["changed"] is True
    assert cmp["per_artifact"]["likelihood_multipliers"]["changed"] is True
    assert cmp["per_artifact"]["propensity"]["n_distinct"] == 3


def test_t4_belief_shape_is_population_invariant():
    """实测结论：posterior **形状**不随群体改变（冻结层标量语义）。"""
    outs, cmp = _cached()
    assert cmp["belief_changed"] is False
    fps = {o["chain"]["dynamic_belief"]["fingerprint"] for o in outs}
    assert len(fps) == 1


def test_t4_belief_scalar_mass_is_population_dependent():
    """mass_posterior 随群体**单调**变化，且严格等于「移除已知牌后质量 × 标量」（可复算）。

    注意：冻结 V3H-4 的 `mass_prior` 是**移除已知牌之前**的总量，
    被乘的基数是 `mass_prior - mass_removed_by_known_cards`（源码 L236-237, L254）。
    """
    outs, _ = _cached()
    by = {}
    for o in outs:
        m = o["chain"]["dynamic_belief"]["mass"]["opp1"]
        mult = o["observation_prior"]["likelihood_multiplier_by_action"]["raise"]
        # 冻结 V3H-4：likelihood_multiplier 就是被乘到 mass 上的那个标量
        assert m["likelihood_multiplier"] == pytest.approx(mult, rel=1e-6)
        pre_scale = m["mass_prior"] - m["mass_removed_by_known_cards"]
        # 容差只为吸收冻结层对 mass_prior/mass_removed 的 round(...,6) 量化
        assert m["mass_posterior"] == pytest.approx(pre_scale * mult, abs=2e-6)
        assert m["likelihood_source"] == "explicit_action_weights"
        by[o["population"]] = m["mass_posterior"]
    assert by["DECLARED_TIGHT_PASSIVE"] < by["DECLARED_BALANCED"]
    assert by["DECLARED_BALANCED"] < by["DECLARED_AGGRESSIVE"]


def test_t4_mass_ratio_equals_population_multiplier_exactly():
    """最强可解释性证据：相对基线的倍数**恰好等于**声明的群体乘数。"""
    outs, _ = _cached()
    post = {o["population"]: o["chain"]["dynamic_belief"]["mass"]["opp1"]
            ["mass_posterior"] for o in outs}
    mult = {o["population"]: o["observation_prior"]
            ["likelihood_multiplier_by_action"]["raise"] for o in outs}
    base = post["DECLARED_BALANCED"]
    for pop in post:
        assert post[pop] / base == pytest.approx(mult[pop], rel=1e-6), pop


def test_t4_mass_prior_is_population_invariant():
    """只有**标量倍数**随群体变；先验质量与已知牌移除量都不变（形状不变）。"""
    outs, _ = _cached()
    priors = {o["chain"]["dynamic_belief"]["mass"]["opp1"]["mass_prior"]
              for o in outs}
    removed = {o["chain"]["dynamic_belief"]["mass"]["opp1"]
               ["mass_removed_by_known_cards"] for o in outs}
    assert len(priors) == 1 and len(removed) == 1
    assert next(iter(priors)) == pytest.approx(52.750001)
    assert next(iter(removed)) == pytest.approx(5.020834)


def test_t4_range_model_is_population_invariant():
    outs, cmp = _cached()
    assert cmp["range_changed"] is False
    assert len({o["chain"]["range_model"]["fingerprint"] for o in outs}) == 1


def test_t4_candidate_ranking_is_population_invariant():
    outs, cmp = _cached()
    assert cmp["candidate_changed"] is False
    for art in ("ranking", "belief_conditioned", "search_eval"):
        assert len({o["chain"][art]["fingerprint"] for o in outs}) == 1, art


def test_t4_combo_weights_byte_identical_across_populations():
    """最强证据：posterior 的 169 个 combo 权重集合**完全相同**。"""
    outs, _ = _cached()
    values = {tuple(o["combo_weight_values"]) for o in outs}
    assert len(values) == 1
    assert {o["n_combo_weights"] for o in outs} == {169}


def test_t4_decision_is_population_invariant():
    outs, cmp = _cached()
    assert cmp["decision_changed"] is False
    prefs = {json.dumps(o["chain"]["assembly"]["preferred"]) for o in outs}
    assert prefs == {"null"}
    assert {o["chain"]["assembly"]["abstained"] for o in outs} == {True}
    assert {o["loop_status"] for o in outs} == {"ABSTAINED"}


def test_t4_difference_scope_is_prior_and_mass_only():
    _, cmp = _cached()
    assert cmp["difference_scope"] == "observation_prior_and_mass"
    assert cmp["explainable_strategy_difference"] is True
    assert "search_eval" in cmp["invariant_artifacts"]
    assert "assembly" in cmp["invariant_artifacts"]


def test_t4_decision_margin_is_zero_when_population_scales_belief():
    """决策**不**依赖 belief 标量：改变标量的唯一通道不触及任何打分。"""
    outs, _ = _cached()
    scores = {json.dumps(o["chain"]["ranking"]["preferred"], sort_keys=True)
              for o in outs}
    assert len(scores) == 1
    s = json.loads(next(iter(scores)))
    assert s["action"] == "bet" and s["amount_bb"] == pytest.approx(100.0)


# ===========================================================================
# T5. 决策未改变 → 阻塞原因定位（必须指名冻结层）
# ===========================================================================
def test_t5_blocked_codes_present_and_localised():
    _, cmp = _cached()
    assert cmp["decision_changed"] is False
    assert cmp["blocked_at"], "unchanged decision must be localised"
    codes = cmp["blocked_codes"]
    assert "belief_shape_invariant" in codes
    for b in cmp["blocked_at"]:
        assert b["code"] in FROZEN_ATTRIBUTION
        assert b["detail"] == FROZEN_ATTRIBUTION[b["code"]]
        assert b["artifact"] in CHAIN_ARTIFACTS


def test_t5_frozen_attribution_names_exact_modules():
    for code, detail in FROZEN_ATTRIBUTION.items():
        assert isinstance(detail, str) and len(detail) > 30, code
    joined = " ".join(FROZEN_ATTRIBUTION.values())
    for mod in ("cash_strategy_dynamic_belief",
                "cash_strategy_belief_update",
                "cash_strategy_action_ranking",
                "cash_strategy_decision_assembly"):
        assert mod in joined, mod


def test_t5_unchanged_decision_never_reported_as_strategy_difference():
    """诚实性：决策未变时不得宣称决策级差异。"""
    _, cmp = _cached()
    assert cmp["decision_changed"] is False
    assert cmp["difference_scope"] != "decision_level"


def test_t5_compare_requires_two_present_evaluations():
    out = compare_populations(evaluations=[_synthetic([("P", {})])[0]])
    assert out["present"] is False
    assert out["reason"] == "at_least_two_present_evaluations_required"


def test_t5_compare_synthetic_detects_upstream_change_only():
    evals = _synthetic([
        ("A", {"propensity": "pa", "belief_update": "ba"}),
        ("B", {"propensity": "pb", "belief_update": "bb"}),
    ])
    cmp = compare_populations(evaluations=evals)
    assert cmp["per_artifact"]["propensity"]["changed"] is True
    assert cmp["per_artifact"]["belief_update"]["changed"] is True
    assert cmp["per_artifact"]["assembly"]["changed"] is False
    assert cmp["decision_changed"] is False


# ===========================================================================
# T6. 全链路 provenance（真实管线）
# ===========================================================================
def test_t6_chain_present_flags_on_real_pipeline():
    outs, _ = _cached()
    for o in outs:
        assert o["present"] is True
        for art in ("propensity", "likelihood_multipliers", "dynamic_belief",
                    "belief_update", "range_model", "ranking",
                    "belief_conditioned", "search_eval", "assembly"):
            assert o["chain"][art]["present"] is True, (o["population"], art)
            assert isinstance(o["chain"][art]["fingerprint"], str)
            assert len(o["chain"][art]["fingerprint"]) == 16


def test_t6_state_is_shared_across_populations():
    outs, _ = _cached()
    shared = {json.dumps(o["state_shared"], sort_keys=True) for o in outs}
    assert len(shared) == 1
    s = json.loads(next(iter(shared)))
    assert s["board"] == ["8h", "Jd", "Ac"]
    assert s["n_players"] == 6
    assert s["hero_id"] == "hero"


def test_t6_provenance_is_model_internal_only():
    outs, _ = _cached()
    for o in outs:
        assert o["evidence_level"] == "DECLARED_MODEL"
        assert o["claimed_evidence_level"] == "MODEL_INTERNAL"
        assert o["executed"] is False
        assert o["result"] is None
        assert o["wired_to_production"] is False
        assert all(v is False for v in o["claim_boundary"].values())


def test_t6_likelihood_rows_carry_full_provenance():
    pr = _prior("DECLARED_AGGRESSIVE")
    for row in pr["likelihood_parameters"]:
        assert set(("action", "street", "multiplier", "source",
                    "evidence_level", "provenance", "uncertainty")) <= set(row)
        assert row["source"] == "declared_population_parameter_table"
        assert row["evidence_level"] == "DECLARED_MODEL"
        prov = row["provenance"]
        assert prov["population"] == "DECLARED_AGGRESSIVE"
        assert prov["derivation"] == \
            "propensity_ratio_vs_baseline_x_observation_factor"
        assert row["uncertainty"]["quantified"] is False


def test_t6_delivery_diagnostic_reports_gates_honestly():
    outs, _ = _cached()
    for o in outs:
        d = o["population_delivery"]
        assert d["state_action_history_length"] == 0
        assert d["gate1_history_present"] is False
        assert d["gate2_parameters_profiled"] is True
        assert d["per_action_multipliers_applied"] == 0
        assert d["population_reaches_belief_update"] is False
        assert d["path"] == "population_not_delivered_to_belief_update"


def test_t6_solver_remains_blocked_and_unused():
    """CASH Solver 仍 BLOCKED：本层不得把它当作可用来源。"""
    from poker_decision_ai.cash_solver_interface import solver_status
    from poker_decision_ai.cash_decision_environment import build_scenario
    st = build_scenario(seed=SEED, street=STREET)["state"]
    s = solver_status(st)
    assert s["solver_status"] == "BLOCKED"
    assert s["available"] is False
    assert s["gto_exact"] is False
    assert s["equilibrium"] is False


# ===========================================================================
# T7. UNKNOWN / 冲突 fail-closed
# ===========================================================================
def test_t7_unknown_population_fail_closed():
    p = population_propensity(population="DECLARED_NOT_A_POPULATION",
                              street=STREET, population_params=POP_PARAMS)
    assert p["present"] is False
    assert "population_not_in_closed_enum" in p["reason"]
    assert p["executed"] is False and p["result"] is None


def test_t7_unknown_street_and_action_fail_closed():
    assert population_propensity(population="DECLARED_BALANCED",
                                 street="not_a_street",
                                 population_params=POP_PARAMS)["present"] is False
    assert _prior("DECLARED_BALANCED", last="not_an_action")["present"] is False


def test_t7_missing_or_malformed_params_fail_closed():
    for bad in (None, {}, "nope", 5):
        r = population_propensity(population="DECLARED_BALANCED", street=STREET,
                                  population_params=bad)
        assert r["present"] is False
        assert r["executed"] is False
    incomplete = {"DECLARED_BALANCED": {"modeled_postflop_aggression_multiplier": 1.0}}
    r = population_propensity(population="DECLARED_BALANCED", street=STREET,
                              population_params=incomplete)
    assert r["present"] is False
    assert "row_incomplete" in r["reason"]


def test_t7_negative_observation_factor_fail_closed():
    r = _prior("DECLARED_BALANCED", obs=-0.1)
    assert r["present"] is False
    assert r["reason"] == "observation_factor_invalid"


def test_t7_conflict_is_rejected_not_silently_resolved():
    from poker_decision_ai.cash_strategy_likelihood_model import (
        action_likelihood_model)
    probe = likelihood_conflict_probe(street=STREET,
                                     population_params=POP_PARAMS)
    assert probe["conflict_injected"] is True
    lm = action_likelihood_model([probe["first_parameter"],
                                  probe["duplicate_parameter"]])
    assert len(lm["conflicts"]) == 1
    assert lm["conflicts"][0]["reason"] == "conflicting_multiplier_for_same_key"
    assert lm["applied"] is False
    assert lm["prior_only"] is True
    assert lm["parameters"] == []
    assert any(u.startswith("conflicting_parameters") for u in lm["unknown"])


def test_t7_invalid_multipliers_never_silently_defaulted():
    from poker_decision_ai.cash_strategy_likelihood_model import (
        action_likelihood_model)
    base = {"action": "raise", "street": STREET, "source": "s",
            "evidence_level": "DECLARED_MODEL", "provenance": {},
            "uncertainty": {}}
    for bad, reason in ((-1.0, "multiplier_negative"),
                        (1000.0, "multiplier_out_of_range"),
                        ("3.0", "multiplier_not_numeric")):
        lm = action_likelihood_model([dict(base, multiplier=bad)])
        assert lm["applied"] is False
        assert [i["reason"] for i in lm["invalid"]] == [reason]
    lm = action_likelihood_model([{"action": "raise", "street": STREET,
                                   "multiplier": 1.0}])
    assert [i["reason"] for i in lm["invalid"]] == ["missing_required_fields"]


def test_t7_probe_unknown_inputs_all_fail_closed():
    probe = likelihood_conflict_probe(street=STREET,
                                     population_params=POP_PARAMS)
    for k, v in probe["unknown_inputs"].items():
        assert v["present"] is False, k
        assert v["reason"], k
        assert v["executed"] is False and v["result"] is None


def test_t7_evaluate_population_fails_closed_on_bad_inputs():
    good = _synthetic([("P", {})])[0]
    assert evaluate_population(
        scenario=None, population="DECLARED_BALANCED", street=STREET,
        last_action=LAST, population_params=POP_PARAMS, pipeline={},
        integrity_context=None, opponent_model=None, range_model=None,
        action_tree=None, transition=None, solver=None, budget=None,
        policy=None, run_decision_loop=None,
        action_likelihood_model=None)["reason"] == "scenario_required"
    assert evaluate_population(
        scenario={"state": {}}, population="NOPE", street=STREET,
        last_action=LAST, population_params=POP_PARAMS, pipeline={},
        integrity_context=None, opponent_model=None, range_model=None,
        action_tree=None, transition=None, solver=None, budget=None,
        policy=None, run_decision_loop=None,
        action_likelihood_model=None)["reason"].startswith(
            "population_not_in_closed_enum")
    assert good is not None


# ===========================================================================
# T8. profile 维度：冻结层不读取（可复算的机制证据）
# ===========================================================================
def test_t8_profile_tagged_rows_are_not_matched_by_frozen_resolver():
    from poker_decision_ai.cash_strategy_likelihood_model import (
        action_likelihood_model, resolve_multiplier)
    lm = action_likelihood_model(_prior("DECLARED_AGGRESSIVE")
                                 ["likelihood_parameters"])
    for a in DECLARED_ACTIONS:
        r = resolve_multiplier(lm["parameters"], action=a, street=STREET,
                               size_bucket=None, profile=None)
        assert r["matched"] is False, a
        assert r["reason"] == "unmatched_no_default_applied"


def test_t8_profile_free_variant_matches_at_action_street_tier():
    from poker_decision_ai.cash_strategy_likelihood_model import (
        action_likelihood_model, resolve_multiplier)
    params = profile_free_parameters(_prior("DECLARED_AGGRESSIVE"))
    lm = action_likelihood_model(params)
    assert lm["applied"] is True
    r = resolve_multiplier(lm["parameters"], action="raise", street=STREET,
                           size_bucket=None, profile=None)
    assert r["matched"] is True
    assert r["tier"] == "action+street"
    assert r["multiplier"] == pytest.approx(1.8)


def test_t8_profile_free_variant_preserves_values_and_attribution():
    pr = _prior("DECLARED_MANIAC")
    free = profile_free_parameters(pr)
    assert len(free) == len(pr["likelihood_parameters"])
    for a, b in zip(free, pr["likelihood_parameters"]):
        assert a["multiplier"] == pytest.approx(b["multiplier"])
        assert "profile" not in a
        assert a["provenance"]["population_attribution"] == "DECLARED_MANIAC"
        assert a["provenance"]["profile_tag_removed"] is True


def test_t8_profile_free_parameters_tolerates_bad_input():
    assert profile_free_parameters(None) == []
    assert profile_free_parameters({}) == []
    assert profile_free_parameters({"likelihood_parameters": "nope"}) == []


def test_t8_state_override_does_not_mutate_caller_state():
    """提供声明式历史的对照必须**不修改**调用方 state。"""
    outs, _ = _cached()
    o = outs[0]
    st = {"game_type": "cash", "street": STREET, "pot": 1.5, "bb": 1.0,
          "hero_id": "hero", "players": [], "action_history": []}
    snapshot = copy.deepcopy(st)
    r = evaluate_population(
        scenario={"state": {"game_type": "cash"}}, population=o["population"],
        street=STREET, last_action=LAST, population_params=POP_PARAMS,
        pipeline={}, integrity_context=None, opponent_model=None,
        range_model=None, action_tree=None, transition=None, solver=None,
        budget=None, policy=None, run_decision_loop=None,
        action_likelihood_model=None, state_override=st)
    assert r["present"] is False           # pipeline 缺失 → fail-closed
    assert st == snapshot, "caller state must not be mutated"


def test_t8_observation_factor_scales_multipliers_linearly():
    a = _prior("DECLARED_AGGRESSIVE", obs=1.0)["likelihood_multiplier_by_action"]
    b = _prior("DECLARED_AGGRESSIVE", obs=2.0)["likelihood_multiplier_by_action"]
    for act in DECLARED_ACTIONS:
        assert b[act] == pytest.approx(2.0 * a[act], rel=1e-6)


def test_t8_observation_factor_zero_is_explicit_and_bounded():
    z = _prior("DECLARED_AGGRESSIVE", obs=0.0)["likelihood_multiplier_by_action"]
    assert set(z.values()) == {0.0}


#: 供 T3 复用（避免测试间重复构造 pipeline 字典）
from poker_decision_ai.cash_strategy_action_policy import build_action_policy as _bap
from poker_decision_ai.cash_strategy_dynamic_belief import (
    build_dynamic_belief as _bdb)
from poker_decision_ai.cash_strategy_belief_update import (
    update_belief_with_likelihood as _ubu)
from poker_decision_ai.cash_strategy_action_ranking import rank_actions as _ra
from poker_decision_ai.cash_strategy_belief_conditioned_eval import (
    evaluate_belief_conditioned as _ebc)
from poker_decision_ai.cash_strategy_search_tree import build_search_tree as _bst
from poker_decision_ai.cash_strategy_search_eval import evaluate_search as _es
from poker_decision_ai.cash_strategy_decision_assembly import (
    assemble_decision as _ad)

_PIPE_FOR_REPEAT = {
    "build_action_policy": _bap,
    "build_dynamic_belief": _bdb,
    "update_belief": _ubu,
    "rank_actions": _ra,
    "evaluate_belief_conditioned": _ebc,
    "build_search_tree": _bst,
    "evaluate_search": _es,
    "assemble_decision": _ad,
}
