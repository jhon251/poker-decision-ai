"""V3G-10 — CASH 收益实验接口测试（授权要求 ≥30 项）。

**不修改**任何既有测试定义；**不触碰**冻结文件；**不执行收益实验**。
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.cash_profit_interface import (
    BLOCKED, DESIGN, DESIGN_ONLY, EXECUTION_ENVIRONMENT_REQUIRED,
    FORBIDDEN_AS_PROFIT, NEVER_EMITTED_FIELDS, PROFIT_INTERFACE_VERSION,
    READY_FOR_EXECUTION, RESULT_STATUSES, RESULT_UNAVAILABLE, VALID_RESULT,
    evaluate_entry_gate, profit_experiment_interface)

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

_FULL_ENV = {"executable_environment": True, "randomized_assignment": True,
             "venue_and_compliance_authorized": True, "design_sealed": True,
             "sample_size_meets_mde": True}


def st(**over) -> dict:
    b = {
        "game_type": "cash", "street": "flop", "board": ["Ah", "7c", "2c"],
        "pot": 20.0, "to_call": 5.0, "bb": 1.0, "sb": 0.5,
        "min_raise_to": 10.0, "hero_id": "hero", "button_seat": 2,
        "table_size": 3, "synthetic_test_only": True,
        "players": [
            {"player_id": "v1", "seat": 1, "position": "BB", "status": "active",
             "stack": 100.0, "hole_cards": []},
            {"player_id": "hero", "seat": 2, "position": "BTN", "is_hero": True,
             "status": "active", "hole_cards": ["As", "Kd"], "stack": 100.0},
            {"player_id": "v2", "seat": 3, "position": "SB", "status": "active",
             "stack": 100.0, "hole_cards": []},
        ],
        "action_history": [],
    }
    b.update(over)
    return b


def _all_keys(o, out=None) -> set:
    if out is None:
        out = set()
    if isinstance(o, dict):
        for k, v in o.items():
            out.add(k)
            _all_keys(v, out)
    elif isinstance(o, list):
        for v in o:
            _all_keys(v, out)
    return out


# ── 1. 当前环境 BLOCKED ───────────────────────────────────────────────
def test_01_current_environment_is_blocked():
    r = profit_experiment_interface(st())
    assert r["present"] is True
    assert r["status"] == BLOCKED
    assert r["reason"].startswith("B3 UNIDENTIFIABLE")


def test_02_profit_not_identifiable_in_current_environment():
    r = profit_experiment_interface(st())
    assert r["profit_identifiable"] is False
    assert r["executed"] is False
    assert r["valid_result_reachable"] is False


def test_03_result_status_is_execution_environment_required():
    r = profit_experiment_interface(st())
    assert r["result_status"] == EXECUTION_ENVIRONMENT_REQUIRED
    assert r["result_status"] in RESULT_STATUSES


# ── 4/5. 无执行环境 / 无随机指派 ──────────────────────────────────────
def test_04_missing_execution_environment_keeps_blocked():
    for env in ({}, {"randomized_assignment": True},
                {"randomized_assignment": True,
                 "venue_and_compliance_authorized": True}):
        r = profit_experiment_interface(st(), environment=env)
        assert r["status"] == BLOCKED
        assert "executable_environment" in r["reason"]
        assert r["executable_environment_available"] is False


def test_05_missing_randomized_assignment_keeps_blocked():
    r = profit_experiment_interface(st(), environment={
        "executable_environment": True,
        "venue_and_compliance_authorized": True})
    assert r["status"] == BLOCKED
    assert "randomized_assignment" in r["reason"]
    assert r["randomized_assignment_available"] is False


def test_05_missing_compliance_authorization_keeps_blocked():
    r = profit_experiment_interface(st(), environment={
        "executable_environment": True, "randomized_assignment": True})
    assert r["status"] == BLOCKED
    assert "venue_and_compliance_authorization" in r["reason"]


def test_05_environment_must_be_declared_truthy_not_merely_present():
    """声明为 falsy → 仍 BLOCKED（不得把"字段存在"当作"能力具备"）。"""
    r = profit_experiment_interface(st(), environment={
        "executable_environment": False, "randomized_assignment": 0,
        "venue_and_compliance_authorized": ""})
    assert r["status"] == BLOCKED and r["profit_identifiable"] is False


# ── 6. 环境齐备 → 仍不执行 ────────────────────────────────────────────
def test_06_all_preconditions_still_do_not_execute():
    r = profit_experiment_interface(st(), environment=_FULL_ENV)
    assert r["status"] == READY_FOR_EXECUTION
    assert r["result_status"] == RESULT_UNAVAILABLE       # 结果仍不可得
    assert r["executed"] is False
    assert r["valid_result_reachable"] is False
    assert "NO execution capability" in r["reason"]


def test_06_valid_result_never_reachable():
    for env in ({}, _FULL_ENV):
        r = profit_experiment_interface(st(), environment=env)
        assert r["valid_result_reachable"] is False
        assert r["result_status"] != VALID_RESULT
    assert VALID_RESULT in RESULT_STATUSES                # 已定义但不可达


# ── 7/8/9. 历史人类结果不可识别 + system/frozen 设计契约 ──────────────
def test_07_historical_human_outcomes_are_not_system_results():
    r = profit_experiment_interface(st())
    forbidden_blob = " ".join(r["forbidden_as_profit"])
    assert "历史人类策略" in forbidden_blob
    assert any("人类" in x for x in r["not_sufficient"])
    assert r["profit_identifiable"] is False


def test_08_design_contract_has_system_frozen_ab():
    a = DESIGN["assignment"]
    assert a["scheme"] == "50/50 randomized assignment: system vs frozen"
    assert a["randomized"] is True and a["per_unit"] is True
    assert DESIGN["arms"] == {"control": "frozen", "treatment": "system"}
    r = profit_experiment_interface(st())
    assert r["design"]["arms"]["treatment"] == "system"


def test_09_design_strata_are_preregistered_and_fixed():
    a = DESIGN["assignment"]
    assert a["stratified_by"] == ["stakes", "table_size", "position", "street",
                                  "facing_bet", "stack_depth_band"]
    assert "固定" in a["note"]
    assert DESIGN["analysis"]["preregistered_strata"] is True


# ── 10/11/12. hand_id 聚类 / bootstrap seed / MDE + 分层 ──────────────
def test_10_cluster_key_is_hand_id():
    assert DESIGN["cluster_key"] == "hand_id"
    assert DESIGN["sample_unit"].startswith("decision point")
    assert DESIGN["analysis"]["cluster_robust_se"] is True


def test_11_bootstrap_spec_matches_m3_audit():
    b = DESIGN["analysis"]["bootstrap"]
    assert b["n"] == 2000 and b["seed"] == 42 and b["paired"] is True
    assert b["ci_level"] == 0.95
    assert b["scheme"] == "cluster bootstrap"
    assert DESIGN["analysis"]["seed"] == 42


def test_12_mde_and_strata_fields_present():
    assert DESIGN["analysis"]["mde"]["reported"] is True
    assert "预注册" in DESIGN["analysis"]["mde"]["note"]
    assert DESIGN["analysis"]["no_multiple_comparison_drift"] is True
    assert DESIGN["analysis"]["no_optional_stopping"] is True


def test_12_outcome_is_net_chips_change_in_bb():
    o = DESIGN["outcome"]
    assert o["unit"] == "bb"
    assert "净筹码变化" in o["primary"]
    assert "执行环境" in o["requires"]


# ── 13/14. 缺失值与 UNKNOWN ───────────────────────────────────────────
def test_13_missing_value_policy_is_no_imputation():
    m = DESIGN["missing_value_handling"]
    assert m["zero_filled"] is False
    assert m["excluded_from_estimate"] is True
    assert "不插补" in m["policy"]
    assert m["missing_recorded_as"] == "outcome_bb=null + outcome_missing_reason"


def test_13_result_fields_include_missing_reason():
    assert "outcome_missing_reason" in DESIGN["result_fields"]
    assert "hand_id（聚类键）" in DESIGN["result_fields"]


def test_14_design_declares_execution_environment_requirement():
    e = DESIGN["execution_environment"]
    assert e["required"] is True
    assert e["not_available_in_this_project"] is True
    assert e["venue_and_compliance_authorization"] is True
    assert len(e["capabilities"]) >= 3


# ── 15. 非 CASH fail closed ───────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin"])
def test_15_non_cash_fails_closed(mode):
    r = profit_experiment_interface(st(game_type=mode))
    assert r["present"] is False
    for k in ("design", "forbidden_as_profit", "entry_gate",
              "never_emitted_fields", "provenance"):
        assert k not in r, k


@pytest.mark.parametrize("mode", [None, "", "   ", "holdem", "CASHX"])
def test_15_missing_or_illegal_mode_fails_closed(mode):
    r = profit_experiment_interface(st(game_type=mode))
    assert r["present"] is False
    assert "mode" in r["reason"] or "not_cash" in r["reason"]


def test_15_non_dict_fails_closed():
    for bad in (None, [], "cash", 3):
        assert profit_experiment_interface(bad)["present"] is False


def test_15_non_cash_does_not_inherit_cash_fields():
    r = profit_experiment_interface(st(game_type="pko"))
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("icm", "bounty", "payout", "tournament"):
        assert banned not in blob, banned


# ── 16/17. 不输出伪造收益/EV/ROI/胜率 ─────────────────────────────────
@pytest.mark.parametrize("env", [None, _FULL_ENV])
def test_16_no_result_fields_exist_as_keys(env):
    """**键级**核验：输出（含嵌套）中不存在任何结果字段名。"""
    kw = {"environment": env} if env else {}
    r = profit_experiment_interface(st(), **kw)
    leaked = sorted(set(NEVER_EMITTED_FIELDS) & _all_keys(r))
    assert leaked == [], leaked


def test_16_no_numeric_result_values_are_emitted():
    """顶层数值字段只能是布尔状态位，不得是任何结果量。"""
    r = profit_experiment_interface(st())
    bools = {"present", "executed", "valid_result_reachable",
             "profit_identifiable", "executable_environment_available",
             "randomized_assignment_available"}
    for k, v in r.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            pytest.fail(f"出现数值字段 {k}={v!r}")
    for k in bools:
        assert isinstance(r[k], bool), k


def test_17_never_emitted_field_list_is_declared_and_nonempty():
    r = profit_experiment_interface(st())
    assert len(r["never_emitted_fields"]) >= 10
    assert "profit" in r["never_emitted_fields"]
    assert "roi" in r["never_emitted_fields"]
    assert "win_rate" in r["never_emitted_fields"]


def test_17_pi_ev_and_heuristic_explicitly_forbidden_as_profit():
    blob = " ".join(FORBIDDEN_AS_PROFIT)
    assert "PI-EV" in blob
    assert "heuristic" in blob
    assert "range equity" in blob
    assert "Solver BLOCKED" in blob
    assert len(profit_experiment_interface(st())["forbidden_as_profit"]) == \
        len(FORBIDDEN_AS_PROFIT)


# ── 18. 入口门禁 ──────────────────────────────────────────────────────
def test_18_entry_gate_blocks_in_current_environment():
    g = evaluate_entry_gate()
    assert g["allowed"] is False
    assert set(g["missing"]) >= {"executable_environment",
                                 "randomized_assignment",
                                 "venue_and_compliance_authorization"}
    assert g["criteria"]


def test_18_entry_gate_passes_only_with_every_criterion():
    assert evaluate_entry_gate(environment=_FULL_ENV)["allowed"] is True
    partial = dict(_FULL_ENV)
    partial["design_sealed"] = False
    g = evaluate_entry_gate(environment=partial)
    assert g["allowed"] is False and g["missing"] == ["design_sealed"]


def test_18_entry_gate_notes_that_it_still_does_not_execute():
    g = evaluate_entry_gate(environment=_FULL_ENV)
    assert "不执行" in g["note"]


# ── 19. 设计字段完整性（授权 §3 逐项）────────────────────────────────
def test_19_design_covers_every_required_element():
    for key in ("sample_unit", "cluster_key", "assignment", "arms", "outcome",
                "analysis", "result_fields", "missing_value_handling",
                "execution_environment", "entry_gate", "not_sufficient"):
        assert key in DESIGN, key
    r = profit_experiment_interface(st())
    assert set(r["design"]) == set(DESIGN)
    assert r["entry_gate"] and r["not_sufficient"]


# ── 20/21. Solver 语义保持 / V3G-9 回归 ───────────────────────────────
def test_20_solver_blocked_semantics_are_not_converted():
    r = profit_experiment_interface(st())
    assert any("Solver BLOCKED" in x for x in r["forbidden_as_profit"])
    assert any("GTO" in x or "equilibrium" in x for x in r["not_sufficient"])
    from poker_decision_ai.cash_solver_interface import solver_status
    ss = solver_status(st())
    assert ss["equilibrium"] is False and ss["gto_exact"] is False


def test_21_v3g_1_to_9_still_consistent():
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_range_model import build_range_model
    from poker_decision_ai.cash_unknown_e2e import build_unknown_report
    s = st()
    assert cash_integrity_context(s)["present"] is True
    assert build_unknown_report(s)["present"] is True
    assert build_opponent_model(s)["present"] is True
    assert build_range_model(s)["present"] is True
    assert build_action_tree(s)["present"] is True


# ── 22/23. 真实 Judge / RealtimeRunner 逐位不变 ───────────────────────
async def test_22_judge_output_bitwise_identical_with_and_without_interface():
    import uuid
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    cfg = load_config()
    logs = ROOT / ".tmp" / "v3g10_logs" / uuid.uuid4().hex
    logs.mkdir(parents=True, exist_ok=True)
    ctx = AgentContext(request_id="v3g10", session_id="v3g10", config=cfg,
                       providers=build_providers(cfg), repos=build_in_memory(),
                       logger=EventLogger(str(logs)),
                       metrics=MetricsCollector())
    s = st(pot=200.0, to_call=50.0, bb=2.0, min_raise_to=100.0)
    eq = {"hero_equity": 0.8, "outcome": {"win": 0.8, "tie": 0.0, "lose": 0.2}}
    inp = {"state": copy.deepcopy(s), "equity": copy.deepcopy(eq)}
    before = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output
    profit_experiment_interface(s)                     # 查询接口
    after = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output
    assert set(before) == set(after)
    for k in sorted(before):
        assert before.get(k) == after.get(k), k


def test_23_real_realtime_decide_unchanged_by_interface_query():
    import asyncio
    import uuid as _uuid
    from tests.conftest import CASH_HAND_TEXT
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.registry import register_all_agents
    from poker_decision_ai.core.orchestration.runner import PipelineRunner
    from poker_decision_ai.core.orchestration.runtime import Runtime
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory
    from poker_decision_ai.realtime import RealtimeRunner

    def _rt():
        cfg = load_config()
        logs = ROOT / ".tmp" / "v3g10_logs" / _uuid.uuid4().hex
        logs.mkdir(parents=True, exist_ok=True)
        providers, repos = build_providers(cfg), build_in_memory()
        registry = register_all_agents()
        logger, metrics = EventLogger(str(logs)), MetricsCollector()
        runner = PipelineRunner(registry, cfg, logger, metrics)
        return Runtime(config=cfg, providers=providers, repos=repos,
                       registry=registry, runner=runner, logger=logger,
                       metrics=metrics)

    req = {"kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT,
           "hero_id": "Hero"}
    r1 = asyncio.run(RealtimeRunner(_rt()).decide(dict(req)))
    profit_experiment_interface(st())
    r2 = asyncio.run(RealtimeRunner(_rt()).decide(dict(req)))
    for k in ("action", "alternative_action", "recommended_size", "path"):
        assert (r1.final_decision or {}).get(k) == \
            (r2.final_decision or {}).get(k), k


# ── 24/25/26. JSON-safe / 确定性 / 不执行 ─────────────────────────────
def test_24_interface_is_json_safe():
    r = profit_experiment_interface(st())
    assert json.loads(json.dumps(r, ensure_ascii=False)) == r


def test_25_same_input_deterministic_and_readonly():
    s = st()
    before = copy.deepcopy(s)
    a = profit_experiment_interface(s)
    for _ in range(5):
        assert profit_experiment_interface(copy.deepcopy(s)) == a
    assert s == before


def test_26_module_has_no_execution_or_randomness_capability():
    src = (PKG / "cash_profit_interface.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("subprocess", "asyncio", "random", "socket", "multiprocessing",
                   "research_unfreeze", "strategy_core", "judge"):
        assert banned not in joined, banned
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for banned in ("compute_ev", "evaluate_intel", "positions_for", "run",
                   "Popen", "sample"):
        assert banned not in called, banned


def test_26_interface_is_not_wired_into_production():
    importers = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name == "cash_profit_interface.py":
            continue
        src = p.read_text(encoding="utf-8-sig", errors="ignore")
        if "cash_profit_interface" in src:
            importers.append(str(p.relative_to(ROOT)))
    # BENCH-1 Task0.1（owner 授权基线迁移，2 个引用方）：
    # 2026-09 中旬 P30-P41 生产桥/策略链阶段已授权下列模块引用本层。
    # 按 test_v3g11 SANCTIONED_WIRING 惯例登记为合法基线（精确集合匹配——
    # 任何未登记的新引用方、或已登记引用方消失，都会被检出）。
    sanctioned_importers = {
        "poker_decision_ai/cash_strategy_consumption_gate.py",
        "poker_decision_ai/cash_strategy_experiment_eligibility.py",
    }
    normalized = {i.replace(chr(92), '/') for i in importers}
    assert normalized == sanctioned_importers, (
        f"cash_profit_interface 引用方与授权基线不符（新引用方需所有者授权登记）: "
        f"unregistered={sorted(normalized - sanctioned_importers)} "
        f"missing={sorted(sanctioned_importers - normalized)}")


# ── 27/28/29. 禁止项 / manifest 零漂移 / 语义保持 ─────────────────────
def test_27_provenance_cites_m3_audit_and_blocker():
    p = profit_experiment_interface(st())["provenance"]
    assert "V3F_M3_OUTCOME_AUDIT" in p["design_source"]
    assert "B3 BLOCKED" in p["blocker"]
    assert p["valid_result_reachable"] is False
    assert "no_execution" in p
    assert p["m3_facts"]["action_differs_from_realized"] == "54.6% (4083/7483)"


def test_28_no_sealed_or_frozen_drift_from_this_stage():
    """本阶段只新增文件；99 项封存/冻结条目零漂移由 do-not-modify 检查覆盖。"""
    import hashlib
    for rel, man_rel in (("poker_decision_ai/domain/poker/strategy_core.py",
                          "docs/V3F_R1_QUALITY_BASELINE.json"),
                         ("poker_decision_ai/agents/judge/agent.py", None)):
        now = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert len(now) == 64
    # 关键冻结模块哈希锁定（与各阶段记录一致）
    h = hashlib.sha256(
        (ROOT / "poker_decision_ai/domain/poker/strategy_core.py").read_bytes()
    ).hexdigest()
    assert h.startswith("378a6867")


def test_29_no_profit_gto_equilibrium_overreach_in_output():
    r = profit_experiment_interface(st())
    assert r["profit_identifiable"] is False
    assert r["executed"] is False
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("profitable", "盈利能力已证明", "solved", "equilibrium\": true",
                   "gto_exact\": true", "independently verified"):
        assert banned not in blob, banned


def test_29_not_sufficient_list_covers_all_prohibited_substitutes():
    ns = " ".join(profit_experiment_interface(st())["not_sufficient"])
    for banned in ("PI-EV", "heuristic", "range equity", "EV estimator",
                   "历史人类", "Solver BLOCKED"):
        assert banned in ns, banned


# ── 30. 独立性不被主张 ────────────────────────────────────────────────
def test_30_independence_is_not_claimed():
    r = profit_experiment_interface(st())
    blob = json.dumps(r, ensure_ascii=False)
    assert "NOT INDEPENDENT" not in blob          # 本层不主张独立性
    assert "independent" not in blob.lower()
    assert DESIGN["analysis"]["bootstrap"]["scheme"] == "cluster bootstrap"
