"""V3G-9 — CASH Solver Fail-Closed Interface 测试（授权清单 1–30）。

**不修改**任何既有测试定义；**不触碰**冻结文件；**不调用真实 Solver**。
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.cash_solver_interface import (
    BLOCKED, CONFIGURED_NOT_RUN, SOLVER_INTERFACE_VERSION, SOLVER_POLICY,
    SOLVER_STATES, UNAVAILABLE, UNKNOWN, VALID_RESULT, solver_status)

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"


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


def _disc(available, *, exe=None, reason="") -> dict:
    ch = {"executable_path": ({"found": True, "path": exe}
                              if exe else {"found": False,
                                           "reason": "POSTFLOP_SOLVER_PATH not set"}),
          "path_search": {"found": False},
          "pip_package": {"found": False},
          "sample_outputs": {"found": False}}
    return {"available": available, "channels": ch, "blocked_reason": reason}


# ── 1/2/3/4. UNAVAILABLE 家族 ─────────────────────────────────────────
def test_01_no_solver_config_is_unavailable_in_discovery():
    r = solver_status(st(), discovery=_disc(False, reason="POSTFLOP_SOLVER_PATH not set"))
    assert r["discovery_state"] == UNAVAILABLE
    assert r["configured"] is False
    assert r["solver_status"] == BLOCKED          # 策略叠加


def test_02_solver_path_env_missing_reason_is_surfaced():
    r = solver_status(st(), discovery=_disc(
        False, reason="POSTFLOP_SOLVER_PATH not set"))
    assert "POSTFLOP_SOLVER_PATH" in r["discovery_reason"]
    assert r["channels"]["executable_path"]["found"] is False


def test_03_solver_path_set_but_not_existing():
    r = solver_status(st(), discovery=_disc(
        False, reason="POSTFLOP_SOLVER_PATH set but missing: C:/nope/solver.exe"))
    assert r["discovery_state"] == UNAVAILABLE
    assert "missing" in r["discovery_reason"]


def test_04_missing_dependency_channel():
    r = solver_status(st(), discovery=_disc(False, reason="postflop_solver not installed"))
    assert r["channels"]["pip_package"]["found"] is False
    assert r["discovery_state"] == UNAVAILABLE


# ── 5. BLOCKED（策略叠加）─────────────────────────────────────────────
def test_05_real_environment_is_blocked_or_unavailable():
    """**当前环境**必须表现为 BLOCKED（或明确 UNAVAILABLE）——不得是可用。"""
    r = solver_status(st())
    assert r["solver_status"] in (BLOCKED, UNAVAILABLE)
    assert r["solver_status"] == BLOCKED                 # 本层策略
    assert r["discovery_state"] in (UNAVAILABLE, UNKNOWN)
    assert r["available"] is not True


def test_05_policy_forbids_use_even_when_executable_present():
    """即使发现到可执行文件，策略仍压制为 BLOCKED（不得调用）。"""
    r = solver_status(st(), discovery=_disc(True, exe="C:/solver/postflop-solver.exe"))
    assert r["solver_status"] == BLOCKED
    assert r["discovery_state"] == CONFIGURED_NOT_RUN
    assert r["configured"] is True
    assert "forbidden" in r["reason"]


# ── 6. UNKNOWN ────────────────────────────────────────────────────────
def test_06_insufficient_evidence_is_unknown_not_defaulted():
    r = solver_status(st(), discovery={"available": None, "channels": {},
                                       "blocked_reason": "probe failed"})
    assert r["discovery_state"] == UNKNOWN
    assert r["configured"] is None                       # 不默认 True/False
    assert "insufficient" in r["discovery_reason"]


def test_06_available_without_executable_is_unknown_not_available():
    r = solver_status(st(), discovery=_disc(True, exe=None))
    assert r["discovery_state"] == UNKNOWN
    assert r["solver_status"] == BLOCKED
    assert r["valid"] is False


# ── 7. CONFIGURED_NOT_RUN ─────────────────────────────────────────────
def test_07_configured_but_not_run_never_claims_verified():
    r = solver_status(st(), discovery=_disc(True, exe="C:/solver/exe"))
    assert r["discovery_state"] == CONFIGURED_NOT_RUN
    assert r["executed"] is False and r["valid"] is False
    assert "NOT executed" in r["discovery_reason"]


# ── 8/9/26. 禁止真实调用 / fake 结果 / 不调用 ─────────────────────────
def test_08_module_has_no_execution_capability():
    """**源码级**：无 subprocess / asyncio / solve 调用 → 物理上不能调用 solver。"""
    src = (PKG / "cash_solver_interface.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("subprocess", "asyncio", "multiprocessing", "shutil"):
        assert banned not in joined, banned
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("solve", "create_subprocess_exec", "run", "Popen",
                   "system"):
        assert banned not in called and banned not in attrs, banned


def test_09_no_fake_or_mock_solver_result_is_produced():
    r = solver_status(st())
    for k in ("strategy", "ev", "frequencies", "actions", "equity"):
        assert k not in r, k
    assert r["valid"] is False
    # 不得出现"结果"字段
    assert "result" not in r and "raw" not in r


def test_26_status_query_performs_no_execution_and_is_repeatable():
    a = solver_status(st())
    for _ in range(3):
        assert solver_status(st()) == a


# ── 10/11. PI-EV / heuristic 不冒充 Solver ────────────────────────────
def test_10_pi_ev_is_not_wired_as_solver():
    src = (PKG / "cash_solver_interface.py").read_text(encoding="utf-8")
    for banned in ("pi_ev", "pi-ev", "PiEvReference", "PI_EV",
                   "reference_complete"):
        assert banned not in src, banned
    r = solver_status(st())
    assert r["valid"] is False and r["solver_status"] != VALID_RESULT


def test_11_heuristic_and_equity_are_not_wired_as_solver():
    src = (PKG / "cash_solver_interface.py").read_text(encoding="utf-8")
    joined = " ".join(imported for imported in
                      [n.module or "" for n in ast.walk(ast.parse(src))
                       if isinstance(n, ast.ImportFrom)])
    for banned in ("strategy_core", "equity", "conditional_model", "fold_equity",
                   "real_engine"):
        assert banned not in joined, banned
    assert "compute_ev" not in src and "equity" not in src.lower().replace(
        "range equity", "")


# ── 12/13. equilibrium / gto_exact ────────────────────────────────────
def test_12_equilibrium_is_false_everywhere():
    assert SOLVER_POLICY["equilibrium"] is False
    for disc in (_disc(False), _disc(True, exe="X"),
                 {"available": None, "channels": {}, "blocked_reason": "x"}):
        assert solver_status(st(), discovery=disc)["equilibrium"] is False


def test_13_gto_exact_is_false_everywhere():
    assert SOLVER_POLICY["gto_exact"] is False
    r = solver_status(st())
    assert r["gto_exact"] is False
    assert r["policy"]["valid_result_reachable"] is False
    blob = json.dumps(r, ensure_ascii=False).lower()
    assert '"gto_exact": false' in blob


def test_13_valid_result_is_unreachable_under_current_policy():
    """`VALID_RESULT` 在当前项目**不可达**（策略禁止 + 无执行能力）。"""
    assert "valid_result_reachable" in SOLVER_POLICY
    assert SOLVER_POLICY["valid_result_reachable"] is False
    assert solver_status(st())["solver_status"] != VALID_RESULT
    assert VALID_RESULT in SOLVER_STATES          # 状态**已定义**，只是不可达


# ── 14–19. CASH / 非 CASH 隔离 ────────────────────────────────────────
def test_14_cash_produces_status():
    r = solver_status(st())
    assert r["present"] is True and r["mode"] == "cash"
    assert r["solver_status"] in SOLVER_STATES


@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin"])
def test_15_to_18_non_cash_fails_closed(mode):
    r = solver_status(st(game_type=mode))
    assert r["present"] is False
    for k in ("channels", "provenance", "policy"):
        assert k not in r, k
    # 不得触发 solver 分支
    assert r["executed"] is False and r["valid"] is False


@pytest.mark.parametrize("mode", [None, "", "   ", "holdem", "CASHX"])
def test_19_missing_or_illegal_mode_fails_closed(mode):
    r = solver_status(st(game_type=mode))
    assert r["present"] is False
    assert "mode" in r["reason"] or "not_cash" in r["reason"]


def test_19_non_dict_fails_closed():
    for bad in (None, [], "cash", 3):
        r = solver_status(bad)
        assert r["present"] is False


def test_19_no_tournament_fields_reachable_for_non_cash():
    r = solver_status(st(game_type="pko"))
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("icm", "bounty", "payout", "tournament"):
        assert banned not in blob, banned


# ── 20. V3F governance 语义一致 ───────────────────────────────────────
def test_20_policy_semantics_match_governance_limitations():
    """本层策略与 `governance.LIMITATIONS['solver']` 语义一致（只读核对）。"""
    from poker_decision_ai.research_unfreeze.governance import LIMITATIONS
    g = LIMITATIONS["solver"]
    assert g["value"] == "BLOCKED"
    assert "equilibrium=false" in g["reference"]
    assert "NOT_exact_gto=true" in g["reference"]
    assert g["value"] == BLOCKED                    # 与枚举字面量一致
    assert SOLVER_POLICY["equilibrium"] is False
    assert SOLVER_POLICY["gto_exact"] is False
    # 本层**不 import** research_unfreeze（隔离），只做语义对齐
    src = (PKG / "cash_solver_interface.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            assert "research_unfreeze" not in (n.module or "")


# ── 21/22/23/24/25. V3G 回归 / 逐位不变 / JSON-safe / 确定性 ──────────
def test_21_v3g2_and_v3g8_still_consistent():
    from poker_decision_ai.cash_integrity_context import cash_integrity_context
    from poker_decision_ai.cash_unknown_e2e import build_unknown_report
    s = st()
    ev = cash_integrity_context(s)
    ur = build_unknown_report(s)
    ss = solver_status(s)
    assert ev["present"] and ur["present"] and ss["present"]
    assert ev["mode"] == ur["mode"] == ss["mode"] == "cash"


async def test_22_judge_output_bitwise_identical_with_and_without_status():
    import uuid
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    cfg = load_config()
    logs = ROOT / ".tmp" / "v3g9_logs" / uuid.uuid4().hex
    logs.mkdir(parents=True, exist_ok=True)
    ctx = AgentContext(request_id="v3g9", session_id="v3g9", config=cfg,
                       providers=build_providers(cfg), repos=build_in_memory(),
                       logger=EventLogger(str(logs)),
                       metrics=MetricsCollector())
    s = st(pot=200.0, to_call=50.0, bb=2.0, min_raise_to=100.0)
    eq = {"hero_equity": 0.8, "outcome": {"win": 0.8, "tie": 0.0, "lose": 0.2}}
    inp = {"state": copy.deepcopy(s), "equity": copy.deepcopy(eq)}
    before = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output
    solver_status(s)                                   # 查询状态
    after = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output
    assert set(before) == set(after)
    for k in sorted(before):
        assert before.get(k) == after.get(k), k


def test_23_real_realtime_decide_output_unchanged_by_status_query():
    """真实 `RealtimeRunner.decide()`：查询 solver 状态不改变最终决策。"""
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
        logs = ROOT / ".tmp" / "v3g9_logs" / _uuid.uuid4().hex
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
    solver_status(st())                                # 中间查询状态
    r2 = asyncio.run(RealtimeRunner(_rt()).decide(dict(req)))
    # 决策字段必须完全一致（"allin" 时 recommended_size 语义见 V3G-7 审计）
    for k in ("action", "alternative_action", "recommended_size", "path"):
        assert (r1.final_decision or {}).get(k) == (r2.final_decision or {}).get(k), k


def test_24_status_is_json_safe():
    r = solver_status(st())
    json.dumps(r)
    assert json.loads(json.dumps(r)) == r


def test_25_same_input_deterministic_and_readonly():
    s = st()
    before = copy.deepcopy(s)
    a = solver_status(s)
    for _ in range(5):
        assert solver_status(copy.deepcopy(s)) == a
    assert s == before


# ── 27/28/29. 不改配置 / 不改 strategy_core & Judge / import 隔离 ────
def test_27_module_does_not_modify_solver_configuration():
    src = (PKG / "cash_solver_interface.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            targets = [getattr(t, "id", None) or getattr(t, "attr", "")
                       for t in n.targets]
            assert "environ" not in targets
            assert "POSTFLOP_SOLVER_PATH" not in targets
    assert "os.environ[" not in src and "os.environ.setdefault" not in src


def test_28_does_not_touch_strategy_core_or_judge():
    src = (PKG / "cash_solver_interface.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("strategy_core", "judge", "research_unfreeze"):
        assert banned not in joined, banned
    assert set(imported) <= {"__future__", "dataclasses", "typing",
                             "cash_integrity_context",
                             ".core.providers.solver",
                             "core.providers.solver"}, imported


def test_29_import_isolation_and_not_wired_into_decision_paths():
    importers = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name == "cash_solver_interface.py":
            continue
        src = p.read_text(encoding="utf-8-sig", errors="ignore")
        if "cash_solver_interface" in src:
            importers.append(str(p.relative_to(ROOT)))
    # BENCH-1 Task0.1（owner 授权基线迁移，2 个引用方）：
    # 2026-09 中旬 P30-P41 生产桥/策略链阶段已授权下列模块引用本层。
    # 按 test_v3g11 SANCTIONED_WIRING 惯例登记为合法基线（精确集合匹配——
    # 任何未登记的新引用方、或已登记引用方消失，都会被检出）。
    sanctioned_importers = {
        "poker_decision_ai/cash_architecture_closure_audit.py",
        "poker_decision_ai/cash_strategy_consumption_gate.py",
    }
    normalized = {i.replace(chr(92), '/') for i in importers}
    assert normalized == sanctioned_importers, (
        f"cash_solver_interface 引用方与授权基线不符（新引用方需所有者授权登记）: "
        f"unregistered={sorted(normalized - sanctioned_importers)} "
        f"missing={sorted(sanctioned_importers - normalized)}")


# ── 30. 不伪造 GTO/equilibrium ────────────────────────────────────────
def test_30_no_gto_or_equilibrium_claim_in_output():
    r = solver_status(st())
    assert r["equilibrium"] is False and r["gto_exact"] is False
    assert r["valid"] is False and r["executed"] is False
    # 输出中不得出现"已求解/均衡成立"之类断言
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("solved", "solution", "equilibrium\": true",
                   "gto_exact\": true", "is_exact"):
        assert banned not in blob, banned
    # 禁止用途必须显式列出（可审计）
    assert len(r["forbidden_uses"]) >= 5
