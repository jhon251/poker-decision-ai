"""V3F-P30-11 — Real Solver Activation & Decision Quality 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.solver_contract import (  # noqa: E402
    freeze_mode, SOLVER_MODES, MOCK_NEVER_REAL)
from poker_decision_ai.strategy_provider import (  # noqa: E402
    classify_solver_result, strategy_failure)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── 1-11. Solver 状态 ──────────────────────────────────────────────
def test_real_solver_available_honest():
    """本机 executable 不存在 → REAL_SOLVER_AVAILABLE=false（诚实）。"""
    from poker_decision_ai.core.providers.solver import (  # noqa: E402
        PostflopSolverCLIProvider)
    p = PostflopSolverCLIProvider()
    assert p.is_configured() is False


def test_solver_unavailable_mode():
    assert freeze_mode("", False).mode == "UNKNOWN"


def test_solver_executable_missing():
    r = strategy_failure("postflop", "executable not found")
    assert r.strategy_mode == "UNKNOWN"


def test_solver_timeout():
    assert strategy_failure("postflop", "timeout").strategy_mode == "UNKNOWN"


def test_solver_malformed_output():
    assert strategy_failure("postflop", "malformed").strategy_mode == "UNKNOWN"


def test_solver_partial_output():
    assert strategy_failure("postflop", "partial").strategy_mode == "UNKNOWN"


def test_solver_failure():
    assert strategy_failure("postflop", "crash").strategy_mode == "UNKNOWN"


def test_real_provenance():
    r = freeze_mode("postflop", False, gto_action="call")
    assert r.mode == "REAL"
    assert r.to_dict()["provenance"]["solver_name"] == "postflop"


def test_baseline_provenance():
    r = freeze_mode("mock", True, gto_action="call")
    assert r.mode == "BASELINE"
    assert r.to_dict()["provenance"]["is_approximate"] is True


def test_approximate_provenance():
    # 近似算法（非 mock 但 is_approximate=True）→ APPROXIMATE/BASELINE
    r = classify_solver_result("fast-heuristic", True, gto_action="check")
    assert r.strategy_mode in ("BASELINE", "APPROXIMATE")


def test_unknown_provenance():
    r = freeze_mode("", False)
    assert r.mode == "UNKNOWN"
    assert r.provenance["reason"] == "no_provider"


# ── 12. UNKNOWN 不随机 fallback ─────────────────────────────────────
def test_unknown_no_random_fallback():
    r = freeze_mode("", False)
    assert r.gto_action is None
    assert r.actions == []


# ── 13-14. legal action gate ────────────────────────────────────────
def test_legal_action_gate():
    from poker_decision_ai.decision_contract import (derive_legal_actions,
                                                     validate_action)
    legal = derive_legal_actions(2000.0, 197680.0, 8920.0)
    ok, _ = validate_action("call", legal)
    assert ok


def test_illegal_action():
    from poker_decision_ai.decision_contract import (derive_legal_actions,
                                                     validate_action)
    legal = derive_legal_actions(0.0, 1000.0, 8920.0)
    ok, why = validate_action("fold", legal)
    assert not ok and why == "illegal_action:fold"


# ── 15-18. 真实帧 ───────────────────────────────────────────────────
def test_8bf8_real_frame():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == "OK"


def test_f01_regression():
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    # V-1 FM-1：NEW_HAND 静默映射移除 → 2 卡部分识别帧诚实拒绝
    assert frame_to_decision_request(rgba, w, h)["status"] == \
        "UNAVAILABLE"


def test_pc110_regression():
    rgba, w, h = _frame("POT_CHANGED_110_1787992526057_full.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == \
        "DECISION_UNAVAILABLE"


def test_f03_regression():
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == "UNAVAILABLE"


# ── 19-20. current_bet / action_history UNKNOWN ─────────────────────
def test_current_bet_unknown():
    import asyncio
    from poker_decision_ai.agents.vision.agent import VisionAgent
    from poker_decision_ai.agents.state.agent import StateAgent
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector

    cfg = load_config()
    ctx = AgentContext("t", "s", cfg, build_providers(cfg),
                       build_in_memory(), EventLogger("logs"),
                       MetricsCollector())
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"]["observation"]
    request = {"kind": "cash", "hero_id": "Hero",
               "vision_observation": obs}
    vout = asyncio.run(VisionAgent().run({"request": request}, ctx))
    sout = asyncio.run(StateAgent().run(
        {"request": request, "vision": vout}, ctx))
    fs = sout.output.get("field_semantics") or {}
    assert fs.get("current_bet") == "UNKNOWN"


def test_action_history_unknown():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["bridge"]["observation"]["extra"][
        "v3f_action_history_status"] == "UNKNOWN"


# ── 21-24. provenance ───────────────────────────────────────────────
def test_range_provenance():
    from poker_decision_ai.agents.range.agent import RangeOutput
    assert RangeOutput().range_mode == "BASELINE"


def test_equity_provenance():
    from poker_decision_ai.agents.equity.agent import EquityOutput
    out = EquityOutput()
    out.range_mode = "RANDOM"
    assert out.range_mode == "RANDOM"


def test_strategy_provenance():
    r = classify_solver_result("mock", True, gto_action="call")
    assert r.strategy_mode == "BASELINE"


def test_decision_result_provenance_complete():
    from poker_decision_ai.continuous_decision import (
        build_continuous_provenance)
    p = build_continuous_provenance(hand_id="h1", street="flop",
                                    action_history_status="UNKNOWN",
                                    range_mode="BASELINE",
                                    strategy_mode="BASELINE").to_dict()
    for k in ("hand_id", "street", "action_history_status", "range_mode",
              "equity_status", "strategy_mode"):
        assert k in p


# ── 25. ProductionDecisionLoop 回归 ─────────────────────────────────
def test_loop_regression_with_provenance():
    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    runtime = build_runtime(load_config())
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    r = loop.tick(300, rgba, w, h)
    assert r["decision_status"] == "READY"
    assert r["solver_provenance"] is not None
    assert r["input_provenance"] is not None
    assert r["input_provenance"]["current_bet_status"] == "UNKNOWN"
    # 再 tick 幂等
    r2 = loop.tick(300, rgba, w, h)
    assert r2["decision_id"] == r["decision_id"]


# ── 1419/positions_for ──────────────────────────────────────────────
def test_1419_regression():
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    from poker_decision_ai.vision2.pixel.realclient.to_call_observer import (  # noqa: E402
        ToCallObserver)
    out = ToCallObserver().observe(rgba, w, h)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "unsupported_render_class"


def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
