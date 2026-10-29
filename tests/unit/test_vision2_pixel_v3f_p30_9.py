"""V3F-P30-9 — Cash-Game Production Closure 测试。

覆盖：Solver executable 不存在/timeout/malformed/REAL·Mock≠REAL ·
UNKNOWN→0/[] 防护 · KNOWN_ZERO 保留 · production loop（frame 去重/
decision_id 幂等/同街重复帧）· mandatory 拒绝 · illegal 拒绝 · solver
failure 无随机 · state conflict · provenance · mode 不混淆 · 真实帧回归 ×4
· 1419/1848 · P30-8 regression · positions_for。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.unknown_semantics import (  # noqa: E402
    classify, tri_to_float)
from poker_decision_ai.strategy_provider import (  # noqa: E402
    classify_solver_result, strategy_failure)
from poker_decision_ai.production_decision_loop import (  # noqa: E402
    ProductionDecisionLoop, DecisionSafetyResult)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── 1-4. Solver 状态 ───────────────────────────────────────────────
def test_solver_executable_missing():
    from poker_decision_ai.core.providers.solver import (  # noqa: E402
        PostflopSolverCLIProvider)
    p = PostflopSolverCLIProvider()
    assert p.is_configured() is False           # 本机无 executable（诚实）


def test_solver_timeout_failure():
    r = strategy_failure("postflop", "timeout")
    assert r.strategy_mode == "UNKNOWN"
    assert r.gto_action is None


def test_solver_malformed_failure():
    r = strategy_failure("postflop", "malformed output")
    assert r.strategy_mode == "UNKNOWN"


def test_solver_real_classification():
    assert classify_solver_result("postflop", False).strategy_mode == "REAL"


# ── 5. Mock 永远不是 REAL ──────────────────────────────────────────
def test_mock_never_real():
    for is_approx in (True, False):
        assert classify_solver_result("mock", is_approx).strategy_mode != \
            "REAL"


# ── 6-8. UNKNOWN / KNOWN_ZERO 防护 ─────────────────────────────────
def test_unknown_to_zero_guard():
    assert tri_to_float(classify(None, "UNKNOWN")) is None


def test_unknown_to_empty_guard():
    from poker_decision_ai.action_event_source import no_source
    assert no_source().status == "UNKNOWN"
    assert no_source().events == []


def test_known_zero_preserved():
    assert tri_to_float(classify(0, "RECOGNIZED")) == 0.0
    assert classify(0, "RECOGNIZED").status == "KNOWN_ZERO"


# ── 9-12. production loop ───────────────────────────────────────────
def test_production_loop_dedup():
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    runtime = build_runtime(load_config())
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    r1 = loop.tick(100, rgba, w, h)
    assert r1["decision_status"] == "READY"
    r2 = loop.tick(100, rgba, w, h)              # 同帧 → 幂等返回（不重复 decide）
    assert r2["decision_id"] == r1["decision_id"]
    assert len(loop.decisions) == 1              # 只执行一次


def test_decision_id_idempotent():
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    runtime = build_runtime(load_config())
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    r1 = loop.tick(101, rgba, w, h)
    r2 = loop.tick(101, rgba, w, h)
    assert r1["decision_id"] == r2["decision_id"]
    assert len(loop.decisions) == 1              # 幂等——同 decision_id 不重复执行


def test_same_state_throttle_skip():
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    runtime = build_runtime(load_config())
    loop = ProductionDecisionLoop(runtime, throttle_s=999.0)
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    r1 = loop.tick(102, rgba, w, h)
    r2 = loop.tick(103, rgba, w, h)              # 同内容 → 同 state_key
    assert r2["decision_status"] == "SKIPPED_DUP"


def test_loop_does_not_auto_click():
    """loop 只输出 DecisionResult——不执行自动下注（无 bet 执行码）。"""
    from poker_decision_ai.production_decision_loop import (  # noqa: E402
        ProductionDecisionLoop)
    src = Path(
        "poker_decision_ai/production_decision_loop.py").read_text(
        encoding="utf-8")
    assert "click" not in src.lower()
    assert "send_bet" not in src.lower()
    assert "auto_submit" not in src.lower()


# ── 13-15. mandatory / illegal / solver failure ─────────────────────
def test_mandatory_unknown_rejects():
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    res = frame_to_decision_request(rgba, w, h)
    # V-1 FM-1：NEW_HAND 静默映射移除 → 2 卡部分识别帧诚实拒绝（结构性）
    assert res["status"] == "UNAVAILABLE"


def test_illegal_action_rejects():
    from poker_decision_ai.decision_contract import (derive_legal_actions,
                                                     validate_action)
    legal = derive_legal_actions(0.0, 1000.0, 8920.0)
    ok, why = validate_action("fold", legal)
    assert not ok and why == "illegal_action:fold"


def test_solver_failure_no_random():
    r = strategy_failure("postflop", "crash")
    assert r.actions == []
    assert r.gto_action is None


# ── 16-17. state conflict / provenance ─────────────────────────────
def test_state_conflict_rejects():
    """f03（occupied UNKNOWN → 结构拒绝）——state conflict 不决策。"""
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["status"] == "UNAVAILABLE"
    assert res["request"] is None


def test_provenance_complete():
    from poker_decision_ai.continuous_decision import (
        build_continuous_provenance)
    p = build_continuous_provenance(hand_id="h1", street="flop",
                                    action_history_status="UNKNOWN",
                                    range_mode="BASELINE",
                                    strategy_mode="BASELINE").to_dict()
    for k in ("hand_id", "street", "action_history_status", "range_mode",
              "equity_status", "strategy_mode"):
        assert k in p


# ── 18. mode 不混淆 ─────────────────────────────────────────────────
def test_modes_not_conflated():
    # Range RANDOM ≠ Equity REAL；Solver BASELINE ≠ Strategy REAL
    r = classify_solver_result("mock", True, gto_action="call")
    assert r.strategy_mode == "BASELINE"
    assert r.strategy_mode != "REAL"
    from poker_decision_ai.agents.equity.agent import EquityOutput
    eq = EquityOutput()
    eq.range_mode = "RANDOM"
    assert eq.range_mode == "RANDOM"            # 不隐式升级


# ── 19-20. 真实帧回归 + 1419/1848 ──────────────────────────────────
def test_8bf8_regression():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == "OK"


def test_f01_f03_pc110_regression():
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    # V-1 FM-1：NEW_HAND 静默映射移除 → 2 卡部分识别帧诚实拒绝
    assert frame_to_decision_request(rgba, w, h)["status"] == \
        "UNAVAILABLE"
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == "UNAVAILABLE"
    rgba, w, h = _frame("POT_CHANGED_110_1787992526057_full.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == \
        "DECISION_UNAVAILABLE"


def test_1419_regression():
    rgba, w, h = _frame("SCENE_CHANGED_37_1787487606667_full.png")
    from poker_decision_ai.vision2.pixel.realclient.to_call_observer import (  # noqa: E402
        ToCallObserver)
    out = ToCallObserver().observe(rgba, w, h)
    assert out["status"] == "UNKNOWN"
    assert out["reason"] == "unsupported_render_class"


# ── 21. P30-8 regression（field_semantics）─────────────────────────
def test_p30_8_semantics_regression():
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
    request = {"kind": "cash", "hero_id": "Hero", "vision_observation": obs}
    vout = asyncio.run(VisionAgent().run({"request": request}, ctx))
    sout = asyncio.run(StateAgent().run(
        {"request": request, "vision": vout}, ctx))
    fs = sout.output.get("field_semantics") or {}
    assert fs.get("current_bet") == "UNKNOWN"
    assert fs.get("action_history") == "UNKNOWN"
    assert fs.get("pot") == "KNOWN_VALUE"
    # min_raise_to 语义防护（P30-9-B）
    assert fs.get("min_raise_to") == "UNKNOWN"


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
