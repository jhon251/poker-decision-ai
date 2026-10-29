"""V3F-P30-10 — Real Solver Activation & Event-Stream Consumption 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.event_stream_consumer import (  # noqa: E402
    EventStreamConsumer, fact_to_action_history)
from poker_decision_ai.strategy_provider import (  # noqa: E402
    classify_solver_result, strategy_failure)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── Solver：executable/valid/timeout/error/malformed/partial/provenance ──
def test_solver_executable_absent():
    from poker_decision_ai.core.providers.solver import (  # noqa: E402
        PostflopSolverCLIProvider)
    p = PostflopSolverCLIProvider()
    assert p.is_configured() is False            # 本机无 executable（诚实）
    assert p.status_reason() != ""


def test_solver_provider_chain_config_default():
    """默认 config → P34 默认 realengine（REAL 规则引擎）；其余为兼容集。"""
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.providers import build_providers
    cfg = load_config()
    prov = build_providers(cfg)
    solver = prov.get_solver()
    assert solver is not None
    assert getattr(solver, "name", "") in ("mock", "postflop", "realengine")


def test_solver_timeout_fail_unknown():
    r = strategy_failure("postflop", "timeout after 120s")
    assert r.strategy_mode == "UNKNOWN"
    assert r.gto_action is None


def test_solver_process_error_unknown():
    r = strategy_failure("postflop", "process failed rc=1")
    assert r.strategy_mode == "UNKNOWN"


def test_solver_malformed_output_unknown():
    r = strategy_failure("postflop", "malformed output: no actions")
    assert r.strategy_mode == "UNKNOWN"


def test_solver_partial_output_unknown():
    r = strategy_failure("postflop", "partial: missing frequencies")
    assert r.strategy_mode == "UNKNOWN"


def test_solver_mock_real_separation():
    assert classify_solver_result("mock", True).strategy_mode == "BASELINE"
    assert classify_solver_result("postflop", False).strategy_mode == "REAL"


def test_solver_provenance():
    r = classify_solver_result("postflop", False, gto_action="call")
    p = r.to_dict()["provenance"]
    assert p["solver_name"] == "postflop"
    assert p["is_approximate"] is False


def test_strategy_mode_propagation():
    """策略模式传播到 DecisionSafetyResult（显式 mock → BASELINE，不 REAL；
    P34 起默认引擎为 realengine——REAL 路径见 test_p34_real_decision）。"""
    import os

    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    old = os.environ.get("SOLVER_PROVIDER")
    os.environ["SOLVER_PROVIDER"] = "mock"
    # SOLVER-1：MC1-R5 后 realtime 链含 frequency_prior agent，负载下
    # 默认 3000ms 死线偶发超时（死线敏感，与被测语义"模式传播"无关）
    # ——测试内提高死线（同 Task 0.3 惯例）。
    try:
        runtime = build_runtime(load_config())
        rt_cfg = dict(runtime.config.realtime_config())
        rt_cfg["deadline_ms"] = 30000
        rt_cfg["agent_timeout_ms"] = 30000
        runtime.config.root._data["realtime"] = rt_cfg
    finally:
        if old is None:
            os.environ.pop("SOLVER_PROVIDER", None)
        else:
            os.environ["SOLVER_PROVIDER"] = old
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    r = loop.tick(200, rgba, w, h)
    assert r["decision_status"] == "READY"
    assert r["strategy_mode"] in ("BASELINE", "UNKNOWN")   # 非 REAL（mock）


# ── Event Stream：HAND_HISTORY/EVENT_STREAM/VISUAL_FACT/NONE/UNKNOWN ──
def test_event_stream_hand_history():
    c = EventStreamConsumer()
    src = c.with_hand_history([{"street": "preflop", "player": "p1",
                                "action": "raise", "amount": 30}])
    assert src.status == "RECOGNIZED"
    assert src.source == "HAND_HISTORY"


def test_event_stream_visual_fact_only():
    c = EventStreamConsumer()
    c.ingest_visual_fact("BET_CHANGED", 10, {"from": 0, "to": 100})
    src = c.current_source()
    assert src.status == "UNKNOWN"               # 无行为源 → UNKNOWN
    assert all(e.status == "VISUAL_FACT_ONLY" for e in c.events)


def test_visual_diff_not_behavior():
    c = EventStreamConsumer()
    ev = c.ingest_visual_fact("STACK_CHANGED", 11, {"seat": "S1"})
    assert ev.action == ""                        # 无行为归因
    assert ev.is_behavior() is False


def test_visual_fact_source_none():
    c = EventStreamConsumer()
    c.ingest_visual_fact("ROI_CHANGED", 12)
    assert c.current_source().source == "NONE"


def test_empty_events_unknown_not_empty():
    c = EventStreamConsumer()
    assert c.current_source().status == "UNKNOWN"
    assert c.current_source().event_count == 0   # 空（非"已知空"）


def test_single_frame_unknown():
    c = EventStreamConsumer()
    src = c.current_source()
    assert src.status == "UNKNOWN"
    assert src.rationale != ""


def test_multiple_visual_facts_ordered():
    c = EventStreamConsumer()
    for i, k in enumerate(["ROI_CHANGED", "BET_CHANGED", "STACK_CHANGED"]):
        c.ingest_visual_fact(k, i + 1)
    assert [e.event_id for e in c.events] == ["vf-1", "vf-2", "vf-3"]
    assert len(c.events) == 3


def test_event_duplicate_allowed_with_provenance():
    c = EventStreamConsumer()
    ev1 = c.ingest_visual_fact("ROI_CHANGED", 1)
    ev2 = c.ingest_visual_fact("ROI_CHANGED", 1)
    assert ev1.event_id == ev2.event_id           # 重复事实可追踪


def test_event_provenance():
    c = EventStreamConsumer()
    ev = c.ingest_visual_fact("ACTION_PANEL_APPEARED", 7,
                              {"seat": "S6"})
    assert ev.provenance["frame_id"] == 7
    assert ev.provenance["kind"] == "ACTION_PANEL_APPEARED"


def test_fact_to_action_history_always_unknown():
    c = EventStreamConsumer()
    c.ingest_visual_fact("BET_CHANGED", 1)
    assert fact_to_action_history(c) == "UNKNOWN"


# ── Continuous Decision ─────────────────────────────────────────────
def test_event_source_state_engine_link():
    """合法事件源 → StateEngine（行为历史——非视觉 diff）。"""
    from poker_decision_ai.domain.poker.state_engine import PokerStateEngine
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "name": "p1", "stack": 2000},
        {"seat": 2, "name": "hero", "stack": 2000}],
        "button_seat": 1, "sb": 5, "bb": 10})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "raise", "to": 30})
    assert len(eng.snapshot()["action_history"]) >= 1


def test_readiness_gate_blocks_incomplete():
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    res = frame_to_decision_request(rgba, w, h)
    # V-1 FM-1：NEW_HAND 静默映射移除 → 2 卡部分识别帧诚实拒绝（结构性）
    assert res["status"] == "UNAVAILABLE"


def test_strategy_legal_action():
    from poker_decision_ai.decision_contract import (derive_legal_actions,
                                                     validate_action)
    legal = derive_legal_actions(2000.0, 197680.0, 8920.0)
    ok, _ = validate_action("call", legal)
    assert ok


def test_illegal_action_rejected():
    from poker_decision_ai.decision_contract import (derive_legal_actions,
                                                     validate_action)
    legal = derive_legal_actions(0.0, 1000.0, 8920.0)
    ok, why = validate_action("fold", legal)
    assert not ok


def test_unknown_safe_degradation():
    r = strategy_failure("postflop", "unavailable")
    assert r.strategy_mode == "UNKNOWN"          # 安全降级（非随机）


def test_loop_visual_fact_trace():
    """loop 携带 event_consumer → provenance 含 visual_fact_trace。"""
    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    runtime = build_runtime(load_config())
    c = EventStreamConsumer()
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0, event_consumer=c)
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    # 模拟前帧视觉事实
    c.ingest_visual_fact("ACTION_PANEL_APPEARED", 100, {"seat": "S6"})
    r = loop.tick(100, rgba, w, h)
    prov = r["provenance"] or {}
    assert "visual_fact_trace" in prov
    assert prov["visual_fact_trace"]["behavior_events"] == 0


# ── 红线扫描 ────────────────────────────────────────────────────────
def test_no_auto_click_scan():
    src = Path(
        "poker_decision_ai/production_decision_loop.py").read_text(
        encoding="utf-8")
    for word in ("click", "send_bet", "submit", "mouse", "keyboard"):
        assert word.lower() not in src.lower()


def test_no_previous_frame_in_event_consumer():
    src = Path(
        "poker_decision_ai/event_stream_consumer.py").read_text(
        encoding="utf-8")
    assert "prev_frame" not in src
    body = src.split('"""')[2] if src.count('"""') >= 2 else src
    assert "previous_frame" not in body


# ── 真实帧回归 ───────────────────────────────────────────────────────
def test_8bf8_regression():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == "OK"


def test_f01_f03_pc110_regression():
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    # V-1 FM-1：NEW_HAND 静默映射移除 → f01 帧（2 卡部分识别）诚实拒绝
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


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
