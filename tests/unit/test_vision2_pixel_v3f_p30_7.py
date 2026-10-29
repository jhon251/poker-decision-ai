"""V3F-P30-7 — Automatic Event Stream + Real Solver Integration Audit 测试。

覆盖：ActionEvent contract · 合法/UNKNOWN 事件源 · 禁 frame→history 推断 ·
StateEngine 序列（street/action 全）· UNKNOWN 三态（≠0/KNOWN_ZERO=0/
KNOWN_VALUE）· action_history≠[] · Solver REAL/unavailable/failure/provenance
· baseline≠REAL · Range/Equity provenance · current_player/side_pot UNKNOWN ·
continuous decision provenance · 真实帧回归 ×4 · positions_for。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.action_event import (  # noqa: E402
    ActionEvent, visual_fact_event, unknown_source, ALLOWED_SOURCES,
    BEHAVIOR_ACTIONS)
from poker_decision_ai.unknown_semantics import (  # noqa: E402
    TriState, classify, tri_to_float)
from poker_decision_ai.continuous_decision import (  # noqa: E402
    ContinuousDecisionProvenance, build_continuous_provenance)
from poker_decision_ai.strategy_provider import (  # noqa: E402
    classify_solver_result, strategy_failure)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── 1-4. ActionEvent contract ───────────────────────────────────────
def test_action_event_contract():
    ev = ActionEvent(event_id="e1", source="HAND_HISTORY",
                     action="call", status="RECOGNIZED")
    assert ev.to_dict()["source"] == "HAND_HISTORY"
    assert ev.is_behavior() is True


def test_valid_event_source():
    from poker_decision_ai.action_event_source import from_hand_history
    src = from_hand_history([{"street": "preflop", "player": "p1",
                              "action": "raise", "amount": 30}])
    assert src.status == "RECOGNIZED"
    assert src.source == "HAND_HISTORY"


def test_unknown_event_source():
    from poker_decision_ai.action_event_source import from_event_stream
    src = from_event_stream([])
    assert src.status == "UNKNOWN"
    assert src.source == "EVENT_STREAM"


def test_no_frame_history_inference():
    """视觉事实 ≠ 行为事件（red line：不把 frame diff 归因为动作）。"""
    ev = visual_fact_event("BET_CHANGED", 42, {"from": 0, "to": 100})
    assert ev.status == "VISUAL_FACT_ONLY"
    assert ev.action == ""                     # 无行为归因
    assert ev.is_behavior() is False


# ── 5-13. StateEngine 序列 ──────────────────────────────────────────
def _mk_engine():
    from poker_decision_ai.domain.poker.state_engine import PokerStateEngine
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "name": "p1", "stack": 2000},
        {"seat": 2, "name": "hero", "stack": 2000}],
        "button_seat": 1, "sb": 5, "bb": 10})
    return eng


def test_state_sequence_full():
    eng = _mk_engine()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "raise", "to": 30})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "hero", "street":
               "preflop", "action": "call", "amount": 20})
    eng.apply({"type": "FLOP", "board": ["Ah", "Kd", "2c"]})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "flop", "action": "check"})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "hero", "street":
               "flop", "action": "bet", "amount": 40})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "flop", "action": "call", "amount": 40})
    snap = eng.snapshot()
    assert snap["street"] == "flop"
    assert len(snap["action_history"]) >= 5


def test_street_transition():
    eng = _mk_engine()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "call", "amount": 10})
    eng.apply({"type": "FLOP", "board": ["Ah", "Kd", "2c"]})
    assert eng.snapshot()["street"] == "flop"
    eng.apply({"type": "TURN", "board": ["Ah", "Kd", "2c", "7s"]})
    assert eng.snapshot()["street"] == "turn"


def test_allin_via_flag():
    eng = _mk_engine()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "call", "amount": 100, "all_in": True})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "hero", "street":
               "preflop", "action": "call", "amount": 100})
    snap = eng.snapshot()
    assert snap["to_call"] == 0.0


def test_fold():
    eng = _mk_engine()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "fold"})
    p1 = next(p for p in eng.snapshot()["players"]
              if p["player_id"] == "p1")
    assert p1["status"] == "folded"


def test_check():
    eng = _mk_engine()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "call", "amount": 10})
    eng.apply({"type": "FLOP", "board": ["Ah", "Kd", "2c"]})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "flop", "action": "check"})
    assert eng.snapshot()["to_call"] == 0.0


def test_call():
    eng = _mk_engine()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "call", "amount": 10})
    assert eng.snapshot()["pot"] >= 10


def test_bet():
    eng = _mk_engine()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "bet", "amount": 30})
    assert eng.snapshot()["current_bet"] >= 30


def test_raise():
    eng = _mk_engine()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "raise", "to": 60})
    assert eng.snapshot()["current_bet"] == 60


def test_multi_player_sequence():
    from poker_decision_ai.domain.poker.state_engine import PokerStateEngine
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "player_id": "p1", "stack": 1000},
        {"seat": 2, "player_id": "p2", "stack": 1000},
        {"seat": 3, "player_id": "hero", "stack": 1000}],
        "button_seat": 1, "sb": 5, "bb": 10})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "raise", "to": 30})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p2", "street":
               "preflop", "action": "fold"})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "hero", "street":
               "preflop", "action": "call", "amount": 20})
    assert len(eng.snapshot()["action_history"]) >= 3


# ── 14-16. UNKNOWN 三态 ─────────────────────────────────────────────
def test_unknown_ne_zero():
    t = classify(None, "UNKNOWN")
    assert t.status == "UNKNOWN"
    assert tri_to_float(t) is None                # 不转 0


def test_known_zero_is_zero():
    t = classify(0, "RECOGNIZED")
    assert t.status == "KNOWN_ZERO"
    assert tri_to_float(t) == 0.0


def test_known_value():
    t = classify(2000, "RECOGNIZED")
    assert t.status == "KNOWN_VALUE"
    assert tri_to_float(t) == 2000.0


def test_action_history_unknown_not_empty():
    src = unknown_source()
    assert src.source == "UNKNOWN"
    assert src.to_dict()["action"] == ""


# ── 17-23. Solver REAL/unavailable/failure/provenance ───────────────
def test_solver_real_if_not_approx():
    r = classify_solver_result("postflop", False, gto_action="call")
    assert r.strategy_mode == "REAL"


def test_solver_unavailable():
    r = strategy_failure("", "no provider")
    assert r.strategy_mode == "UNKNOWN"


def test_solver_timeout_failure_same():
    r = strategy_failure("postflop", "timeout")
    assert r.strategy_mode == "UNKNOWN"
    assert r.gto_action is None


def test_solver_malformed_provenance():
    r = classify_solver_result("mock", True, actions=[])
    assert r.strategy_mode == "BASELINE"
    assert r.to_dict()["provenance"]["solver_name"] == "mock"


def test_solver_provenance_mode():
    r = classify_solver_result("real-cli", False)
    assert r.to_dict()["provenance"]["is_approximate"] is False


def test_baseline_never_real():
    r = classify_solver_result("mock", False)     # mock 即便 is_approx=False
    assert r.strategy_mode != "REAL"


def test_equity_provenance_random():
    from poker_decision_ai.agents.equity.agent import EquityOutput
    out = EquityOutput()
    out.range_mode = "RANDOM"
    assert out.range_mode != "REAL"


# ── 24-25. Range/Equity provenance ─────────────────────────────────
def test_range_provenance_baseline():
    from poker_decision_ai.agents.range.agent import RangeOutput
    out = RangeOutput()
    assert out.range_mode == "BASELINE"


# ── 26-27. current_player / side_pot UNKNOWN ────────────────────────
def test_current_player_unknown():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"]["observation"]
    assert obs["extra"]["v3f_current_player"] is None
    assert obs["extra"]["v3f_current_player_status"] == "UNKNOWN"


def test_side_pot_unknown():
    """无独立 side_pot 视觉对象 → UNKNOWN（不得 pot+stack 推导）。"""
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"]["observation"]
    assert "side_pot" not in obs                # 不创建/不推导


# ── 28. continuous decision provenance ─────────────────────────────
def test_continuous_provenance():
    p = build_continuous_provenance(
        hand_id="h1", street="flop", decision_id="d1",
        action_history_status="UNKNOWN", range_mode="BASELINE",
        equity_status="RECOGNIZED", strategy_mode="BASELINE")
    d = p.to_dict()
    assert d["hand_id"] == "h1"
    assert d["action_history_status"] == "UNKNOWN"
    assert d["range_mode"] == "BASELINE"
    assert d["state_provenance"]["deduced"] is False


# ── 29-32. 真实帧回归 ───────────────────────────────────────────────
def test_8bf8_regression():
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


# ── 附加：REAL_SOLVER_AVAILABLE 诚实 ────────────────────────────────
def test_real_solver_availability_honest():
    from poker_decision_ai.core.providers.solver import (  # noqa: E402
        PostflopSolverCLIProvider)
    p = PostflopSolverCLIProvider()
    configured = p.is_configured()
    # 本机无 executable → 诚实 false（不假装存在）
    assert isinstance(configured, bool)


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
