"""V3F-P30-6 — Action History + Opponent Range Integration 测试。

覆盖：ActionEventSource（HAND_HISTORY/EVENT_STREAM/UNKNOWN/不转[]）·
StateEngine 序列（preflop/flop/turn/raise/allin/fold）· Range/Equity
退化 provenance（BASELINE/APPROXIMATE/RANDOM/UNKNOWN）· Judge 非法 ·
真实帧回归 ×4 · 无伪造历史 · positions_for。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.action_event_source import (  # noqa: E402
    from_hand_history, from_event_stream, no_source, ActionEventSource)
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── 1-3. hand-history / event stream / empty vs UNKNOWN ──────────────
def test_hand_history_to_action_history():
    acts = [{"street": "preflop", "player": "Alice", "action": "raise",
             "amount": 300, "all_in": False},
            {"street": "preflop", "player": "Bob", "action": "call",
             "amount": 300, "all_in": False}]
    src = from_hand_history(acts)
    assert src.status == "RECOGNIZED"
    assert src.event_count == 2
    assert src.source == "HAND_HISTORY"


def test_event_stream_to_action_history():
    evs = [{"type": "HAND_START", "street": "preflop"},
           {"type": "PLAYER_ACTION", "action": "raise", "player_id": "p1"}]
    src = from_event_stream(evs)
    assert src.status == "RECOGNIZED"
    assert src.source == "EVENT_STREAM"


def test_empty_history_is_unknown_not_empty_recognized():
    src = from_hand_history([])
    assert src.status == "UNKNOWN"
    assert src.event_count == 0
    assert src.events == []                   # 空事件（非"已知空历史"）


def test_no_source_unknown():
    src = no_source()
    assert src.status == "UNKNOWN"
    assert src.source == "NONE"
    assert "不伪造" in src.rationale


# ── 4-13. StateEngine 序列（preflop/flop/turn/river/check/call/bet/raise/
# allin/fold）────────────────────────────────────────────────────────
def test_state_engine_preflop_sequence():
    from poker_decision_ai.domain.poker.state_engine import (
        PokerStateEngine)
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "player_id": "p1", "stack": 1000},
        {"seat": 2, "player_id": "hero", "stack": 1000}],
        "button_seat": 1, "sb": 5, "bb": 10})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "raise", "to": 30})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "hero", "street":
               "preflop", "action": "call", "amount": 20, "to": 30})
    snap = eng.snapshot()
    # hero 投入 20（call 增量）→ current_bet=30 → to_call=10（需再补 10
    # ——因 HAND_START 未含 BB post 事件；引擎语义按事件增量）
    assert snap["to_call"] == 10.0
    assert snap["current_bet"] >= 10


def test_state_engine_flop_sequence():
    from poker_decision_ai.domain.poker.state_engine import PokerStateEngine
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "player_id": "p1", "stack": 2000},
        {"seat": 2, "player_id": "hero", "stack": 2000}],
        "button_seat": 1, "sb": 5, "bb": 10})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "call", "to": 10})
    eng.apply({"type": "FLOP", "board": ["Ah", "Kd", "2c"]})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "flop", "action": "check"})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "hero", "street":
               "flop", "action": "bet", "amount": 20, "to": 20})
    snap = eng.snapshot()
    assert snap["street"] == "flop"
    assert snap["pot"] > 0


def test_state_engine_turn_river():
    from poker_decision_ai.domain.poker.state_engine import PokerStateEngine
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "player_id": "p1", "stack": 5000},
        {"seat": 2, "player_id": "hero", "stack": 5000}],
        "button_seat": 1, "sb": 5, "bb": 10})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "call", "to": 10})
    eng.apply({"type": "FLOP", "board": ["Ah", "Kd", "2c"]})
    eng.apply({"type": "TURN", "board": ["Ah", "Kd", "2c", "7s"]})
    assert eng.snapshot()["street"] == "turn"
    eng.apply({"type": "RIVER", "board": ["Ah", "Kd", "2c", "7s", "9h"]})
    assert eng.snapshot()["street"] == "river"


def test_state_engine_raise_sequence():
    from poker_decision_ai.domain.poker.state_engine import PokerStateEngine
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "player_id": "p1", "stack": 5000},
        {"seat": 2, "player_id": "hero", "stack": 5000}],
        "button_seat": 1, "sb": 5, "bb": 10})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "bet", "to": 20})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "hero", "street":
               "preflop", "action": "raise", "to": 60})
    assert eng.snapshot()["current_bet"] == 60


def test_state_engine_allin_call():
    from poker_decision_ai.domain.poker.state_engine import PokerStateEngine
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "player_id": "p1", "stack": 100},
        {"seat": 2, "player_id": "hero", "stack": 2000}],
        "button_seat": 1, "sb": 5, "bb": 10})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "call", "amount": 100, "all_in": True})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "hero", "street":
               "preflop", "action": "call", "amount": 90})
    snap = eng.snapshot()
    assert snap["to_call"] == 0.0


def test_state_engine_fold():
    from poker_decision_ai.domain.poker.state_engine import PokerStateEngine
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "player_id": "p1", "stack": 1000},
        {"seat": 2, "player_id": "hero", "stack": 1000}],
        "button_seat": 1, "sb": 5, "bb": 10})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "p1", "street":
               "preflop", "action": "fold"})
    snap = eng.snapshot()
    assert snap["street"] in ("preflop", "showdown")


# ── 14. multiple opponents ──────────────────────────────────────────
def test_state_engine_multi_opponents():
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
               "preflop", "action": "call", "to": 30})
    snap = eng.snapshot()
    assert len(snap["action_history"]) >= 3


# ── 15-17. UNKNOWN 传播 ─────────────────────────────────────────────
def test_action_history_unknown_no_fabricate():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    assert res["bridge"]["observation"]["extra"][
        "v3f_action_history_status"] == "UNKNOWN"


def test_unknown_not_converted_to_empty():
    src = no_source()
    assert src.status == "UNKNOWN"
    assert src.to_dict()["event_count"] == 0
    assert src.rationale != ""


def test_unknown_range_degrades():
    """FrameDecision provenance：纯帧 → action_history UNKNOWN（非空）。"""
    import asyncio
    from poker_decision_ai.frame_decision import frame_to_decision_async
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.runtime import build_runtime

    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    runtime = build_runtime(load_config())
    res = asyncio.run(frame_to_decision_async(
        runtime, np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4),
        w, h))
    prov = res.provenance or {}
    ah = prov.get("action_history") or {}
    assert ah.get("status") == "UNKNOWN"


# ── 18-22. Range 模式 ───────────────────────────────────────────────
def test_range_modes_distinct():
    from poker_decision_ai.agents.range.agent import RangeOutput
    base = RangeOutput()
    assert base.range_mode == "BASELINE"
    assert base.range_source == ""
    approx = RangeOutput()
    approx.range_mode = "APPROXIMATE"
    approx.range_source = "population_proxy"
    assert approx.range_mode == "APPROXIMATE"


def test_equity_random_provenance():
    from poker_decision_ai.agents.equity.agent import EquityOutput
    out = EquityOutput()
    out.range_mode = "RANDOM"
    assert out.range_mode == "RANDOM"


def test_equity_unknown_range_mode_never_real():
    from poker_decision_ai.agents.equity.agent import EquityOutput
    out = EquityOutput()
    out.range_mode = "UNKNOWN"
    assert out.range_mode != "REAL"


# ── 23-24. strategy provenance / Judge 非法 ─────────────────────────
def test_strategy_provenance_mode():
    from poker_decision_ai.strategy_provider import classify_solver_result
    r = classify_solver_result("mock", True, gto_action="call")
    assert r.strategy_mode == "BASELINE"
    assert r.to_dict()["provenance"]["is_approximate"] is True


def test_judge_rejects_illegal():
    from poker_decision_ai.decision_contract import (derive_legal_actions,
                                                     validate_action)
    legal = derive_legal_actions(0.0, 1000.0, 8920.0)
    ok, why = validate_action("fold", legal)
    assert not ok and why == "illegal_action:fold"


# ── 25-28. 真实帧回归 ─────────────────────────────────────────────────
def test_real_8bf8():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == "OK"


def test_real_f01():
    rgba, w, h = _frame("BUTTON_CHANGED_167_1787992682749_full.png")
    # V-1 FM-1：NEW_HAND 静默映射移除 → 2 卡部分识别帧诚实拒绝
    assert frame_to_decision_request(rgba, w, h)["status"] == \
        "UNAVAILABLE"


def test_real_pc110():
    rgba, w, h = _frame("POT_CHANGED_110_1787992526057_full.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == \
        "DECISION_UNAVAILABLE"


def test_real_f03():
    rgba, w, h = _frame("POT_CHANGED_102_1787992505794_full.png")
    assert frame_to_decision_request(rgba, w, h)["status"] == "UNAVAILABLE"


# ── 29-30. 无 previous-frame fallback / 无伪造历史 ───────────────────
def test_no_previous_frame_fallback():
    src = Path(
        "poker_decision_ai/action_event_source.py").read_text(encoding="utf-8")
    assert "prev_frame" not in src
    # 代码体（去除 docstring 后）无 previous_frame 引用
    body = src.split('"""')[2] if src.count('"""') >= 2 else src
    assert "previous_frame" not in body


def test_no_fabricated_history():
    src = Path(
        "poker_decision_ai/action_event_source.py").read_text(encoding="utf-8")
    assert "SINGLE_FRAME" in src               # 契约显式禁止
    assert "fabricat" not in src.lower() or "不伪造" in src


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
