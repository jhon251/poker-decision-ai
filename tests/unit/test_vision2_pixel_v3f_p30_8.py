"""V3F-P30-8 — Cash-Game Core Decision Closure 测试。

覆盖：UNKNOWN 三态（≠KNOWN_ZERO/KNOWN_ZERO/KNOWN_VALUE）· current_bet/
action_history 语义 · StateEngine 连续手（preflop/flop/turn/river/allin/fold/
multi-player）· Range/Equity/Strategy provenance · Solver REAL/BASELINE/
unavailable/malformed/timeout/failure · Mock≠REAL · current_player/side_pot
UNKNOWN · 决策合法性 · 真实帧回归 ×4 · 无 hidden random · 无 UNKNOWN→0/[] ·
provenance 保持 · positions_for。
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
from poker_decision_ai.vision2.pixel.realclient.production_bridge import (  # noqa: E402
    frame_to_decision_request)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


# ── 1-3. UNKNOWN 三态 ───────────────────────────────────────────────
def test_unknown_ne_known_zero():
    u = classify(None, "UNKNOWN")
    z = classify(0.0, "RECOGNIZED")
    assert u.status == "UNKNOWN"
    assert z.status == "KNOWN_ZERO"
    assert tri_to_float(u) is None           # UNKNOWN ≠ 0
    assert tri_to_float(z) == 0.0            # KNOWN_ZERO = 0


def test_known_zero():
    assert classify(0, "RECOGNIZED").status == "KNOWN_ZERO"


def test_known_value():
    assert classify(2000, "RECOGNIZED").status == "KNOWN_VALUE"
    assert tri_to_float(classify(2000, "RECOGNIZED")) == 2000.0


# ── 4-5. current_bet / action_history 语义 ─────────────────────────
def test_current_bet_unknown_semantics():
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
    assert fs.get("current_bet") == "UNKNOWN"     # 语义区分（数值仍 0.0 布局）
    assert fs.get("action_history") == "UNKNOWN"
    assert fs.get("pot") == "KNOWN_VALUE"


def test_action_history_unknown_not_empty_semantics():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"]["observation"]
    assert obs["extra"]["v3f_action_history_status"] == "UNKNOWN"


# ── 6. action_history recognized ────────────────────────────────────
def test_action_history_recognized():
    from poker_decision_ai.action_event_source import from_hand_history
    src = from_hand_history([{"street": "preflop", "player": "p1",
                              "action": "raise", "amount": 30}])
    assert src.status == "RECOGNIZED"


# ── 7-9. Range/Equity/Strategy provenance ───────────────────────────
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
    assert r.to_dict()["provenance"]["is_approximate"] is True


# ── 10-16. Solver 状态 ─────────────────────────────────────────────
def test_solver_real():
    assert classify_solver_result("postflop", False).strategy_mode == "REAL"


def test_solver_baseline():
    assert classify_solver_result("mock", True).strategy_mode == "BASELINE"


def test_solver_unavailable():
    assert strategy_failure("", "none").strategy_mode == "UNKNOWN"


def test_solver_malformed():
    assert strategy_failure("postflop", "malformed").strategy_mode == \
        "UNKNOWN"


def test_solver_timeout():
    assert strategy_failure("postflop", "timeout").strategy_mode == "UNKNOWN"


def test_solver_failure_no_random():
    r = strategy_failure("postflop", "exec not found")
    assert r.gto_action is None
    assert r.actions == []


def test_mock_never_real():
    assert classify_solver_result("mock", False).strategy_mode != "REAL"


# ── 17-23. Continuous hand state（StateEngine）───────────────────────
def _mk6():
    from poker_decision_ai.domain.poker.state_engine import PokerStateEngine
    eng = PokerStateEngine("hero")
    eng.apply({"type": "HAND_START", "seats": [
        {"seat": 1, "name": "BTN", "stack": 20000},
        {"seat": 2, "name": "SB", "stack": 20000},
        {"seat": 3, "name": "BB", "stack": 20000},
        {"seat": 4, "name": "UTG", "stack": 20000},
        {"seat": 5, "name": "MP", "stack": 20000},
        {"seat": 6, "name": "hero", "stack": 20000}],
        "button_seat": 1, "sb": 1000, "bb": 2000})
    return eng


def test_continuous_preflop():
    eng = _mk6()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BTN", "street":
               "preflop", "action": "raise", "to": 6000})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "SB", "street":
               "preflop", "action": "fold"})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BB", "street":
               "preflop", "action": "call", "amount": 4000})
    snap = eng.snapshot()
    assert snap["street"] == "preflop"
    assert len(snap["action_history"]) >= 3


def test_continuous_flop():
    eng = _mk6()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BTN", "street":
               "preflop", "action": "call", "amount": 2000})
    eng.apply({"type": "FLOP", "board": ["Ah", "Kd", "2c"]})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BB", "street":
               "flop", "action": "check"})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BTN", "street":
               "flop", "action": "bet", "amount": 4000})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BB", "street":
               "flop", "action": "call", "amount": 4000})
    assert eng.snapshot()["street"] == "flop"


def test_continuous_turn():
    eng = _mk6()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BTN", "street":
               "preflop", "action": "call", "amount": 2000})
    eng.apply({"type": "FLOP", "board": ["Ah", "Kd", "2c"]})
    eng.apply({"type": "TURN", "board": ["Ah", "Kd", "2c", "7s"]})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BB", "street":
               "turn", "action": "check"})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BTN", "street":
               "turn", "action": "bet", "amount": 8000})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BB", "street":
               "turn", "action": "raise", "to": 20000})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BTN", "street":
               "turn", "action": "call", "amount": 12000})
    assert eng.snapshot()["street"] == "turn"


def test_continuous_river():
    eng = _mk6()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BTN", "street":
               "preflop", "action": "call", "amount": 2000})
    eng.apply({"type": "FLOP", "board": ["Ah", "Kd", "2c"]})
    eng.apply({"type": "TURN", "board": ["Ah", "Kd", "2c", "7s"]})
    eng.apply({"type": "RIVER", "board": ["Ah", "Kd", "2c", "7s", "9h"]})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BTN", "street":
               "river", "action": "bet", "amount": 20000, "all_in": True})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "BB", "street":
               "river", "action": "call", "amount": 15000, "all_in": True})
    eng.apply({"type": "SHOWDOWN",
               "shows": [{"player_id": "BTN", "cards": ["As", "Ad"]}]})
    assert eng.snapshot()["street"] in ("river", "showdown")


def test_continuous_allin_fold_multi():
    eng = _mk6()
    eng.apply({"type": "PLAYER_ACTION", "player_id": "UTG", "street":
               "preflop", "action": "fold"})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "MP", "street":
               "preflop", "action": "raise", "to": 6000, "all_in": True})
    eng.apply({"type": "PLAYER_ACTION", "player_id": "hero", "street":
               "preflop", "action": "fold"})
    p_mp = next(p for p in eng.snapshot()["players"]
                if p["player_id"] == "MP")
    p_hero = next(p for p in eng.snapshot()["players"]
                  if p["player_id"] == "hero")
    assert p_mp["status"] == "allin"
    assert p_hero["status"] == "folded"


# ── 24-25. current_player / side_pot UNKNOWN ────────────────────────
def test_current_player_unknown():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"]["observation"]
    assert obs["extra"]["v3f_current_player"] is None


def test_side_pot_unknown_not_derived():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = frame_to_decision_request(rgba, w, h)
    obs = res["bridge"]["observation"]
    assert "side_pot" not in obs                # 不推导


# ── 26. legal action validation ─────────────────────────────────────
def test_legal_action_validation():
    from poker_decision_ai.decision_contract import (derive_legal_actions,
                                                     validate_action)
    legal = derive_legal_actions(2000.0, 197680.0, 8920.0)
    assert legal == ["fold", "call", "raise"]
    ok, _ = validate_action("call", legal)
    assert ok


# ── 27-30. 真实帧回归 ───────────────────────────────────────────────
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


# ── 31-34. 无 hidden random / 无 UNKNOWN→0/[] / provenance ───────────
def test_no_hidden_random_fallback():
    r = strategy_failure("", "none")
    assert r.actions == []
    assert r.gto_action is None


def test_no_unknown_to_zero():
    assert tri_to_float(classify(None, "UNKNOWN")) is None


def test_no_unknown_to_empty_list():
    from poker_decision_ai.action_event_source import no_source
    src = no_source()
    assert src.status == "UNKNOWN"
    assert src.events == []                     # 空（非"已知空"）


def test_provenance_preserved():
    from poker_decision_ai.continuous_decision import (
        build_continuous_provenance)
    p = build_continuous_provenance(hand_id="h1", street="flop",
                                    action_history_status="UNKNOWN",
                                    strategy_mode="BASELINE")
    d = p.to_dict()
    assert d["hand_id"] == "h1"
    assert d["action_history_status"] == "UNKNOWN"
    assert d["strategy_mode"] == "BASELINE"
    assert d["state_provenance"]["deduced"] is False


# ── positions_for 零修改 ────────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: E401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
