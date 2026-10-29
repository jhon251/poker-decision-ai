"""Phase 6 (V2) Event Driven State Engine 测试：12 类事件 + 增量更新 + 与解析器等价。"""
from __future__ import annotations

import pytest

from poker_decision_ai.domain.poker.hand_history import parse_hand_history
from poker_decision_ai.domain.poker.state_engine import (
    PokerStateEngine,
    StateEngineError,
    events_from_parsed_hand,
)

from tests.conftest import CASH_HAND_TEXT


def _seats() -> list:
    return [
        {"seat": 1, "name": "Alice", "stack": 25.0, "is_hero": False},
        {"seat": 2, "name": "Bob", "stack": 31.25, "is_hero": False},
        {"seat": 3, "name": "Hero", "stack": 24.38, "is_hero": True},
        {"seat": 4, "name": "Carol", "stack": 20.5, "is_hero": False},
        {"seat": 5, "name": "Dave", "stack": 28.25, "is_hero": False},
        {"seat": 6, "name": "Eve", "stack": 25.0, "is_hero": False},
    ]


def _started_engine() -> PokerStateEngine:
    eng = PokerStateEngine(hero_id="Hero")
    eng.apply({"type": "HAND_START", "hand_id": "h1", "game_type": "cash",
               "sb": 0.10, "bb": 0.25, "button_seat": 3, "seats": _seats()})
    return eng


def test_hand_start_hole_cards():
    eng = _started_engine()
    s = eng.snapshot()
    assert s["street"] == "preflop" and s["bb"] == 0.25
    assert len(s["players"]) == 6
    hero = next(p for p in s["players"] if p["is_hero"])
    assert hero["position"] == "BTN"          # 按钮 seat 3 → BTN（轮转）
    eng.apply({"type": "HOLE_CARDS", "player_id": "Hero", "cards": ["As", "Kd"]})
    s = eng.snapshot()
    hero_hole = next(p for p in s["players"] if p["is_hero"])["hole_cards"]
    assert hero_hole == ["As", "Kd"]


def test_incremental_betting_and_streets():
    eng = _started_engine()
    # 盲注（post 事件）
    eng.apply({"type": "PLAYER_ACTION", "street": "preflop", "player_id": "Carol",
               "action": "post", "amount": 0.10})
    eng.apply({"type": "PLAYER_ACTION", "street": "preflop", "player_id": "Dave",
               "action": "post", "amount": 0.25})
    # Bob call 0.25
    eng.apply({"type": "CALL", "street": "preflop", "player_id": "Bob", "amount": 0.25})
    # Hero raise to 0.75（增量 0.75）
    eng.apply({"type": "RAISE", "street": "preflop", "player_id": "Hero",
               "amount": 0.50, "to": 0.75})
    eng.apply({"type": "CALL", "street": "preflop", "player_id": "Bob", "amount": 0.50})
    s = eng.snapshot()
    assert s["pot"] == pytest.approx(1.85)      # 0.10+0.25+0.25+0.75+0.50
    assert s["current_bet"] == pytest.approx(0.75)
    # 翻牌
    eng.apply({"type": "FLOP", "street": "flop", "board": ["Qh", "7d", "2c"]})
    s = eng.snapshot()
    assert s["street"] == "flop" and s["board"] == ["Qh", "7d", "2c"]
    assert s["current_bet"] == 0.0 and s["to_call"] == 0.0    # 新街重置
    # Bob 下注 1.50
    eng.apply({"type": "BET", "street": "flop", "player_id": "Bob", "amount": 1.50, "to": 1.50})
    s = eng.snapshot()
    assert s["pot"] == pytest.approx(3.35)
    assert s["to_call"] == pytest.approx(1.50)
    assert s["min_raise_to"] == pytest.approx(3.00)
    assert s["last_aggressor"] == "Bob"
    assert s["available_actions"] == ["fold", "call", "raise"]


def test_fold_and_allin_status():
    eng = _started_engine()
    eng.apply({"type": "FOLD", "street": "preflop", "player_id": "Alice"})
    eng.apply({"type": "ALL_IN", "street": "preflop", "player_id": "Bob",
               "action": "raise", "amount": 30.0, "to": 31.25, "all_in": True})
    s = eng.snapshot()
    by_name = {p["player_id"]: p for p in s["players"]}
    assert by_name["Alice"]["status"] == "folded"
    assert by_name["Bob"]["status"] == "allin"
    assert len(s["action_history"]) == 2


def test_street_order_violation_raises():
    eng = _started_engine()
    with pytest.raises(StateEngineError):
        eng.apply({"type": "TURN", "street": "turn", "board": ["Qh", "7d", "2c", "3s"]})


def test_unknown_event_raises():
    eng = _started_engine()
    with pytest.raises(StateEngineError):
        eng.apply({"type": "WARP_SPEED"})


def test_events_from_hand_history_match_state_agent():
    """事件流重放 CASH_HAND_TEXT → 与 State Agent 的解析结果一致。"""
    import asyncio

    from poker_decision_ai.agents.state.agent import StateAgent
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    cfg = load_config()
    ctx = AgentContext("t", "s", cfg, build_providers(cfg), build_in_memory(),
                       EventLogger("logs"), MetricsCollector())
    env = asyncio.run(StateAgent().run(
        {"request": {"hand_history_text": CASH_HAND_TEXT, "hero_id": "Hero"}}, ctx))
    agent_state = env.output

    hand = parse_hand_history(CASH_HAND_TEXT)
    eng = PokerStateEngine(hero_id="Hero")
    for ev in events_from_parsed_hand(hand):
        eng.apply(ev)
    eng_state = eng.snapshot()

    assert eng_state["street"] == agent_state["street"] == "flop"
    assert eng_state["board"] == agent_state["board"]
    assert eng_state["pot"] == pytest.approx(agent_state["pot"])          # 3.35
    assert eng_state["to_call"] == pytest.approx(agent_state["to_call"])  # 1.50
    assert eng_state["last_aggressor"] == agent_state["last_aggressor"]
    eng_hero = next(p for p in eng_state["players"] if p["is_hero"])
    agent_hero = next(p for p in agent_state["players"] if p["is_hero"])
    assert eng_hero["position"] == agent_hero["position"] == "BTN"


def test_events_list_grows_incrementally():
    eng = _started_engine()
    for _ in range(3):
        eng.apply({"type": "FOLD", "street": "preflop", "player_id": "Alice"})
    assert len(eng.events) == 4    # HAND_START + 3
    assert eng.snapshot()["street"] == "preflop"
