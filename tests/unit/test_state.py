"""Unit tests for the State Agent (hand-history / vision -> PokerState)."""
from __future__ import annotations

import pytest

from poker_decision_ai.agents.state.agent import StateAgent, StateOutput
from poker_decision_ai.core.schemas import json_schema_for, validate_instance

# 6-max NL0.05/NL0.10（配置里映射为 NL4）；按钮 seat#4，Hero 在 BTN。
# 期望：pot=3.95, current_bet=1.50, to_call=1.50, min_raise_to=3.00, stakes=NL4。
HAND_HISTORY = """PokerStars Hand #123456: No Limit Hold'em ($0.05/$0.10 USD) - 2025/01/01 12:00:00 ET
Table 'Test' 6-max Seat #4 is the button
Seat 1: Alice ($10.00)
Seat 2: Bob ($12.50)
Seat 3: Carol ($8.20)
Seat 4: Hero ($9.75)
Seat 5: Dave ($11.30)
Seat 6: Eve ($10.00)
Dave: posts small blind $0.05
Eve: posts big blind $0.10
*** HOLE CARDS ***
Dealt to Hero [As Kd]
Alice: folds
Bob: folds
Carol: folds
Hero: raises $0.20 to $0.30
Dave: folds
Eve: raises $0.90 to $1.20
Hero: calls $0.90
*** FLOP *** [Qh 7d 2c]
Eve: bets $1.50
"""


@pytest.fixture
def agent():
    return StateAgent()


async def test_schema_contract(agent):
    schema = json_schema_for(StateOutput)
    assert schema["type"] == "object"
    for key in ("hand_id", "game_type", "street", "board", "pot", "to_call",
                "players", "available_actions", "cash", "tournament", "confidence"):
        assert key in schema["properties"]
    out = StateOutput(hand_id="h", game_type="cash", street="preflop")
    validate_instance(StateOutput, out)


async def test_not_applicable_empty(agent, ctx):
    ok, reason = await agent.applicable({}, ctx)
    assert not ok


async def test_applicable_hand_history(agent, ctx):
    ok, _ = await agent.applicable({"request": {"hand_history_text": HAND_HISTORY}}, ctx)
    assert ok


async def test_hand_history_state(agent, ctx):
    env = await agent.run({"request": {"hand_history_text": HAND_HISTORY, "kind": "cash_hand"},
                           "vision": None}, ctx)
    assert env.status in ("ok", "degraded")
    out = env.output

    assert out["street"] == "flop"
    assert out["board"] == ["Qh", "7d", "2c"]
    assert out["pot"] == pytest.approx(3.95)
    assert out["current_bet"] == pytest.approx(1.50)
    assert out["to_call"] == pytest.approx(1.50)
    assert out["min_raise_to"] == pytest.approx(3.00)
    assert out["game_type"] == "cash"
    assert out["cash"]["stakes"] == "NL4"          # sb/bb 反查配置（0.05/0.10 -> NL4）
    assert out["cash"]["sb"] == 0.05
    assert out["cash"]["bb"] == 0.10

    # hero / 位置
    assert out["hero_id"] == "Hero"
    assert out["hero_seat"] == 4
    by_pid = {p["player_id"]: p for p in out["players"]}
    assert by_pid["Hero"]["position"] == "BTN"     # 按钮 seat#4 -> BTN
    assert by_pid["Hero"]["hole_cards"] == ["As", "Kd"]
    assert by_pid["Eve"]["position"] == "BB"       # seat#6 = BB（positions_for 固定顺序）

    # 玩家状态：folded / active（玩家 id = 名字，跨手牌稳定）
    assert by_pid["Hero"]["status"] == "active"
    assert by_pid["Eve"]["status"] == "active"     # Eve（BB）在行动
    assert by_pid["Alice"]["status"] == "folded"
    assert by_pid["Dave"]["status"] == "folded"

    # available_actions / last_aggressor
    assert out["available_actions"] == ["fold", "call", "raise"]
    assert out["last_aggressor"] == "Eve"          # Eve 最后 bet


async def test_hand_history_deterministic(agent, ctx):
    inp = {"request": {"hand_history_text": HAND_HISTORY, "kind": "cash_hand"}, "vision": None}
    e1 = await agent.run(inp, ctx)
    e2 = await agent.run(inp, ctx)
    assert e1.output == e2.output


async def test_vision_observation_mapping(agent, ctx):
    obs = {
        "source": "mock", "hero_seat": 3, "button_seat": 2, "street": "flop",
        "hole_cards": ["As", "Kd"], "board": ["Qh", "7d", "2c"],
        "seats": [
            {"seat": 1, "name": "Alice", "stack": 10.0, "in_hand": False},
            {"seat": 2, "name": "Bob", "stack": 12.5, "in_hand": True},
            {"seat": 3, "name": "Hero", "stack": 9.75, "in_hand": True},
        ],
        "pot": 5.95, "current_bet": 1.5, "to_call": 1.5,
        "blinds": {"sb": 0.05, "bb": 0.10}, "ante": 0.0,
        "available_actions": ["fold", "call", "raise"],
    }
    env = await agent.run({"request": {"vision_observation": obs, "kind": "cash_hand"},
                           "vision": None}, ctx)
    out = env.output
    assert out["game_type"] == "cash"
    assert out["street"] == "flop"
    assert out["board"] == ["Qh", "7d", "2c"]
    assert out["pot"] == pytest.approx(5.95)
    assert out["to_call"] == pytest.approx(1.5)
    assert out["hero_seat"] == 3
    assert out["source"] == "vision"
    assert len(out["players"]) == 3
    hero_p = next(p for p in out["players"] if p["is_hero"])
    assert hero_p["hole_cards"] == ["As", "Kd"]
    assert hero_p["status"] == "active"


async def test_vision_output_priority(agent, ctx):
    """vision 输出里的 observation 作为第二优先级来源。"""
    obs = {"source": "mock", "street": "turn", "board": ["Qh", "7d", "2c", "3s"],
           "pot": 10.0, "blinds": {"sb": 0.05, "bb": 0.10}}
    env = await agent.run({"request": {"kind": "cash_hand"},
                           "vision": {"observation": obs, "provider": "mock"}}, ctx)
    out = env.output
    assert out["street"] == "turn"
    assert out["board"] == ["Qh", "7d", "2c", "3s"]
    assert out["source"] == "vision"
