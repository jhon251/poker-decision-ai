"""Shared pytest fixtures for all agent tests.

测试卫生（重要）：测试必须完全离线、与 .env 解耦。conftest 在导入时强制
Mock Provider 环境变量（load_dotenv 默认不覆盖已存在的环境变量），
确保真实 API Key 存在时测试依然走 Mock，不产生网络调用与费用。
"""
from __future__ import annotations

import os

# ---- 强制离线：pytest 中任何 config 加载都得到 Mock Provider ----
os.environ["LLM_PROVIDER"] = "mock"
os.environ["LLM_API_KEY"] = ""
os.environ["LLM_MODEL"] = ""
os.environ["LLM_DEEP_MODEL"] = ""
os.environ["VISION_PROVIDER"] = "mock"
os.environ["SOLVER_PROVIDER"] = "mock"
# SOLVER-1：同理解耦真实求解器——生产 .env 配置 postflop，但测试默认
# 必须回到"无真实 solver"的 fail-closed 前置（测试内显式构造 provider
# 或 monkeypatch 环境来测真求解路径）。
os.environ.pop("POSTFLOP_SOLVER_PATH", None)


import pytest

from poker_decision_ai.core.config import AppConfig, load_config
from poker_decision_ai.core.logging import EventLogger, MetricsCollector
from poker_decision_ai.core.orchestration.context import AgentContext
from poker_decision_ai.core.providers import build_providers
from poker_decision_ai.core.repositories import build_in_memory


@pytest.fixture(autouse=True)
def _solver_env_isolated():
    """SOLVER-1：load_config→load_dotenv 会在测试中途把生产 .env 的
    POSTFLOP_SOLVER_PATH 灌回环境（dotenv 不覆盖已存在变量，但本变量
    未预先存在）——每测前弹出，保持无真实 solver 的测试前置。"""
    os.environ.pop("POSTFLOP_SOLVER_PATH", None)
    yield


@pytest.fixture
def cfg() -> AppConfig:
    return load_config()


@pytest.fixture
def ctx(cfg, tmp_path):
    """A fully-offline AgentContext (mock providers, in-memory repos)."""
    return AgentContext(
        request_id="test-req", session_id="test-session", config=cfg,
        providers=build_providers(cfg), repos=build_in_memory(),
        logger=EventLogger(str(tmp_path / "logs")), metrics=MetricsCollector(),
    )


@pytest.fixture
def sample_cash_state() -> dict:
    """A canonical 6-max NL10 cash state (hero on BTN facing a bet on the flop)."""
    return {
        "hand_id": "h-test-1",
        "game_type": "cash",
        "street": "flop",
        "board": ["Qh", "7d", "2c"],
        "pot": 10.10,
        "current_bet": 3.75,
        "to_call": 3.75,
        "min_raise_to": 7.50,
        "sb": 0.10,
        "bb": 0.25,
        "ante": 0.0,
        "players": [
            {"player_id": "p1", "name": "Alice", "seat": 1, "stack": 25.0, "status": "folded", "hole_cards": [], "position": "MP", "is_hero": False},
            {"player_id": "p2", "name": "Bob", "seat": 2, "stack": 31.25, "status": "active", "hole_cards": [], "position": "CO", "is_hero": False},
            {"player_id": "hero", "name": "Hero", "seat": 3, "stack": 24.38, "status": "active", "hole_cards": ["As", "Kd"], "position": "BTN", "is_hero": True},
            {"player_id": "p4", "name": "Carol", "seat": 4, "stack": 20.5, "status": "folded", "hole_cards": [], "position": "SB", "is_hero": False},
            {"player_id": "p5", "name": "Dave", "seat": 5, "stack": 28.25, "status": "folded", "hole_cards": [], "position": "BB", "is_hero": False},
            {"player_id": "p6", "name": "Eve", "seat": 6, "stack": 25.0, "status": "folded", "hole_cards": [], "position": "UTG", "is_hero": False},
        ],
        "button_seat": 3,
        "hero_id": "hero",
        "hero_seat": 3,
        "action_history": [
            {"street": "preflop", "player_id": "p2", "action": "raise", "amount": 1.20, "all_in": False, "seq": 1},
            {"street": "preflop", "player_id": "hero", "action": "call", "amount": 1.10, "all_in": False, "seq": 2},
            {"street": "flop", "player_id": "p2", "action": "bet", "amount": 1.50, "all_in": False, "seq": 3},
        ],
        "street_actions": {
            "preflop": [
                {"street": "preflop", "player_id": "p2", "action": "raise", "amount": 1.20, "all_in": False, "seq": 1},
                {"street": "preflop", "player_id": "hero", "action": "call", "amount": 1.10, "all_in": False, "seq": 2},
            ],
            "flop": [
                {"street": "flop", "player_id": "p2", "action": "bet", "amount": 1.50, "all_in": False, "seq": 3},
            ],
        },
        "available_actions": ["fold", "call", "raise"],
        "last_aggressor": "p2",
        "rake": 0.15,
        "source": "hand_history",
        "cash": {"stakes": "NL10", "sb": 0.10, "bb": 0.25, "currency": "USD", "rake_percent": 5.0},
        "tournament": None,
    }


@pytest.fixture
def sample_tournament_state(sample_cash_state) -> dict:
    s = dict(sample_cash_state)
    s["game_type"] = "mtt"
    s["cash"] = None
    s["tournament"] = {
        "tournament_id": "T-1", "name": "Test MTT", "total_players": 100,
        "players_remaining": 45, "total_chips": 100000.0,
        "payouts": [50.0, 30.0, 20.0, 15.0, 10.0, 8.0, 5.0, 4.0, 3.0, 2.0],
        "next_payout": 2.0, "itm_players": 10, "satellite_seats": None,
        "stage": "bubble", "avg_stack": 2222.0,
    }
    s["players"] = [
        {"player_id": "p1", "name": "Alice", "seat": 1, "stack": 1500.0, "status": "folded", "hole_cards": [], "position": "MP", "is_hero": False},
        {"player_id": "p2", "name": "Bob", "seat": 2, "stack": 3000.0, "status": "active", "hole_cards": [], "position": "CO", "is_hero": False},
        {"player_id": "hero", "name": "Hero", "seat": 3, "stack": 2500.0, "status": "active", "hole_cards": ["As", "Kd"], "position": "BTN", "is_hero": True},
        {"player_id": "p4", "name": "Carol", "seat": 4, "stack": 2000.0, "status": "folded", "hole_cards": [], "position": "SB", "is_hero": False},
        {"player_id": "p5", "name": "Dave", "seat": 5, "stack": 1800.0, "status": "folded", "hole_cards": [], "position": "BB", "is_hero": False},
        {"player_id": "p6", "name": "Eve", "seat": 6, "stack": 1200.0, "status": "folded", "hole_cards": [], "position": "UTG", "is_hero": False},
    ]
    s["sb"] = 50.0
    s["bb"] = 100.0
    s["pot"] = 600.0
    s["current_bet"] = 200.0
    s["to_call"] = 200.0
    return s


@pytest.fixture
def runtime(cfg, tmp_path):
    """Full offline Runtime: mock providers + in-memory repos (no disk writes)."""
    from poker_decision_ai.core.orchestration.runtime import Runtime
    from poker_decision_ai.core.orchestration.registry import register_all_agents
    from poker_decision_ai.core.orchestration.runner import PipelineRunner

    providers = build_providers(cfg)
    repos = build_in_memory()
    registry = register_all_agents()
    logger = EventLogger(str(tmp_path / "logs"))
    metrics = MetricsCollector()
    runner = PipelineRunner(registry, cfg, logger, metrics)
    return Runtime(config=cfg, providers=providers, repos=repos, registry=registry,
                   runner=runner, logger=logger, metrics=metrics)


CASH_HAND_TEXT = """PokerStars Hand #9001: No Limit Hold'em ($0.10/$0.25 USD) - 2025/01/10 12:00:00 ET
Table 'T' 6-max Seat #3 is the button
Seat 1: Alice ($25.00)
Seat 2: Bob ($31.25)
Seat 3: Hero ($24.38)
Seat 4: Carol ($20.50)
Seat 5: Dave ($28.25)
Seat 6: Eve ($25.00)
Carol: posts small blind $0.10
Dave: posts big blind $0.25
*** HOLE CARDS ***
Dealt to Hero [As Kd]
Eve: folds
Alice: folds
Bob: calls $0.25
Hero: raises $0.50 to $0.75
Carol: folds
Dave: folds
Bob: calls $0.50
*** FLOP *** [Qh 7d 2c]
Bob: bets $1.50
"""

TOURNAMENT_HAND_TEXT = """PokerStars Hand #9002: Tournament #9, $10+$1 USD Hold'em No Limit - Level VI (50/100) - 2025/01/10 13:00:00 ET
Table 'B' 6-max Seat #6 is the button
Seat 1: Alice (1500 in chips)
Seat 2: Bob (3000 in chips)
Seat 3: Carol (2000 in chips)
Seat 4: Dave (1800 in chips)
Seat 5: Eve (1200 in chips)
Seat 6: Hero (2500 in chips)
Alice: posts small blind $50
Bob: posts big blind $100
*** HOLE CARDS ***
Dealt to Hero [As Kd]
Carol: folds
Dave: raises $150 to $250
Eve: folds
Hero: calls $250
Alice: folds
Bob: folds
*** FLOP *** [Qh 7d 2c]
Dave: bets $400
*** SUMMARY ***
Tournament players remain: 45 (of 100) | Next payout $2.00
Payouts: 50 30 20 15 10 8 5 4 3 2
10 players paid
"""


@pytest.fixture
def cash_hand_text() -> str:
    return CASH_HAND_TEXT


@pytest.fixture
def tournament_hand_text() -> str:
    return TOURNAMENT_HAND_TEXT
