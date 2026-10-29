"""Phase 9 (V2) Replay Engine 测试：逐街重建 + 全决策点记录 + 结果/质量分离。"""
from __future__ import annotations

from pathlib import Path

from poker_decision_ai.replay_engine import ReplayEngine, _is_decision_point
from poker_decision_ai.domain.poker.hand_history import parse_hand_history
from poker_decision_ai.domain.poker.state_engine import (
    PokerStateEngine, events_from_parsed_hand,
)

ROOT = Path(__file__).parent.parent.parent
FULL = (ROOT / "examples/hands/cash_full_hand.txt").read_text(encoding="utf-8")
SHOWDOWN = (ROOT / "examples/hands/cash_showdown_hand.txt").read_text(encoding="utf-8")


def _events(text: str):
    hand = parse_hand_history(text)
    return events_from_parsed_hand(hand), hand


def test_decision_point_detection_rule():
    evs, _hand = _events(FULL)
    eng = PokerStateEngine(hero_id="Hero")
    points = []
    for i, ev in enumerate(evs):
        eng.apply(ev)
        if _is_decision_point(eng, ev):
            points.append((eng.snapshot()["street"],
                           (ev.get("action") or ev.get("type") or "").lower()))
    # preflop Bob limp、flop check、turn check、turn raise、river bet → 5 个决策点
    assert points == [("preflop", "call"), ("flop", "check"), ("turn", "check"),
                      ("turn", "raise"), ("river", "bet")], points


async def test_replay_full_hand_records_all_decision_points(runtime):
    eng = ReplayEngine(runtime)
    result = await eng.replay_hand(FULL, hero_id="Hero")
    assert result.hand_id == "6001"
    assert len(result.decision_points) == 5
    streets = [d.street for d in result.decision_points]
    assert streets == ["preflop", "flop", "turn", "turn", "river"]

    # §十八 字段全集
    for d in result.decision_points:
        assert d.state and d.state["hero_id"] == "Hero"
        assert d.recommended_action in ("fold", "check", "call", "bet", "raise", "allin")
        assert d.equity is not None and 0.0 <= d.equity <= 1.0
        assert d.pot_odds["required_equity"] is not None
        assert isinstance(d.player_model, dict)
        assert isinstance(d.population_model, dict)
        assert "solver_available" in d.solver_result
        assert d.bluff_probability is not None and 0.0 <= d.bluff_probability <= 1.0
        assert d.confidence is not None
        assert d.game_context and d.game_context["game_mode"] == "cash"
        assert d.provider_status and d.provider_status["solver"]["status"] in ("mock", "real", "unavailable")

    # 真实行动对照
    actuals = [d.actual_action for d in result.decision_points]
    assert actuals == ["raise", "bet", "bet", "call", "fold"], actuals
    # 无摊牌：showdown 仅记录信息
    assert result.showdown_result["happened"] is False


async def test_replay_showdown_hand_records_result(runtime):
    eng = ReplayEngine(runtime)
    result = await eng.replay_hand(SHOWDOWN, hero_id="Hero")
    assert len(result.decision_points) == 5
    assert result.decision_points[-1].actual_action == "call"
    sd = result.showdown_result
    assert sd["happened"] is True
    assert sd["hero_shown"] is True
    assert sd["won"] is False
    assert sd["villain_shown"] and "Queens" in str(sd["villain_shown"])
    # showdown_result 回填到每个决策点
    assert all(d.showdown_result == sd for d in result.decision_points)


async def test_replay_decision_points_persisted_to_research(runtime):
    eng = ReplayEngine(runtime)
    await eng.replay_hand(FULL, hero_id="Hero")
    records = runtime.repos.research.list("decision_point")
    assert len(records) == 5
    assert all(r.get("hand_id") == "6001" for r in records)
    # 决策点包含 actual/recommended 分离
    assert any(r.get("actual_action") == "fold" for r in records)


async def test_ev_estimate_present(runtime):
    eng = ReplayEngine(runtime)
    result = await eng.replay_hand(FULL, hero_id="Hero")
    for d in result.decision_points:
        assert d.ev_estimate is not None
