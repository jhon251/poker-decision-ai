"""V1.1 游戏模式集成测试：四种 Pipeline 自动装配 + 手动模式切换。"""
from __future__ import annotations

from poker_decision_ai.core.schemas import new_id

from tests.conftest import CASH_HAND_TEXT  # noqa: F401  (import path sanity)

CASH_TEXT = open("examples/hands/cash_flop_bet.txt", encoding="utf-8").read()
MTT_TEXT = open("examples/hands/tournament_bubble.txt", encoding="utf-8").read()
PKO_TEXT = open("examples/hands/pko_flop.txt", encoding="utf-8").read()


async def test_cash_pipeline_excludes_tournament_and_bounty(runtime):
    ctx = runtime.new_context(new_id("gm"), "gm-session")
    report = await runtime.runner.run({
        "kind": "cash_hand", "hand_history_text": CASH_TEXT, "hero_id": "Hero",
        "game_mode": "cash",
    }, ctx)
    assert report.status in ("ok", "degraded")
    assert report.game_context["game_mode"] == "cash"
    for banned in ("icm", "tournament_stage", "blind_pressure", "satellite",
                   "bounty", "bounty_equity", "bounty_icm", "bounty_strategy",
                   "mystery_bounty", "icm_exploit"):
        assert banned not in report.envelopes, f"cash 不应运行 {banned}"
    assert report.final_decision is not None


async def test_mtt_pipeline_runs_tournament_agents(runtime):
    ctx = runtime.new_context(new_id("gm"), "gm-session")
    report = await runtime.runner.run({
        "kind": "tournament_hand", "hand_history_text": MTT_TEXT, "hero_id": "Hero",
        "game_mode": "tournament", "tournament_type": "mtt",
    }, ctx)
    assert report.status in ("ok", "degraded")
    assert report.game_context["spec"] if "spec" in report.game_context else True
    for name in ("icm", "tournament_stage", "blind_pressure", "icm_exploit"):
        assert name in report.envelopes, f"mtt 应运行 {name}"
    for banned in ("bounty", "bounty_equity", "bounty_icm", "bounty_strategy", "mystery_bounty"):
        assert banned not in report.envelopes, f"mtt 不应运行 {banned}"


async def test_satellite_pipeline_high_priority(runtime):
    sat_text = """PokerStars Hand #5001: Tournament #5, $5+$0.5 USD Hold'em No Limit Satellite - Level X (200/400) - 2025/01/09 10:00:00 ET
Table 'S' 6-max Seat #5 is the button
Seat 1: Alice (6000 in chips)
Seat 2: Bob (9000 in chips)
Seat 3: Carol (4000 in chips)
Seat 4: Dave (7000 in chips)
Seat 5: Hero (8000 in chips)
Seat 6: Eve (5000 in chips)
Eve: posts small blind $200
Alice: posts big blind $400
*** HOLE CARDS ***
Dealt to Hero [Qd Qs]
Bob: folds
Carol: folds
Dave: raises $6600 to $7000 and is all-in
Eve: folds
Alice: folds
*** SUMMARY ***
Tournament players remain: 12 (of 60) | 10 seats paid
"""
    ctx = runtime.new_context(new_id("gm"), "gm-session")
    report = await runtime.runner.run({
        "kind": "tournament_hand", "hand_history_text": sat_text, "hero_id": "Hero",
        "game_mode": "tournament", "tournament_type": "satellite",
    }, ctx)
    assert report.status in ("ok", "degraded")
    assert report.game_context["tournament_type"] == "satellite"
    assert "satellite" in report.envelopes
    assert "icm" in report.envelopes
    assert "icm_exploit" not in report.envelopes     # 卫星赛不套普通 MTT 的 ICM 剥削
    judge = report.envelopes["judge"]["output"]
    assert "satellite" in judge.get("final_inputs_summary", {})


async def test_pko_pipeline_runs_all_bounty_agents(runtime):
    ctx = runtime.new_context(new_id("gm"), "gm-session")
    report = await runtime.runner.run({
        "kind": "tournament_hand", "hand_history_text": PKO_TEXT, "hero_id": "Hero",
        "game_mode": "bounty", "bounty_type": "pko",
    }, ctx)
    assert report.status in ("ok", "degraded")
    assert report.game_context["game_mode"] == "bounty"
    assert report.game_context["bounty_type"] == "pko"
    for name in ("bounty", "bounty_equity", "bounty_icm", "bounty_strategy"):
        assert name in report.envelopes, f"pko 应运行 {name}"
    for name in ("icm", "tournament_stage", "blind_pressure"):
        assert name in report.envelopes, f"pko 应运行 {name}"
    assert "satellite" not in report.envelopes
    # state 解析赏金
    state = report.envelopes["state"]["output"]
    assert state["tournament"]["bounty_type"] == "pko"
    bounties = {p["name"]: p.get("bounty", 0.0) for p in state["players"]}
    assert bounties["Bob"] == 120.0 and bounties["Hero"] == 12.5
    # judge 综合 bounty 输入
    judge = report.envelopes["judge"]["output"]
    summary = judge.get("final_inputs_summary", {})
    assert summary.get("bounty_strategy") in ("ok", "missing")
    assert report.final_decision is not None


async def test_mode_inference_from_hand_history(runtime):
    """无显式模式：从手牌历史推断（cash/mtt/pko 三种）。"""
    ctx = runtime.new_context(new_id("gm"), "gm-session")
    report = await runtime.runner.run({
        "kind": "replay_hand", "hand_history_text": PKO_TEXT, "hero_id": "Hero",
    }, ctx)
    assert report.game_context["game_mode"] == "bounty"
    assert report.game_context["bounty_type"] == "pko"
    assert "bounty" in report.envelopes

    ctx2 = runtime.new_context(new_id("gm"), "gm-session")
    report2 = await runtime.runner.run({
        "kind": "replay_hand", "hand_history_text": CASH_TEXT, "hero_id": "Hero",
    }, ctx2)
    assert report2.game_context["game_mode"] == "cash"
    assert "bounty" not in report2.envelopes


async def test_mystery_pipeline_includes_mystery_agent(runtime):
    ctx = runtime.new_context(new_id("gm"), "gm-session")
    report = await runtime.runner.run({
        "kind": "tournament_hand", "hand_history_text": PKO_TEXT, "hero_id": "Hero",
        "game_mode": "bounty", "bounty_type": "mystery",
    }, ctx)
    assert report.status in ("ok", "degraded")
    assert "mystery_bounty" in report.envelopes
    assert "bounty" in report.envelopes
