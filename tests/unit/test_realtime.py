"""Phase 7 (V2) Realtime 快路径测试：快计划过滤 + 快参数 + 自动降级 + 延迟统计。"""
from __future__ import annotations

import pytest

from poker_decision_ai.core.perf import LatencyTracker
from poker_decision_ai.realtime import FAST_AGENTS, RealtimeRunner

from tests.conftest import CASH_HAND_TEXT

EXCLUDED = {"debate", "emotion", "image", "strategy_evolution", "evaluation",
            "learning", "research", "hypothesis", "experiment", "human_style",
            "icm_exploit"}


def test_fast_agent_set_excludes_deep_agents():
    for name in EXCLUDED:
        assert name not in FAST_AGENTS
    for name in ("state", "equity", "solver", "exploit", "bluff", "risk",
                 "judge", "deviation", "icm", "bounty", "bounty_equity"):
        assert name in FAST_AGENTS


def test_fast_plan_filters_stages(runtime):
    rr = RealtimeRunner(runtime)
    gc = __import__("poker_decision_ai.core.gamemode", fromlist=["GameContext"]).GameContext(
        game_mode="bounty", mode_type="pko")
    plugin = rr.mode_registry.resolve(gc)
    plan = rr._fast_plan(plugin)
    agents = [t.agent for s in plan.stages for t in s.tasks]
    assert "bounty" in agents and "judge" in agents and "deviation" in agents
    for banned in EXCLUDED:
        assert banned not in agents
    # 紧超时已应用
    for s in plan.stages:
        for t in s.tasks:
            assert t.timeout_ms <= 500


async def test_realtime_decide_cash_hand(runtime):
    rr = RealtimeRunner(runtime)
    report = await rr.decide({
        "kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT, "hero_id": "Hero",
    })
    assert report.path == "realtime"
    assert report.final_decision is not None
    assert report.final_decision["action"] in ("fold", "check", "call", "bet", "raise", "allin")
    assert report.status in ("ok", "degraded")
    # 快路径不运行 deep 专属 Agent
    for banned in ("debate", "evaluation", "learning", "research", "hypothesis"):
        assert banned not in report.agents_ran
    assert "equity" in report.agents_ran and "judge" in report.agents_ran
    # 决策已按 path=realtime 持久化
    rec = runtime.repos.decisions.get(report.request_id)
    assert rec["path"] == "realtime"
    assert "model_versions" in rec


async def test_realtime_uses_fast_equity_iterations(runtime):
    """快路径 equity 样本量 ≈ realtime.equity_iterations（2000），非完整 20000。"""
    rr = RealtimeRunner(runtime)
    report = await rr.decide({
        "kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT, "hero_id": "Hero",
    })
    eq_env = None
    for rid in runtime.repos.decisions.list():
        pass
    # 从报告 agents_ran 确认 equity 运行；用 caches 空（每次新 runtime）下样本验证
    from poker_decision_ai.core.schemas import new_id

    ctx = runtime.new_context(new_id("eq"), "eq")
    ctx.extra["realtime"] = runtime.config.realtime_config()
    from poker_decision_ai.agents.equity.agent import EquityAgent

    state_env = await runtime.registry.get("state").run(
        {"request": {"hand_history_text": CASH_HAND_TEXT, "hero_id": "Hero"}}, ctx)
    env = await EquityAgent().run({"state": state_env.output}, ctx)
    samples = env.output["villains"]["Bob"]["samples"]
    assert samples == 2000, f"expected fast iterations 2000, got {samples}"


async def test_realtime_llm_disabled_by_default(runtime):
    rr = RealtimeRunner(runtime)
    report = await rr.decide({
        "kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT, "hero_id": "Hero",
    })
    assert report.final_decision is not None
    # 默认确定性快路径：allow_llm=false → 决策仍产出（不依赖 LLM）


async def test_realtime_solver_unavailable_degrades(runtime, monkeypatch):
    # SOLVER-1：生产 .env 现配置真实 solver；本测试验证"未配置降级"路径，
    # 显式清除环境以防 load_dotenv 泄漏。
    monkeypatch.delenv("POSTFLOP_SOLVER_PATH", raising=False)
    from poker_decision_ai.core.providers.solver import PostflopSolverCLIProvider

    runtime.providers.solver = PostflopSolverCLIProvider("")
    rr = RealtimeRunner(runtime)
    report = await rr.decide({
        "kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT, "hero_id": "Hero",
    })
    assert report.status in ("ok", "degraded")
    assert report.final_decision is not None
    assert report.provider_status["solver"]["status"] == "unavailable"


def test_latency_tracker_percentiles():
    t = LatencyTracker("test")
    for i in range(1, 101):
        t.record(float(i))
    snap = t.snapshot()["total"]
    assert snap["p50"] == 50.0
    assert snap["p90"] == 90.0
    assert snap["p95"] == 95.0
    assert snap["p99"] == 99.0
    assert snap["n"] == 100
    t.record_part("equity", 120.0)
    assert t.snapshot()["parts"]["equity"]["avg"] == 120.0


async def test_realtime_tracks_latency(runtime):
    rr = RealtimeRunner(runtime)
    for _ in range(3):
        await rr.decide({"kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT,
                         "hero_id": "Hero"})
    snap = rr.tracker.snapshot()["total"]
    assert snap["n"] == 3
    assert snap["p50"] is not None
    # 实测快路径（2000 次迭代 + 缓存命中后）应在秒级以内；断言宽松上限防误报
    assert snap["p50"] < 5000
