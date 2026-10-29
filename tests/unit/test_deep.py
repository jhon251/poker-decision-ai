"""Phase 8 (V2) Deep/Background + Decision Revision 测试。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from poker_decision_ai.deep import DecisionRevision, DeepRunner
from poker_decision_ai.realtime import RealtimeRunner

from tests.conftest import CASH_HAND_TEXT


def test_decision_revision_record_shape():
    rev = DecisionRevision(
        revision_id="r1:revision:deep", request_id="r1",
        realtime_action="CALL", deep_action="FOLD",
        realtime_confidence=0.7, deep_confidence=0.55,
        confidence_change=-0.15, reason="deep 复核", evidence=["e1"],
        model_versions={"bluff": "bluff_v1"},
    )
    d = rev.to_dict()
    assert d["realtime_action"] == "CALL" and d["deep_action"] == "FOLD"
    assert d["confidence_change"] == -0.15
    assert d["model_versions"]["bluff"] == "bluff_v1"
    assert d["created_at"]


def test_deep_plan_is_full_pipeline(runtime):
    dr = DeepRunner(runtime)
    gc = __import__("poker_decision_ai.core.gamemode", fromlist=["GameContext"]).GameContext(
        game_mode="cash")
    plugin = dr.mode_registry.resolve(gc)
    plan = dr._deep_plan(plugin)
    agents = [t.agent for s in plan.stages for t in s.tasks]
    # Deep 路径包含完整 Agent 集（含 debate/evaluation/learning）
    for name in ("debate", "evaluation", "learning", "research", "emotion",
                 "image", "strategy_evolution", "judge", "deviation"):
        assert name in agents
    # 宽松超时
    for s in plan.stages:
        for t in s.tasks:
            assert t.timeout_ms >= 5000


async def test_realtime_then_deep_full_cycle(runtime):
    """Realtime 决策 → Deep 后台复核：两条记录并存，realtime 不被覆盖。"""
    rr = RealtimeRunner(runtime)
    rt_report = await rr.decide({
        "kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT, "hero_id": "Hero",
    })
    rt_rec = runtime.repos.decisions.get(rt_report.request_id)
    assert rt_rec["path"] == "realtime"

    dr = DeepRunner(runtime)
    deep_report = await dr.deepen({
        "kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT, "hero_id": "Hero",
    }, rt_report)
    assert deep_report.status in ("ok", "degraded")
    assert deep_report.final_decision is not None
    # deep 记录独立存在
    deep_rec = runtime.repos.decisions.get(deep_report.request_id)
    assert deep_rec["path"] == "deep"
    # realtime 记录原样保留（未被覆盖）
    rt_rec_after = runtime.repos.decisions.get(rt_report.request_id)
    assert rt_rec_after["path"] == "realtime"
    assert rt_rec_after["final_decision"]["action"] == rt_rec["final_decision"]["action"]
    # revision：动作一致时为空；若不一致则记录完整
    if deep_report.revision:
        assert deep_report.revision["realtime_action"]
        assert deep_report.revision["deep_action"]
        rev_rec = runtime.repos.decisions.get(deep_report.revision["revision_id"])
        assert rev_rec["path"] == "revision"


def test_revision_writer_preserves_realtime_when_actions_differ(runtime):
    """构造 CALL→FOLD 差异：revision 独立记录，realtime 记录不动。"""
    from poker_decision_ai.core.schemas import utc_now_iso

    rt_id = "rt-x"
    runtime.repos.decisions.save({
        "request_id": rt_id, "path": "realtime",
        "base_decision": {"action": "call", "confidence": 0.7},
        "final_decision": {"action": "call", "confidence": 0.7},
        "created_at": utc_now_iso(),
    })
    rev = DecisionRevision(
        revision_id=f"{rt_id}:revision:deep", request_id=rt_id,
        realtime_action="call", deep_action="fold",
        realtime_confidence=0.7, deep_confidence=0.5,
        confidence_change=-0.2, reason="deep 复核",
        evidence=["e"], model_versions={"bluff": "bluff_v1"},
    )
    runtime.repos.decisions.save({
        "request_id": rev.revision_id, "path": "revision",
        "revision": rev.to_dict(), "created_at": utc_now_iso(),
    })
    assert runtime.repos.decisions.get(rt_id)["path"] == "realtime"
    assert runtime.repos.decisions.get(rt_id)["final_decision"]["action"] == "call"
    got = runtime.repos.decisions.get(rev.revision_id)
    assert got["revision"]["deep_action"] == "fold"


async def test_spawn_runs_in_background(runtime):
    rr = RealtimeRunner(runtime)
    rt_report = await rr.decide({
        "kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT, "hero_id": "Hero",
    })
    dr = DeepRunner(runtime)
    task = dr.spawn({"kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT,
                     "hero_id": "Hero"}, rt_report)
    assert isinstance(task, asyncio.Task)
    deep_report = await task
    assert deep_report.final_decision is not None


async def test_llm_tier_resolution_in_deep_context(ctx):
    """Deep 上下文（ctx.extra["deep"]）→ llm_json 自动用 deep tier（mock 下安全）。"""
    @dataclass
    class Simple:
        note: str

    ctx.extra["deep"] = {"llm_tier": "deep", "allow_llm": True}
    r = await ctx.llm_json(system="s", user="u", schema_cls=Simple)
    assert r is not None
    # 无 deep 上下文 → fast
    ctx.extra.pop("deep")
    r2 = await ctx.llm_json(system="s", user="u", schema_cls=Simple)
    assert r2 is not None
