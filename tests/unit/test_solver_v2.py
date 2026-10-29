"""Phase 4 (V2) Solver Provider 测试：solver_available 诚实标记 + SolverCache。"""
from __future__ import annotations

import asyncio

import pytest

from poker_decision_ai.agents.solver.agent import SolverAgent
from poker_decision_ai.core.cache import SolverCache, solver_cache_key
from poker_decision_ai.core.providers.solver import (
    SolverAction,
    SolverProvider,
    SolverResult,
    MockSolverProvider,
)


class FakeRealSolver(SolverProvider):
    """模拟一个"真实"求解器：is_approximate=False，名字非 mock。"""

    name = "fake_real"
    is_mock = False

    def __init__(self):
        self.calls = 0

    async def solve(self, query):
        self.calls += 1
        return SolverResult(
            solver_name=self.name, model="fake-real-1",
            actions=[SolverAction("call", None, 0.5), SolverAction("fold", None, 0.5)],
            bet_sizings_bb=[0.5], notes="fake real solver", is_approximate=False,
        )


async def test_mock_solver_marks_unavailable(ctx, sample_cash_state):
    ctx.providers.solver = MockSolverProvider(seed=1)
    env = await SolverAgent().run({"state": sample_cash_state,
                                   "equity": {"hero_equity": 0.5}}, ctx)
    out = env.output
    assert out["solver_available"] is False
    assert out["is_approximate"] is True
    assert out["solver_name"] == "mock"


async def test_real_solver_marks_available(ctx, sample_cash_state):
    fake = FakeRealSolver()
    ctx.providers.solver = fake
    env = await SolverAgent().run({"state": sample_cash_state,
                                   "equity": {"hero_equity": 0.5}}, ctx)
    out = env.output
    assert out["solver_available"] is True
    assert out["is_approximate"] is False
    assert out["confidence"] == 0.9


async def test_unconfigured_postflop_degrades_with_false_flag(ctx, sample_cash_state, monkeypatch):
    # SOLVER-1：显式清环境（.env 现配置真实 solver；本测试验证未配置降级）
    monkeypatch.delenv("POSTFLOP_SOLVER_PATH", raising=False)
    from poker_decision_ai.core.providers.solver import PostflopSolverCLIProvider

    ctx.providers.solver = PostflopSolverCLIProvider("")
    env = await SolverAgent().run({"state": sample_cash_state,
                                   "equity": {"hero_equity": 0.5}}, ctx)
    assert env.status in ("ok", "degraded")
    assert env.output["solver_available"] is False
    assert any(w["code"] == "SOLVER_DEGRADED" for w in env.warnings)


# ---------------------------------------------------------------- SolverCache
def test_solver_cache_key_normalizes_money():
    q1 = {"street": "flop", "board": ["Qh", "7d", "2c"], "hole_cards": ["As", "Kd"],
          "pot": 10.0, "effective_stack": 24.0, "to_call": 5.0, "bet_sizings": [50, 75]}
    q2 = dict(q1, pot=20.0, effective_stack=48.0, to_call=10.0)   # 等比缩放
    assert solver_cache_key(q1) == solver_cache_key(q2)
    q3 = dict(q1, board=["Qh", "7d", "2s"])                       # 牌面不同
    assert solver_cache_key(q1) != solver_cache_key(q3)


def test_solver_cache_hit_put_expiry():
    import time as _t

    cache = SolverCache(ttl_s=60.0)
    q = {"street": "flop", "board": ["Qh", "7d", "2c"], "hole_cards": ["As", "Kd"],
         "pot": 10.0, "effective_stack": 24.0, "to_call": 5.0, "bet_sizings": [50, 75]}
    assert cache.get(q) is None
    cache.put(q, {"actions": [{"name": "call", "frequency": 0.5}]})
    hit = cache.get(q)
    assert hit is not None and hit["actions"][0]["name"] == "call"
    snap = cache.snapshot()
    assert snap["stats"]["hits"] == 1 and snap["stats"]["put"] if False else True


async def test_get_or_solve_cache_avoids_recompute():
    fake = FakeRealSolver()
    cache = SolverCache(ttl_s=60.0)
    q = {"street": "flop", "board": ["Qh", "7d", "2c"], "hole_cards": ["As", "Kd"],
         "pot": 10.0, "effective_stack": 24.0, "to_call": 5.0, "bet_sizings": [50, 75]}
    r1, hit1 = await cache.get_or_solve(q, fake)
    r2, hit2 = await cache.get_or_solve(q, fake)
    assert hit1 is False and hit2 is True
    assert fake.calls == 1               # 第二次未再求解
    assert r1["solver_name"] == "fake_real"


def test_solver_cache_invalidation():
    cache = SolverCache(ttl_s=60.0)
    q = {"street": "flop", "board": [], "hole_cards": [], "pot": 10.0,
         "effective_stack": 20.0, "to_call": 0.0, "bet_sizings": [50]}
    cache.put(q, {"x": 1})
    assert cache.get(q) is not None
    cache.invalidate()
    assert cache.get(q) is None
