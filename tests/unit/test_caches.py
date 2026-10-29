"""Phase 5 (V2) Cache 体系测试：六个缓存 + CacheManager + Agent 接线。"""
from __future__ import annotations

import asyncio
import time

from poker_decision_ai.core.cache import (
    CacheManager,
    EquityCache,
    FeatureCache,
    PlayerModelCache,
    PopulationCache,
    RangeCache,
    SolverCache,
)


def test_ttl_expiry_and_invalidation():
    c = FeatureCache(ttl_s=0.05)
    c.put("k", {"v": 1})
    assert c.get("k") == {"v": 1}
    time.sleep(0.08)
    assert c.get("k") is None            # TTL 过期
    snap = c.snapshot()
    assert snap["stats"]["expirations"] == 1


def test_version_bump_invalidates():
    m = CacheManager()
    m.population.put("NL10:cash", {"vpip": {"value": 0.2, "samples": 100}})
    assert m.population.get("NL10:cash") is not None
    m.bump_version("population", "population_v2")
    assert m.population.get("NL10:cash") is None
    assert m.version("population") == "population_v2"
    assert m.population.default_version == "population_v2"


def test_targeted_invalidation():
    m = CacheManager()
    m.player.put("Bob", {"v": 1})
    m.player.put("Eve", {"v": 2})
    m.invalidate("player", "Bob")
    assert m.player.get("Bob") is None
    assert m.player.get("Eve") is not None


def test_equity_cache_key_signature():
    k1 = EquityCache.key(["As", "Kd"], ["Qh", "7d", "2c"], "sig-a", 20000, 42)
    k2 = EquityCache.key(["As", "Kd"], ["Qh", "7d", "2c"], "sig-a", 20000, 42)
    k3 = EquityCache.key(["As", "Kd"], ["Qh", "7d", "2c"], "sig-b", 20000, 42)
    k4 = EquityCache.key(["As", "Kd"], ["Qh", "7d", "2c"], "sig-a", 5000, 42)
    assert k1 == k2 and k1 != k3 and k1 != k4


def test_solver_cache_in_manager():
    m = CacheManager()
    q = {"street": "flop", "board": [], "hole_cards": [], "pot": 10.0,
         "effective_stack": 20.0, "to_call": 0.0, "bet_sizings": [50]}
    m.solver.put(q, {"actions": []})
    assert m.solver.get(q) is not None
    snap = m.snapshot()
    assert snap["model_versions"]["bluff"] == "bluff_v1"
    assert "solver" in snap


async def test_equity_agent_uses_cache(runtime, cash_hand_text):
    """同输入两次运行：第二次命中 EquityCache（缓存命中计数 +1，输出一致）。"""
    from poker_decision_ai.core.schemas import new_id

    runtime.caches = CacheManager()
    ctx1 = runtime.new_context(new_id("c1"), "c")
    report1 = await runtime.runner.run({
        "kind": "cash_hand", "hand_history_text": cash_hand_text, "hero_id": "Hero",
    }, ctx1)
    eq1 = report1.envelopes["equity"]["output"]
    ctx2 = runtime.new_context(new_id("c2"), "c")
    report2 = await runtime.runner.run({
        "kind": "cash_hand", "hand_history_text": cash_hand_text, "hero_id": "Hero",
    }, ctx2)
    eq2 = report2.envelopes["equity"]["output"]
    assert abs(eq1["hero_equity"] - eq2["hero_equity"]) < 1e-9
    stats = runtime.caches.equity.snapshot()["stats"]
    assert stats["hits"] >= 1        # 第二次运行至少命中一次
    # 第二次运行明显更快（缓存跳过 MC）
    assert report2.latency_ms < report1.latency_ms


def test_player_and_range_cache_basic():
    m = CacheManager()
    m.player.put("Hero", {"sample_size": 1200})
    m.range.put(RangeCache.key("BTN", "open", "Hero", "range_v1"), {"combo_count": 200})
    assert m.player.get("Hero")["sample_size"] == 1200
    assert m.range.get(RangeCache.key("BTN", "open", "Hero", "range_v1")) is not None
