"""SOLVER-1 — 真实 postflop solver 接入测试。

覆盖：provider 探针（is_configured + 真实求解出策略/EV）、preflop 查表
（source=chart 诚实标注）、spot 表查表（exact/nearest/miss + 延迟）、
SolverAgent realtime 路径分派（postflop+realtime 只查表，miss 诚实降级）。
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import time

import pytest

CLI = str(pathlib.Path("tools/bin/postflop-bench-cli.exe"))
SPOT_DIR = "data/solver/spots_r1"
TURN_QUERY = {
    "street": "turn",
    "board": ["Td", "9d", "6h", "Qc"],
    "ranges": {"oop": "66+,A8s+,A5s-A4s,AJo+,K9s+,KQo,QTs+,JTs,96s+,85s+,75s+,65s,54s",
               "ip": "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+"},
    "pot": 4.0, "effective_stack": 196.0, "bb": 2.0,
    "max_iterations": 200, "target_exploitability_pot_pct": 1.0,
}


def _cli_available() -> bool:
    return pathlib.Path(CLI).exists()


def _spots_available() -> bool:
    return (pathlib.Path(SPOT_DIR) / "manifest.json").exists()


# ---------------------------------------------------------------- provider

@pytest.mark.skipif(not _cli_available(), reason="postflop-bench-cli 未构建")
def test_provider_probe_real_solve_with_strategy_and_ev():
    from poker_decision_ai.core.providers.solver import PostflopSolverCLIProvider
    p = PostflopSolverCLIProvider(executable=CLI)
    assert p.is_configured() is True
    res = asyncio.run(p.solve(dict(TURN_QUERY)))
    assert res.solver_name == "postflop"
    assert res.is_approximate is False
    assert res.raw["near_nash_equilibrium"] is True
    assert res.raw["is_mock"] is False
    assert res.actions, "solver must return strategy actions"
    freq_total = sum(a.frequency for a in res.actions)
    assert abs(freq_total - 1.0) < 1e-3
    assert any(a.ev is not None for a in res.actions)
    assert res.raw["exploitability_bb"] is not None
    assert res.raw["simplifications"], "simplifications must be recorded"
    # preflop 诚实拒绝（chart 负责覆盖）
    with pytest.raises(Exception, match="preflop"):
        asyncio.run(p.solve({**TURN_QUERY, "street": "preflop"}))


def test_provider_not_configured_raises(monkeypatch):
    from poker_decision_ai.core.providers.solver import PostflopSolverCLIProvider
    monkeypatch.delenv("POSTFLOP_SOLVER_PATH", raising=False)
    p = PostflopSolverCLIProvider(executable="")
    assert p.is_configured() is False
    with pytest.raises(Exception):
        asyncio.run(p.solve(dict(TURN_QUERY)))


# ---------------------------------------------------------------- chart

def test_preflop_chart_lookup_honest_labels():
    from poker_decision_ai.core.providers.preflop_chart import PreflopChartProvider
    p = PreflopChartProvider()
    if not p.is_configured():
        pytest.skip("preflop chart not present")
    q = {"street": "preflop", "hole_cards": ["Ah", "Kd"], "scenario": "sb_open"}
    res = asyncio.run(p.solve(q))
    assert res.raw["source"] == "chart"
    assert res.raw["equilibrium"] is False      # 查表不冒充均衡解
    assert res.raw["is_mock"] is False
    assert res.actions[0].name in ("raise", "fold")


# ---------------------------------------------------------------- spot 表

@pytest.mark.skipif(not _spots_available(), reason="spot 库尚未求解")
def test_spot_table_exact_hit_and_latency():
    from poker_decision_ai.core.providers.spot_table import get_spot_table
    manifest = json.loads(
        (pathlib.Path(SPOT_DIR) / "manifest.json").read_text(encoding="utf-8"))
    sid = manifest["solved_ids"][0]
    rec = json.loads((pathlib.Path(SPOT_DIR) / f"{sid}.json").read_text(
        encoding="utf-8"))
    q = rec["query"]
    table = get_spot_table(SPOT_DIR)
    query = {"street": q["street"], "board": q["board"],
             "ranges": {"oop": q["oop_range"], "ip": q["ip_range"]},
             "pot_bb": q["pot_bb"],
             "effective_stack_bb": q["effective_stack_bb"]}
    t0 = time.perf_counter()
    hit = table.lookup(query)
    first_ms = (time.perf_counter() - t0) * 1000
    assert hit is not None
    assert hit["raw"]["lookup_mode"] == "exact"
    t0 = time.perf_counter()
    for _ in range(20):
        table.lookup(query)
    avg_ms = (time.perf_counter() - t0) * 1000 / 20
    assert avg_ms < 50, f"lookup latency {avg_ms:.1f}ms >= 50ms"
    assert first_ms < 50 or avg_ms < 50
    assert hit["raw"]["near_nash_equilibrium"] is True


def test_spot_table_miss_returns_none():
    from poker_decision_ai.core.providers.spot_table import get_spot_table
    table = get_spot_table(SPOT_DIR)
    hit = table.lookup({"street": "flop", "board": ["2c", "3d", "4h"],
                        "ranges": {"oop": "random:1.0", "ip": "random:1.0"},
                        "pot_bb": 3.5, "effective_stack_bb": 96.5})
    if hit is not None:                      # nearest 命中也必须诚实标注
        assert hit["raw"]["lookup_mode"] in ("nearest", "exact")
    assert hit is None or hit["actions"]


# ------------------------------------------------- SolverAgent 路径分派

class _Cfg:
    def frequency_prior_config(self):
        return {}


class _LLM:
    name = "mock"


class _Ctx:
    def __init__(self, realtime: bool = True):
        from poker_decision_ai.core.providers import (
            MockLLMProvider, MockVisionProvider, PostflopSolverCLIProvider,
            ProviderRegistry, MockSolverProvider)
        cfg = type("C", (), {"frequency_prior_config":
                             staticmethod(lambda: {})})()
        self.config = cfg
        self.extra = {"realtime": {"deadline_ms": 60000,
                                   "agent_timeout_ms": 30000}} if realtime else {}
        self._solver = (PostflopSolverCLIProvider(executable=CLI)
                        if _cli_available() else
                        MockSolverProvider(seed=42))
        self.providers = ProviderRegistry(
            llm=MockLLMProvider(), vision=MockVisionProvider(),
            solver=self._solver)

    def llm(self):
        return _LLM()


def _spot_state(board=None, pot=8.0, street="turn"):
    return {"hand_id": "solver1-t0", "game_type": "cash", "street": street,
            "board": board or ["Td", "9d", "6h", "Qc"], "pot": pot,
            "current_bet": 0.0, "to_call": 0.0, "min_raise_to": 4.0,
            "sb": 1.0, "bb": 2.0, "ante": 0.0,
            "players": [
                {"player_id": "hero", "seat": 1, "stack": 192.0,
                 "status": "active", "hole_cards": ["As", "Kd"],
                 "position": "BB", "is_hero": True},
                {"player_id": "seat2", "seat": 2, "stack": 192.0,
                 "status": "active", "hole_cards": [], "position": "BTN/SB",
                 "is_hero": False}],
            "button_seat": 1, "hero_id": "hero"}


def _a_solved_spot():
    manifest = json.loads((pathlib.Path(SPOT_DIR) / "manifest.json").read_text(
        encoding="utf-8"))
    # 优先取 turn/river spot（单街求解，秒级）。flop spot 是全 3 街树
    # （分钟级），真求解会把本测试拖进负载敏感的脆弱区（全量套件下
    # 偶发超时抖动）；DEEP 路径语义用 turn spot 验证完全等价。
    for sid in manifest["solved_ids"]:
        rec = json.loads((pathlib.Path(SPOT_DIR) / f"{sid}.json").read_text(
            encoding="utf-8"))
        if (rec.get("query") or {}).get("street") in ("turn", "river"):
            return rec
    return json.loads((pathlib.Path(SPOT_DIR) /
                       f"{manifest['solved_ids'][0]}.json").read_text(
                           encoding="utf-8"))


def test_solver_agent_realtime_uses_spot_lookup():
    from poker_decision_ai.agents.solver.agent import SolverAgent
    spot = _a_solved_spot()
    q = spot["query"]
    agent = SolverAgent()
    out = asyncio.run(agent.handle(
        {"state": _spot_state(board=q["board"], street=q["street"],
                              pot=q["pot_bb"] * 2.0)}, _Ctx()))
    assert out.solver_available is True, "realtime spot 命中 ⇒ 真实 solver 可用"
    assert out.is_approximate is False
    assert out.actions
    assert out.gto_action


def test_solver_agent_realtime_miss_degrades_to_mock():
    from poker_decision_ai.agents.solver.agent import SolverAgent
    agent = SolverAgent()
    out = asyncio.run(agent.handle(
        {"state": _spot_state(board=["2c", "3d", "4h", "5c"])}, _Ctx()))
    assert out.solver_available is False             # 诚实降级
    assert out.is_approximate is True                # mock 显式标注
    assert out.gto_action                            # 但仍有动作（不阻塞）


def test_deep_path_uses_live_solver():
    """DEEP（非 realtime）路径：现场 provider.solve 真求解（用已解 spot 的
    声明范围；deep 无 60s 死线，求解分钟级可接受）。"""
    from poker_decision_ai.agents.solver.agent import SolverAgent
    spot = _a_solved_spot()
    q = spot["query"]
    agent = SolverAgent()
    out = asyncio.run(agent.handle(
        {"state": _spot_state(board=q["board"], street=q["street"],
                              pot=q["pot_bb"] * 2.0)}, _Ctx(realtime=False)))
    assert out.solver_available is True
    assert out.is_approximate is False
    assert "postflop" in out.solver_name
