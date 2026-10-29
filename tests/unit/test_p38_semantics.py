"""V3F-P38 — F1/F2 语义修复测试。

F1：REAL 仅指外部可验证 solver（postflop CLI）；P34 realengine（确定性
规则引擎，非 GTO）→ 诚实 RULES；mock/无法证明来源 → BASELINE；绝不伪造
REAL。F2：solver_provenance.mode 与 strategy_mode 同源同值，solver_source
表达真实 provider。UNKNOWN / Judge safety gate 行为不变。

不修改 P34 RealEngine 算法、solver 本身、冻结 UNKNOWN 语义。
"""
from __future__ import annotations

import os
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data2")
F_OK = "8bf82f134651db0add626be0304a2ff1.png"


def _frame(fid):
    _cand = EV / fid
    if not _cand.exists():
        _cand = Path("data/vision2/coinpoker_v3") / fid
    w, h, rgba = decode_png(_cand.read_bytes())
    return rgba, w, h


def _runtime_with_solver(cfg_override: dict | None = None):
    """构建 runtime；SOLVER_PROVIDER/覆盖项按需注入（与 P34 同口径）。"""
    from poker_decision_ai.core.config import load_config, NestedDict
    old = os.environ.get("SOLVER_PROVIDER")
    if cfg_override is None:
        os.environ["SOLVER_PROVIDER"] = "realengine"
    try:
        cfg = load_config()
        if cfg_override:
            cfg.root = cfg.root.merged(NestedDict(cfg_override))
        from poker_decision_ai.core.orchestration.runtime import build_runtime
        return build_runtime(cfg)
    finally:
        if old is None:
            os.environ.pop("SOLVER_PROVIDER", None)
        else:
            os.environ["SOLVER_PROVIDER"] = old


def _tick(cfg_override=None, provider=None, solve_patch=None):
    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    runtime = _runtime_with_solver(cfg_override)
    if provider is not None:
        solver = runtime.providers.get_solver()
        assert solver is not None and solver.name == provider.name
    if solve_patch is not None:
        ctx = mock.patch.object(type(solver), "solve", solve_patch)
        ctx.start()
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame(F_OK)
    r = loop.tick(3801, rgba, w, h)
    if solve_patch is not None:
        ctx.stop()
    return r, runtime


# ── F1-1. realengine 不再被标记为外部 REAL ──────────────────────────
def test_f1_realengine_labeled_rules_not_real():
    r, rt = _tick()
    assert rt.providers.get_solver().name == "realengine"
    assert r["decision_status"] == "READY"
    assert r["strategy_mode"] == "RULES"          # 诚实非-GTO 规则引擎
    assert r["strategy_mode"] != "REAL"
    assert r["solver_mode"] == "RULES"
    assert r["action_validated"] is True          # 计算逻辑/安全门不变
    assert r["solver_provenance"]["solver_source"] == "realengine"


# ── F1-2. 外部 postflop 真实结果 → REAL ─────────────────────────────
# SOLVER-1 契约演进：realtime 路径禁止现场求解（3s deadline），其唯一
# "外部可验证 solver"通道 = postflop spot 库查表命中（可复核产物：
# spot_id + exploitability + manifest）；现场 provider.solve 仅限 DEEP。
# SOLVER-2 契约演进：preflop 走图表基线（source=chart，**非 solver 产物**
# → 永不 REAL）；spot 查表只作用于 postflop 首入节点（to_call=0）。
# 本测试注入 flop 状态（vision 请求桥接处）+ exact 查表命中（与
# SpotTable._to_result 输出同构），验证 P38 语义保持：可验证 solver
# 输出 → REAL；preflop chart / 无命中 → BASELINE（F1-3/F1-4）。
def test_f1_external_postflop_real(monkeypatch):
    import poker_decision_ai.core.providers.spot_table as st

    class _StubSpotTable:
        def lookup(self, query):
            return {
                "solved": True, "solver_name": "postflop",
                "model": "spot_table:SRP-flop-wet_twotone mode=exact",
                "actions": [{"name": "check", "size_bb": None,
                             "frequency": 0.72, "ev": 1.2}],
                "bet_sizings_bb": [],
                "notes": "postflop spot 库查表（真实 CFR 均衡解）",
                "is_approximate": False,
                "raw": {"source": "postflop_spot_lookup",
                        "near_nash_equilibrium": True, "is_mock": False,
                        "lookup_mode": "exact",
                        "spot_id": "SRP-flop-wet_twotone",
                        "exploitability_bb": 0.03, "iterations": 300,
                        "simplifications": ["heads-up (2 players)"]},
            }

    monkeypatch.setitem(st._CACHE, st.DEFAULT_SPOT_DIR, _StubSpotTable())

    # SOLVER-2：vision 帧是 preflop（无 board）；在 bridge 请求处注入
    # flop 首入状态（to_call=0），使命中 spot 库的 postflop 通道可达。
    import poker_decision_ai.vision2.pixel.realclient.production_bridge as pb
    orig_bridge = pb.frame_to_decision_request

    def _flop_bridge(frame_rgba, w, h, hero_id="Hero"):
        got = orig_bridge(frame_rgba, w, h, hero_id=hero_id)
        if got.get("status") == "OK":
            obs = dict(got["request"]["vision_observation"])
            obs["board"] = ["Td", "9d", "6h"]
            obs["street"] = "flop"
            obs["to_call"] = 0.0
            obs["current_bet"] = 0.0
            # spot 套件 = HU only：帧是 6-max 桌——收缩到 hero+1 个对手
            seats = obs.get("seats") or []
            hero_seat = obs.get("hero_seat")
            keep = [s for s in seats if s.get("seat") == hero_seat]
            for s in seats:
                if s.get("seat") != hero_seat and len(keep) < 2:
                    keep.append(s)
            obs["seats"] = [dict(s, in_hand=True) for s in keep]
            got["request"]["vision_observation"] = obs
        return got

    monkeypatch.setattr(pb, "frame_to_decision_request", _flop_bridge)

    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    runtime = _runtime_with_solver(
        {"solver": {"provider": "postflop",
                    "path": "C:/fake/postflopsolver.exe"}})
    solver = runtime.providers.get_solver()
    assert solver.name == "postflop"
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame(F_OK)
    r = loop.tick(3802, rgba, w, h)
    assert r["decision_status"] == "READY"
    assert r["strategy_mode"] == "REAL"           # 可验证 solver 产物才 REAL
    assert r["solver_mode"] == "REAL"
    assert r["solver_provenance"]["solver_source"] == "postflop"
    assert r["solver_provenance"]["mode"] == "REAL"


def test_f1_preflop_chart_never_real(monkeypatch):
    """SOLVER-2：preflop 图表基线（source=chart）**非 solver 产物**——
    即使 provider=postflop，preflop 决策也绝不 REAL（诚实 BASELINE）。"""
    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    runtime = _runtime_with_solver(
        {"solver": {"provider": "postflop",
                    "path": "C:/fake/postflopsolver.exe"}})
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame(F_OK)                    # preflop 帧（无 board）
    r = loop.tick(3806, rgba, w, h)
    assert r["decision_status"] == "READY"
    assert r["strategy_mode"] == "BASELINE"
    assert r["strategy_mode"] not in ("REAL", "RULES")
    assert r["solver_provenance"]["mode"] == "BASELINE"


# ── F1-3/F1-4. 未配置 / 异常 → 诚实降级 BASELINE ────────────────────
def test_f1_solver_unconfigured_degrades_baseline():
    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    runtime = _runtime_with_solver(
        {"solver": {"provider": "postflop"}})     # path 未设置 → 未配置
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
    rgba, w, h = _frame(F_OK)
    r = loop.tick(3803, rgba, w, h)
    assert r["strategy_mode"] == "BASELINE"       # 诚实降级
    assert r["strategy_mode"] != "REAL"
    assert r["solver_provenance"]["mode"] == "BASELINE"


def test_f1_solver_exception_degrades_baseline():
    from poker_decision_ai.core.errors import ProviderError
    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    runtime = _runtime_with_solver(
        {"solver": {"provider": "postflop",
                    "path": "C:/fake/postflopsolver.exe"}})
    solver = runtime.providers.get_solver()

    async def boom(self, query):
        raise ProviderError("external solver exploded")

    with mock.patch.object(type(solver), "solve", boom):
        loop = ProductionDecisionLoop(runtime, throttle_s=0.0)
        rgba, w, h = _frame(F_OK)
        r = loop.tick(3804, rgba, w, h)
    assert r["strategy_mode"] == "BASELINE"       # 异常 → 降级，不伪装 REAL
    assert r["strategy_mode"] != "REAL"
    assert r["solver_provenance"]["mode"] == "BASELINE"


# ── F1-5. mock 来源永不为 REAL/RULES ────────────────────────────────
def test_f1_mock_never_real_or_rules():
    r, rt = _tick(cfg_override={"solver": {"provider": "mock"}})
    assert rt.providers.get_solver().name == "mock"
    assert r["strategy_mode"] == "BASELINE"
    assert r["strategy_mode"] not in ("REAL", "RULES")


# ── F2. provenance 一致性（所有路径） ────────────────────────────────
def test_f2_provenance_mode_matches_strategy_mode():
    for cfg in (None, {"solver": {"provider": "mock"}},
                {"solver": {"provider": "postflop"}}):
        r, _ = _tick(cfg_override=cfg)
        assert r["solver_provenance"]["mode"] == r["strategy_mode"]
        assert r["solver_mode"] == r["strategy_mode"]
        src = r["solver_provenance"]["solver_source"]
        assert src in ("realengine", "postflop", "mock")


def test_f2_provenance_source_is_real_provider():
    r, rt = _tick()
    assert r["solver_provenance"]["solver_source"] == \
        rt.providers.get_solver().name


# ── UNKNOWN / Judge safety gate 行为不变 ────────────────────────────
def test_judge_gate_and_unknown_semantics_unchanged():
    from poker_decision_ai.production_decision_loop import (
        ProductionDecisionLoop)
    runtime = _runtime_with_solver()
    loop = ProductionDecisionLoop(runtime, throttle_s=0.0)

    class FakeReport:
        def to_dict(self):
            return {"final_decision": {"action": "bet_the_moon",
                                       "amount": None},
                    "agents_ran": [], "degraded": [], "skipped": []}

    async def fake_decide(self, request):
        return FakeReport()

    from poker_decision_ai import realtime as rt
    with mock.patch.object(rt.RealtimeRunner, "decide", fake_decide):
        rgba, w, h = _frame(F_OK)
        r = loop.tick(3805, rgba, w, h)
    assert r["decision_status"] == "UNAVAILABLE"  # Judge gate 拒绝
    assert r["action_validated"] is False
    assert "judge_safety_gate" in (r["warning"] or "")
    # Hero UNKNOWN 帧 → UNAVAILABLE（readiness gate 不变）
    rgba, w, h = _frame("TABLE_CHANGED_53_1791197130197_full.png")
    r2 = loop.tick(3806, rgba, w, h)
    assert r2["decision_status"] == "UNAVAILABLE"
    # V-1 FM-1：street 层诚实拒绝（NEW_HAND 不可表示）——readiness 不变
    # V-2 更新（test-side only）：召回修复后该帧（目视 GT 4 卡）可计数控 4 / FULL 3 → 拒绝原因 street_board_contradiction（诚实拒绝语义不变）。
    assert ("street_new_hand_unconfirmed" in (r2["warning"] or "")
            or "street_board_contradiction" in (r2["warning"] or ""))


# ── realengine 算法零修改（raw 语义仍由 provider 层诚实携带） ────────
def test_realengine_algorithm_untouched():
    import asyncio
    from poker_decision_ai.core.providers.real_engine import (
        RealEngineSolverProvider)
    p = RealEngineSolverProvider()
    assert p.name == "realengine" and p.is_mock is False
    res = asyncio.run(p.solve({
        "hole_cards": ["As", "Kd"], "hero_equity": 0.65,
        "hero_equity_source": "computed", "pot": 1000.0,
        "effective_stack": 5000.0, "bb": 100.0, "street": "preflop",
        "to_call": 0.0, "game": "cash"}))
    assert res.is_approximate is False            # 供应商层诚实标记不变
    assert res.raw["gto"] is False                # 非 GTO 声明不变
    assert res.raw["deterministic"] is True
