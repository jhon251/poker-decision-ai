"""P34 — REAL Decision Engine 单元测试（确定性/诚实降级/规则表）。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.core.errors import ProviderNotConfiguredError  # noqa: E402
from poker_decision_ai.core.providers.real_engine import (  # noqa: E402
    ENGINE_MODEL, RealEngineSolverProvider)


def _q(**over):
    base = {"board": [], "hole_cards": ["5c", "3c"], "pot": 3.0,
            "effective_stack": 200.0, "street": "preflop", "to_call": 0.0,
            "hero_equity": 0.55, "hero_equity_source": "computed",
            "bet_sizings": [50, 75], "game": "cash", "bb": 2.0}
    base.update(over)
    return base


def _solve(q):
    return asyncio.run(RealEngineSolverProvider().solve(q))


# ── 确定性与身份 ──────────────────────────────────────────────────────
def test_deterministic_repeat():
    p = RealEngineSolverProvider()
    a = asyncio.run(p.solve(_q()))
    b = asyncio.run(p.solve(_q()))
    assert a.to_dict() == b.to_dict()          # 无随机数 → 逐字段一致


def test_provider_identity_honest():
    p = RealEngineSolverProvider()
    assert p.name == "realengine"
    assert p.is_mock is False
    assert p.status().value == "real"          # REAL（非 mock 冒名）
    assert p.is_configured() is True


def test_engine_flags_not_gto():
    r = _solve(_q())
    assert r.raw["gto"] is False
    assert r.raw["deterministic"] is True
    assert "NOT GTO" in r.model
    assert r.is_approximate is False           # 非 mock 占位（真实计算）


def test_strategy_mode_classification_real():
    from poker_decision_ai.strategy_provider import classify_solver_result
    assert classify_solver_result("realengine", False).strategy_mode == "REAL"


# ── 输入门：缺/伪造输入 → ProviderNotConfiguredError（诚实降级） ──────
@pytest.mark.parametrize("over", [
    {"hole_cards": ["5c"]},
    {"hole_cards": []},
    {"hero_equity": None},
    {"hero_equity": "0.5"},
    {"hero_equity_source": "missing"},          # 伪造胜率拒绝
    {"pot": 0.0},
    {"effective_stack": 0.0},
    {"bb": 0.0},
    {"street": "showdown"},
    {"game": "tournament"},
])
def test_input_gate_rejects(over):
    with pytest.raises(ProviderNotConfiguredError):
        _solve(_q(**over))


# ── 规则表：preflop ──────────────────────────────────────────────────
def test_preflop_open_raise():
    r = _solve(_q(to_call=0.0, pot=3.0))
    a = r.actions[0]
    assert a.name == "raise"
    assert a.size_bb == 2.5                     # 2.5bb 开注
    assert a.frequency == 1.0                   # 单点策略（不伪造混合）
    assert r.raw["rule"] == "preflop_open_raise"


def test_preflop_open_short_stack_allin():
    r = _solve(_q(to_call=0.0, pot=3.0, effective_stack=5.0))   # 2.5bb 栈
    assert r.actions[0].name == "allin"


def test_preflop_raise_vs_call_vs_fold():
    # req = to_call/(pot+2*to_call) = 4/(3+8) ≈ 0.3636
    q = {"to_call": 4.0, "pot": 3.0}
    assert _solve(_q(**q, hero_equity=0.70)).actions[0].name == "raise"
    assert _solve(_q(**q, hero_equity=0.45)).actions[0].name == "call"
    assert _solve(_q(**q, hero_equity=0.28)).actions[0].name == "fold"


def test_preflop_cheap_defend_call():
    # margin 略负但 to_call ≤ 1bb → 低价防守 call
    # req = 1/(2+2)=0.25；eq 0.22 → margin -0.03 ∈ [-0.05, 0)
    r = _solve(_q(to_call=1.0, pot=2.0, hero_equity=0.22))
    assert r.actions[0].name == "call"
    assert r.raw["rule"] == "preflop_call_cheap_defend"


# ── 规则表：postflop ─────────────────────────────────────────────────
def test_postflop_value_bet():
    r = _solve(_q(street="flop", board=["Ah", "Kd", "7c"],
                  to_call=0.0, pot=10.0, hero_equity=0.80))
    a = r.actions[0]
    assert a.name == "bet"
    assert a.size_bb == 0.66                    # 占池比（既有约定）
    assert r.raw["rule"] == "postflop_value_bet"


def test_postflop_small_bet_needs_spr():
    assert _solve(_q(street="flop", board=["Ah", "Kd", "7c"], to_call=0.0,
                     pot=10.0, effective_stack=60.0,
                     hero_equity=0.55)).raw["rule"] == "postflop_small_bet"
    # SPR < 3 → 不做次级下注 → check
    assert _solve(_q(street="flop", board=["Ah", "Kd", "7c"], to_call=0.0,
                     pot=10.0, effective_stack=20.0,
                     hero_equity=0.55)).actions[0].name == "check"


def test_postflop_no_random_bluff():
    r = _solve(_q(street="turn", board=["Ah", "Kd", "7c", "2s"],
                  to_call=0.0, pot=10.0, hero_equity=0.30))
    assert r.actions[0].name == "check"          # 无读数不做随机诈唬


def test_postflop_facing_bet_call_ev():
    # req = 5/(10+10)=0.25；eq 0.35 → margin 0.10 ∈ [0, 0.15) → call
    # EV = 0.35*20 - 5 = 2.0
    r = _solve(_q(street="flop", board=["Ah", "Kd", "7c"],
                  to_call=5.0, pot=10.0, hero_equity=0.35))
    a = r.actions[0]
    assert a.name == "call"
    assert a.ev == pytest.approx(2.0, abs=1e-6)  # 显式假设下的近似 EV
    assert r.raw["rule"] == "postflop_call_pot_odds"


def test_postflop_raise_then_fold():
    q = {"street": "flop", "board": ["Ah", "Kd", "7c"], "to_call": 5.0,
         "pot": 10.0}
    assert _solve(_q(**q, hero_equity=0.85)).actions[0].name == "raise"
    assert _solve(_q(**q, hero_equity=0.10)).actions[0].name == "fold"


def test_call_ev_zero_for_fold_none_for_aggressive():
    r_fold = _solve(_q(to_call=4.0, pot=3.0, hero_equity=0.10))
    assert r_fold.actions[0].ev == 0.0
    r_raise = _solve(_q(to_call=4.0, pot=3.0, hero_equity=0.80))
    assert r_raise.actions[0].ev is None         # 无弃牌率假设——诚实省略


# ── 动作空间一致性（与冻结 safety gate 契约对齐） ─────────────────────
def test_action_space_never_illegal():
    p = RealEngineSolverProvider()
    for to_call in (0.0, 1.0, 4.0, 50.0):
        for equity in (0.05, 0.3, 0.45, 0.55, 0.7, 0.9):
            for eff in (8.0, 40.0, 200.0):
                r = asyncio.run(p.solve(_q(
                    to_call=to_call, hero_equity=equity,
                    effective_stack=eff,
                    street="flop", board=["Ah", "Kd", "7c"])))
                act = r.actions[0].name
                if to_call > 0:
                    assert act in ("fold", "call", "raise", "allin")
                    assert act not in ("check", "bet")   # 面对下注不可能
                else:
                    assert act in ("check", "bet", "allin")
                    assert act not in ("fold", "call")   # 免费局面不弃/跟


def test_confidence_bounded_and_monotone():
    lo = _solve(_q(to_call=4.0, pot=3.0, hero_equity=0.38)).actions[0]
    hi = _solve(_q(to_call=4.0, pot=3.0, hero_equity=0.90)).actions[0]
    # 直接比较引擎 confidence：经 SolverOutput 包装后看 raw margin 单测
    from poker_decision_ai.core.providers.real_engine import (
        RealEngineSolverProvider as E)
    assert E()._confidence(-1.0) == 0.65
    assert E()._confidence(1.0) == 0.85          # 上限 0.85——不自称满置信
