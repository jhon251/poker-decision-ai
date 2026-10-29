"""V3F-P41 — Strategy Core v1 (EVCore) 测试。

覆盖：CASH EV 公式、尺寸 EV 比较、排序/最优、equity 缺失/非法输入诚实
不可用、UNKNOWN 传播、Judge 接入（primary source）、RULES/旧记分板回退、
solver provenance 不被伪造、确定性、栈封顶、P30–P40 回归（由全量套件保证）。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from poker_decision_ai.domain.poker.strategy_core import compute_ev


def _state(**over):
    st = {"game_type": "cash", "street": "flop", "pot": 200.0, "bb": 2.0,
          "to_call": 0.0,
          "players": [{"player_id": "Hero", "is_hero": True, "stack": 400.0}]}
    st.update(over)
    return st


def _eq(v=0.6):
    return {"hero_equity": v, "confidence": 0.9}


# ── 1. 正常 CASH action EV（公式逐项核对） ───────────────────────────
def test_call_ev_formula():
    # pot=100bb, to_call=25bb, eq=0.5 → EV(call)=0.5*100−0.5*25=37.5
    r = compute_ev(_state(pot=200.0, to_call=50.0), _eq(0.5))
    assert r.available is True
    call = next(c for c in r.candidates if c.action == "call")
    assert abs(call.ev_bb - 37.5) < 1e-9
    fold = next(c for c in r.candidates if c.action == "fold")
    assert fold.ev_bb == 0.0


def test_check_ev_formula():
    # to_call=0, eq=0.6, pot=100bb → EV(check)=60
    r = compute_ev(_state(pot=200.0, to_call=0.0), _eq(0.6))
    check = next(c for c in r.candidates if c.action == "check")
    assert abs(check.ev_bb - 60.0) < 1e-9


# ── 2. 不同 bet size 的 EV 比较（斜率 = 2·eq − 1） ───────────────────
def test_big_bets_better_when_eq_above_half():
    # eq=0.6 → 斜率 +0.2：1.0×pot 优于 0.33×pot 优于 check
    r = compute_ev(_state(pot=200.0, to_call=0.0), _eq(0.6))
    evs = {round(c.amount_bb, 6): c.ev_bb for c in r.candidates
           if c.action == "bet"}
    assert evs[200.0] > evs[min(0.33 * 100, 200)]     # 100bb > 66bb 尺寸
    best_bet = max((c for c in r.candidates if c.action == "bet"),
                   key=lambda c: c.ev_bb)
    check = next(c for c in r.candidates if c.action == "check")
    assert best_bet.ev_bb > check.ev_bb


def test_check_better_when_eq_below_half():
    # eq=0.45 → 斜率 −0.1：所有 bet 差于 check
    r = compute_ev(_state(pot=200.0, to_call=0.0), _eq(0.45))
    assert r.best_action == "check"
    check = next(c for c in r.candidates if c.action == "check")
    for c in r.candidates:
        if c.action == "bet":
            assert c.ev_bb < check.ev_bb


def test_eq_half_neutral_prefers_check():
    # eq=0.5 → 斜率 0：bet EV 全部等于 check EV → tiebreak 取先注册的 check
    r = compute_ev(_state(pot=200.0, to_call=0.0), _eq(0.5))
    assert r.best_action == "check"


# ── 3/4. EV 排序与最优 ────────────────────────────────────────────────
def test_candidates_sorted_desc_and_best_consistent():
    r = compute_ev(_state(pot=200.0, to_call=50.0), _eq(0.7))
    evs = [c.ev_bb for c in r.candidates]
    assert evs == sorted(evs, reverse=True)
    assert r.best_action == r.candidates[0].action
    assert r.best_ev_bb == r.candidates[0].ev_bb


def test_raise_grid_with_min_raise_to():
    # to_call=25bb, min_raise_to=50bb, eff=200bb → raise 候选含 2.5x/3x/allin
    r = compute_ev(_state(pot=200.0, to_call=50.0, min_raise_to=100.0),
                   _eq(0.7))
    raises = sorted(round(c.amount_bb, 6) for c in r.candidates
                    if c.action == "raise")
    assert 62.5 in raises and 75.0 in raises and 200.0 in raises


# ── 5. equity 缺失 → 诚实不可用（不伪造 EV/fold 决策） ────────────────
def test_equity_missing_unavailable():
    r = compute_ev(_state(), {})
    assert r.available is False
    assert "hero_equity" in r.unavailable_reason
    assert r.candidates == [] and r.best_action is None


def test_equity_out_of_range_unavailable():
    r = compute_ev(_state(), _eq(1.5))
    assert r.available is False


# ── 6. 非法输入 / 非 CASH 目标 → 不可用 ──────────────────────────────
@pytest.mark.parametrize("over,reason_key", [
    ({"bb": 0.0}, "bb"),
    ({"bb": -2.0}, "bb"),
    ({"pot": 0.0}, "pot"),
    ({"pot": -5.0}, "pot"),
    ({"to_call": -1.0}, "to_call"),
    ({"game_type": ""}, "game_type"),      # P44：objective 缺失 → 不可用
    ({"game_type": None}, "game_type"),
])
def test_illegal_inputs_unavailable(over, reason_key):
    r = compute_ev(_state(**over), _eq(0.6))
    assert r.available is False
    assert reason_key in r.unavailable_reason


# ── 7. 栈封顶 / allin 语义 ────────────────────────────────────────────
def test_bet_capped_by_stack():
    # pot=100bb, stack=20bb → 所有 bet 封顶 20bb（去重后单一尺寸）
    r = compute_ev(_state(pot=200.0, to_call=0.0,
                          players=[{"player_id": "Hero", "is_hero": True,
                                    "stack": 40.0}]), _eq(0.6))
    bets = [c for c in r.candidates if c.action == "bet"]
    assert bets and all(c.amount_bb <= 20.0 + 1e-9 for c in bets)


def test_call_capped_allin():
    # to_call=150bb > stack=20bb → call 金额封顶 20bb
    r = compute_ev(_state(pot=200.0, to_call=300.0,
                          players=[{"player_id": "Hero", "is_hero": True,
                                    "stack": 40.0}]), _eq(0.5))
    call = next(c for c in r.candidates if c.action == "call")
    assert abs(call.amount_bb - 20.0) < 1e-9


# ── 8. 确定性：重复输入 → 完全一致 ────────────────────────────────────
def test_deterministic_repeatable():
    a = compute_ev(_state(pot=200.0, to_call=50.0), _eq(0.7)).to_summary()
    b = compute_ev(_state(pot=200.0, to_call=50.0), _eq(0.7)).to_summary()
    assert a == b


# ── 9. Judge 接入：EVCore 成为 CASH primary source ───────────────────
async def test_judge_evcore_primary_source(ctx):
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({
        "state": _state(pot=200.0, to_call=50.0, min_raise_to=100.0),
        "equity": _eq(0.8),
    }, ctx)
    out = env.output
    assert out["final_inputs_summary"]["ev_core"] == "ok"
    assert out["weights_used"]["ev_core"] == 0.5
    assert out["ev_core"]["available"] is True
    assert out["ev_core"]["best"]["action"] in ("call", "raise")
    # 强 equity → 价值行为最优；raise/allin 尺寸优先采用 EVCore 最优尺寸
    if out["action"] in ("raise", "allin"):
        assert abs(out["recommended_size"]
                   - out["ev_core"]["best"]["amount_bb"]) < 1e-6


async def test_judge_evcore_weak_equity_folds(ctx):
    # eq=0.1, to_call=25bb → EV(call)=0.1*100−0.9*25<0 < fold → fold
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({
        "state": _state(pot=200.0, to_call=50.0),
        "equity": _eq(0.1),
    }, ctx)
    out = env.output
    assert out["action"] == "fold"
    best = out["ev_core"]["best"]
    assert best["action"] == "fold"


# ── 10. 回退：equity 缺失 → 旧记分板原样；MTT 无 ICM → 诚实 chip_ev（P44） ──
async def test_judge_fallback_non_cash_keeps_scorecard(ctx):
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({
        "state": {"game_type": "mtt", "pot": 200.0, "bb": 2.0,
                  "to_call": 50.0,
                  "players": [{"player_id": "Hero", "is_hero": True,
                               "stack": 400.0}]},
        "equity": _eq(0.8),
        "solver": {"gto_action": "raise", "is_approximate": False,
                   "confidence": 0.9,
                   "actions": [{"name": "raise", "frequency": 0.8}]},
    }, ctx)
    out = env.output
    # P44：MTT EVCore 可用，但无 ICM 输入 → 诚实 chip_ev（icm_available=False）
    assert out["final_inputs_summary"]["ev_core"] == "ok"
    assert out["ev_core"]["objective"] == "chip_ev"
    assert out["ev_core"]["icm_available"] is False
    assert out["weights_used"]["solver"] == 0.35     # 旧 source 权重不变
    assert out["final_inputs_summary"]["solver"] == "ok"


async def test_judge_fallback_missing_equity_conservative(ctx):
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({"state": _state(), "equity": {}}, ctx)
    out = env.output
    assert out["final_inputs_summary"]["ev_core"] == "unavailable"
    # 无任何来源 → 保守动作（原语义不变）
    assert out["action"] in ("fold", "check")


# ── 11. solver provenance 不被 EVCore 伪造 ────────────────────────────
async def test_solver_provenance_untouched_by_evcore(ctx):
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({
        "state": _state(pot=200.0, to_call=50.0, min_raise_to=100.0),
        "equity": _eq(0.8),
        "solver": {"gto_action": "call", "is_approximate": False,
                   "confidence": 0.9,
                   "actions": [{"name": "call", "frequency": 0.7}]},
    }, ctx)
    out = env.output
    # solver 来源标签保持 judge 原语义（ok/degraded），EVCore 不改写
    assert out["final_inputs_summary"]["solver"] == "ok"
    assert out["weights_used"]["solver"] == 0.35
    # EVCore 块只携带确定性 EV + 假设声明
    assert any(a.startswith("A1") for a in out["ev_core"]["assumptions"])
    assert all("GTO" not in c["rule"] for c in out["ev_core"]["candidates"])


# ── 12. UNKNOWN 传播（ev_core 不可用原因进入输出；P44：equity 缺失） ──
async def test_evcore_unavailable_reason_propagates(ctx):
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({
        "state": {"game_type": "mtt", "pot": 200.0, "bb": 2.0,
                  "to_call": 50.0,
                  "players": [{"player_id": "Hero", "is_hero": True,
                               "stack": 400.0}]},
        "equity": {},                               # equity 缺失 → UNKNOWN
    }, ctx)
    out = env.output
    assert out["ev_core"]["available"] is False
    assert "hero_equity" in out["ev_core"]["unavailable_reason"]
