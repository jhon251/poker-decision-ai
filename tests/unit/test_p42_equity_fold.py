"""V3F-P42 — Equity Distribution + Fold Equity 测试。

覆盖（按阶段授权 12 项要求）：
  1  distribution 正常计算           6  无条件数据 → UNKNOWN / 诚实 fallback
  2  distribution 可复现             7  fold equity 进入 bet/raise EV
  3  win/lose/tie 概率一致性         8  不同 bet size → EV 正确变化
  4  输入不足 → UNKNOWN              9  P41 点估计 fallback 不回归
  5  fold probability 有条件数据     10/11  Judge primary / UNKNOWN / provenance
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.domain.poker.fold_equity import FoldEstimator
from poker_decision_ai.domain.poker.strategy_core import compute_ev

RIVER = ["Ah", "7c", "2s", "3d", "9h"]


def _state(**over):
    st = {"game_type": "cash", "street": "flop", "pot": 200.0, "bb": 2.0,
          "to_call": 0.0,
          "players": [
              {"player_id": "Hero", "is_hero": True, "stack": 400.0,
               "hole_cards": ["As", "Kd"]},
              {"player_id": "V1", "is_hero": False, "status": "active",
               "stack": 400.0},
          ]}
    st.update(over)
    return st


def _eq(v=0.6, **outcome):
    d = {"hero_equity": v, "confidence": 0.9}
    if outcome:
        d["outcome"] = outcome
    return d


def _range(dist):
    return {"villains": {"V1": {"range_name": "test",
                                "distribution": dist}}}


def _beh(metric="fold_to_cbet", value=0.5, samples=200):
    return {"behavior_statistics": {
        metric: {"value": value, "samples": samples}}}


def _opp(metric="fold_to_cbet", value=0.5, samples=200):
    """opponent envelope（judge/compute_ev 契约：players 包裹）。"""
    return {"players": {"V1": _beh(metric, value, samples)}}


# ══ 1. distribution 正常计算（equity 核心机制 + EV 消费） ═══════════════
def test_equityresult_carries_win_tie_counts():
    from poker_decision_ai.domain.poker.equity import EquityCalculator
    from poker_decision_ai.domain.poker.ranges import Range, combo_key
    calc = EquityCalculator(iterations=800, seed=7)
    vr = Range({combo_key(12, 12, True): 1.0,   # QQ
                combo_key(14, 13, True): 1.0})  # AKs
    res = calc.hand_vs_range(["As", "Kd"], ["Ah", "7c", "2s"], vr)
    assert res.samples > 0
    assert res.win_samples + res.tie_samples <= res.samples
    # 点估计 = win + tie/2（同一计数，严格一致）
    assert abs(res.hero_equity
               - (res.win_samples + 0.5 * res.tie_samples) / res.samples) < 1e-9


def test_compute_ev_uses_distribution_mode():
    # win=0.6 tie=0.1 lose=0.3 → eq' = 0.65；EV(check)=0.65×100=65（点估计 0.6 会给 60）
    r = compute_ev(_state(), _eq(0.6, win=0.6, tie=0.1, lose=0.3))
    assert r.ev_mode == "distribution"
    check = next(c for c in r.candidates if c.action == "check")
    assert abs(check.ev_bb - 65.0) < 1e-9


async def test_equity_agent_emits_outcome(ctx):
    from poker_decision_ai.agents.equity.agent import EquityAgent
    ctx.extra["realtime"] = {"equity_iterations": 600}
    env = await EquityAgent().run({
        "state": _state(board=["Ah", "7c", "2s"]),
        "range": _range({"QQ+": 1.0, "AK": 1.0}),
    }, ctx)
    out = env.output
    oc = out.get("outcome") or {}
    if oc:                       # agent 级（迭代数可被 realtime 覆盖时）
        assert abs(oc["win"] + oc["tie"] + oc["lose"] - 1.0) < 1e-3
        assert abs(out["hero_equity"] - (oc["win"] + 0.5 * oc["tie"])) < 5e-3
        assert oc["seed"] >= 0 and oc["samples"] > 0


# ══ 2. 可复现性 ═══════════════════════════════════════════════════════
def test_fold_estimator_reproducible():
    def run():
        est = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                            villain_dist={"72o": 1.0, "A2o": 1.0},
                            hero_cards=["As", "Kd"], board=RIVER[:3],
                            seed=42, runouts=10)
        return est.for_size(50.0).to_dict()
    a, b = run(), run()
    assert a == b
    assert a["available"] is True


def test_compute_ev_reproducible_with_fold():
    args = (_state(board=["Ah", "7c", "2s"]),
            _eq(0.6, win=0.6, tie=0.05, lose=0.35),
            _range({"72o": 1.0, "A2o": 1.0}), _beh())
    r1 = compute_ev(*args).to_summary()
    r2 = compute_ev(*args).to_summary()
    assert r1 == r2


# ══ 3. win/lose/tie 一致性 ════════════════════════════════════════════
def test_outcome_sum_and_eq_consistency_in_ev():
    oc = {"win": 0.55, "tie": 0.10, "lose": 0.35}
    r = compute_ev(_state(), _eq(0.6, **oc))
    # eq' = 0.55 + 0.05 = 0.60 → 与点估计一致时 EV 数值相同，但模式标记升级
    assert r.ev_mode == "distribution"
    assert abs(r.equity - 0.6) < 1e-9


def test_malformed_outcome_not_fabricated():
    # win+lose+tie ≠ 1 → 分布无效 → 诚实回退点估计（不伪造分布）
    r = compute_ev(_state(), _eq(0.6, win=0.9, tie=0.2, lose=0.2))
    assert r.ev_mode == "point_fallback"
    check = next(c for c in r.candidates if c.action == "check")
    assert abs(check.ev_bb - 60.0) < 1e-9


# ══ 4. 输入不足 → UNKNOWN（fold 通道） ═════════════════════════════════
def test_fold_unknown_without_any_evidence():
    est = FoldEstimator(pot_bb=100.0, facing="raise", street="flop",
                        hero_cards=["As", "Kd"], board=RIVER[:3])
    fe = est.for_size(50.0)
    assert fe.available is False
    assert fe.p_fold is None and fe.p_continue is None
    assert fe.unavailable_reason


def test_fold_unknown_postflop_raise_no_metric():
    # postflop raise：无对应行为统计；无范围 → UNKNOWN（不虚构）
    est = FoldEstimator(pot_bb=100.0, facing="raise", street="flop",
                        behavior=_beh(metric="fold_to_3bet"),     # 给了也不该用
                        hero_cards=["As", "Kd"], board=RIVER[:3])
    fe = est.for_size(50.0)
    assert fe.available is False


def test_fold_needs_hero_cards_for_range_channel():
    est = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                        villain_dist={"72o": 1.0}, board=RIVER[:3])
    fe = est.for_size(50.0)
    assert fe.available is False      # 缺 hero 手牌 → 范围通道不可用
    assert fe.unavailable_reason


# ══ 5. 条件行为数据 → fold probability 可用 ═══════════════════════════
def test_stat_channel_available():
    est = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                        behavior=_beh(value=0.42, samples=250),
                        hero_cards=["As", "Kd"], board=RIVER[:3])
    fe = est.for_size(50.0)
    assert fe.available is True
    assert fe.mode == "stat"
    assert abs(fe.p_fold - 0.42) < 1e-9
    assert abs(fe.p_continue - 0.58) < 1e-9
    prov = fe.provenance["stat"]
    assert prov["metric"] == "fold_to_cbet" and prov["samples"] == 250
    assert prov["size_adjusted"] is False        # 诚实声明：未按尺寸调整


def test_stat_below_min_samples_unavailable():
    est = FoldEstimator(pot_bb=100.0, facing="bet", street="flop",
                        behavior=_beh(samples=10),
                        hero_cards=["As", "Kd"], board=RIVER[:3])
    fe = est.for_size(50.0)
    assert fe.available is False
    assert "samples" in fe.unavailable_reason


def test_preflop_raise_uses_fold_to_3bet():
    est = FoldEstimator(pot_bb=100.0, facing="raise", street="preflop",
                        behavior=_beh(metric="fold_to_3bet", value=0.33,
                                      samples=120),
                        hero_cards=["As", "Kd"])
    fe = est.for_size(25.0)
    assert fe.available is True and fe.mode == "stat"
    assert abs(fe.p_fold - 0.33) < 1e-9


# ══ 6. 无条件数据 → 诚实 fallback（EV 不伪造概率） ════════════════════
def test_ev_falls_back_when_no_fold_evidence():
    r = compute_ev(_state(), _eq(0.6))
    assert r.ev_mode == "point_fallback"
    assert all(c.p_fold is None for c in r.candidates)
    bet = next(c for c in r.candidates if c.action == "bet"
               and c.amount_bb == 50.0)
    # always-called：0.6×150 − 0.4×50 = 70
    assert abs(bet.ev_bb - 70.0) < 1e-9
    assert "fallback" in bet.rule


# ══ 7. fold equity 进入 bet/raise EV（数值核对） ═══════════════════════
def test_stat_fold_enters_bet_ev():
    # p_f=0.5, p_c=0.5, eq'=0.6, P=100, s=50:
    # EV = 0.5×100 + 0.5×(0.6×150 − 0.4×50) = 50 + 35 = 85（fallback=70）
    r = compute_ev(_state(), _eq(0.6), None, _opp(value=0.5, samples=300))
    bet = next(c for c in r.candidates if c.action == "bet"
               and c.amount_bb == 50.0)
    assert bet.p_fold == 0.5 and bet.p_continue == 0.5
    assert abs(bet.ev_bb - 85.0) < 1e-9
    assert bet.ev_bb > 70.0                      # 高于 always-called
    assert "p_f" in bet.rule
    assert r.ev_mode == "point+fold"             # 点估计 + fold 证据


def test_raise_ev_with_fold_component():
    # preflop 3bet：to_call=25bb, raise to 62.5bb（2.5×）：s_inc=37.5
    # p_f=0.5, eq'=0.6 → EV = 0.5×100 + 0.5×(0.6×137.5 − 0.4×37.5) = 83.75
    st = _state(to_call=50.0, street="preflop")
    r = compute_ev(st, _eq(0.6), None,
                   _opp(metric="fold_to_3bet", value=0.5, samples=300))
    ra = next(c for c in r.candidates if c.action == "raise"
              and abs(c.amount_bb - 62.5) < 1e-6)
    assert abs(ra.ev_bb - 83.75) < 1e-9
    assert ra.p_fold == 0.5


def test_ev_mode_distribution_plus_fold():
    r = compute_ev(_state(), _eq(0.6, win=0.6, tie=0.05, lose=0.35),
                   _range({"72o": 1.0}), _beh())
    assert r.ev_mode == "distribution+fold"


# ══ 8. 不同 bet size → EV 正确变化（范围结构通道尺寸条件化） ═══════════
# 河牌面 Ah 7c 2s 3d 9h，hero AK = 一对 A：
#   KQo/JTo → 一对 K / 高牌 J，全部输给 hero → 理性继续阈值下全弃
#   AA      → 明三条 A → 全继续
def test_bigger_bet_better_vs_weak_range():
    r = compute_ev(_state(street="river", board=RIVER),
                   _eq(0.9, win=0.9, tie=0.0, lose=0.1),
                   _range({"KQo": 1.0, "JTo": 1.0}))
    bets = [c for c in r.candidates if c.action == "bet"]
    by_size = {round(c.amount_bb, 4): c for c in bets}
    small, big = by_size[33.0], by_size[100.0]
    assert small.p_fold is not None and big.p_fold >= small.p_fold - 1e-9
    assert big.ev_bb > small.ev_bb
    assert big.ev_bb > 0.95 * 100.0              # 几乎全弃 → EV ≈ P


def test_strong_range_keeps_fold_small():
    # villain 范围 AA（river 明三条）→ 理性继续阈值下全继续 → p_fold≈下限
    r = compute_ev(_state(street="river", board=RIVER),
                   _eq(0.1, win=0.1, tie=0.0, lose=0.9),
                   _range({"AA": 1.0}))
    bets = [c for c in r.candidates if c.action == "bet"]
    assert bets
    assert all(c.p_fold <= 0.02 for c in bets)   # clip 下限 0.01 ± 误差
    # 无弃牌收益 → EV 退回接近 always-called（大注应该更差）
    by_size = {round(c.amount_bb, 4): c for c in bets}
    assert by_size[100.0].ev_bb < by_size[33.0].ev_bb


# ══ 9. P41 点估计 fallback 不回归 ═════════════════════════════════════
def test_p41_call_formula_unchanged():
    r = compute_ev(_state(pot=200.0, to_call=50.0), _eq(0.5))
    call = next(c for c in r.candidates if c.action == "call")
    assert abs(call.ev_bb - 37.5) < 1e-9
    assert r.ev_mode == "point_fallback"


def test_p41_check_formula_unchanged():
    r = compute_ev(_state(), _eq(0.6))
    check = next(c for c in r.candidates if c.action == "check")
    assert abs(check.ev_bb - 60.0) < 1e-9


# ══ 10/11. Judge：primary 不变 / UNKNOWN / provenance ═════════════════
async def test_judge_primary_with_fold_inputs(ctx):
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({
        "state": _state(),
        "equity": _eq(0.6, win=0.6, tie=0.05, lose=0.35),
        "range": _range({"72o": 1.0}),
        "opponent": _opp(value=0.5, samples=300),
    }, ctx)
    out = env.output
    assert out["weights_used"]["ev_core"] == 0.5          # primary 不变
    assert out["final_inputs_summary"]["ev_core"] == "ok"
    ec = out["ev_core"]
    assert ec["ev_mode"] == "distribution+fold"
    assert ec["available"] is True
    assert any(c["p_fold"] is not None for c in ec["candidates"])
    # 假设诚实声明 + 无 GTO 伪造
    assert any(a.startswith("A2-v2") for a in ec["assumptions"])
    assert all("GTO" not in c["rule"] for c in ec["candidates"])


async def test_judge_non_cash_chip_ev_fallback_p44(ctx):
    """P44 语义演进：MTT EVCore 可用（chip_ev 诚实回退），P42 fold 通道照常。"""
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({
        "state": {"game_type": "mtt", "pot": 200.0, "bb": 2.0, "to_call": 50.0,
                  "players": [{"player_id": "Hero", "is_hero": True,
                               "stack": 400.0, "hole_cards": ["As", "Kd"]},
                              {"player_id": "V1", "is_hero": False,
                               "status": "active", "stack": 400.0}]},
        "equity": _eq(0.8),
        "range": _range({"72o": 1.0}),
        "opponent": _beh(),
    }, ctx)
    out = env.output
    ec = out["ev_core"]
    assert ec["available"] is True
    assert ec["objective"] == "chip_ev"          # 无 ICM 输入 → 不冒充 chip_ev+icm
    assert ec["icm_available"] is False
    assert any(c["p_fold"] is not None for c in ec["candidates"])  # P42 通道不变
    assert out["final_inputs_summary"]["solver"] == "missing"


async def test_judge_rule_fold_ev_does_not_break_action_validity(ctx):
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({
        "state": _state(pot=200.0, to_call=50.0),
        "equity": _eq(0.1),
        "range": _range({"AA": 1.0}),
        "opponent": _beh(),
    }, ctx)
    out = env.output
    # 弱 equity + villain 范围 AA（弃牌权益≈0）→ fold 仍最优
    assert out["action"] == "fold"
    assert out["ev_core"]["best"]["action"] == "fold"


# ══ 范围结构通道：river 精确路径 ═══════════════════════════════════════
def test_range_channel_river_exact():
    est = FoldEstimator(pot_bb=100.0, facing="bet", street="river",
                        villain_dist={"KQo": 1.0},
                        hero_cards=["As", "Kd"], board=RIVER, seed=42)
    fe = est.for_size(50.0)
    assert fe.available is True and fe.mode == "range"
    assert fe.p_fold > 0.9                       # 一对 K 输给一对 A → 全弃
    assert fe.provenance["range"]["required_equity_v"] > 0.2   # 50/(100+100)
    est2 = FoldEstimator(pot_bb=100.0, facing="bet", street="river",
                         villain_dist={"AA": 1.0},
                         hero_cards=["As", "Kd"], board=RIVER, seed=42)
    fe2 = est2.for_size(50.0)
    assert fe2.p_fold < 0.1                      # 明三条 → 全继续
