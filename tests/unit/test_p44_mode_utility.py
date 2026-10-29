"""V3F-P44 — Mode-aware Utility / EV Core 测试。

覆盖（按阶段授权 14 项）：
  1  CASH regression（utility==ev 逐位、P41/P42 数值冻结）
  2  MTT utility（chip_ev+icm，k± 非对称）
  3  ICM 影响 action/size 排序（CASH call / MTT fold 翻转）
  4  PKO bounty utility（仅淘汰型候选）
  5  KO（全额赏金）
  6  Mystery（揭晓用、未揭晓不伪造）
  7  bounty missing（诚实降级）
  8  ICM missing（诚实降级 chip_ev）
  9  objective missing（game_type 缺失 → 不可用）
  10 UNKNOWN 传播（compute_ev + judge）
  11 provenance（objective/mode/icm_available/bounty_available/assumptions）
  12 P41–P43 regression（数值冻结 + 套件级）
  13 确定性（同输入 → 同输出）
  14 多 action/size utility 比较
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.domain.poker.strategy_core import compute_ev

PAYOUTS = [1000.0, 500.0, 200.0, 100.0, 50.0, 30.0]


def _eq(v=0.6):
    return {"hero_equity": v, "confidence": 0.9}


def _stacks(stacks, bounties=None):
    ps = []
    for i, s in enumerate(stacks):
        p = {"player_id": f"P{i}", "is_hero": i == 0, "stack": float(s),
             "status": "active", "hole_cards": ["As", "Kd"]}
        if bounties and i < len(bounties) and bounties[i]:
            p["bounty"] = float(bounties[i])
        ps.append(p)
    return ps


def _mtt(**over):
    st = {"game_type": "mtt", "bb": 2.0, "pot": 100.0, "to_call": 0.0,
          "tournament": {"payouts": PAYOUTS},
          "players": _stacks([60, 120, 120, 120, 250, 250])}
    st.update(over)
    return st


def _cand(r, action, amount=None):
    for c in r.candidates:
        if c.action != action:
            continue
        if amount is None or (c.amount_bb is not None
                              and abs(c.amount_bb - amount) < 1e-6):
            return c
    raise AssertionError(f"candidate {action} {amount} not found")


# ── 1. CASH regression：utility == ev 逐位相等（冻结兼容） ─────────────
def _cash_state(**over):
    st = {"game_type": "cash", "bb": 2.0, "pot": 200.0, "to_call": 0.0,
          "players": _stacks([400, 400])}          # P41 同款 200bb 栈
    st.update(over)
    return st


def test_cash_utility_equals_ev_bitwise():
    r = compute_ev(_cash_state(), _eq(0.6))          # to_call=0 → bet 分支
    assert r.available and r.objective == "chip_ev" and r.mode == "cash"
    assert r.icm_available is False and r.bounty_available is False
    for c in r.candidates:
        assert c.utility_bb == c.ev_bb              # k+=k-=1 → 逐位一致
        assert c.bounty_util_bb is None
    # P42 冻结数值原样：bet 0.5×pot，always-called → 0.6×150−0.4×50=70
    assert abs(_cand(r, "bet", 50.0).ev_bb - 70.0) < 1e-9


def test_cash_p42_stat_value_frozen():
    r = compute_ev(_cash_state(), _eq(0.6), opponent_src={
        "players": {"P1": {"behavior_statistics": {
            "fold_to_cbet": {"value": 0.5, "samples": 300}}}}})
    assert abs(_cand(r, "bet", 50.0).ev_bb - 85.0) < 1e-9   # P42 数值不变
    assert abs(_cand(r, "bet", 50.0).utility_bb - 85.0) < 1e-9


# ── 2. MTT utility：chip_ev+icm + k± 非对称 ───────────────────────────
def test_mtt_objective_icm_kappa():
    r = compute_ev(_mtt(pot=100.0, to_call=60.0), _eq(0.5))
    assert r.available and r.mode == "mtt"
    assert r.objective == "chip_ev+icm" and r.icm_available is True
    call = _cand(r, "call")
    # ICM 凹性：损失被放大（k->1）、盈利被折价（k+<1）→ |utility| ≤ |ev|
    assert call.utility_bb < call.ev_bb          # call 的负损失项被放大
    assert any("A6" in a for a in r.assumptions)  # 边际线性化声明


def test_mtt_utility_discount_monotone_in_size():
    # eq<0.5：下注越大，ICM 折价越重（(util−ev) 随尺寸单调下降）
    r = compute_ev(_mtt(pot=100.0, to_call=0.0), _eq(0.45))
    bets = [c for c in r.candidates if c.action == "bet"]
    assert len(bets) >= 2
    pens = [(c.amount_bb, c.utility_bb - c.ev_bb) for c in bets]
    pens.sort()
    for (s1, p1), (s2, p2) in zip(pens, pens[1:]):
        assert p2 <= p1 + 1e-9                   # 尺寸↑ → ICM 罚更重


# ── 3. ICM 对 action 排序产生影响（确定性翻转 fixture） ────────────────
FLIP_STATE = {"game_type": "mtt", "bb": 2.0, "pot": 100.0, "to_call": 60.0,
              "tournament": {"payouts": PAYOUTS},
              "players": _stacks([60, 120, 120, 120, 250, 250])}


def test_icm_flips_action_ranking():
    # chip EV(call)=+0.08>0（CASH 会 call），ICM utility<0 → MTT 选 fold
    cash = dict(FLIP_STATE, game_type="cash")
    cash.pop("tournament")
    r_cash = compute_ev(cash, _eq(0.376))
    assert r_cash.best_action == "call"          # CASH：chip EV 为王
    r_mtt = compute_ev(FLIP_STATE, _eq(0.376))
    assert r_mtt.best_action == "fold"           # MTT：ICM 风险厌恶翻转
    call = _cand(r_mtt, "call")
    assert call.ev_bb > 0 > call.utility_bb      # 排序键确实换了目标函数


# ── 4/5/6. PKO / KO / Mystery bounty utility ─────────────────────────
def _bounty_state(btype, **tour_over):
    t = {"payouts": PAYOUTS, "bounty_type": btype}
    t.update(tour_over)
    return _mtt(tournament=t,
                players=_stacks([100, 80, 300, 300], [0, 100, 0, 0]),
                pot=100.0, to_call=0.0)


def test_pko_bounty_utility_elimination_only():
    r = compute_ev(_bounty_state("pko", bounty_rate=0.5), _eq(0.6))
    assert r.objective == "chip_ev+icm+bounty" and r.bounty_available is True
    allin = _cand(r, "bet", 50.0)                # eff=50bb → allin 候选
    non_allin = _cand(r, "bet", 25.0)
    assert allin.bounty_util_bb is not None and allin.bounty_util_bb > 0
    assert non_allin.bounty_util_bb is None      # 非淘汰型 → 无赏金（≠额外equity）
    # P(elim)=eq(covered)×gain=0.6×(0.5×100/2)=15bb
    assert abs(allin.bounty_util_bb - 15.0) < 1e-6
    # utility = ICM 折算(ev) + bounty（折算后 ≤ ev+bounty，凹性折价）
    assert 0 < (allin.ev_bb + allin.bounty_util_bb
                - allin.utility_bb) < 1.0


def test_ko_full_bounty_beats_pko_share():
    r_ko = compute_ev(_bounty_state("ko"), _eq(0.6))
    r_pko = compute_ev(_bounty_state("pko", bounty_rate=0.5), _eq(0.6))
    assert r_ko.bounty_available and r_pko.bounty_available
    ko = _cand(r_ko, "bet", 50.0)
    pko = _cand(r_pko, "bet", 50.0)
    assert abs(ko.bounty_util_bb - 30.0) < 1e-6     # 全额 100 chips/2 = 50bb×0.6
    assert ko.bounty_util_bb > pko.bounty_util_bb   # KO 全额 > PKO 份额


def test_mystery_bounty_honesty():
    # 揭晓阶段：用已揭晓值（同 KO 全额口径）
    r_on = compute_ev(_bounty_state("mystery", mystery_phase_active=True),
                      _eq(0.6))
    assert r_on.bounty_available is True
    assert abs(_cand(r_on, "bet", 50.0).bounty_util_bb - 30.0) < 1e-6
    # 未揭晓：绝不伪造 → bounty utility 不可用
    r_off = compute_ev(_bounty_state("mystery"), _eq(0.6))
    assert r_off.bounty_available is False
    assert r_off.objective == "chip_ev+icm"       # 诚实不含 bounty
    assert all(c.bounty_util_bb is None for c in r_off.candidates)
    assert any("未揭晓" in a for a in r_off.assumptions)


# ── 7/8/9. 输入缺失的诚实降级 ─────────────────────────────────────────
def test_bounty_missing_degrades_honestly():
    r = compute_ev(_mtt(pot=100.0, to_call=0.0), _eq(0.6))
    assert r.mode == "mtt" and r.bounty_available is False
    assert r.objective == "chip_ev+icm"           # 不冒充 +bounty
    assert all(c.bounty_util_bb is None for c in r.candidates)


def test_icm_missing_degrades_to_chip_ev():
    r = compute_ev(_mtt(tournament={}, pot=100.0, to_call=60.0), _eq(0.5))
    assert r.available and r.icm_available is False
    assert r.objective == "chip_ev"               # 不冒充 chip_ev+icm
    for c in r.candidates:
        assert c.utility_bb == c.ev_bb            # k=1（未伪造 ICM 折算）


def test_objective_missing_unavailable():
    for gt in ("", None):
        st = _mtt(pot=100.0, to_call=60.0)
        st["game_type"] = gt
        r = compute_ev(st, _eq(0.5))
        assert r.available is False
        assert "game_type" in r.unavailable_reason


# ── 10. UNKNOWN 传播 ─────────────────────────────────────────────────
def test_unknown_equity_propagates():
    r = compute_ev(_mtt(pot=100.0, to_call=60.0), {})
    assert r.available is False
    assert "hero_equity" in r.unavailable_reason


async def test_judge_mtt_unknown_propagation(ctx):
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({"state": _mtt(pot=100.0, to_call=60.0),
                                  "equity": {}}, ctx)
    out = env.output
    assert out["ev_core"]["available"] is False
    assert out["final_inputs_summary"]["ev_core"] == "unavailable"


# ── 11. provenance ───────────────────────────────────────────────────
def test_provenance_fields_complete():
    s = compute_ev(_mtt(pot=100.0, to_call=60.0), _eq(0.5)).to_summary()
    assert s["objective"] == "chip_ev+icm" and s["mode"] == "mtt"
    assert s["icm_available"] is True and s["bounty_available"] is False
    assert s["best"]["utility_bb"] is not None
    assert any("k+" in a for a in s["assumptions"])   # ICM 折算系数可审计
    b = compute_ev(_bounty_state("pko", bounty_rate=0.5), _eq(0.6)).to_summary()
    assert b["objective"] == "chip_ev+icm+bounty" and b["bounty_available"] is True
    assert any("A7" in a for a in b.assumptions if isinstance(a, str)) \
        if False else any("A7" in a for a in b["assumptions"])


async def test_judge_mtt_evcore_primary(ctx):
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    env = await JudgeAgent().run({"state": _mtt(pot=100.0, to_call=60.0),
                                  "equity": _eq(0.5)}, ctx)
    out = env.output
    assert out["final_inputs_summary"]["ev_core"] == "ok"
    assert out["weights_used"]["ev_core"] == 0.5
    assert out["ev_core"]["objective"] == "chip_ev+icm"
    assert out["ev_core"]["best"]["action"] == "call"   # eq=0.5 > 赔率盈亏平衡


# ── 12. P41–P43 regression（数值冻结；套件级由全量 pytest 保障） ──────
def test_p41_p42_values_frozen():
    r = compute_ev(_cash_state(pot=200.0, to_call=0.0), _eq(0.6))
    assert abs(_cand(r, "check").ev_bb - 60.0) < 1e-9        # P41 check=eq×P
    r2 = compute_ev(_cash_state(pot=200.0, to_call=50.0), _eq(0.6))
    assert abs(_cand(r2, "call", 25.0).ev_bb - 50.0) < 1e-9  # P41 call
    # P41 raise：r=2.5×to_call=62.5bb，s_inc=37.5 → 0.6×137.5−0.4×62.5=57.5
    assert abs(_cand(r2, "raise", 62.5).ev_bb - 57.5) < 1e-9


# ── 13. 确定性 ───────────────────────────────────────────────────────
def test_deterministic_repeatable():
    a = compute_ev(_mtt(pot=100.0, to_call=60.0), _eq(0.45)).to_summary()
    b = compute_ev(_mtt(pot=100.0, to_call=60.0), _eq(0.45)).to_summary()
    assert a == b
    c = compute_ev(_bounty_state("pko", bounty_rate=0.5), _eq(0.45)).to_summary()
    d = compute_ev(_bounty_state("pko", bounty_rate=0.5), _eq(0.45)).to_summary()
    assert c == d


# ── 14. 多 action/size 比较 ──────────────────────────────────────────
def test_multi_action_size_comparison():
    r = compute_ev(_mtt(pot=100.0, to_call=0.0), _eq(0.45))
    by_amt = {round(c.amount_bb, 4): c for c in r.candidates
              if c.action == "bet"}
    assert len(by_amt) >= 3
    utils = [c.utility_bb for c in by_amt.values()]
    assert len(set(utils)) == len(utils)            # 不同尺寸 → 不同 utility
    # 排序按 utility（降序），且与 chip EV 排序可不同（此处至少可解释排序）
    ordered = sorted(r.candidates,
                     key=lambda c: -(c.utility_bb if c.utility_bb is not None
                                     else c.ev_bb))
    assert r.candidates == ordered                  # 结果顺序=utility 降序
    # 最大下注的 ICM 罚最重
    big = max(by_amt.values(), key=lambda c: c.amount_bb)
    small = min(by_amt.values(), key=lambda c: c.amount_bb)
    assert (big.utility_bb - big.ev_bb) < (small.utility_bb - small.ev_bb)
