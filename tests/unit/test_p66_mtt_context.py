"""V3F-P66 — MTT Deep Strategy Context 测试。

覆盖：M-ratio 三区（short/medium/big）；bubble gap（含 hero short 拒绝
激进）；final-table / payout_sensitivity 报告性（不调整，防 ICM 双计）；
mode 门控（mtt 应用 / cash·pko 拒绝并给出 reason）；缺失上下文 UNKNOWN；
fusion 分账本（exploit_bb / mtt_context_bb）；恒等式（含 mtt 分量）；
确定性；CASH/PKO 隔离；P62/P63/P65 回归。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai.mtt_context import (
    mtt_context_to_entries,
    mode_application_gate,
    resolve_mtt_context,
)
from poker_decision_ai.strategy_intel import evaluate_intel
from poker_decision_ai.strategy_robustness import make_state


def _mtt_state(stage: str = "middle", itm: int | None = 45,
               remaining: int | None = 50, avg_stack: float = 5000.0,
               hero_stack: float = 5000.0, bb: float = 100.0):
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check", game_type="mtt")
    st["bb"] = bb
    st["sb"] = bb / 2
    st["tournament"] = {"payouts": [50, 30, 20], "stage": stage,
                        "itm_players": itm,
                        "players_remaining": remaining,
                        "total_players": 100,
                        "avg_stack": avg_stack}
    for p in st["players"]:
        p["stack"] = hero_stack
    return st


def _ctx(state):
    return resolve_mtt_context(state)


# ------------------------------------------------------------ 因素解析
def test_m_ratio_zones():
    short = _ctx(_mtt_state(hero_stack=400.0, bb=100.0))     # M≈2.7
    mid = _ctx(_mtt_state(hero_stack=5000.0, bb=100.0))      # M≈33
    sp_s = short["factors"]["stack_pressure"]
    sp_m = mid["factors"]["stack_pressure"]
    assert sp_s["status"] == "usable" and sp_s["zone"] == "short"
    assert sp_m["status"] == "usable" and sp_m["zone"] == "big"


def test_missing_blinds_stack_pressure_unknown():
    st = _mtt_state()
    st["bb"] = None
    st["sb"] = None
    d = _ctx(st)
    assert d["factors"]["stack_pressure"]["status"] == "UNKNOWN"


def test_bubble_gap_usable_and_hero_short_rejected():
    near = _ctx(_mtt_state(stage="bubble", itm=10, remaining=11))
    assert near["factors"]["bubble_pressure"]["status"] == "usable"
    assert near["factors"]["bubble_pressure"]["gap"] == 1
    # hero short at bubble → 激进倾斜拒绝（UNKNOWN，不伪推）
    st_short = _mtt_state(stage="bubble", itm=10, remaining=11,
                          hero_stack=300.0, bb=100.0)
    d_short = _ctx(st_short)
    assert d_short["factors"]["bubble_pressure"]["status"] == "UNKNOWN"
    assert "hero short" in d_short["factors"]["bubble_pressure"]["reason"]


def test_bubble_not_near_unknown():
    d = _ctx(_mtt_state(itm=10, remaining=50))
    assert d["factors"]["bubble_pressure"]["status"] == "UNKNOWN"
    assert "not near bubble" in d["factors"]["bubble_pressure"]["reason"]


def test_missing_itm_unknown():
    st = _mtt_state(itm=None, remaining=50)
    d = _ctx(st)
    assert d["factors"]["bubble_pressure"]["status"] == "UNKNOWN"
    assert "itm_players" in d["factors"]["bubble_pressure"]["reason"]


def test_report_only_factors_anti_double_count():
    """final-table / payout_sensitivity 为 context_only——显式不调整
    （风险厌恶已由 ICM k± 表达），防与 ICM 双计。"""
    d = _ctx(_mtt_state(stage="final_table", itm=9, remaining=9))
    assert d["factors"]["final_table"]["status"] == "context_only"
    assert "k±" in d["factors"]["final_table"]["reason"] or \
        "ICM" in d["factors"]["final_table"]["reason"]
    ps = d["factors"]["payout_sensitivity"]
    assert ps["status"] == "context_only"
    assert "chip_ev+icm" in ps["expressed_by"]


# ------------------------------------------------------------ 门控与 entries
def test_gate_mtt_applies_others_rejected():
    assert mode_application_gate("mtt") == {"applied": True, "reason": ""}
    g_cash = mode_application_gate("cash")
    assert g_cash["applied"] is False
    assert "forbidden" in g_cash["reason"]
    g_pko = mode_application_gate("pko")
    assert g_pko["applied"] is False
    assert "not applicable" in g_pko["reason"]


def test_entries_short_stack_orientations():
    st = _mtt_state(hero_stack=400.0, bb=100.0)             # M≈2.7 short
    ctx = _ctx(st)
    entries = mtt_context_to_entries(ctx)
    by_act = {e["action"]: e["delta_bb"] for e in entries}
    assert by_act["bet"] > 0 and by_act["raise"] > 0
    assert by_act["call"] < 0                                # call 被支配
    assert all(e["source"] == "mtt_context" for e in entries)


def test_entries_bubble_boost_only_when_usable():
    near = mtt_context_to_entries(_ctx(_mtt_state(stage="bubble", itm=10,
                                                  remaining=11)))
    assert any(e["action"] in ("bet", "raise") and e["delta_bb"] > 0
               for e in near)
    none_bubble = mtt_context_to_entries(_ctx(_mtt_state(itm=10,
                                                         remaining=50)))
    assert not any("mtt_bubble_pressure" in e["evidence_streams"]
                   for e in none_bubble)


def test_no_entries_from_report_only_factors():
    """final_table / payout_sensitivity 是报告项——绝不产生 adjustment。"""
    ctx = _ctx(_mtt_state(stage="final_table", itm=9, remaining=9,
                          hero_stack=8000.0, bb=100.0))
    entries = mtt_context_to_entries(ctx)
    final_table_sources = [e for e in entries
                           if "final_table" in e["evidence_streams"]]
    payout_sources = [e for e in entries
                      if "payout_sensitivity" in e["evidence_streams"]]
    assert not final_table_sources and not payout_sources


# ------------------------------------------------------------ intel 集成
def _intel_with_ctx(mode="mtt", **state_kw):
    st = _mtt_state(**state_kw)
    st["game_type"] = mode
    r = evaluate_intel(st, {"hero_equity": 0.45},
                       range_src={"villains": {"v2": {"distribution": {
                           "AA": 0.1, "KK": 0.1, "AQo": 0.2, "KQo": 0.4}}}},
                       mtt_context=resolve_mtt_context(st))
    return r


def test_mtt_context_applied_only_in_mtt():
    """mode 门控：MTT 应用（adjustments.mtt_context_bb 出现）；
    CASH/PKO 携带上下文 → 拒绝（no mtt_context_bb + gate reason）。"""
    r_mtt = _intel_with_ctx(mode="mtt")
    mtt_applied = [c for c in r_mtt["candidates"]
                   if "mtt_context_bb" in c["adjustments"]]
    assert r_mtt["mtt_context_gate"]["applied"] is True
    assert mtt_applied, "short-stack MTT 应有 mtt_context 调整"
    for mode in ("cash", "pko"):
        r_x = _intel_with_ctx(mode=mode)
        assert r_x["mtt_context_gate"]["applied"] is False
        assert "mode" in r_x["mtt_context_gate"]["reason"]
        assert all("mtt_context_bb" not in c["adjustments"]
                   for c in r_x["candidates"])


def test_mtt_identity_includes_context_component():
    """chip 恒等式（base+cv==final_ev）+ utility 加项恒等式
    （final−before == exploit_bb + mtt_context_bb）。MTT utility 经 k±
    缩放——base+Σadj==utility 仅在 k=1（CASH）成立，P64 冻结语义。"""
    r = _intel_with_ctx(mode="mtt")
    for c in r["candidates"]:
        adj = c["adjustments"]
        exp_bb = adj.get("exploit_bb") or 0.0
        mtt_bb = adj.get("mtt_context_bb") or 0.0
        # chip 空间恒等式（全模式）
        assert c["base_ev_bb"] + adj["saca_cv_bb"] == \
            pytest.approx(c["final_ev_bb"], abs=3e-6)
        # utility 空间恒等式（P62/P66 语义）：final−before == 融合总量
        if c.get("utility_before_exploit_bb") is not None:
            total = c["final_utility_bb"] - \
                c["utility_before_exploit_bb"]
            assert total == pytest.approx(exp_bb + mtt_bb, abs=3e-6)


def test_cash_isolation_unchanged_vs_p60():
    """CASH + mtt_context → 结果与 P60 无上下文完全一致（完全隔离）。"""
    st = _mtt_state()
    st["game_type"] = "cash"
    ctx = resolve_mtt_context(st)
    kw = {"range_src": {"villains": {"v2": {"distribution": {
        "AA": 0.1, "QQ": 0.2, "AQo": 0.3, "KQo": 0.4}}}}}
    base = evaluate_intel(copy.deepcopy(st), {"hero_equity": 0.45},
                          **{k: copy.deepcopy(v) for k, v in kw.items()})
    with_ctx = evaluate_intel(copy.deepcopy(st), {"hero_equity": 0.45},
                              mtt_context=ctx,
                              **{k: copy.deepcopy(v)
                                 for k, v in kw.items()})
    # 决策语义零差异；差异仅在 gate 块（隔离自身的记录）
    strip = lambda r: {k: v for k, v in r.items()
                       if k != "mtt_context_gate"}
    assert strip(base) == strip(with_ctx)       # 零行为差异
    assert with_ctx["mtt_context_gate"]["applied"] is False
    assert "cash" in with_ctx["mtt_context_gate"]["reason"]


def test_pko_isolation_no_mtt_adjustments():
    """PKO 携带 mtt_context → 被门控拒绝（无 mtt_context_bb）。"""
    st = _mtt_state()
    st["game_type"] = "pko"
    st["tournament"]["bounty_type"] = "pko"
    ctx = resolve_mtt_context(st)
    r = evaluate_intel(st, {"hero_equity": 0.45},
                       mtt_context=ctx,
                       range_src={"villains": {"v2": {"distribution": {
                           "AA": 0.1, "QQ": 0.2, "AQo": 0.3, "KQo": 0.4}}}})
    assert r["mtt_context_gate"]["applied"] is False
    assert all("mtt_context_bb" not in c["adjustments"]
               for c in r["candidates"])


def test_deterministic():
    r1 = _intel_with_ctx(mode="mtt")
    r2 = _intel_with_ctx(mode="mtt")
    assert r1 == r2


def test_unknown_context_factors_no_adjustment():
    """全部因素 UNKNOWN（缺盲注/缺 itm）→ 零 mtt entries。"""
    st = _mtt_state()
    st["bb"] = None
    st["sb"] = None
    st["tournament"]["itm_players"] = None
    ctx = resolve_mtt_context(st)
    entries = mtt_context_to_entries(ctx)
    assert entries == []
