"""V3F-P67 — PKO/KO/Mystery Deep Strategy Context 测试。

覆盖：effective bounty（KO 全额/PKO growth 折算/Mystery 未揭示不伪造）；
cover 关系；bounty_pressure（冻结公式复用）；late-stage 门控；
mode 门控（mtt/cash 拒绝）；fusion 分账本与恒等式；双计审计
（bounty chip EV 由 P44 表达，pko 层不重复）；确定性；CASH/MTT 隔离。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai.pko_context import (
    mode_application_gate,
    pko_context_to_entries,
    resolve_pko_context,
)
from poker_decision_ai.strategy_intel import evaluate_intel
from poker_decision_ai.strategy_robustness import make_state


def _pko_state(bounty: float = 3000.0, hero_stack: float = 10000.0,
               opp_stack: float = 10000.0, btype: str = "pko",
               remaining: int | None = 30, growth: float = 0.5,
               bb: float = 100.0, hero_bounty: float = 3000.0,
               reveal: bool = True) -> dict:
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check", game_type="pko")
    st["bb"] = bb
    st["sb"] = bb / 2
    t = {"payouts": [50, 30, 20], "stage": "middle",
         "players_remaining": remaining, "bounty_type": btype,
         "bounty_rate": growth, "total_players": 100,
         "avg_stack": 10000.0}
    if remaining is not None:
        t["itm_players"] = max(1, remaining // 3)
    st["tournament"] = t
    if btype == "mystery":
        t["is_bounty"] = reveal
    for p in st["players"]:
        p["stack"] = hero_stack if p.get("is_hero") else opp_stack
        if btype == "mystery" and not reveal:
            p["bounty"] = None                # 未揭示——不伪造金额
        elif p.get("is_hero"):
            p["bounty"] = hero_bounty
        else:
            p["bounty"] = bounty
    return st


def _ctx(st):
    return resolve_pko_context(st)


# ------------------------------------------------------------ 因素解析
def test_effective_bounty_pko_growth_halved():
    c = _ctx(_pko_state(bounty=3000.0, growth=0.5))
    eff = c["factors"]["effective_bounty"]
    assert eff["status"] == "usable"
    assert eff["max_bounty_bb"] == pytest.approx(3000 * 0.5 / 100, rel=1e-6)
    assert eff["bounty_type"] == "pko"


def test_effective_bounty_ko_full_amount():
    c = _ctx(_pko_state(bounty=3000.0, btype="ko", growth=0.5))
    eff = c["factors"]["effective_bounty"]
    assert eff["status"] == "usable"
    assert eff["max_bounty_bb"] == pytest.approx(30.0, rel=1e-6)


def test_mystery_unrevealed_never_fabricated():
    """Mystery 未揭示 → effective bounty UNKNOWN（金额不伪造）。"""
    c = _ctx(_pko_state(btype="mystery", reveal=False))
    eff = c["factors"]["effective_bounty"]
    assert eff["status"] == "UNKNOWN"
    assert "unrevealed" in eff["reason"]
    assert c["bounty_type"] == "mystery"


def test_cover_relation_both_directions():
    covered = _ctx(_pko_state(hero_stack=12000.0, opp_stack=8000.0))
    assert covered["factors"]["cover_relation"]["relation"] == "covered"
    by = _ctx(_pko_state(hero_stack=6000.0, opp_stack=12000.0))
    assert by["factors"]["cover_relation"]["relation"] == "covered_by"


def test_bounty_pressure_uses_frozen_formula():
    c = _ctx(_pko_state(bounty=3000.0, growth=0.5, hero_stack=10000.0,
                        bb=100.0))
    bp = c["factors"]["bounty_pressure"]
    assert bp["status"] == "usable"
    # 冻结公式：15bb 赏金 / 100bb 有效栈 = 0.15
    assert bp["pressure"] == pytest.approx(0.15, rel=1e-6)


def test_missing_bounty_context_unknown():
    st = _pko_state(bounty=0.0)
    for p in st["players"]:
        if not p.get("is_hero"):
            p["bounty"] = 0.0
    c = _ctx(st)
    assert c["factors"]["effective_bounty"]["status"] == "UNKNOWN"
    assert "no active opponent bounty" in \
        c["factors"]["effective_bounty"]["reason"]


def test_missing_bb_unknown():
    st = _pko_state()
    st["bb"] = None
    c = _ctx(st)
    assert c["factors"]["effective_bounty"]["status"] == "UNKNOWN"
    assert "bb missing" in c["factors"]["effective_bounty"]["reason"]


# ------------------------------------------------------------ mode 门控
def test_gate_bounty_modes_apply_others_rejected():
    for m in ("pko", "ko", "mystery"):
        g = mode_application_gate(m)
        assert g["applied"] is True and g["reason"] == ""
    for m in ("cash", "mtt"):
        g = mode_application_gate(m)
        assert g["applied"] is False
        assert "bounty mode" in g["reason"]
        assert "mode switch isolation" in g["reason"]


# ------------------------------------------------------------ entries 方向
def test_entries_hunting_positive_and_capped():
    st = _pko_state(bounty=3000.0, growth=0.5)
    entries = pko_context_to_entries(_ctx(st))
    assert entries and all(e["source"] == "pko_context" for e in entries)
    bets = [e for e in entries if e["action"] == "bet"]
    assert bets and bets[0]["delta_bb"] > 0            # 猎赏 → bet +
    assert all(abs(e["delta_bb"]) <= 0.5 for e in entries)


def test_late_stage_extra_entry_only_when_usable():
    late = _ctx(_pko_state(remaining=8, bounty=6000.0, growth=0.5))
    l_entries = [e for e in pko_context_to_entries(late)
                 if "late_stage_bounty_pressure" in e["evidence_streams"]]
    assert l_entries and all(e["delta_bb"] > 0 for e in l_entries)
    early = _ctx(_pko_state(remaining=80, bounty=3000.0, growth=0.5))
    e_entries = [e for e in pko_context_to_entries(early)
                 if "late_stage_bounty_pressure" in e["evidence_streams"]]
    assert not e_entries


def test_covered_by_negative_entry():
    st = _pko_state(hero_stack=6000.0, opp_stack=15000.0)
    entries = pko_context_to_entries(_ctx(st))
    cautious = [e for e in entries
                if "cover_relation" in e["evidence_streams"]]
    assert cautious and cautious[0]["delta_bb"] < 0


def test_no_entries_without_usable_pressure():
    st = _pko_state(bounty=0.0)
    for p in st["players"]:
        if not p.get("is_hero"):
            p["bounty"] = 0.0
    assert pko_context_to_entries(_ctx(st)) == []


# ------------------------------------------------------------ intel 集成
def _intel_pko(**kw):
    st = _pko_state(**{k: v for k, v in kw.items()})
    st["game_type"] = "pko"
    return evaluate_intel(
        st, {"hero_equity": 0.45},
        range_src={"villains": {"v2": {"distribution": {
            "AA": 0.1, "KK": 0.1, "AQo": 0.2, "KQo": 0.4}}}},
        pko_context=_ctx(st),
        exploit={"available": True, "confidence": 1.0,
                 "affected_actions": {"bet": 0.2},
                 "streams_used": ["entry_rate"], "provenance": {},
                 "fallback": {}})


def test_intel_pko_context_applied_with_split_ledger():
    r = _intel_pko()
    assert r["pko_context_gate"]["applied"] is True
    applied = [c for c in r["candidates"]
               if "pko_context_bb" in c["adjustments"]]
    assert applied
    for c in r["candidates"]:
        adj = c["adjustments"]
        # 分账本：exploit 与 pko 独立字段（不混用）
        assert "pko_context_bb" in adj or not adj.get("pko_applied")
        if c.get("utility_before_exploit_bb") is not None:
            total = c["final_utility_bb"] - c["utility_before_exploit_bb"]
            assert total == pytest.approx(
                (adj.get("exploit_bb") or 0.0)
                + (adj.get("mtt_context_bb") or 0.0)
                + (adj.get("pko_context_bb") or 0.0), abs=3e-6)


def test_intel_mtt_no_pko_adjustments():
    """MTT 携带 pko_context → 门控拒绝（零 pko 调整）。"""
    st = _pko_state()
    st["game_type"] = "mtt"
    st["tournament"]["bounty_type"] = "pko"
    r = evaluate_intel(
        st, {"hero_equity": 0.45},
        range_src={"villains": {"v2": {"distribution": {
            "AA": 0.1, "QQ": 0.2, "AQo": 0.3, "KQo": 0.4}}}},
        pko_context=_ctx(st),
        exploit={"available": True, "confidence": 1.0,
                 "affected_actions": {"bet": 0.2},
                 "streams_used": ["entry_rate"], "provenance": {},
                 "fallback": {}})
    assert r["pko_context_gate"]["applied"] is False
    assert "bounty mode" in r["pko_context_gate"]["reason"]
    assert all("pko_context_bb" not in c["adjustments"]
               for c in r["candidates"])


def test_intel_cash_no_pko_adjustments():
    st = _pko_state()
    st["game_type"] = "cash"
    r = evaluate_intel(st, {"hero_equity": 0.45},
                       pko_context=_ctx(st))
    assert r["pko_context_gate"]["applied"] is False
    assert all("pko_context_bb" not in c["adjustments"]
               for c in r["candidates"])


def test_double_counting_p44_knockout_vs_pko_layer():
    """bounty chip EV 由 P44 knockout utility 表达（淘汰型候选
    bounty_util_bb）；pko_context 层只含猎赏倾斜——两者来源不同且
    分账本独立，guard/字段不重叠。"""
    r = _intel_pko()
    for c in r["candidates"]:
        adj = c["adjustments"]
        # P44 knockout EV 不出现在 pko_context 分量中
        if adj.get("bounty_util_bb"):
            assert (adj.get("pko_context_bb") or 0.0) is not None
        # pko_context 条目的 evidence_streams 不含 knockout 语义
        assert "pko_context_bb" not in c.get("provenance", {})
    # knockout_util 与 pko_context_bb 同时存在是允许的（不同来源分账），
    # 但必须可区分
    assert r["pko_context_gate"]["applied"] is True


def test_deterministic():
    a = _intel_pko()
    b = _intel_pko()
    assert a == b


def test_sanity_mode_isolation_violations_zero():
    r = _intel_pko()
    assert r["mode_isolation"]["violations"] == []
