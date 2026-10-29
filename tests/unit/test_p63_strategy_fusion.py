"""V3F-P63 — Strategy Fusion 测试。

覆盖：ledger 条目；同向叠加；反向按强度裁决；不可裁决冲突 → 不调整 +
标注；通配/显式动作域；FUSION_CAP；exploit→entries 往返；intel 融合集
成（identity、重排序、conflicted、双计审计扩展 range/exploit 重叠）；
三种模式透传；确定性；P60/P62 回归。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai.strategy_fusion import (
    FUSION_CAP_BB,
    RESOLVE_MARGIN,
    exploit_to_entries,
    fuse_entries,
    fused_delta_for_action,
    future_solver_to_entries,
    make_entry,
)
from poker_decision_ai.strategy_intel import evaluate_intel
from poker_decision_ai.strategy_robustness import make_state


def _intel_with_exploit(entry_frac_p=0.3, entry_frac_player=0.9):
    # P66 修正：默认 fixture 改为真实偏差（玩家 90% entry vs 基线 30%）
    # ——原 0.9/0.9 双侧同率 → 零偏差 → 空融合（P66 语义下为 None）
    from poker_decision_ai.domain.poker.range_posterior import (
        RangePosteriorIndex)
    from poker_decision_ai.exploit_strategy import (
        compute_deviations, compute_exploit_opportunity)

    def rpi(frac, n=400, prefix="r"):
        idx = RangePosteriorIndex()
        for i in range(n):
            entered = (i % 10) < frac * 10
            idx.add_events([{"hand_id": f"{prefix}{i}", "player": "v",
                             "street": "preflop", "facing": "none",
                             "position": "CO",
                             "action": "raise" if (
                                 entered and (i % 10) < 5)
                             else ("call" if entered else "fold")}])
        return idx

    d = compute_deviations(player_rpi=rpi(entry_frac_player),
                           pop_rpi=rpi(entry_frac_p),
                           position="CO")
    opp = compute_exploit_opportunity(d)
    r = evaluate_intel(
        make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                   "check"),
        {"hero_equity": 0.45},
        range_src={"villains": {"v2": {"distribution": {
            "AA": 0.1, "KK": 0.1, "AKs": 0.1, "QQ": 0.1, "AQo": 0.2,
            "KQo": 0.4}}}},
        evidence={"range_mode": "POSTERIOR",
                  "opponent_mode": "player_history",
                  "population_source": "uoft_synthetic"},
        exploit=opp)
    return r, opp


# ------------------------------------------------------------ 融合规则
def test_fuse_same_direction_stacks():
    f = fuse_entries([make_entry("exploit", "bet", 0.2, 1.0),
                      make_entry("future_solver", "bet", 0.1, 1.0)])
    assert f["actions"]["bet"]["delta_bb"] == pytest.approx(0.3)
    assert f["actions"]["bet"]["resolution"] == "stack"
    assert f["actions"]["bet"]["conflicted"] is False


def test_fuse_opposite_resolved_by_strength():
    """反向：强证据（高 conf × 大 delta）裁决方向，净量取代数和。"""
    strong = make_entry("exploit", "bet", 0.4, 1.0)
    weak = make_entry("future_solver", "bet", -0.1, 0.5)
    f = fuse_entries([strong, weak])
    a = f["actions"]["bet"]
    assert a["delta_bb"] == pytest.approx(0.3, abs=1e-6)
    assert a["resolution"] == "resolved_by_strength"
    assert a["conflicted"] is False


def test_fuse_unresolvable_conflict_no_adjustment():
    """|net| < RESOLVE_MARGIN×max → 冲突不可裁决 → 0 + conflicted 标注
    （UNKNOWN，不猜测）。"""
    a = make_entry("exploit", "bet", 0.3, 0.9)
    b = make_entry("future_solver", "bet", -0.3, 0.9)
    f = fuse_entries([a, b])
    act = f["actions"]["bet"]
    assert act["delta_bb"] == 0.0
    assert act["conflicted"] is True
    assert act["resolution"] == "unresolvable_conflict_no_adjustment"
    assert "bet" in f["conflicted_actions"]


def test_fuse_cap_applied():
    f = fuse_entries([make_entry("exploit", "bet", 0.4, 1.0),
                      make_entry("future_solver", "bet", 0.4, 1.0)])
    assert f["actions"]["bet"]["delta_bb"] == FUSION_CAP_BB
    assert "bet" in f["capped_actions"]


def test_fuse_deterministic():
    entries = [make_entry("exploit", "bet", 0.2, 0.8),
               make_entry("future_solver", "raise", -0.1, 0.6)]
    a = fuse_entries(copy.deepcopy(entries))
    b = fuse_entries(copy.deepcopy(entries))
    assert a == b


def test_fused_delta_for_action_wildcard_plus_explicit():
    f = fuse_entries([make_entry("exploit", None, 0.2, 1.0),
                      make_entry("future_solver", "bet", 0.1, 1.0)])
    assert fused_delta_for_action(f, "bet")[0] == pytest.approx(0.3)
    assert fused_delta_for_action(f, "raise")[0] == pytest.approx(0.2)


# ------------------------------------------------------------ exploit 往返
def test_exploit_to_entries_and_roundtrip():
    from poker_decision_ai.domain.poker.range_posterior import (
        RangePosteriorIndex)
    from poker_decision_ai.exploit_strategy import (
        compute_deviations, compute_exploit_opportunity)

    def rpi(frac, prefix):
        idx = RangePosteriorIndex()
        for i in range(400):
            entered = (i % 10) < frac * 10
            idx.add_events([{"hand_id": f"{prefix}{i}", "player": "v",
                             "street": "preflop", "facing": "none",
                             "position": "CO",
                             "action": "raise" if (
                                 entered and (i % 10) < 5)
                             else ("call" if entered else "fold")}])
        return idx

    d = compute_deviations(player_rpi=rpi(0.9, "l"), pop_rpi=rpi(0.3, "p"))
    opp = compute_exploit_opportunity(d)
    entries = exploit_to_entries(opp)
    assert entries and all(e["source"] == "exploit" for e in entries)
    f = fuse_entries(entries)
    for act, bb in opp["affected_actions"].items():
        if bb:
            assert f["actions"][act]["delta_bb"] == pytest.approx(
                bb, abs=1e-6)


# ------------------------------------------------------------ intel 集成
def test_intel_fusion_applied_identity_and_rerank():
    r, opp = _intel_with_exploit()
    assert r["exploit"]["available"] is True
    assert r["fusion"] is not None
    for c in r["candidates"]:
        adj = c["adjustments"]
        assert "exploit_bb" in adj
        total = adj["saca_cv_bb"] + (adj.get("exploit_bb") or 0.0)
        assert c["base_ev_bb"] + total == \
            pytest.approx(c["final_utility_bb"], abs=3e-6)
        assert "fusion" in c
    utils = [c["final_utility_bb"] for c in r["candidates"]]
    assert utils == sorted(utils, reverse=True)
    assert r["best_action_exploited"] == r["candidates"][0]["action"]


def test_intel_no_exploit_no_fusion_block():
    r = evaluate_intel(make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"],
                                  "IP", 10.0, "check"),
                       {"hero_equity": 0.45})
    assert r["exploit"] == {"available": False}
    assert r["fusion"] is None
    assert r["double_counting_guard"]["overlap_detected"] is False


def test_intel_fusion_range_exploit_overlap_audited():
    """exploit 消费 entry/pfr 流且 range posterior 激活 → range/exploit
    重叠被审计标注（同一历史双路由）。"""
    r, _ = _intel_with_exploit()
    g = r["double_counting_guard"]
    assert g.get("range_exploit_overlap", {}).get("detected") is True
    assert "route once" in g.get("range_exploit_overlap", {}).get("note")


def test_intel_conflicted_exploit_not_applied():
    """冲突不可裁决的机会（available=False）→ intel 不应用任何 exploit
    调整（UNKNOWN 纪律：不猜测方向）。"""
    conflicted = {"available": False, "conflict": True,
                  "reason": "conflicting evidence directions across "
                            "streams",
                  "streams_used": ["entry_rate", "conditional_fold"]}
    r = evaluate_intel(make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"],
                                  "IP", 10.0, "check"),
                       {"hero_equity": 0.45}, exploit=conflicted)
    assert r["exploit"]["available"] is False
    assert r["exploit"]["conflict"] is True
    for c in r["candidates"]:
        assert "exploit_bb" not in c["adjustments"]
    assert r["double_counting_guard"]["exploit_streams"] == [
        "entry_rate", "conditional_fold"]


def _cfi_conflict(fold_frac: float, hid: str):
    """返回 ConditionalFoldIndex 对象（compute_deviations 调 .estimate）。

    n=400：保证收缩后 fold 偏差 tilt 幅值超冲突阈值（0.25）。"""
    from poker_decision_ai.domain.poker.conditional_model import (
        ConditionalFoldIndex)
    cp = ConditionalFoldIndex()
    for i in range(400):
        f = 1 if (i % 100) / 100.0 < fold_frac else 0
        cp.add_events([{"player": "v", "street": "flop", "facing": "bet",
                        "size_bucket": "m<=1.0", "position": "CO",
                        "texture": "wet",
                        "action": "fold" if f else "call", "fold": f,
                        "hand_id": f"{hid}{i}"}])
    return cp


# ------------------------------------------------------------ solver 往返
def test_future_solver_entries_source_label():
    entries = future_solver_to_entries({"action": "bet",
                                        "utility_bb": 0.2})
    assert entries and entries[0]["source"] == "solver"
    assert entries[0]["provenance"].startswith("solver_label")


# ------------------------------------------------------------ 三模式透传
@pytest.mark.parametrize("gt,extra", [
    ("cash", {}),
    ("mtt", {"tournament": {"payouts": [50, 30, 20]}}),
    ("pko", {"bounty_type": "pko"}),
])
def test_fusion_across_modes(gt, extra):
    r, _ = _intel_with_exploit()
    st = make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"], "IP", 10.0,
                    "check", game_type=gt)
    st.update(copy.deepcopy(extra))
    if gt == "pko":
        for p in st["players"]:
            p["bounty"] = 50.0
    r2 = evaluate_intel(st, {"hero_equity": 0.45},
                        range_src={"villains": {"v2": {"distribution": {
                            "AA": 0.1, "QQ": 0.2, "AQo": 0.3,
                            "KQo": 0.4}}}},
                        exploit=_intel_with_exploit()[1])
    assert r2["mode"] == ("cash" if gt == "cash" else "mtt")   # P44 语义
    # objective 语义：mtt 带 payouts → chip_ev+icm；pko 无 payouts 上下文
    # → ICM 诚实降级 chip_ev（P44 fallback，候选 provenance 标注）
    assert r2["objective"] == ("chip_ev+icm" if gt == "mtt" else "chip_ev")
    for c in r2["candidates"]:
        assert c["provenance"]["mode"] == r2["mode"]
        assert "exploit_bb" in c["adjustments"]


def _intel_with_exploit_pair():
    return _intel_with_exploit()


# ------------------------------------------------------------ P60/P62 回归
def test_p60_regression_identity_without_fusion():
    r = evaluate_intel(make_state("flop", ["9c", "4c"], ["Ah", "7c", "2c"],
                                  "IP", 10.0, "check"),
                       {"hero_equity": 0.45},
                       range_src={"villains": {"v2": {"distribution": {
                           "AA": 0.1, "QQ": 0.2, "AQo": 0.3, "KQo": 0.4}}}})
    for c in r["candidates"]:
        adj = c["adjustments"]
        assert c["base_ev_bb"] + adj["saca_cv_bb"] == \
            pytest.approx(c["final_ev_bb"], abs=3e-6)
        assert "exploit_bb" not in adj
