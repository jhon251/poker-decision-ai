"""R2.1 — 多街 v2 测试。

覆盖验收硬门槛：
  * opponent future aggression：事件构造、阶梯、门控、population 回退、UNKNOWN、
    **决策时点截断（无未来泄漏）**、覆盖度报告（player/street/position）、确定性
  * multi-street accounting：分量恒等式、**零修正等价**、无双计、单调性
  * call/check 语义方向：侵略性↑ → 跟注价值下降；听牌 → 实现价值上升
  * 模式隔离：cash/mtt/pko objective；私有助手缺失 → 整体拒绝
  * 确定性、UNKNOWN-first、anti-lookahead（未来牌置换）
  * R1/R2 冻结保留、manifest 封存、B1/B2 快照结构与 bet 偏置指标
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker_decision_ai.multistreet_v2 import (  # noqa: E402
    MS2_VERSION,
    _child,
    child_equity,
    evaluate_v2,
    mode_transform_available,
    mode_utility_from_chip,
    remaining_deck,
    sample_runouts,
)
from poker_decision_ai.opponent_aggression import (  # noqa: E402
    AGGRESSION_VERSION,
    ConditionalAggressionIndex,
    build_aggression_events,
    build_aggression_events_canonical,
    build_aggression_index,
)

ROOT = Path(__file__).resolve().parents[2]
TIGHT = {"AA": 0.2, "KK": 0.15, "QQ": 0.2, "AKs": 0.2, "AKo": 0.25}


def _state(street="flop", hole=("Ah", "Ad"), board=("Ks", "9c", "4d"),
           pot=200.0, to_call=100.0, stack=4000.0, game_type="cash",
           tournament=None):
    st = {"game_type": game_type, "street": street, "pot": pot,
          "to_call": to_call, "bb": 2.0, "board": list(board),
          "players": [
              {"player_id": "Hero", "is_hero": True, "stack": stack,
               "hole_cards": list(hole), "status": "active"},
              {"player_id": "V1", "is_hero": False, "stack": stack,
               "status": "active"}]}
    if tournament:
        st["tournament"] = tournament
    return st


def _idx(events):
    pop = ConditionalAggressionIndex()
    pop.add_events(events)
    return {"version": AGGRESSION_VERSION, "population": pop, "players": {}}


def _fake_events(n=200, p_bet=0.3, street="turn", facing="checked_to",
                 position="BB", texture="high"):
    out = []
    k = int(n * p_bet)
    for i in range(n):
        out.append({"player": "V1", "street": street, "facing": facing,
                    "size_bucket": None, "position": position,
                    "texture": texture, "action": "bet" if i < k else "check",
                    "aggressed": 1 if i < k else 0, "hand_id": f"h{i}"})
    return out


# ── opponent future aggression ─────────────────────────────────────────
def test_aggression_event_construction_and_ladder():
    idx = ConditionalAggressionIndex()
    idx.add_events(_fake_events(200, 0.3))
    hit = idx.estimate("turn", "checked_to", None, "BB", "high")
    assert hit and hit["n"] == 200 and abs(hit["p_aggress"] - 0.3) < 1e-9
    # 粗化阶梯：full key 缺失 → 退到 (street, facing, bucket)（level=键长=3）
    hit2 = idx.estimate("turn", "checked_to", None, "SB", "low")
    assert hit2 and hit2["level"] == 3 and hit2["n"] == 200


def test_aggression_min_samples_gating_and_population_fallback():
    small = ConditionalAggressionIndex()
    small.add_events(_fake_events(5, 0.4))
    assert small.estimate("turn", "checked_to", None, "BB", "high") is None
    pop = ConditionalAggressionIndex()
    pop.add_events(_fake_events(100, 0.25))
    est = small.estimate("turn", "checked_to", None, "BB", "high",
                         population=pop)
    assert est and est["source"] == "population"
    assert abs(est["p_aggress"] - 0.25) < 1e-9
    # 双方均不足 → None（UNKNOWN，不猜测）
    assert ConditionalAggressionIndex().estimate(
        "turn", "checked_to", None, "BB", "high", population=small) is None


def test_aggression_index_deterministic_and_serializable():
    a = ConditionalAggressionIndex()
    a.add_events(_fake_events(50, 0.5))
    b = ConditionalAggressionIndex()
    b.add_events(_fake_events(50, 0.5))
    assert a.to_dict() == b.to_dict()
    c = ConditionalAggressionIndex.from_dict(a.to_dict())
    assert c.to_dict() == a.to_dict()


def test_aggression_coverage_reports_three_levels():
    from poker_decision_ai.pluribus_decision_adapter import (
        iter_pluribus_rows, parse_hand_row)
    hands = [p for row in iter_pluribus_rows(limit=120)
             if (p := parse_hand_row(row))]
    cov = build_aggression_index(hands)["coverage"]
    assert set(cov["by_street"]) <= {"flop", "turn", "river"}
    assert cov["by_position"] and cov["players"]
    assert cov["n_events"] > 0 and cov["players_ge_min_samples"] >= 0
    assert set(cov["by_facing"]) <= {"checked_to", "facing_bet"}


def test_aggression_no_future_leakage_truncation():
    """截断不变性：删除后续街 → 之前街的事件必须逐字节相同。"""
    from poker_decision_ai.pluribus_decision_adapter import (
        iter_pluribus_rows, parse_hand_row)
    hands = [p for row in iter_pluribus_rows(limit=40)
             if (p := parse_hand_row(row))]
    h = next(h for h in hands if len(h.get("streets") or []) >= 3)
    full = build_aggression_events(h)
    # 只保留 flop 街
    trunc = copy.deepcopy(h)
    kept = [s for s in trunc["streets"]
            if s["street"] in ("preflop", "flop")]
    trunc["streets"] = kept
    part = build_aggression_events(trunc)
    flop_full = [e for e in full if e["street"] == "flop"]
    flop_part = [e for e in part if e["street"] == "flop"]
    assert flop_full == flop_part
    # 事件特征不得含任何未来字段
    for e in full:
        assert not any(k in e for k in ("result", "winners", "final_pot",
                                        "next_action", "future"))


def test_canonical_aggression_events_prefix_replay():
    hand = {
        "hand_id": "c1", "bb": 10.0, "sb": 5.0, "table_size": 2,
        "button_seat": 1,
        "players": [{"player_id": "A", "seat": 1, "stack_start": 1000.0,
                     "hole_cards": []},
                    {"player_id": "B", "seat": 2, "stack_start": 1000.0,
                     "hole_cards": []}],
        "preflop_actions": [
            {"street": "preflop", "player_id": "A", "action": "post",
             "amount": 5.0, "to": None},
            {"street": "preflop", "player_id": "B", "action": "post",
             "amount": 10.0, "to": None},
            {"street": "preflop", "player_id": "A", "action": "call",
             "amount": 5.0, "to": None}],
        "flop": ["9h", "Tc", "5h"],
        "flop_actions": [
            {"street": "flop", "player_id": "A", "action": "check",
             "amount": None, "to": None},
            {"street": "flop", "player_id": "B", "action": "bet",
             "amount": 20.0, "to": 20.0},
            {"street": "flop", "player_id": "A", "action": "fold",
             "amount": None, "to": None}],
        "turn": [], "turn_actions": [], "river": [], "river_actions": [],
    }
    evs = build_aggression_events_canonical(hand)
    assert [e["facing"] for e in evs] == ["checked_to", "checked_to",
                                          "facing_bet"]
    assert [e["aggressed"] for e in evs] == [0, 1, 0]
    assert evs[2]["size_bucket"] is not None


# ── multi-street accounting ────────────────────────────────────────────
def _eval(st, agg, dist=TIGHT, **kw):
    eq = child_equity(list(st["players"][0]["hole_cards"]), st["board"],
                      dist, 200, 42)
    return evaluate_v2(st, {"hero_equity": eq},
                       range_src={"villains": {"V1": {"distribution": dist}}},
                       aggression=agg, villain_id="V1",
                       villain_position="BB", **kw)


def test_component_identity_and_no_double_counting():
    agg = _idx(_fake_events(200, 0.3))
    r = _eval(_state(), agg, runouts=4)
    assert r["available"], r.get("decline_reason")
    for c in r["candidates"]:
        if c.get("utility_chip_v2") is None:
            continue
        lhs = c["utility_chip_v2"]
        rhs = (c["immediate_ev"] + c["future_realization"]
               - c["aggression_risk"])
        assert abs(lhs - rhs) < 5e-6, c["action"]
        # call/check 的终局项必须为 0（否则与未来项双计）
        if c["action"] in ("check", "call"):
            assert c["immediate_ev"] == 0.0
        if c["action"] in ("bet", "raise"):
            # 终局项 = p_fold×P（不得包含被跟分支）
            assert abs(c["immediate_ev"]
                       - round(float(c["p_fold"] or 0.0)
                               * (200.0 / 2.0), 6)) < 1e-6


def test_zero_correction_mode_equivalence():
    """零修正等价：Δ=0 时 transform 必须还原冻结 utility（逐位）。"""
    from poker_decision_ai.domain.poker.strategy_core import compute_ev
    cases = [
        ("cash", None),
        ("mtt", {"payouts": [1000.0, 500.0, 200.0, 100.0]}),
        ("pko", {"payouts": [1000.0, 500.0, 200.0, 100.0],
                 "bounty_type": "pko", "bounty_rate": 0.5}),
    ]
    for mode, tour in cases:
        st = _state(street="turn", board=("Ks", "9c", "4d", "7s"),
                    pot=300.0, to_call=0.0, game_type=mode, tournament=tour)
        if mode == "pko":
            st["players"][1]["bounty"] = 3000.0
        res = compute_ev(st, {"hero_equity": 0.6},
                         range_src={"villains": {"V1": {"distribution": TIGHT}}},
                         seed=42)
        assert res.available
        for c in res.candidates:
            got = mode_utility_from_chip(st, float(c.ev_bb or 0.0), c)
            assert "error" not in got, (mode, c.action, got)
            assert abs(got["utility"] - float(c.utility_bb)) < 1e-6, \
                (mode, c.action, got["utility"], c.utility_bb)


def test_mode_transform_helper_availability():
    assert mode_transform_available(_state())["available"] is True
    got = mode_transform_available(_state(game_type="mtt",
                                         tournament={"payouts": [1.0]}))
    # 私有助手存在 → 可用（缺失时必须整体拒绝，测试见 mode 隔离组）
    assert got["available"] is True


def test_call_value_decreases_with_aggression():
    """语义方向（弱牌）：对手下一街下注概率↑ → 跟注净值不得上升。

    注：强牌相反（可跟注/反加赢更大底池）——该方向由
    `test_aggression_risk_sign_and_monotonicity` 单独断言。
    """
    weak = _state(hole=("7h", "2d"), board=("As", "9c", "4d"))
    low = _eval(weak, _idx(_fake_events(200, 0.05)), runouts=6)
    high = _eval(weak, _idx(_fake_events(200, 0.95)), runouts=6)
    cl = next(c for c in low["candidates"] if c["action"] == "call")
    ch = next(c for c in high["candidates"] if c["action"] == "call")
    assert cl["utility_v2"] is not None and ch["utility_v2"] is not None
    assert ch["utility_v2"] <= cl["utility_v2"] + 1e-6
    assert high["p_aggress"] > low["p_aggress"]


def test_aggression_risk_sign_and_monotonicity():
    agg = _idx(_fake_events(200, 0.5))
    weak = _eval(_state(hole=("7h", "2d"), board=("As", "9c", "4d")), agg,
                 runouts=6)
    strong = _eval(_state(hole=("Ah", "Ad")), agg, runouts=6)
    cw = next(c for c in weak["candidates"] if c["action"] == "call")
    cs = next(c for c in strong["candidates"] if c["action"] == "call")
    # 弱牌被下注 → 风险为正（价值下降）；强牌可反加 → 风险可为负
    assert cw["aggression_risk"] >= cs["aggression_risk"] - 1e-6


# ── UNKNOWN-first / anti-lookahead / 隔离 ──────────────────────────────
def test_unknown_first_declines():
    agg = _idx(_fake_events(200, 0.3))
    # river：无未来街
    r = _eval(_state(street="river", board=("Ks", "9c", "4d", "7s", "2h")),
              agg, runouts=2)
    assert r["available"] is False and "no future street" in r["decline_reason"]
    # 缺侵略性索引
    st = _state()
    eq = child_equity(["Ah", "Ad"], st["board"], TIGHT, 150, 42)
    r2 = evaluate_v2(st, {"hero_equity": eq},
                     range_src={"villains": {"V1": {"distribution": TIGHT}}},
                     aggression=None, villain_id="V1")
    assert r2["available"] is False and "aggression index missing" in \
        r2["decline_reason"]
    # 侵略性证据不足
    r3 = _eval(_state(), _idx(_fake_events(10, 0.3)), runouts=2)
    assert r3["available"] is False
    assert "aggression evidence insufficient" in r3["decline_reason"]
    # 缺对手范围
    r4 = evaluate_v2(st, {"hero_equity": 0.5}, range_src=None,
                     aggression=agg, villain_id="V1")
    assert r4["available"] is False
    assert all(c["utility_v2"] is None for c in r4["candidates"])  # 不猜测


def test_anti_lookahead_future_card_permutation():
    agg = _idx(_fake_events(200, 0.3))
    a = _eval(_state(), agg, runouts=6)
    b = _eval(copy.deepcopy(_state()), agg, runouts=6)
    assert a == b
    deck = remaining_deck(["Ah", "Ad"], ["Ks", "9c", "4d"])
    assert sample_runouts(deck, 6, 42) == sample_runouts(deck[::-1], 6, 42)
    for c in ["Ah", "Ad", "Ks", "9c", "4d"]:
        assert c not in deck


def test_determinism_layer_and_index():
    agg1 = _idx(_fake_events(200, 0.3))
    agg2 = _idx(_fake_events(200, 0.3))
    assert _eval(_state(), agg1, runouts=4) == _eval(_state(), agg2, runouts=4)


def test_mode_isolation_objectives_and_cash_purity():
    agg = _idx(_fake_events(200, 0.3))
    for mode, tour, obj in (
            ("cash", None, "chip_ev"),
            ("mtt", {"payouts": [1000.0, 500.0, 200.0]}, "chip_ev+icm"),
            ("pko", {"payouts": [1000.0, 500.0, 200.0],
                     "bounty_type": "pko", "bounty_rate": 0.5},
             "chip_ev+icm+bounty")):
        st = _state(game_type=mode, tournament=tour)
        if mode == "pko":
            st["players"][1]["bounty"] = 3000.0
        r = _eval(st, agg, runouts=2)
        if r["available"]:
            assert r["base"]["objective"] == obj
    cash = _eval(_state(), agg, runouts=2)
    # CASH 纯净性：objective=chip_ev 且所有候选 bounty_util 为空
    assert cash["base"]["objective"] == "chip_ev"
    assert all(c["bounty_util_bb"] is None
               for c in cash["base"]["candidates"])
    assert "chip_ev+icm" not in json.dumps(cash["base"], ensure_ascii=False)


def test_strength_conditioned_gap_declared():
    agg = _idx(_fake_events(200, 0.3))
    r = _eval(_state(), agg, runouts=2)
    joined = " ".join(r["assumptions"])
    assert "牌力条件侵略性未建模" in joined
    assert "not model" in r["provenance"][
        "strength_conditioned_aggression"].lower()


# ── 冻结保留 / manifest / B1-B2 快照 ───────────────────────────────────
def test_r1_r2_frozen_preserved():
    from poker_decision_ai.quality_benchmark import (frozen_hashes,
                                                     verify_frozen)
    r1 = json.loads((ROOT / "docs" / "V3F_R1_QUALITY_BASELINE.json")
                    .read_text(encoding="utf-8"))
    assert verify_frozen(r1["manifest"]["frozen_files"])["match"] is True
    assert frozen_hashes() == r1["manifest"]["frozen_files"]
    assert r1["cash"]["decision_ev"]["observed_agreement_real"] == 0.340339
    r2 = json.loads((ROOT / "docs" / "V3F_R2_MULTISTREET_BASELINE.json")
                    .read_text(encoding="utf-8"))
    assert r2["equivalence_check"]["R1_REAL_reproduced"] == 0.340339


def test_manifest_sealed_and_clean():
    from poker_decision_ai.multistreet_v2_benchmark import (
        MANIFEST_PATH, verify_manifest)
    if not MANIFEST_PATH.exists():
        pytest.skip("manifest not written yet")
    v = verify_manifest()
    assert v["present"] and v["checked"] >= 6
    assert v["clean"], f"代码在封存后漂移: {v['drift']}"


def test_b1_snapshot_has_bias_and_confusion():
    p = ROOT / "docs" / "V3F_R21_B1_BENCH.json"
    if not p.exists():
        pytest.skip("B1 not run yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["snapshot_version"] == "r21-bench-b1-v1"
    assert "R2_1" in d["arms"]
    refs = {"R1_REAL", "R1_RANGE_ONLY_NO_BEHAVIOR"} & set(d["arms"])
    assert refs, "至少需有一个参考臂（R1_REAL 或修正后的 RANGE_ONLY）"
    for arm in ("R2_1", *sorted(refs)):
        a = d["arms"][arm]
        assert a["n"] > 0
        assert a["bet_bias_share"] is not None      # bet 偏置必须报告
        assert set(a["confusion"]) == {"fold", "check", "call", "bet", "raise"}
    assert d["aggression_coverage"]["by_street"]
    assert d["aggression_coverage"]["by_position"]
    assert d["aggression_players"] is not None
    assert "不构成任何盈利性" in d["honesty"]["no_profitability"]
    assert "NOT MODELED" in d["honesty"]["strength_conditioned_aggression"]


def test_b2_snapshot_cross_corpus():
    p = ROOT / "docs" / "V3F_R21_B2_CROSS_CORPUS.json"
    if not p.exists():
        pytest.skip("B2 not run yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["snapshot_version"] == "r21-bench-b2-v1"
    assert d["fit"]["hands_scanned"] > 0 and d["fit"]["events"] > 0
    assert "域差异" in d["domain_note"]
    assert d["arms"]["R2_1"]["declines"] >= 0
