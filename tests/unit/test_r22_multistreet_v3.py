"""R2.2 — 多街 v3 测试（层内语义修正 + 诊断常驻化）。

覆盖硬约束：恒等式 / 零修正模式等价 / 确定性 / 模式隔离 / **位置对称性（F1）**
/ 三分支（F2）/ all-in 守卫（F3）/ 诊断常驻（F4）/ bounty 标注（F5）/
anti-lookahead / 截断无泄漏 / R1·R2·R2.1 只读不变 / manifest 封存。
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker_decision_ai.multistreet_v3 import (  # noqa: E402
    MS3_VERSION,
    evaluate_v3,
    hero_in_position,
)
from poker_decision_ai.opponent_aggression import (  # noqa: E402
    ConditionalAggressionIndex,
    build_aggression_events,
)

ROOT = Path(__file__).resolve().parents[2]
TIGHT = {"AA": 0.2, "KK": 0.15, "QQ": 0.2, "AKs": 0.2, "AKo": 0.25}


def _idx(events=None):
    pop = ConditionalAggressionIndex()
    pop.add_events(events or _fake(300, 0.3, "turn", "checked_to"))
    pop.add_events(_fake(200, 0.08, "flop", "facing_bet"))
    return {"version": "test", "population": pop, "players": {}}


def _fake(n, p, street, facing, position="BB"):
    k = int(n * p)
    return [{"player": "V1", "street": street, "facing": facing,
             "size_bucket": None, "position": position, "texture": "high",
             "action": "raise" if (i < k and facing == "facing_bet")
             else ("bet" if i < k else "check"),
             "aggressed": 1 if i < k else 0, "hand_id": f"h{i}"}
            for i in range(n)]


def _state(ip=True, street="flop", hole=("Ah", "Ad"), board=("Ks", "9c", "4d"),
           pot=200.0, to_call=100.0, stack=4000.0, game_type="cash",
           tournament=None, positions=True):
    hero_pos = ("BTN" if ip else "BB") if positions else None
    vil_pos = ("BB" if ip else "BTN") if positions else None
    hp = {"player_id": "Hero", "is_hero": True, "stack": stack,
          "hole_cards": list(hole), "status": "active"}
    vp = {"player_id": "V1", "is_hero": False, "stack": stack,
          "status": "active"}
    if positions:
        hp["position"] = hero_pos
        vp["position"] = vil_pos
    st = {"game_type": game_type, "street": street, "pot": pot,
          "to_call": to_call, "bb": 2.0, "board": list(board),
          "players": [hp, vp]}
    if tournament:
        st["tournament"] = tournament
    return st


def _eval(st, *, agg=None, eq=0.694, runouts=4, **kw):
    return evaluate_v3(st, {"hero_equity": eq},
                       range_src={"villains": {"V1": {"distribution": TIGHT}}},
                       aggression=_idx() if agg is None else agg,
                       villain_id="V1",
                       villain_position=("BB" if st["players"][0].get(
                           "position") == "BTN" else "BTN"),
                       runouts=runouts, equity_iterations=100, **kw)


# ── F1 位置对称性 ──────────────────────────────────────────────────────
def test_f1_position_direction_and_symmetry():
    ip = _eval(_state(ip=True))
    oop = _eval(_state(ip=False))
    assert ip["available"] and oop["available"]
    assert ip["provenance"]["node_order"] == "villain_first"
    assert oop["provenance"]["node_order"] == "hero_first"
    assert ip["hero_ip"] is True and oop["hero_ip"] is False
    # 同一手牌/同一底池：两侧节点方向必须不同（F1 核心断言）
    assert ip["provenance"]["node_order"] != oop["provenance"]["node_order"]
    # 分量有限
    for r in (ip, oop):
        for c in r["candidates"]:
            if c.get("utility_chip_v3") is not None:
                assert math.isfinite(c["utility_chip_v3"])


def test_f1_position_unverifiable_declines():
    r = _eval(_state(positions=False))
    assert r["available"] is False
    assert "position unverifiable" in r["decline_reason"]
    assert all(c["utility_v3"] is None for c in r["candidates"])  # 不猜


def test_hero_in_position_helper():
    assert hero_in_position(_state(ip=True), "V1") is True
    assert hero_in_position(_state(ip=False), "V1") is False
    assert hero_in_position(_state(positions=False), "V1") is None


# ── F2 三分支 ─────────────────────────────────────────────────────────
def test_f2_three_branch_distribution():
    r = _eval(_state())
    bets = [c for c in r["candidates"]
            if c["action"] in ("bet", "raise") and not c["allin_guard"]]
    assert bets
    for c in bets:
        d = c["opponent_response_distribution"]
        assert set(d) == {"fold", "call", "raise"}
        assert abs(sum(d.values()) - 1.0) < 1e-6
        assert d["raise"] > 0          # 加注分支必须存在（此前缺失）
        assert c["leaf"]["raise_rate_facing_bet"] > 0
    # check/call 候选（to_call=0 的场景）的对手分布是 checked_to 口径
    r0 = _eval(_state(to_call=0.0))
    chk = next(c for c in r0["candidates"] if c["action"] == "check")
    assert set(chk["opponent_response_distribution"]) == {"bet", "check"}


def test_f2_raise_branch_lowers_aggressive_value():
    """对手加注率越高，hero 的下注价值不得更高（方向性）。"""
    hi = dict(_idx())
    hi["population"] = ConditionalAggressionIndex()
    hi["population"].add_events(_fake(300, 0.3, "turn", "checked_to"))
    hi["population"].add_events(_fake(400, 0.5, "flop", "facing_bet"))
    lo = dict(_idx())
    lo["population"] = ConditionalAggressionIndex()
    lo["population"].add_events(_fake(300, 0.3, "turn", "checked_to"))
    lo["population"].add_events(_fake(400, 0.01, "flop", "facing_bet"))
    a = _eval(_state(), agg=hi)
    b = _eval(_state(), agg=lo)
    ra = next(c for c in a["candidates"] if c["action"] == "raise")
    rb = next(c for c in b["candidates"] if c["action"] == "raise")
    assert ra["utility_v3"] <= rb["utility_v3"] + 1e-6


# ── F3 all-in 守卫 ────────────────────────────────────────────────────
def test_f3_allin_guard_skips_node_and_keeps_frozen():
    r = _eval(_state(stack=12.0, pot=200.0, to_call=100.0))
    assert r["available"]
    guarded = [c for c in r["candidates"] if c["allin_guard"]]
    assert guarded, "短栈下必须出现 all-in 守卫候选"
    for c in guarded:
        assert c["leaf"]["node_skipped"] is True
        assert c["future_realization"] == 0.0 and c["aggression_risk"] == 0.0
        assert abs(c["utility_chip_v3"] - c["frozen_ev_bb"]) < 1e-9


def test_f3_large_stack_allin_candidate_guarded():
    r = _eval(_state())
    allins = [c for c in r["candidates"]
              if c["amount_bb"] and c["amount_bb"] >= 4000.0 / 2.0 - 1e-6]
    assert allins and all(c["allin_guard"] for c in allins)


# ── 恒等式 / 零修正模式等价 / 确定性 ───────────────────────────────────
def test_identity_holds():
    r = _eval(_state())
    checked = 0
    for c in r["candidates"]:
        if c.get("utility_chip_v3") is None:
            continue
        gap = abs(c["utility_chip_v3"] - (c["immediate_ev"]
                                          + c["future_realization"]
                                          - c["aggression_risk"]))
        assert gap <= 2e-6, (c["action"], gap)
        checked += 1
    assert checked >= 3


def test_zero_correction_mode_equivalence():
    """零修正等价：Δ=0 时模式变换必须逐位还原冻结核 utility。"""
    from poker_decision_ai.domain.poker.strategy_core import compute_ev
    from poker_decision_ai.multistreet_v2 import mode_utility_from_chip
    for mode, tour in (("cash", None),
                       ("mtt", {"payouts": [1000.0, 500.0, 200.0]}),
                       ("pko", {"payouts": [1000.0, 500.0, 200.0],
                                "bounty_type": "pko", "bounty_rate": 0.5})):
        st = _state(street="turn", board=("Ks", "9c", "4d", "7s"), pot=300.0,
                    to_call=0.0, game_type=mode, tournament=tour)
        if mode == "pko":
            st["players"][1]["bounty"] = 3000.0
        res = compute_ev(st, {"hero_equity": 0.6},
                         range_src={"villains": {"V1": {"distribution": TIGHT}}},
                         seed=42)
        for c in res.candidates:
            got = mode_utility_from_chip(st, float(c.ev_bb or 0.0), c)
            assert "error" not in got, (mode, c.action, got)
            assert abs(got["utility"] - float(c.utility_bb)) < 1e-6


def test_determinism():
    assert _eval(_state()) == _eval(_state())


# ── 模式隔离 / F5 ─────────────────────────────────────────────────────
def test_mode_isolation_and_f5_annotation():
    for mode, tour, obj in (
            ("cash", None, "chip_ev"),
            ("mtt", {"payouts": [1000.0, 500.0, 200.0]}, "chip_ev+icm"),
            ("pko", {"payouts": [1000.0, 500.0, 200.0],
                     "bounty_type": "pko", "bounty_rate": 0.5},
             "chip_ev+icm+bounty")):
        st = _state(game_type=mode, tournament=tour)
        if mode == "pko":
            st["players"][1]["bounty"] = 3000.0
        r = _eval(st)
        if r["available"]:
            assert r["base"]["objective"] == obj
    cash = _eval(_state())
    assert cash["bounty_semantics"] == "n/a"
    assert all((c.get("mode_transform") or {}).get("bounty_util") in (0.0, None)
               for c in cash["candidates"])
    mtt = _eval(_state(game_type="mtt",
                       tournament={"payouts": [1000.0, 500.0, 200.0]}))
    if mtt["available"]:
        assert "frozen_value_carried" in mtt["bounty_semantics"]


# ── UNKNOWN-first / anti-lookahead / 截断 ─────────────────────────────
def test_unknown_first():
    st = _state()
    eq = {"hero_equity": 0.5}
    r = evaluate_v3(st, eq, range_src={"villains": {"V1": {
        "distribution": TIGHT}}}, aggression=None, villain_id="V1")
    assert r["available"] is False and "aggression index missing" in \
        r["decline_reason"]
    r2 = _eval(_state(street="river",
                      board=("Ks", "9c", "4d", "7s", "2h")))
    assert r2["available"] is False and "no future street" in \
        r2["decline_reason"]
    r3 = evaluate_v3(st, eq, range_src=None, aggression=_idx(),
                     villain_id="V1")
    assert r3["available"] is False
    assert all(c["utility_v3"] is None for c in r3["candidates"])
    # 证据不足
    small = {"version": "t", "population": ConditionalAggressionIndex(),
             "players": {}}
    r4 = _eval(_state(), agg=small)
    assert r4["available"] is False
    assert "aggression evidence insufficient" in r4["decline_reason"]


def test_anti_lookahead_and_truncation():
    from poker_decision_ai.multistreet_v2 import remaining_deck, sample_runouts
    a = _eval(_state())
    b = _eval(copy.deepcopy(_state()))
    assert a == b
    deck = remaining_deck(["Ah", "Ad"], ["Ks", "9c", "4d"])
    assert sample_runouts(deck, 6, 42) == sample_runouts(deck[::-1], 6, 42)
    for c in ("Ah", "Ad", "Ks", "9c", "4d"):
        assert c not in deck
    # 侵略性事件严格决策时点截断
    from poker_decision_ai.pluribus_decision_adapter import (
        iter_pluribus_rows, parse_hand_row)
    hands = [p for row in iter_pluribus_rows(limit=40)
             if (p := parse_hand_row(row))]
    h = next(h for h in hands if len(h.get("streets") or []) >= 3)
    full = build_aggression_events(h)
    trunc = copy.deepcopy(h)
    trunc["streets"] = [s for s in trunc["streets"]
                        if s["street"] in ("preflop", "flop")]
    part = build_aggression_events(trunc)
    assert [e for e in full if e["street"] == "flop"] == \
        [e for e in part if e["street"] == "flop"]


# ── 冻结保留 / manifest / 基准快照 ────────────────────────────────────
def test_r1_r2_r21_readonly_preserved():
    from poker_decision_ai.quality_benchmark import (frozen_hashes,
                                                     verify_frozen)
    r1 = json.loads((ROOT / "docs" / "V3F_R1_QUALITY_BASELINE.json")
                    .read_text(encoding="utf-8"))
    assert verify_frozen(r1["manifest"]["frozen_files"])["match"] is True
    assert frozen_hashes() == r1["manifest"]["frozen_files"]
    r2 = json.loads((ROOT / "docs" / "V3F_R2_MULTISTREET_BASELINE.json")
                    .read_text(encoding="utf-8"))
    assert r2["equivalence_check"]["R1_REAL_reproduced"] == 0.340339
    r21 = json.loads((ROOT / "docs" / "V3F_R21_B1_BENCH_pass2.json")
                     .read_text(encoding="utf-8"))
    assert r21["arms"]["R1_REAL"]["observed_action_agreement"] == 0.340339
    assert r21["arms"]["R2_1"]["observed_action_agreement"] == 0.354915
    # R2.1 manifest 封存仍成立（其文件未被修改）
    from poker_decision_ai.multistreet_v2_benchmark import (
        verify_manifest as verify_r21)
    v = verify_r21()
    if v.get("present"):
        assert v["clean"], f"R2.1 封存被破坏: {v['drift']}"


def test_r22_manifest_sealed():
    from poker_decision_ai.multistreet_v3_benchmark import verify_manifest
    v = verify_manifest()
    if not v.get("present"):
        pytest.skip("R2.2 manifest not written yet")
    assert v["checked"] >= 6
    assert v["clean"], f"R2.2 代码封存后漂移: {v['drift']}"


def test_b1_snapshot_has_diagnostics_and_symmetry():
    p = ROOT / "docs" / "V3F_R22_B1.json"
    if not p.exists():
        pytest.skip("B1′ not run yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["snapshot_version"] == "r22-bench-b1-v1"
    assert "R2_2" in d["arms"] and d["arms"]["R2_2"]["n"] > 0
    diag = d["diagnostics"]
    for key in ("per_action_quantiles", "ablation_agreement", "identity_check"):
        assert key in diag
    assert set(diag["ablation_agreement"]) == {"frozen_ss", "v3_baseline",
                                               "no_aggression"}
    assert diag["identity_check"]["ok_within_rounding"] is True
    q = diag["per_action_quantiles"].get("call", {})
    if q:
        assert q["future_realization"]["median"] is not None
    assert "预注册" in json.dumps(d["config"], ensure_ascii=False) or \
        "不承诺" in json.dumps(d["config"], ensure_ascii=False)
    assert d["counts"]["legal"] > 0 and d["counts"]["illegal"] == 0


def test_b3_snapshot_position_strata():
    p = ROOT / "docs" / "V3F_R22_B3_POSITION.json"
    if not p.exists():
        pytest.skip("B3′ not run yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["snapshot_version"] == "r22-bench-b3-position-v1"
    sym = d["symmetry"]
    if sym["ip_node_order"]:
        assert set(sym["ip_node_order"]) == {"villain_first"}
    if sym["oop_node_order"]:
        assert set(sym["oop_node_order"]) == {"hero_first"}


def test_legality_scope_audit_pinned():
    """legality 口径审计（R2.2 报告疑点）：前缀 2,950 == B1 counts.legal；
    全集 22,947 == R1 快照 legality 合计；两口径 illegal=0、unknown=0。"""
    p = ROOT / "docs" / "V3F_R22_LEGALITY_AUDIT.json"
    if not p.exists():
        pytest.skip("legality audit not generated")
    a = json.loads(p.read_text(encoding="utf-8"))
    v = a["verdict"]
    assert v["B1_legal_equals_points"] is True
    assert v["prefix_actions_match_B1_legal"] is True
    assert v["full_actions_match_R1_published"] is True
    assert v["illegal_zero_everywhere"] is True
    assert v["unknown_la_none_present"] == 0
    assert a["decision_prefix"]["human_actions"] == 2950
    assert a["full_eval"]["human_actions"] == 22947
    b1 = json.loads((ROOT / "docs" / "V3F_R22_B1.json")
                    .read_text(encoding="utf-8"))
    assert b1["counts"]["legal"] == b1["counts"]["points"] == 2950
    assert b1["counts"]["illegal"] == 0


def test_b2_snapshot_cross_corpus():
    p = ROOT / "docs" / "V3F_R22_B2.json"
    if not p.exists():
        pytest.skip("B2′ not run yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["snapshot_version"] == "r22-bench-b2-v1"
    assert d["fit"]["hands_scanned"] > 0 and d["fit"]["events"] > 0
    assert "域差异" in d["domain_note"]
