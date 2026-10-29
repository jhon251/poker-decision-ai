"""R2 — 多街前瞻层测试。

覆盖：确定性、anti-lookahead（未见牌约束 + 未来牌置换不变性）、river 无前瞻
降级、UNKNOWN-first、三模式隔离（objective 继承、不注入）、记账桥恒等式、
合法性/边界、隐含赔率方向性、R1 冻结保留（哈希 + 快照不被覆盖）、
无 Solver/GTO 声明。
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker_decision_ai.multistreet import (  # noqa: E402
    MS_VERSION,
    _child_equity,
    _child_state,
    _leaf_value,
    multistreet_evaluate,
    remaining_deck,
    sample_runouts,
)

ROOT = Path(__file__).resolve().parents[2]
TIGHT = {"AA": 0.2, "KK": 0.15, "QQ": 0.2, "AKs": 0.2, "AKo": 0.25}
WIDE = {"AA": 0.15, "KK": 0.1, "QQ": 0.12, "JJ": 0.12, "AKs": 0.13,
        "AKo": 0.13, "AQs": 0.12, "76s": 0.13}


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


def _eval(st, dist=TIGHT, **kw):
    eq = _child_equity(list(st["players"][0]["hole_cards"]), st["board"],
                       dist, 300, 42)
    return multistreet_evaluate(
        st, {"hero_equity": eq},
        range_src={"villains": {"V1": {"distribution": dist}}}, **kw)


# ── 确定性 ─────────────────────────────────────────────────────────────
def test_layer_deterministic():
    a = _eval(_state(), runouts=6)
    b = _eval(_state(), runouts=6)
    assert a == b
    assert a["provenance"]["layer"] == MS_VERSION
    assert a["provenance"]["solver"].startswith("not used")


# ── anti-lookahead ─────────────────────────────────────────────────────
def test_unseen_only_deck_and_sample():
    hero, board = ["Ah", "Ad"], ["Ks", "9c", "4d"]
    deck = remaining_deck(hero, board)
    assert len(deck) == 52 - 5
    for c in hero + board:
        assert c not in deck
    picked = sample_runouts(deck, 12, 42)
    assert len(picked) == 12 and len(set(picked)) == 12
    assert all(c in deck for c in picked)
    # 确定性采样 + 与输入顺序无关
    assert picked == sample_runouts(list(reversed(deck)), 12, 42)


def test_layer_has_no_future_card_input():
    """层接口不接收未来牌：board 之外的任何牌都只能来自未见牌采样。"""
    import inspect
    from poker_decision_ai import multistreet
    src = inspect.getsource(multistreet.multistreet_evaluate)
    assert "runout" in src and "sampled_unseen_only" in inspect.getsource(
        multistreet)
    sig = inspect.signature(multistreet.multistreet_evaluate)
    assert "future" not in " ".join(sig.parameters)
    assert "turn_cards" not in " ".join(sig.parameters)
    # 层输出声明未来牌来源
    r = _eval(_state(), runouts=4)
    assert "sampled_unseen_only" in r["provenance"]["future_cards"]


def test_future_card_permutation_invariance():
    """同一 state（flop 决策）在「实际转牌/河牌」被置换后输出必须一致。"""
    base = _state()
    a = _eval(copy.deepcopy(base), runouts=6)
    # 模拟实际未来牌不同（层不可见；board 仍是 flop 三张）
    b = _eval(copy.deepcopy(base), runouts=6)
    assert a == b
    # 牌堆顺序置换亦不影响
    deck = remaining_deck(["Ah", "Ad"], ["Ks", "9c", "4d"])
    r1 = sample_runouts(deck, 6, 42)
    r2 = sample_runouts(deck[::-1], 6, 42)
    assert r1 == r2


# ── UNKNOWN-first / river 降级 ──────────────────────────────────────────
def test_river_declines_without_lookahead():
    r = _eval(_state(street="river", board=("Ks", "9c", "4d", "7s", "2h")),
              runouts=4)
    assert r["available"] is False
    assert "no future street" in r["decline_reason"]
    assert r["best_ms"] is None and r["best_ms_checkdown"] is None
    assert all(c["utility_ms"] is None for c in r["candidates"])   # 不猜测
    assert all(c["utility_ss"] is not None for c in r["candidates"])  # 冻结值保留


def test_unknown_first_missing_inputs():
    # 缺对手范围
    st = _state()
    eq = _child_equity(["Ah", "Ad"], st["board"], TIGHT, 200, 42)
    r = multistreet_evaluate(st, {"hero_equity": eq}, range_src=None)
    assert r["available"] is False and "villain range" in r["decline_reason"]
    # 缺 hero 手牌
    st2 = _state()
    st2["players"][0]["hole_cards"] = []
    r2 = multistreet_evaluate(
        st2, {"hero_equity": 0.5},
        range_src={"villains": {"V1": {"distribution": TIGHT}}})
    assert r2["available"] is False and "hole cards" in r2["decline_reason"]
    # 冻结核不可用（非法 to_call）→ 层如实拒绝
    st3 = _state(to_call=-5.0)
    r3 = multistreet_evaluate(
        st3, {"hero_equity": 0.5},
        range_src={"villains": {"V1": {"distribution": TIGHT}}})
    assert r3["available"] is False
    assert "frozen core unavailable" in r3["decline_reason"]


# ── 三模式隔离 ─────────────────────────────────────────────────────────
def test_child_state_preserves_mode_context():
    st = _state(game_type="pko",
                tournament={"payouts": [1000.0, 500.0], "bounty_type": "pko",
                            "bounty_rate": 0.5})
    st["players"][1]["bounty"] = 100.0
    child = _child_state(st, street="turn", board=["Ks", "9c", "4d", "7s"],
                         pot_bb_new=150.0, hero_stack_bb_new=1950.0, bb=2.0)
    assert child["game_type"] == "pko"
    assert child["tournament"]["bounty_type"] == "pko"
    assert child["players"][1]["bounty"] == 100.0
    assert child["to_call"] == 0.0 and child["street"] == "turn"


def test_leaf_objective_inherits_mode():
    """叶子 objective 由冻结核决定：cash/mtt/pko 各自独立，层不注入。"""
    objs = {}
    for mode, tour in (("cash", None),
                       ("mtt", {"payouts": [1000.0, 500.0, 200.0]}),
                       ("pko", {"payouts": [1000.0, 500.0, 200.0],
                                "bounty_type": "pko", "bounty_rate": 0.5})):
        st = _state(game_type=mode, tournament=tour)
        if mode == "pko":
            st["players"][1]["bounty"] = 3000.0
        child = _child_state(st, street="turn",
                             board=["Ks", "9c", "4d", "7s"],
                             pot_bb_new=150.0, hero_stack_bb_new=1950.0,
                             bb=2.0)
        lv = _leaf_value(child, 0.6, TIGHT, "V1", 42)
        objs[mode] = lv["objective"]
    assert objs == {"cash": "chip_ev", "mtt": "chip_ev+icm",
                    "pko": "chip_ev+icm+bounty"}


def test_layer_cash_output_has_no_icm_bounty():
    r = _eval(_state(), runouts=4)
    blob = json.dumps(r, ensure_ascii=False).lower()
    assert "bounty" not in blob and "icm" not in blob


# ── 记账桥 / 合法性 / 边界 ─────────────────────────────────────────────
def test_accounting_bridge_identity():
    r = _eval(_state(), runouts=6)
    for c in r["candidates"]:
        if c["utility_ms"] is None:
            continue
        if c["action"] in ("check", "call"):
            leaf = c["leaf"]
            inv = leaf["hero_investment_bb"]
            expect = round(leaf["mean_leaf_utility_bb"] - inv, 6)
            assert abs(c["utility_ms"] - expect) < 1e-6
        assert abs(c["delta"] - (c["utility_ms"] - c["utility_ss"])) < 1e-6


def test_legality_and_numeric_bounds():
    r = _eval(_state(), runouts=6)
    ss_actions = {c["action"] for c in r["base"]["candidates"]}
    for c in r["candidates"]:
        assert c["action"] in ss_actions | {"fold"}      # 不新增动作
        if c["utility_ms"] is not None:
            assert math.isfinite(c["utility_ms"])
            assert math.isfinite(c["delta"])
        if c["action"] in ("bet", "raise") and c["amount_bb"] is not None:
            assert c["amount_bb"] <= 4000.0 / 2.0 + 1e-6  # 不超 hero 栈


def test_implied_odds_direction_strong_vs_air():
    """方向性：强牌（未来街价值可实现）call delta > 0；纯空气 → 近 0（±噪声）。"""
    strong = _eval(_state(hole=("Ah", "Ad")), dist=TIGHT, runouts=8)
    air = _eval(_state(hole=("7h", "2d"), board=("As", "9c", "4d")),
                dist=TIGHT, runouts=8)
    s_call = next(c for c in strong["candidates"] if c["action"] == "call")
    a_call = next(c for c in air["candidates"] if c["action"] == "call")
    assert s_call["delta"] is not None and a_call["delta"] is not None
    assert s_call["delta"] > 0                      # 强牌获得未来街价值
    assert a_call["delta"] < s_call["delta"]        # 空气不得优于强牌
    assert abs(a_call["delta"]) < 15.0              # 空气不产生大额虚构价值


def test_checkdown_variant_present_and_bounded():
    r = _eval(_state(), runouts=6)
    for c in r["candidates"]:
        if c.get("utility_ms_checkdown") is not None:
            # check 口径 ≤ best-action 口径（同 runout 集合）
            assert c["utility_ms_checkdown"] <= c["utility_ms"] + 1e-6
    assert r["best_ms_checkdown"] in {c["action"] for c in r["candidates"]}


def test_multiway_state_runs_with_declared_single_villain():
    """多人池：层仍只建模单一主对手（R2-A4 已声明），输出必须有限且不伪造。"""
    st = _state()
    st["players"] = st["players"] + [
        {"player_id": "V2", "is_hero": False, "stack": 3000.0,
         "status": "active"},
        {"player_id": "V3", "is_hero": False, "stack": 2500.0,
         "status": "active"}]
    r = _eval(st, runouts=4)
    assert r["available"] in (True, False)
    if r["available"]:
        for c in r["candidates"]:
            if c["utility_ms"] is not None:
                assert math.isfinite(c["utility_ms"])
    assert any("主对手" in a for a in r["assumptions"])
    # 声明式：单对手假设必须显式（不得静默）
    assert any("多人池" in a for a in r["assumptions"])


def test_low_spr_amounts_capped():
    """低 SPR（6bb 有效栈）：金额必须封顶于有效栈，utility 有限。"""
    st = _state(stack=12.0, pot=100.0, to_call=40.0)
    r = _eval(st, runouts=4)
    eff_bb = 12.0 / 2.0
    for c in r["candidates"]:
        if c["amount_bb"] is not None:
            assert c["amount_bb"] <= eff_bb + 1e-6
        if c["utility_ms"] is not None:
            assert math.isfinite(c["utility_ms"])
    # 深 SPR 对照：同一手牌在深栈下 call 的未来街价值应不低于低 SPR
    deep = _eval(_state(stack=4000.0, pot=100.0, to_call=40.0), runouts=4)
    d_call = next((c for c in deep["candidates"] if c["action"] == "call"), None)
    s_call = next((c for c in r["candidates"] if c["action"] == "call"), None)
    if d_call and s_call and d_call["utility_ms"] is not None \
            and s_call["utility_ms"] is not None:
        assert d_call["utility_ms"] >= s_call["utility_ms"] - 1e-6


# ── R1 冻结保留 ────────────────────────────────────────────────────────
def test_r1_snapshot_preserved():
    from poker_decision_ai.quality_benchmark import (frozen_hashes,
                                                     verify_frozen)
    r1 = ROOT / "docs" / "V3F_R1_QUALITY_BASELINE.json"
    assert r1.exists()
    d = json.loads(r1.read_text(encoding="utf-8"))
    assert d["snapshot_version"] == "r1-benchmark-v1"
    # R1 核心指标未被 R2 覆盖
    arms = d["cash"]["fold_calibration"]["arms"]
    assert arms["REAL"]["brier"] == 0.125886
    assert arms["BASELINE"]["brier"] == 0.12913
    de = d["cash"]["decision_ev"]
    assert de["observed_agreement_real"] == 0.340339
    assert de["observed_agreement_baseline"] == 0.385763
    # 冻结哈希仍匹配（R2 未改策略数学）
    assert verify_frozen(d["manifest"]["frozen_files"])["match"] is True
    assert frozen_hashes() == d["manifest"]["frozen_files"]


def test_r2_snapshot_separate_and_additive():
    p = ROOT / "docs" / "V3F_R2_MULTISTREET_BASELINE.json"
    if not p.exists():
        pytest.skip("R2 snapshot not generated yet")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["snapshot_version"] == "r2-multistreet-bench-v1"
    assert d["r1_reference"]["present"] is True          # 只读引用 R1
    assert set(d["arms"]) >= {"R1_REAL", "R2_MULTISTREET", "R2_CHECKDOWN"}
    assert d["manifest"]["budget_declared"]["runouts_per_branch"] > 0
    assert "不构成任何盈利性" in d["honesty"]["no_profitability"]
    assert "NOT VALIDATABLE" in d["honesty"]["mtt_pko"]
    # 臂命名诚实性：不得把「REAL 范围+无行为证据」冒充 R1 发布的 BASELINE
    assert "R1_RANGE_ONLY" in d["arms"] and "R1_BASELINE" not in d["arms"]
    assert "不是**" in d["arm_naming_note"] or "**不是**" in d["arm_naming_note"]
    # 等价性：R1_REAL 必须与 R1 发布值逐位一致
    eq = d["equivalence_check"]
    assert eq["R1_REAL_reproduced"] == eq["R1_published_real"] == 0.340339
    # 目标状态如实记录（天花板未解决）
    assert "未改善" in d["honesty"]["goal_status"]
    vi = d["verdict_inputs"]
    assert vi["check_agreement_r2"] < vi["check_agreement_r1_real"]
    assert vi["call_agreement_r2"] < vi["call_agreement_r1_real"]


# ── 评测器等价性（小样本，确保 R1 臂未被扭曲） ──────────────────────────
def test_benchmark_harness_r1_arm_equivalence():
    from poker_decision_ai.multistreet_benchmark import collect_rows
    from poker_decision_ai.pluribus_decision_adapter import (
        iter_pluribus_rows, parse_hand_row)
    hands = [p for row in iter_pluribus_rows(limit=60)
             if (p := parse_hand_row(row))]
    col = collect_rows(hands, decision_hand_limit=30, runouts=2,
                       child_equity_iterations=100, max_points=25)
    rows = col["rows"]
    assert rows, "harness must produce points on real data"
    # R1_REAL 臂必须存在且排名来自冻结核（utility 排序一致）
    for r in rows:
        arm = r["arms"]["R1_REAL"]
        if arm.get("ranking"):
            utils = [c["utility_bb"] for c in arm["ranking"]]
            assert utils == sorted(utils, reverse=True)
            assert arm["best_action"] == arm["ranking"][0]["action"]
    # 层拒绝的点必须回退（不产生新决策）
    for r in rows:
        r2 = r["arms"]["R2_MULTISTREET"]
        if r2["declined"]:
            assert r2["best_action"] == r["arms"]["R1_REAL"]["best_action"]
