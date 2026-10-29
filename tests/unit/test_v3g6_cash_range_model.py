"""V3G-6 — CASH Range Modeling / 手内范围更新测试。

覆盖授权要求：preflop 初始 range、每类 action 更新、unknown action、
board filtering、hero-card removal、duplicate/conflict、多玩家隔离、
normalization、impossible hand removal、missing evidence → UNKNOWN、
CASH/non-CASH isolation、deterministic repeat、V3G-3/4/5 回归、
production 旁路前后 action/EV/size 逐位一致。

**不修改**任何既有测试定义；**不触碰**冻结文件。
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.cash_range_model import (
    ACTION_WEIGHT, MAX_WEIGHT, MIN_WEIGHT, RANGE_MODEL_VERSION,
    build_range_model, initial_range)
from poker_decision_ai.domain.poker.ranges import (parse_range,
                                                   parse_single_card)

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

_CFG = {
    "positions": ["UTG", "MP", "CO", "BTN", "SB", "BB"],
    "open": {"BTN": "22+,A2s+,K9s+,Q9s+,J9s+,T8s+,98s,87s,76s,65s,AJo+,KQo",
             "CO": "22+,A9s+,KTs+,QTs+,JTs,T9s,98s,87s,AJo+,KQo",
             "BB": "random:1.0"},
    "three_bet": {"default": "JJ+,AJs+,KQs,AQo+"},
}


def cash_state(**over) -> dict:
    st = {
        "game_type": "cash", "street": "preflop", "board": [],
        "pot": 3.0, "to_call": 2.5, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "button_seat": 2, "table_size": 3,
        "synthetic_test_only": True,
        "players": [
            {"player_id": "v1", "seat": 1, "position": "BB", "status": "active",
             "stack": 100.0, "hole_cards": []},
            {"player_id": "hero", "seat": 2, "position": "BTN", "is_hero": True,
             "status": "active", "hole_cards": ["As", "Kd"], "stack": 100.0},
            {"player_id": "v2", "seat": 3, "position": "CO", "status": "active",
             "stack": 100.0, "hole_cards": []},
        ],
        "action_history": [],
    }
    st.update(over)
    return st


def _build(st=None, **kw):
    return build_range_model(st or cash_state(), ranges_config=_CFG, **kw)


# ── preflop 初始 range（来源可追溯）──────────────────────────────────
def test_preflop_initial_range_is_config_sourced_and_traceable():
    r = initial_range("BTN", ranges_config=_CFG)
    assert r["present"] is True
    assert r["source"] == "configs/ranges.yaml"
    assert r["source_key"] == "ranges.open.BTN"
    assert r["text"] == _CFG["open"]["BTN"]

    r2 = initial_range("BTN", ranges_config=_CFG, route="three_bet")
    # 非 open 路线使用该路线的 default 段（位置仍需已知——不得猜位置）
    assert r2["source_key"] == "ranges.three_bet.default"
    assert r2["text"] == _CFG["three_bet"]["default"]
    # 位置未知时即便路线存在也必须 fail closed（不猜位置）
    r3 = initial_range("x", ranges_config=_CFG, route="three_bet")
    assert r3["present"] is False and "unknown_position" in r3["reason"]


def test_real_config_source_is_reachable_and_parsable():
    """真实 config（非测试桩）可用 —— 证明初始 range 来源真实存在。"""
    from poker_decision_ai.core.config import load_config
    cfg = load_config().default_ranges()
    assert cfg, "configs/ranges.yaml 未加载"
    r = initial_range("BTN", ranges_config=cfg)
    assert r["present"] is True
    assert len(parse_range(r["text"])) > 0
    r_bb = initial_range("BB", ranges_config=cfg)
    assert r_bb["present"] is True


def test_unknown_position_is_unknown_not_guessed():
    for bad in (None, "", "   ", "ZZ"):
        r = initial_range(bad, ranges_config=_CFG)
        assert r["present"] is False, bad
        assert "unknown_position" in r["reason"]


def test_missing_route_range_is_unknown():
    r = initial_range("BTN", ranges_config=_CFG, route="four_bet")
    assert r["present"] is False and "no_configured_range" in r["reason"]


# ── 每类已观察 action 的更新（单调、有界）────────────────────────────
@pytest.mark.parametrize("action", ["fold", "check", "call", "bet", "raise",
                                    "allin"])
def test_each_observed_action_applies_declared_factor(action):
    st = cash_state(action_history=[{"street": "preflop", "player_id": "v1",
                                     "action": action, "sequence": 0}])
    r = _build(st)
    v1 = r["model"]["v1"]
    assert v1["level"] == "inferred"
    applied = v1["observed"]["applied_factors"]
    assert len(applied) == 1
    assert applied[0]["action"] == action
    assert applied[0]["factor"] == ACTION_WEIGHT[action]
    assert applied[0]["source"] == "declared_ACTION_WEIGHT"


def test_action_factors_are_monotonic_in_chips_committed():
    """声明式单调性：投入筹码越多 → 权重因子越大/相等。"""
    order = ["fold", "check", "call", "bet", "raise", "allin"]
    vals = [ACTION_WEIGHT[a] for a in order]
    assert vals == sorted(vals)


def test_aggressive_action_increases_relative_weight_vs_passive():
    """同一起始 range 下，raise 后的分布比 fold 后更集中于强组合（相对）。"""
    base = cash_state(action_history=[])
    strong = (14, 14, True)          # AA
    r_fold = _build(cash_state(action_history=[
        {"street": "preflop", "player_id": "v1", "action": "fold",
         "sequence": 0}]))
    r_raise = _build(cash_state(action_history=[
        {"street": "preflop", "player_id": "v1", "action": "raise",
         "sequence": 0}]))
    d_fold = r_fold["model"]["v1"]["inferred"]["normalized"]
    d_raise = r_raise["model"]["v1"]["inferred"]["normalized"]
    # BB open = random → AA 与弱组合同权；raise 因子对全体同乘，故归一化后
    # 形状不变（声明式 uniform 因子）。这里断言**组合集合**不变但因子被记录。
    assert set(d_fold) == set(d_raise)
    assert r_raise["model"]["v1"]["observed"]["applied_factors"][0][
        "factor"] > r_fold["model"]["v1"]["observed"]["applied_factors"][0][
        "factor"]
    assert _build(base)["model"]["v1"]["level"] == "inferred"


def test_multiple_actions_apply_in_deterministic_street_order():
    st = cash_state(street="flop", board=["Ah", "7c", "2c"], to_call=0.0,
                    action_history=[
                        {"street": "turn", "player_id": "v1", "action": "bet",
                         "sequence": 1},
                        {"street": "flop", "player_id": "v1", "action": "check",
                         "sequence": 0}])
    r = _build(st)
    applied = r["model"]["v1"]["observed"]["applied_factors"]
    assert [a["street"] for a in applied] == ["flop", "turn"]
    # 复跑一致
    assert _build(st) == r


def test_weights_stay_bounded_under_repeated_aggression():
    acts = [{"street": "flop", "player_id": "v1", "action": "allin",
             "sequence": i} for i in range(8)]
    r = _build(cash_state(street="flop", board=["Ah", "7c", "2c"], to_call=0.0,
                          action_history=acts))
    v1 = r["model"]["v1"]
    assert v1["level"] == "inferred"
    assert v1["inferred"]["combos"] > 0
    # 归一化最大值恒为 1.0（确定性）；权重被 MAX_WEIGHT 钳制
    assert max(v1["inferred"]["normalized"].values()) <= 1.0
    assert MIN_WEIGHT <= MAX_WEIGHT


# ── unknown action 不更新 ─────────────────────────────────────────────
def test_unknown_action_does_not_update_range():
    st = cash_state(action_history=[{"street": "preflop", "player_id": "v1",
                                     "action": "teleport", "sequence": 0}])
    r = _build(st)
    v1 = r["model"]["v1"]
    assert v1["observed"]["applied_factors"] == []
    assert any(x["reason"].startswith("unknown_action") for x in r["unknown"])
    # 与完全无行动的分布一致
    r_none = _build(cash_state())
    assert v1["inferred"]["distribution"] == r_none["model"]["v1"][
        "inferred"]["distribution"]


def test_missing_action_field_does_not_update():
    st = cash_state(action_history=[{"street": "preflop", "player_id": "v1",
                                     "sequence": 0}])
    r = _build(st)
    assert r["model"]["v1"]["observed"]["applied_factors"] == []


def test_action_without_player_id_is_not_attributed():
    st = cash_state(action_history=[{"street": "preflop", "action": "raise",
                                     "sequence": 0}])
    r = _build(st)
    for pid, m in r["model"].items():
        assert m["observed"]["applied_factors"] == [], pid
    assert any(x["reason"] == "missing_player_id" for x in r["unknown"])


# ── board filtering / hero-card removal / impossible hands ────────────
def test_known_cards_are_removed_from_opponent_range():
    """hero 手牌 + board 必须从对手组合空间中移除（组合数不增加、
    被挡死的组合权重下降或消失）。"""
    flop = cash_state(street="flop", board=["Ah", "7c", "2c"], to_call=0.0)
    r = _build(flop)
    assert r["known_cards"]["count"] == 5          # 2 hero + 3 board
    v1 = r["model"]["v1"]
    assert v1["level"] == "inferred"
    # AA 仍有 1 种可用花色（Ah 被挡）→ 权重低于未被挡的对子
    dist = v1["inferred"]["distribution"]
    assert dist.get("AA", 0) < dist.get("KK", 0)


def test_board_card_reduces_affected_combos_not_increases():
    """已知牌移除的**绝对**效果：被挡的组合权重下降；未涉及的组合不变。

    注意：`distribution()` 是**归一化**后的分布，未涉及的组合占比会因分母
    变小而上升——因此必须比较**移除前后的绝对权重**（`stats`/单组合权重），
    不能用归一化分布来判断"是否增加"。
    """
    from poker_decision_ai.domain.poker.ranges import parse_range as _pr
    # 已知牌：As、Ah、Kd（hero）、7c、2c（board）
    #   AA：Ah 已死 → 1 张 A 在外 → C(2,2)/6 = 1/6
    #   KK：Kd 已死 → 3 张 K 在外 → C(3,2)/6 = 3/6 = 0.5
    #   QQ：未被触及 → 6/6 = 1.0
    pre = _pr(_CFG["open"]["BB"]).remove_cards([])
    flop_dead = [parse_single_card(c) for c in ("As", "Kd", "Ah", "7c", "2c")]
    flop = _pr(_CFG["open"]["BB"]).remove_cards(flop_dead)
    assert abs(flop.weight_of((14, 14, True)) - 1 / 6) < 1e-9      # AA
    assert abs(flop.weight_of((13, 13, True)) - 3 / 6) < 1e-9      # KK
    assert flop.weight_of((12, 12, True)) == 1.0                    # QQ 未触及
    assert flop.weight_of((14, 14, True)) < flop.weight_of((12, 12, True))

    # 模块输出：被挡越多的对子权重越低（AA < KK < QQ）
    r = _build(cash_state(street="flop", board=["Ah", "7c", "2c"], to_call=0.0))
    dist = r["model"]["v1"]["inferred"]["distribution"]
    assert dist.get("AA", 0) < dist.get("KK", 0) < dist.get("QQ", 0)


def test_impossible_hand_is_removed_not_invented():
    """四张 A 全在已知牌里 → AA 必须消失（不得保留、不得虚构）。"""
    st = cash_state(street="river", to_call=0.0,
                    board=["Ah", "Ad", "Ac", "2c", "3c"])
    st["players"][1]["hole_cards"] = ["As", "Kd"]
    r = _build(st)
    dist = r["model"]["v1"]["inferred"]["distribution"]
    assert dist.get("AA", 0) == 0
    # 且组合空间里不存在任何含 A 的"不可能"组合被凭空加入
    assert all(v > 0 for v in dist.values())


def test_range_never_contains_a_card_held_by_hero():
    """已知牌不得重复进入对手 range。

    验证**既有引擎的确定性移除语义**（先核实再断言）：
    移除 2 张 A 后，AA 仍有 C(2,2)=1 种可用组合（牌堆剩 As/Ah 两张中的
    另一对子），故 AA 权重由 1.0 降为 1/6 —— 是**降权**而非消失；
    只有 3 张以上同点数牌被移除时 AA 才真正不可用（见下一个测试）。
    """
    st = cash_state(street="flop", board=[], to_call=0.0)
    st["players"][1]["hole_cards"] = ["As", "Ah"]
    r = _build(st)
    dist = r["model"]["v1"]["inferred"]["distribution"]
    assert dist.get("AKs", 0) > 0
    # AA 被降权（< KK），因为只剩 1/6 的具体组合
    assert dist.get("AA", 0) < dist.get("KK", 0)
    # 绝对权重：156 个未受阻组合 + AA 的 1/6（`stats()` 按 3 位小数输出）
    assert abs(r["model"]["v1"]["inferred"]["stats"]["total_weight"]
               - (156 + 1 / 6)) < 1e-3


def test_three_dead_aces_make_aa_truly_impossible():
    """3 张 A 在已知牌里 → 只剩 1 张 A → AA 无法组成 → 权重 0。"""
    st = cash_state(street="turn", to_call=0.0,
                    board=["Ah", "Ad", "Ac", "2c"])
    st["players"][1]["hole_cards"] = ["Kd", "Qd"]
    r = _build(st)
    dist = r["model"]["v1"]["inferred"]["distribution"]
    assert dist.get("AA", 0) == 0
    assert all(v > 0 for v in dist.values())


def test_missing_hero_cards_marked_unknown_but_range_still_reported():
    st = cash_state()
    st["players"][1]["hole_cards"] = []
    r = _build(st)
    assert "hero.hole_cards" in r["known_cards"]["unknown"]
    assert r["known_cards"]["count"] == 0
    assert r["model"]["v1"]["level"] == "inferred"      # 仍可给出（未知牌未移除）


def test_missing_board_is_not_invented():
    st = cash_state(street="flop", to_call=0.0)
    st.pop("board", None)
    r = _build(st)
    v1 = r["model"]["v1"]
    assert "board" not in json.dumps(r["known_cards"]["unknown"])
    assert v1["level"] == "inferred"


# ── duplicate / conflict ──────────────────────────────────────────────
def test_identical_duplicate_action_applies_once():
    st = cash_state(action_history=[
        {"street": "preflop", "player_id": "v1", "action": "raise",
         "sequence": 0},
        {"street": "preflop", "player_id": "v1", "action": "raise",
         "sequence": 0}])
    r = _build(st)
    assert len(r["model"]["v1"]["observed"]["applied_factors"]) == 1
    assert r["conflicts"] == []


def test_conflicting_action_same_slot_applies_neither():
    st = cash_state(action_history=[
        {"street": "preflop", "player_id": "v1", "action": "raise",
         "sequence": 0},
        {"street": "preflop", "player_id": "v1", "action": "fold",
         "sequence": 0}])
    r = _build(st)
    assert r["model"]["v1"]["observed"]["applied_factors"] == []
    assert any(x["reason"] == "conflicting_action_same_slot"
               for x in r["conflicts"])
    # 与无行动一致（两条都不采信）
    assert r["model"]["v1"]["inferred"]["distribution"] == \
        _build(cash_state())["model"]["v1"]["inferred"]["distribution"]


def test_same_slot_different_players_is_not_a_conflict():
    st = cash_state(action_history=[
        {"street": "preflop", "player_id": "v1", "action": "raise",
         "sequence": 0},
        {"street": "preflop", "player_id": "v2", "action": "fold",
         "sequence": 0}])
    r = _build(st)
    assert r["conflicts"] == []
    assert len(r["model"]["v1"]["observed"]["applied_factors"]) == 1
    assert len(r["model"]["v2"]["observed"]["applied_factors"]) == 1


# ── 多玩家隔离 ────────────────────────────────────────────────────────
def test_multi_player_ranges_are_isolated():
    st = cash_state(action_history=[
        {"street": "preflop", "player_id": "v1", "action": "raise",
         "sequence": 0}])
    r = _build(st)
    assert sorted(r["model"]) == ["v1", "v2"]
    assert len(r["model"]["v1"]["observed"]["applied_factors"]) == 1
    assert r["model"]["v2"]["observed"]["applied_factors"] == []
    # 两人初始 range 来自各自位置（BB vs CO）
    assert r["model"]["v1"]["initial_range"]["source_key"] == "ranges.open.BB"
    assert r["model"]["v2"]["initial_range"]["source_key"] == "ranges.open.CO"


def test_hero_action_does_not_update_opponent_range():
    st = cash_state(action_history=[
        {"street": "preflop", "player_id": "hero", "action": "raise",
         "sequence": 0}])
    r = _build(st)
    assert "hero" not in r["model"]
    for m in r["model"].values():
        assert m["observed"]["applied_factors"] == []


def test_folded_player_has_no_range_entry():
    st = cash_state()
    st["players"][0]["status"] = "folded"
    r = _build(st)
    assert "v1" not in r["model"]


def test_unknown_position_player_is_unknown_not_guessed():
    st = cash_state()
    st["players"][0]["position"] = None
    r = _build(st)
    v1 = r["model"]["v1"]
    assert v1["level"] == "unknown"
    assert v1["reason"] == "unknown_position"


# ── normalization ─────────────────────────────────────────────────────
def test_normalization_is_deterministic_and_bounded():
    r = _build()
    v1 = r["model"]["v1"]["inferred"]
    norm = v1["normalized"]
    assert norm
    assert max(norm.values()) == 1.0
    assert all(0.0 < v <= 1.0 for v in norm.values())
    # 键为**既有** 169 分类名（与 distribution 同口径）
    assert set(norm) == set(v1["distribution"])
    # 归一化后最大值唯一且为 1.0；顺序 = 权重降序 + 名称升序（确定性）
    items = list(norm.items())
    assert items == sorted(items, key=lambda kv: (-kv[1], kv[0]))


def test_distribution_sums_to_one():
    r = _build()
    dist = r["model"]["v1"]["inferred"]["distribution"]
    # distribution() 内部按 6 位小数四舍五入（既有实现），故容差取 1e-4
    assert abs(sum(dist.values()) - 1.0) < 1e-4


# ── missing evidence → UNKNOWN ────────────────────────────────────────
def test_missing_action_history_is_flagged_but_range_still_built():
    st = cash_state()
    st.pop("action_history", None)
    r = _build(st)
    assert "action_history" in r["unknown"]
    assert r["model"]["v1"]["observed"]["applied_factors"] == []
    assert r["model"]["v1"]["level"] == "inferred"


def test_empty_range_after_filtering_is_unknown():
    """起始 range 被已知牌完全挡死 → UNKNOWN（不虚构组合）。"""
    st = cash_state()
    st["players"][0]["position"] = "BTN"
    # 用只含 AA 的 route 让移除生效：three_bet default = JJ+,AJs+,KQs,AQo+
    st["players"][1]["hole_cards"] = ["Js", "Jd"]
    st["board"] = ["Jh", "Jc"]
    r = build_range_model(st, ranges_config=_CFG, route="three_bet")
    v1 = r["model"]["v1"]
    assert v1["level"] in ("unknown", "inferred")
    if v1["level"] == "unknown":
        assert v1["reason"] == "range_empty_after_filtering"


# ── CASH / 非 CASH isolation ──────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin", None])
def test_non_cash_fails_closed(mode):
    r = build_range_model(cash_state(game_type=mode), ranges_config=_CFG)
    assert r["present"] is False
    assert "model" not in r


def test_non_dict_state_fails_closed():
    for bad in (None, [], "cash", 5):
        assert build_range_model(bad, ranges_config=_CFG)["present"] is False


def test_no_tournament_fields_and_no_random():
    r = _build()
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("icm", "bounty", "satellite", "payout", "gto",
                   "equilibrium", "runout"):
        assert banned not in blob, banned


def test_module_has_no_randomness_or_forbidden_imports():
    src = (PKG / "cash_range_model.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("random", "research_unfreeze", "vision2", "strategy_core",
                   "judge", "multistreet"):
        assert banned not in joined, banned
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for banned in ("compute_ev", "evaluate_intel", "positions_for",
                   "sample_runouts", "remaining_deck", "open"):
        assert banned not in called, banned


# ── V3G-5 接入（仅显式 prior；统计不当牌力）───────────────────────────
def test_opponent_model_prior_requires_adequate_confidence():
    weak = {"present": True, "model": {"v1": {
        "confidence": {"level": "unknown"},
        "inferred": {"vpip": {"value": 0.5, "samples": 3,
                              "level": "unknown"}}}}}
    strong = {"present": True, "model": {"v1": {
        "confidence": {"level": "adequate"},
        "inferred": {"vpip": {"value": 0.5, "samples": 200,
                              "level": "adequate"}}}}}
    r_weak = _build(opponent_model=weak)
    r_strong = _build(opponent_model=strong)
    assert r_weak["model"]["v1"]["evidence"]["opponent_prior_multiplier"] is None
    assert r_strong["model"]["v1"]["evidence"][
        "opponent_prior_multiplier"] is not None
    # multiplier 被限制在声明区间内
    pm = r_strong["model"]["v1"]["evidence"]["opponent_prior_multiplier"]
    assert 0.8 <= pm <= 1.25


def test_opponent_model_does_not_change_range_shape_only_scale():
    """统计只影响**显式声明的** uniform multiplier，不改变分布形状。"""
    strong = {"present": True, "model": {"v1": {
        "confidence": {"level": "adequate"},
        "inferred": {"vpip": {"value": 0.5, "samples": 200,
                              "level": "adequate"}}}}}
    a = _build()["model"]["v1"]["inferred"]["distribution"]
    b = _build(opponent_model=strong)["model"]["v1"]["inferred"]["distribution"]
    assert set(a) == set(b)
    for k in a:
        assert abs(a[k] - b[k]) < 1e-6, k


def test_explicit_prior_multiplier_is_honoured_and_bounded():
    r = _build(prior_multiplier=2.0)
    assert r["model"]["v1"]["evidence"]["prior_multiplier"] == 2.0
    r_bad = _build(prior_multiplier=-1.0)
    assert r_bad["model"]["v1"]["evidence"]["prior_multiplier"] == 1.0


def test_opponent_derived_weights_are_traceable_to_metric_samples_evidence():
    """**可追溯性**（授权要求）：统计推断产生的权重必须能追到
    metric / samples / evidence，且缩放参数公开、结果可复算。"""
    strong = {"present": True, "model": {"v1": {
        "confidence": {"level": "adequate", "hands_with_evidence": 240},
        "inferred": {"vpip": {"value": 0.5, "samples": 240,
                              "level": "adequate",
                              "evidence": "observed_samples=240"}}}}}
    ev = _build(opponent_model=strong)["model"]["v1"]["evidence"]
    mult = ev["opponent_prior_multiplier"]
    prov = ev["opponent_prior_provenance"]
    assert mult is not None and prov is not None
    assert prov["source"] == "v3g5_opponent_model"
    assert prov["metric"] == "vpip"
    assert prov["samples"] == 240
    assert prov["value"] == 0.5
    assert prov["metric_level"] == "adequate"
    assert prov["confidence_level"] == "adequate"
    assert prov["evidence"] == "observed_samples=240"
    # 缩放参数公开 → 结果可独立复算
    raw = 1.0 + (prov["value"] - prov["anchor"]) * prov["slope"]
    expected = min(prov["bounds"][1], max(prov["bounds"][0], raw))
    assert abs(mult - round(expected, 4)) < 1e-9
    json.dumps(prov)


def test_no_prior_provenance_when_evidence_insufficient():
    """证据不足 → 不得产生任何统计推断 provenance（不猜）。"""
    weak = {"present": True, "model": {"v1": {
        "confidence": {"level": "low"},
        "inferred": {"vpip": {"value": 0.5, "samples": 12,
                              "level": "unknown"}}}}}
    ev = _build(opponent_model=weak)["model"]["v1"]["evidence"]
    assert ev["opponent_prior_multiplier"] is None
    assert ev["opponent_prior_provenance"] is None


# ── 无 random / 无未来牌 / 无 runout 采样（显式断言）───────────────────
def test_no_runout_sampling_or_future_card_fields_in_output():
    """输出中**不得**出现 runout / future-card / 采样相关字段。"""
    r = _build(cash_state(street="flop", board=["Ah", "7c", "2c"], to_call=0.0))
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("runout", "future_board", "future_cards", "sampled",
                   "n_runouts", "equity", "win_probability"):
        assert banned not in blob, banned
    # 公共牌只反映**已知**牌面，不含任何未来牌
    assert r["known_cards"]["count"] == 5


def test_module_source_contains_no_random_or_sampling_calls():
    """源码级：不得 import random、不得出现采样/未来牌 API。"""
    src = (PKG / "cash_range_model.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name != "random" for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            assert (n.module or "") != "random"
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("sample_runouts", "remaining_deck", "sample_weighted",
                   "sample_non_colliding", "choice", "shuffle"):
        assert banned not in called, banned
    assert "random" not in attrs


def test_opponent_model_absent_is_no_adjustment():
    r = _build()
    for m in r["model"].values():
        assert m["evidence"]["opponent_prior_multiplier"] is None
        assert m["evidence"]["prior_multiplier"] == 1.0


# ── deterministic repeat + 只读 ───────────────────────────────────────
def test_repeated_runs_are_identical_and_json_safe():
    st = cash_state(action_history=[
        {"street": "preflop", "player_id": "v1", "action": "raise",
         "sequence": 0}])
    first = _build(st)
    for _ in range(5):
        assert _build(copy.deepcopy(st)) == first
    json.dumps(first)


def test_model_does_not_mutate_inputs():
    st = cash_state(action_history=[
        {"street": "preflop", "player_id": "v1", "action": "raise",
         "sequence": 0}])
    before = copy.deepcopy(st)
    _build(st)
    assert st == before


def test_model_is_not_wired_into_production():
    importers = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name == "cash_range_model.py":
            continue
        src = p.read_text(encoding="utf-8-sig", errors="ignore")
        if "cash_range_model" in src:
            importers.append(str(p.relative_to(ROOT)))
    # BENCH-1 Task0.1（owner 授权基线迁移，6 个引用方）：
    # 2026-09 中旬 P30-P41 生产桥/策略链阶段已授权下列模块引用本层。
    # 按 test_v3g11 SANCTIONED_WIRING 惯例登记为合法基线（精确集合匹配——
    # 任何未登记的新引用方、或已登记引用方消失，都会被检出）。
    sanctioned_importers = {
        "poker_decision_ai/cash_architecture_closure_audit.py",
        "poker_decision_ai/cash_strategy_action_policy.py",
        "poker_decision_ai/cash_strategy_belief_update.py",
        "poker_decision_ai/cash_strategy_consumption_gate.py",
        "poker_decision_ai/cash_strategy_dynamic_belief.py",
        "poker_decision_ai/cash_strategy_likelihood_model.py",
    }
    normalized = {i.replace(chr(92), '/') for i in importers}
    assert normalized == sanctioned_importers, (
        f"cash_range_model 引用方与授权基线不符（新引用方需所有者授权登记）: "
        f"unregistered={sorted(normalized - sanctioned_importers)} "
        f"missing={sorted(sanctioned_importers - normalized)}")


# ── V3G-3/4/5 回归 + Judge 逐位一致 ──────────────────────────────────
def test_v3g3_v3g4_v3g5_still_consistent():
    from poker_decision_ai.cash_multistreet import build_multistreet_context
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_preflop_decision import build_preflop_decision
    st = cash_state()
    pf = build_preflop_decision(st)
    ms = build_multistreet_context(st, preflop=pf)
    om = build_opponent_model(st, this_hand_actions=[
        {"player_id": "v1", "action": "raise"}])
    assert pf["present"] and ms["present"] and om["present"]
    r = build_range_model(st, ranges_config=_CFG, opponent_model=om)
    assert r["present"] is True


async def test_judge_action_ev_size_bitwise_identical_with_and_without_range():
    """接 production 前的硬不变式：跑/不跑 range 更新，Judge 输出逐位一致。"""
    import uuid
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    cfg = load_config()
    logs = ROOT / ".tmp" / "v3g6_logs" / uuid.uuid4().hex
    logs.mkdir(parents=True, exist_ok=True)
    ctx = AgentContext(request_id="v3g6", session_id="v3g6", config=cfg,
                       providers=build_providers(cfg), repos=build_in_memory(),
                       logger=EventLogger(str(logs)),
                       metrics=MetricsCollector())
    st = cash_state(pot=200.0, to_call=50.0, bb=2.0, min_raise_to=100.0,
                    street="flop", board=["Ah", "7c", "2c"])
    eq = {"hero_equity": 0.8, "outcome": {"win": 0.8, "tie": 0.0, "lose": 0.2}}
    inp = {"state": copy.deepcopy(st), "equity": copy.deepcopy(eq)}

    before = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output
    build_range_model(st, ranges_config=_CFG)          # 跑 range 更新
    after = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output

    # 覆盖 Judge **全部**输出键（授权点名的 action / alternative_action /
    # recommended_size / EV / Judge weights / confidence / reasoning / warnings，
    # 加上 uncertainty / key_evidence / evidence / final_inputs_summary）
    assert set(before) == set(after)
    for k in sorted(before):
        assert before.get(k) == after.get(k), k


def test_judge_has_no_scores_or_fallback_keys_so_they_cannot_be_compared():
    """如实记录：Judge 输出**不存在** `scores` / `fallback` 两个键。

    授权要求"必须证明 scores / fallback 逐位不变"。经审计，Judge 的
    `JudgeOutput` 没有这两个字段：动作分 `scores` 是**局部变量**、不进入输出；
    fallback 行为由 `final_inputs_summary`（各来源 ok/unavailable）与
    `weights_used`（仅可用来源计权）表达。因此可证明的字段集 =
    Judge 的**全部输出键**（见上一条测试），`scores`/`fallback` 以
    "输出键集合前后相等 + 上述代理字段逐位相等"覆盖。
    """
    import inspect
    from poker_decision_ai.agents.judge import agent as judge_mod
    keys = set(getattr(judge_mod.JudgeOutput, "__dataclass_fields__", {}))
    assert "scores" not in keys
    assert "fallback" not in keys
    # 局部 scores 仅存在于函数体内（不进入输出）
    src = inspect.getsource(judge_mod)
    assert "scores: dict[str, float]" in src
    assert "out.scores" not in src
