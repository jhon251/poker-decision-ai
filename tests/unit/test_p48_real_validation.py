"""P48 — Real-data Strategy Validation 测试。

覆盖：固定 manifest；BASELINE vs REAL 对照（Range/Fold/Equity/EV）；
方向 sanity（收紧/宽松/position 隔离/player-population 隔离/样本递增
无跳变/最小样本回退/REAL 不混合）；MTT 机制检查 + PKO NOT VALIDATABLE；
provenance 一致；deterministic；posterior 合法归一化；真实文件冒烟。
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from poker_decision_ai.domain.poker import evidence_gate as eg
from poker_decision_ai.domain.poker.range_posterior import RangePosteriorIndex
from poker_decision_ai.real_validation import (
    LAMBDA_JUMP_LIMIT,
    build_evidence,
    collect_real_sample,
    compare_scenario,
    direction_check,
    lambda_curve,
    make_state,
    manifest,
    pick_target_player,
    scenarios,
    total_variation,
)
from poker_decision_ai.core.config import load_config

UOFT_DIR = Path(__file__).resolve().parents[2] / "data" / "uoft"
CHUNK0 = UOFT_DIR / "normalized" / "chunk_000.jsonl"
CFG = load_config()


# ---------------------------------------------------------------- fixtures
def mini_events(n_hands: int, player: str, position: str, entry_frac: float,
                fold_frac: float = 0.3, prefix: str = "h") -> list[dict]:
    """经 extract_events 的真实事件流（与 P43/P46 同管）。

    entry_frac 控制翻前入池率（未入池手无后续街——与真实牌局一致）；
    fold_frac 控制入池手面对 flop 下注的弃牌率（facing=bet 事件仅来自
    入池手——条件证据量 = n_hands × entry_frac，测试须保证 ≥30）。
    """
    from poker_decision_ai.domain.poker.conditional_model import extract_events
    # 5-max 座位：player 固定 seat 2（seats 索引 i=1）；
    # positions_for 冻结语义 k=(i-b-3)%n → player 位置随 button_seat 旋转：
    # b(seat1)→SB, b(seat2=自己)→BTN, b(seat3)→CO, b(seat4)→UTG, b(seat5)→BB
    btn_seat = {"BTN": 2, "CO": 3, "UTG": 4, "BB": 5, "SB": 1}[position]
    seats = [{"seat": 1, "name": "o1"}, {"seat": 2, "name": player},
             {"seat": 3, "name": "o3"}, {"seat": 4, "name": "o4"},
             {"seat": 5, "name": "o5"}]
    out = []
    for i in range(n_hands):
        entered = (i % 100) / 100.0 < entry_frac
        pf_action = "raise" if entered else "fold"
        actions = [
            {"street": "preflop", "player": player, "action": pf_action,
             "amount": 2.0, "to": 2.0},
        ]
        if entered:
            f = 1 if (i % 100) / 100.0 < fold_frac else 0
            actions += [
                {"street": "flop", "player": "o1", "action": "bet",
                 "amount": 4.0, "to": 4.0},
                {"street": "flop", "player": player,
                 "action": "fold" if f else "call", "amount": 0.0,
                 "to": None},
            ]
        parsed = {
            "hand_id": f"{prefix}{i}", "button_seat": btn_seat,
            "seats": seats,
            "actions": actions,
            "street_boards": {"flop": ["Ah", "Kd", "7c"]},
        }
        out.append({"hand_id": f"{prefix}{i}",
                    "events": extract_events(parsed),
                    "quality": {"status": "warning", "warnings": []},
                    "verdicts": eg.classify_canonical(
                        {"hand_id": f"{prefix}{i}", "game_type": "cash",
                         "stakes": "NL100", "button_seat": 1,
                         "players": [{"player_id": player, "seat": 1,
                                      "hole_cards": [], "position": None},
                                     {"player_id": "other", "seat": 2,
                                      "hole_cards": [], "position": None}],
                         "preflop_actions": [{"street": "preflop", "seq": 0,
                                              "player_id": player,
                                              "action": "raise",
                                              "amount": 2.0}],
                         "flop_actions": [{"street": "flop", "seq": 1,
                                           "player_id": player,
                                           "action": "call", "amount": 0.0}],
                         "meta": {"venue": "PokerStars"}}, None)})
    return out


def all_hands_with(*player_groups: tuple[str, list[dict]]) -> dict:
    out: dict[str, list[dict]] = {}
    for pid, hands in player_groups:
        out[pid] = hands
    return out


@pytest.fixture
def loose_evidence():
    hands = mini_events(80, "v1", "CO", 0.5, fold_frac=0.2)   # 50% entry, 爱跟
    return build_evidence(hands, all_hands_with(("v1", hands)))


@pytest.fixture
def tight_evidence():
    hands = mini_events(80, "v1", "CO", 0.1, fold_frac=0.9)   # 10% entry, 紧凶弃
    return build_evidence(hands, all_hands_with(("v1", hands)))


@pytest.fixture
def base_profile(loose_evidence):
    from poker_decision_ai.real_validation import real_profile
    return real_profile("villain", loose_evidence)


# ------------------------------------------------------------ manifest 与场景
def test_manifest_fixed_and_declares_not_validatable():
    m = manifest()
    # P49/P51 语义演进（授权）：manifest → p51-real-v3（P48 场景保持前 17 项）
    assert m["manifest_version"] == "p51-real-v3"
    assert [s["id"] for s in manifest()["scenarios"][:17]] ==         [s["id"] for s in scenarios()]
    assert [s["id"] for s in m["scenarios"]] == \
        [s["id"] for s in manifest()["scenarios"]]
    nv = {e["evidence_type"]: e["status"] for e in m["not_validatable"]}
    assert nv["bounty_context"] == "NOT VALIDATABLE WITH CURRENT DATA"
    assert nv["combo_level_range_ground_truth"] == \
        "NOT VALIDATABLE WITH CURRENT DATA"
    assert nv["mtt_strategy_transfer"] == "MECHANISM_CHECK_ONLY"
    # PKO 场景不得出现在可验证场景中
    assert not any("pko" in s["id"].lower() or "ko" in s["id"].lower()
                   for s in m["scenarios"])


def test_scenarios_cover_matrix():
    ids = [s["id"] for s in scenarios()]
    for street in ("preflop", "flop", "turn", "river"):
        assert any(street in i for i in ids)
    for pos in ("CO", "BTN", "SB", "BB"):
        assert any(pos in i for i in ids)


# ------------------------------------------------------------ 对照核心
async def test_loose_player_loosens_range_with_provenance(base_profile):
    # P50-2 语义演进：loose 的定义回到起始范围通道（entry 率 50% vs book
    # 14%）——preflop 场景；flop continuing 方向由 P49 三臂测试覆盖
    scen = {"id": "cash_preflop_CO", "street": "preflop", "position": "CO",
            "route": "open", "game_type": "cash"}
    cmp1 = await compare_scenario(CFG, scen, "villain", base_profile, None)
    m = cmp1["metrics"]
    assert m["range_tv"] > 0.0                       # 范围确实变化
    r = cmp1["real"]
    assert r["posterior_available"] is True
    assert r["posterior_status"] == "player"
    assert r["lambda"] < 0                           # 松玩家 → 放宽倾斜
    assert r["data_kind"] == "REAL"
    # fold 概率变化被记录且合法
    for size, row in m["fold"].items():
        if row["diff"] is not None:
            assert -1.0 <= row["diff"] <= 1.0
    assert m["equity"]["base"] is not None and m["equity"]["real"] is not None
    assert m["best_action_change"] in (True, False)


async def test_tight_player_tightens_range(tight_evidence):
    scen = {"id": "cash_preflop_BTN", "street": "preflop", "position": "BTN",
            "route": "open", "game_type": "cash"}
    from poker_decision_ai.real_validation import real_profile
    cmp1 = await compare_scenario(CFG, scen, "villain",
                                  real_profile("villain", tight_evidence), None)
    r = cmp1["real"]
    assert r["posterior_available"] is True
    assert r["lambda"] > 0                           # 紧玩家 → 收紧倾斜


async def test_direction_check_helper_consistent(loose_evidence, tight_evidence):
    """flop continuing 方向：fold 率高 → 收紧（λ>0）；fold 率低 → 放宽。"""
    scen = {"street": "flop", "position": "CO", "route": "open"}
    all_h = all_hands_with(("v1", mini_events(80, "v1", "CO", 0.5, 0.2)))
    tight = build_evidence(mini_events(80, "v1", "CO", 0.5, 0.9), all_h)
    loose = build_evidence(mini_events(80, "v1", "CO", 0.5, 0.2), all_h)
    out = await direction_check(CFG, scen, tight, loose)
    assert out["tight"].get("direction_ok") is True
    assert out["loose"].get("direction_ok") is True


async def test_fallback_below_min_samples():
    """n=20（< MIN_SAMPLES=30）→ 诚实 baseline，不因真实数据绕过门槛。"""
    hands = mini_events(20, "v1", "CO", 3)
    ev = build_evidence(hands, all_hands_with(("v1", hands)))
    from poker_decision_ai.real_validation import real_profile
    scen = {"id": "cash_flop_CO", "street": "flop", "position": "CO",
            "route": "open", "game_type": "cash"}
    cmp1 = await compare_scenario(CFG, scen, "villain",
                                  real_profile("villain", ev), None)
    r = cmp1["real"]
    assert r["posterior_available"] is False
    assert r["posterior_status"] == "baseline"
    assert r["fallback_reason"]


async def test_population_fallback_when_player_sparse():
    """player <30 但 population ≥30 → population 级 posterior。"""
    few = mini_events(10, "v1", "CO", 0.5, 0.3)
    pop = mini_events(80, "pop1", "CO", 0.5, 0.2) +         mini_events(80, "pop2", "BTN", 0.5, 0.2, prefix="p")
    ev = build_evidence(few, all_hands_with(("v1", few), ("pop1", pop[:80]),
                                            ("pop2", pop[80:])))
    from poker_decision_ai.real_validation import real_profile
    scen = {"id": "cash_flop_CO", "street": "flop", "position": "CO",
            "route": "open", "game_type": "cash"}
    cmp1 = await compare_scenario(CFG, scen, "villain",
                                  real_profile("villain", ev), None)
    assert cmp1["real"]["posterior_status"] == "population"


async def test_position_condition_isolation():
    """BTN 率高、CO 率低 → CO posterior 用 CO 精确键，不被 BTN 污染。"""
    btn_hands = mini_events(80, "v1", "BTN", 0.6, prefix="b")  # 60% entry
    co_hands = mini_events(80, "v1", "CO", 0.1, prefix="c")    # 10% entry
    merged = btn_hands + co_hands
    ev = build_evidence(merged, all_hands_with(("v1", merged)))
    from poker_decision_ai.real_validation import real_profile
    lam_btn = (await compare_scenario(
        CFG, {"id": "x", "street": "preflop", "position": "BTN",
              "route": "open", "game_type": "cash"},
        "villain", real_profile("villain", ev), None))["real"]["lambda"]
    lam_co = (await compare_scenario(
        CFG, {"id": "y", "street": "preflop", "position": "CO",
              "route": "open", "game_type": "cash"},
        "villain", real_profile("villain", ev), None))["real"]["lambda"]
    # CO 位置取 CO 的紧证据（λ>0），BTN 取 BTN 的松证据（λ<0）——无交叉污染
    assert lam_btn < 0 < lam_co


async def test_posterior_distribution_legal_fixed(base_profile):
    from poker_decision_ai.agents.range.agent import RangeAgent
    from poker_decision_ai.real_validation import _RangeCtx
    scen = {"street": "flop", "position": "CO", "route": "open"}
    state = make_state(scen["street"], scen["position"], scen["route"])
    out = await RangeAgent().handle({"state": state,
                                     "opponent": {"players": {"villain":
                                                              base_profile}}},
                                    _RangeCtx(CFG))
    v = out.villains["villain"]
    dist = v["distribution"]
    assert all(w >= 0 for w in dist.values())
    assert abs(sum(dist.values()) - 1.0) < 1e-2


async def test_mtt_mechanism_check_only(base_profile):
    """MTT：只验证 objective/icm 机制与 provenance（不做策略迁移结论）。"""
    scen = {"id": "mtt_mech_flop_CO", "street": "flop", "position": "CO",
            "route": "open", "game_type": "mtt",
            "payouts": [50, 30, 20, 15, 10, 8, 6, 5, 4, 3],
            "mechanism_only": True}
    cmp1 = await compare_scenario(CFG, scen, "villain", base_profile, None)
    m = cmp1["metrics"]
    assert m["ev_objective"]["base"] == "chip_ev+icm"
    assert m["ev_objective"]["real"] == "chip_ev+icm"
    assert m["mechanism_only"] is True
    assert cmp1["real"]["data_kind"] == "REAL"


async def test_deterministic_comparison(base_profile):
    scen = {"id": "cash_river_BTN", "street": "river", "position": "BTN",
            "route": "open", "game_type": "cash"}
    a = await compare_scenario(CFG, scen, "villain", base_profile, None)
    b = await compare_scenario(CFG, scen, "villain", base_profile, None)
    assert a == b


async def test_lambda_curve_smooth_and_threshold_honest():
    """样本递增 → λ 曲线无异常跳变；n<30 段诚实 baseline。"""
    hands = mini_events(120, "v1", "CO", 0.5, 0.3)
    all_h = all_hands_with(("v1", hands))
    scen = {"street": "preflop", "position": "CO", "route": "open"}
    curve = await lambda_curve(hands, all_h, CFG, (20, 40, 60, 120), scen)
    assert curve["no_abnormal_jump"] is True
    assert curve["max_jump"] is not None and curve["max_jump"] <= \
        LAMBDA_JUMP_LIMIT
    # n=20：player 证据不足 → population fallback（全量语料 ≥30）
    assert curve["curve"][0]["status"] == "population"
    assert curve["curve"][-1]["status"] == "player"
    # population 也稀疏时 → 诚实 baseline（P47 纪律）
    sparse = await lambda_curve(hands[:20], all_hands_with(("v1", hands[:20])),
                                CFG, (20,), scen)
    assert sparse["curve"][0]["status"] == "baseline"


def test_total_variation_symmetry():
    d1 = {"A": 0.6, "B": 0.4}
    d2 = {"A": 0.2, "B": 0.8}
    assert total_variation(d1, d2) == pytest.approx(0.4)
    assert total_variation(d1, d1) == 0.0


# ------------------------------------------------------------ 真实文件冒烟
@pytest.mark.skipif(not CHUNK0.exists(),
                    reason="UofT data files not present")
async def test_real_uoft_end_to_end_validation():
    """真实 UofT 样本端到端：采集→证据→对照→确定性（有界扫描）。"""
    hands = collect_real_sample(str(CHUNK0), scan_limit=1500)
    assert hands
    pid = pick_target_player(hands)
    assert pid
    player_hands = hands[pid]
    ev = build_evidence(player_hands, hands)
    from poker_decision_ai.real_validation import real_profile
    prof = real_profile(pid, ev)
    assert prof["data_kind"] == "REAL"
    assert ev["n_hands"] == len(player_hands)
    scen = {"id": "cash_flop_CO", "street": "flop", "position": "CO",
            "route": "open", "game_type": "cash"}
    # state 中 villain 的 player_id 固定为 "villain"；profile 以该键挂接
    cmp1 = await compare_scenario(CFG, scen, "villain", prof, None)
    assert cmp1["metrics"]["equity"]["base"] is not None
    r = cmp1["real"]
    # 真实玩家：要么 posterior 生效（player/population + REAL provenance），
    # 要么诚实 baseline（证据不足——扫描窗口内手数可能 < MIN_SAMPLES）
    assert r["posterior_status"] in ("player", "population", "baseline")
    if r["posterior_available"]:
        assert r["data_kind"] == "REAL"
        assert r["lambda"] is not None
        assert -1.2 <= r["lambda"] <= 1.2               # LAMBDA_CAP 契约
    else:
        assert r["posterior_status"] == "baseline"
        assert r["fallback_reason"]
    a = await compare_scenario(CFG, scen, "villain", prof, None)
    assert a == cmp1                                     # deterministic
    # λ 曲线在真实数据上无异常跳变
    curve = await lambda_curve(player_hands, hands, CFG,
                         tuple(n for n in (20, 60, min(len(player_hands), 200))
                               if n <= len(player_hands)), scen)
    assert curve["no_abnormal_jump"] is True
