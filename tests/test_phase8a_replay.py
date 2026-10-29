"""Phase 8A Replay Engine 测试：确定性/街序/底池/盲注/终局/摊牌/all-in/
位置推导/信息集隔离/反向注入/两臂共享 RNG/臂网格冻结/源码红线。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from poker_decision_ai.domain.poker.canonical import (  # noqa: E402
    CanonicalAction, CanonicalHand, CanonicalPlayer)
import phase8a_replay as p8  # noqa: E402


# ---------------- 合成手构造 ----------------
def _p(pid, stack=1000.0):
    return CanonicalPlayer(player_id=pid, seat=None, stack_start=stack)


def mk_hand(sb=25.0, bb=50.0, blinds=None, pre=None, flop_cards=None,
            flop=None, turn_card=None, turn=None, river_card=None, river=None,
            showdown=None, winners=None, all_in=None):
    players = [_p(f"p{i}") for i in range(1, 7)]
    return CanonicalHand(
        hand_id="synth-1", game_mode="cash", game_type="cash", sb=sb, bb=bb,
        table_size=6,
        players=players,
        preflop_actions=pre or [],
        flop=flop_cards or [], flop_actions=flop or [],
        turn=turn_card or [], turn_actions=turn or [],
        river=river_card or [], river_actions=river or [],
        showdown=showdown, winners=winners or [], all_in=all_in or [],
    )


def A(street, seq, pid, action, amount=None, to=None):
    return CanonicalAction(street=street, seq=seq, player_id=pid,
                           action=action, amount=amount, to=to)


def standard_pre():
    return [A("preflop", 1, "p1", "post", 25.0),
            A("preflop", 2, "p2", "post", 50.0),
            A("preflop", 3, "p3", "fold"),
            A("preflop", 4, "p4", "fold"),
            A("preflop", 5, "p5", "call", 50.0),
            A("preflop", 6, "p6", "fold"),
            A("preflop", 7, "p1", "fold"),
            A("preflop", 8, "p2", "check")]


def standard_pos():
    return p8.derive_positions_orbit([f"p{i}" for i in range(1, 7)],
                                     [25, 50, 0, 0, 0, 0], standard_pre())


# ---------------- 位置推导 ----------------
def test_position_orbit_labels():
    pos = standard_pos()
    assert pos is not None
    assert pos["labels"] == {"p1": "SB", "p2": "BB", "p6": "BTN",
                             "p3": "EP", "p4": "EP", "p5": "LP"}
    assert pos["button_seat"] == pos["seats"]["p6"]


def test_position_blinds_orbit_mismatch_excluded():
    # blinds 说 SB=p3、BB=p4，但 orbit 说 SB=p1、BB=p2 → 歧义 → None
    pos = p8.derive_positions_orbit([f"p{i}" for i in range(1, 7)],
                                    [0, 0, 25, 50, 0, 0], standard_pre())
    assert pos is None


def test_position_incomplete_orbit_excluded():
    pre = standard_pre()[:6]          # 缺少 p1/p2 的行动 → orbit 不完整
    pos = p8.derive_positions_orbit([f"p{i}" for i in range(1, 7)],
                                    [25, 50, 0, 0, 0, 0], pre)
    assert pos is None


def test_position_straddle_excluded():
    pos = p8.derive_positions_orbit([f"p{i}" for i in range(1, 7)],
                                    [25, 50, 100, 0, 0, 0], standard_pre())
    assert pos is None


def test_position_hu_rule():
    players = ["a", "b"]
    pos = p8.derive_positions_orbit(players, [50, 100], [])
    assert pos is not None
    assert pos["labels"]["a"] == "BTN"       # HU：SB = button
    assert pos["labels"]["b"] == "BB"


def test_position_seats_path_uses_frozen_rules():
    pos = p8.derive_position_map([f"p{i}" for i in range(1, 7)],
                                 [1, 2, 3, 4, 5, 6], [25, 50, 0, 0, 0, 0],
                                 standard_pre())
    assert pos is not None
    assert pos["labels"]["p1"] == "SB" and pos["labels"]["p2"] == "BB"
    assert pos["labels"]["p6"] == "BTN"


# ---------------- replay：街序/底池/盲注/终局/摊牌/all-in ----------------
def _full_hand():
    return mk_hand(
        pre=standard_pre(),
        flop_cards=["Ah", "Kd", "2c"],
        flop=[A("flop", 1, "p2", "bet", 100.0, to=100.0),
              A("flop", 2, "p5", "call", 100.0)],
        turn_card=["3s"],
        turn=[A("turn", 1, "p2", "check"),
              A("turn", 2, "p5", "check")],
        river_card=["7h"],
        river=[A("river", 1, "p2", "bet", 200.0, to=200.0),
               A("river", 2, "p5", "call", 200.0)],
        showdown={"happened": True,
                  "shows": [{"player_id": "p2"}, {"player_id": "p5"}]},
        winners=[{"player_id": "p2", "amount": 725.0}],
    )


def test_replay_street_and_pot_updates():
    hand = _full_hand()
    trace = p8.replay_hand(hand, "p2", standard_pos(), seed=42)
    assert not trace["excluded"], trace["exclude_reason"]
    streets = [d["street"] for d in trace["decision_points"]]
    assert streets == ["turn"]             # hero=p2 唯一决策点（镜像冻结判定）
    obs = trace["decision_points"][0]["observation"]
    assert obs["street"] == "turn"
    assert obs["pot"] == pytest.approx(325.0)     # 125 + 100 + 100
    assert obs["board"] == ["Ah", "Kd", "2c", "3s"]
    final = trace["final_state"]
    assert final["street"] == "showdown"          # SHOWDOWN 事件生效
    assert final["pot"] == pytest.approx(725.0)   # 325 + 200 + 200
    assert any(r["action"] == "post" for r in final["action_history"])   # 盲注入池


def test_replay_fold_status():
    hand = mk_hand(pre=standard_pre())
    trace = p8.replay_hand(hand, "p2", standard_pos(), seed=42)
    assert not trace["excluded"]
    final = trace["final_state"]
    by = {p["player_id"]: p for p in final["players"]}
    assert by["p3"]["status"] == "folded"
    assert by["p4"]["status"] == "folded"
    assert by["p1"]["status"] == "folded"


def test_replay_all_in_status():
    hand = mk_hand(
        pre=[A("preflop", 1, "p1", "post", 25.0),
             A("preflop", 2, "p2", "post", 50.0),
             A("preflop", 3, "p3", "fold"),
             A("preflop", 4, "p4", "fold"),
             A("preflop", 5, "p5", "call", 1000.0),
             A("preflop", 6, "p6", "fold"),
             A("preflop", 7, "p1", "fold"),
             A("preflop", 8, "p2", "fold")],
        all_in=["p5"],
    )
    trace = p8.replay_hand(hand, "p2", standard_pos(), seed=42)
    assert not trace["excluded"]
    by = {p["player_id"]: p for p in trace["final_state"]["players"]}
    assert by["p5"]["status"] == "allin"


def test_replay_legal_action_ordering_via_engine():
    """乱序事件（turn 在 flop 前）→ 引擎抛 StateEngineError → excluded。"""
    hand = mk_hand(
        pre=standard_pre(),
        turn_card=["3s"],
        turn=[A("turn", 1, "p2", "check")],       # 无 flop 直接 turn
    )
    trace = p8.replay_hand(hand, "p2", standard_pos(), seed=42)
    assert trace["excluded"] is True
    assert "engine_error" in trace["exclude_reason"]


# ---------------- 确定性 ----------------
def test_deterministic_replay_trace():
    hand = _full_hand()
    pos = standard_pos()
    a = p8.replay_hand(hand, "p2", pos, seed=42)
    b = p8.replay_hand(hand, "p2", pos, seed=42)
    assert json.dumps(a, sort_keys=True, default=str) == \
        json.dumps(b, sort_keys=True, default=str)
    assert a["decision_points"] and a["decision_points"][0]["obs_hash"]


def test_different_seed_same_observation_hash():
    hand = _full_hand()
    pos = standard_pos()
    a = p8.replay_hand(hand, "p2", pos, seed=1)
    b = p8.replay_hand(hand, "p2", pos, seed=2)
    assert [d["obs_hash"] for d in a["decision_points"]] == \
        [d["obs_hash"] for d in b["decision_points"]]   # 观测与种子无关


# ---------------- 信息集隔离 ----------------
def test_observation_whitelist_only():
    hand = _full_hand()
    trace = p8.replay_hand(hand, "p5", standard_pos(), seed=42)
    assert len(trace["decision_points"]) >= 2      # flop/turn/river 对手行动触发
    for dp in trace["decision_points"]:
        obs = dp["observation"]
        assert set(obs) <= set(p8.ALLOWED_OBS_KEYS)
        for pl in obs["players"]:
            assert set(pl) <= set(p8.ALLOWED_PLAYER_KEYS)


def test_villain_hole_cards_absent_structurally():
    hand = _full_hand()
    trace = p8.replay_hand(hand, "p5", standard_pos(), seed=42)
    for p in trace["final_state"]["players"]:
        if not p["is_hero"]:
            assert p["hole_cards"] == []        # 引擎从未收到对手手牌


def test_observation_forbidden_injection_hard_fail():
    hand = _full_hand()
    trace = p8.replay_hand(hand, "p5", standard_pos(), seed=42)
    obs = dict(trace["decision_points"][0]["observation"])
    for bad in ("winners", "net_winnings", "future_actions", "showdown",
                "outcome", "villain_hole_cards"):
        obs2 = dict(obs)
        obs2[bad] = [{"player_id": "p2"}]
        with pytest.raises(AssertionError):
            p8.assert_observation_clean(obs2)
    # players 条目注入 hole_cards
    obs3 = json.loads(json.dumps(obs))
    obs3["players"][0]["hole_cards"] = ["As", "Ad"]
    with pytest.raises(AssertionError):
        p8.assert_observation_clean(obs3)
    # 白名单外键
    obs4 = dict(obs)
    obs4["equity"] = 0.5
    with pytest.raises(AssertionError):
        p8.assert_observation_clean(obs4)


def test_observation_contains_only_legal_info():
    hand = _full_hand()
    trace = p8.replay_hand(hand, "p5", standard_pos(), seed=42)
    obs = trace["decision_points"][0]["observation"]
    assert obs["hero_hole_cards"] == []          # 合成手未给 p5 手牌 → 如实为空
    assert obs["hero_position"] == "LP"
    assert obs["blinds"] == {"sb": 25.0, "bb": 50.0, "ante": 0.0}
    assert obs["button_seat"] == standard_pos()["button_seat"]
    assert obs["action_history"]                 # 已发生的公开行动可见
    assert obs["available_actions"]


# ---------------- 两臂共享 RNG 契约 ----------------
def test_arms_identical_when_no_recommendation():
    hand = _full_hand()
    trace = p8.replay_hand(hand, "p2", standard_pos(), seed=42,
                           recommendation_provider=lambda obs: None)
    assert trace["decision_points"]
    for dp in trace["decision_points"]:
        assert dp["baseline_action"] == dp["exploit_action"]


def test_arms_ineligible_recommendation_identical():
    hand = _full_hand()
    trace = p8.replay_hand(hand, "p2", standard_pos(), seed=42,
                           recommendation_provider=lambda obs: {"eligible": False})
    for dp in trace["decision_points"]:
        assert dp["baseline_action"] == dp["exploit_action"]


def test_exploit_multiplier_math_frozen():
    import random as _r
    rng = _r.Random(42)
    rec = {"eligible": True,
           "mapping": {"action_class": "raise", "sign": +1},
           "confidence": 0.9, "deviation_magnitude": 0.1}
    arm = p8.ExploitArm(rng, k=1.0, eps=1.0,
                        recommendation_provider=lambda obs: rec)
    obs = {"available_actions": ["fold", "call", "raise"]}
    probs = arm.action_probs(obs)
    base = p8.BaselineArm(_r.Random(42)).action_probs(obs)
    mult = 1.0 + 1.0 * 0.9 * 0.1 * 1.0          # = 1.09
    expected_raise = base["raise"] * mult
    total = base["fold"] + base["call"] + expected_raise
    assert probs["raise"] == pytest.approx(expected_raise / total)


def test_arm_grid_frozen():
    import random as _r
    for k in (0.3, 1.5, 0.0):
        with pytest.raises(AssertionError):
            p8.ExploitArm(_r.Random(1), k=k, eps=1.0)
    for eps in (0.1, 0.7, 2.0):
        with pytest.raises(AssertionError):
            p8.ExploitArm(_r.Random(1), k=1.0, eps=eps)
    assert p8.PRIMARY_ARM == (1.0, 1.0)
    assert set(p8.FROZEN_K) == {0.5, 1.0}
    assert set(p8.FROZEN_EPS) == {0.25, 0.5, 1.0}


def test_shared_rng_stream_contract():
    """同 seed 下两臂在相同观测+相同概率表时产生相同行动序列（共享 RNG 流）。"""
    import random as _r
    rng = _r.Random(7)
    base = p8.BaselineArm(rng)
    rng2 = _r.Random(7)
    expl = p8.ExploitArm(rng2, k=1.0, eps=1.0,
                         recommendation_provider=lambda obs: None)
    obs = {"available_actions": ["fold", "check", "call", "bet", "raise"]}
    for _ in range(50):
        assert base.decide(obs) == expl.decide(obs)


# ---------------- 真实数据 smoke ----------------
def test_real_pluribus_row_replays(pluribus_row):
    hand, _ = p8.map_row_to_canonical(pluribus_row, 0)
    assert hand is not None
    pos = p8.derive_position_map(
        [str(p.player_id) for p in hand.players], None,
        pluribus_row["blinds_or_straddles"], hand.preflop_actions)
    assert pos is not None
    hero = str(hand.players[0].player_id)
    assert hand.players[0].hole_cards                       # Pluribus 全员真手牌
    trace = p8.replay_hand(hand, hero, pos, seed=42)
    assert not trace["excluded"], trace["exclude_reason"]
    a = json.dumps(trace, sort_keys=True, default=str)
    b = json.dumps(p8.replay_hand(hand, hero, pos, seed=42),
                   sort_keys=True, default=str)
    assert a == b


@pytest.fixture
def pluribus_row():
    import glob

    import pyarrow.parquet as pq
    files = sorted(glob.glob(r"C:\PokerDataset\UofT_CPRG\data\*.parquet"))
    if not files:
        pytest.skip("dataset not available")
    # 快速列投影定位含 pluribus 的分片，再读完整行
    for fp in files:
        t = pq.read_table(fp, columns=["source_file"])
        if any("pluribus" in (s.as_py() or "").lower() for s in t["source_file"]):
            full = pq.read_table(fp)
            for r in full.to_pylist():
                if "pluribus" in (r.get("source_file") or "").lower():
                    return r
    pytest.skip("no pluribus row found")


# ---------------- 源码红线 ----------------
def test_source_redlines():
    src = open(Path(p8.__file__), encoding="utf-8").read()
    assert "exploit_utility_proven" not in src
    assert "ev_gain" not in src
    assert "phase6g" not in src
    assert "get_profile" not in src
    assert "population_priors" not in src
    assert "half_life" not in src
    assert "gmm" not in src
    assert "hero_equity" not in src
    assert "icm" not in src.lower()
    assert "solver" not in src.lower()
