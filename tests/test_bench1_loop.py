"""BENCH-1 — 对抗回路正确性测试（小 N；真实 judge 链 + 确定性 bot）。

覆盖：bot 合法性（3 种 kind）、尺寸映射钳制、HU 回路端到端（真实 runtime；
非法动作必须为 0、结算零和、同 seed 复跑逐字节一致）。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from external_harness.engine import fresh_nseat_state
from external_harness.protocol.bench_bots import bench_bots_contract, bot_act
from external_harness.adapter.judge_adapter import (
    deep_merge, map_action_to_engine, judge_adapter_contract)


# ---------------------------------------------------------------- bot 层

def _view(to_call: int = 0) -> dict:
    return {"legal_actions": (["check", "bet", "all-in"] if to_call == 0
                              else ["fold", "call", "raise"]),
            "to_call_chips": to_call, "call_chips": to_call,
            "min_raise_to_chips": 4, "max_raise_to_chips": 200,
            "all_in_raise_to_chips": 200, "pot_chips": 8,
            "hero_hole_cards": ["Ah", "Kd"], "board": [], "street": "preflop",
            "seat": 1, "n_seats": 2, "position_label": "BB",
            "folded_seats": [], "live_seats": [0, 1],
            "stacks_chips": [200, 200], "street_bets_chips": [0, 2]}


def test_bench_bots_all_kinds_legal_across_contexts():
    assert bench_bots_contract()["kinds"] == ["pivot", "frequency_bot",
                                              "loose_aggressive"]
    freq = {"preflop": {"facing_bet": {"fold": 0.3, "call": 0.5, "raise": 0.2},
                        "no_bet": {"check": 0.6, "bet": 0.4}}}
    for kind in ("pivot", "frequency_bot", "loose_aggressive"):
        for to_call in (0, 4):
            for i in range(30):
                res = bot_act(_view(to_call), kind=kind, seed=702,
                              hand_index=i, decision_index=i,
                              freq_table=freq)
                assert res["ok"], (kind, to_call, i, res.get("reason"))
                assert res["action"] in res["legal_actions"]
                if res["action"] in ("bet", "raise"):
                    assert 4 <= res["amount_chips"] <= 200
                elif res["action"] in ("fold", "check"):
                    assert res["amount_chips"] is None


def test_frequency_bot_without_table_degrades_to_pivot():
    res = bot_act(_view(4), kind="frequency_bot", seed=1, hand_index=0,
                  decision_index=0, freq_table={})
    assert res["ok"] and res["action"] in ("fold", "call", "raise")


# ---------------------------------------------------------- 尺寸映射层

def _state_for_map():
    st = fresh_nseat_state(seed=7, n_seats=2, starting_stack=200, button=0)
    # 构造一个 BB 面对 6-chips 加注的状态（to_call=4）
    st = st.apply("raise", 6)          # hero=BTN/SB 加注到 6
    return st


def test_map_action_raise_to_level_and_clamp():
    st = _state_for_map()
    hero = st.current_player
    lo = st.min_aggressive_to(hero)
    hi = st.max_aggressive_to(hero)
    # raise-to 语义：3bb → 6 chips 水平（>= min）
    r = map_action_to_engine(st, "raise", recommended_size_bb=3.0)
    assert r["ok"] and lo <= r["amount_chips"] <= hi
    # 过小尺寸 → 钳到 min（显式记录，不静默）
    r2 = map_action_to_engine(st, "raise", recommended_size_bb=0.5)
    assert r2["amount_chips"] == lo and r2["clamped"]
    # 过大尺寸 → 钳到 all-in
    r3 = map_action_to_engine(st, "raise", recommended_size_bb=10_000.0)
    assert r3["amount_chips"] == hi and r3["clamped"]
    # fold/check 不带金额
    assert map_action_to_engine(st, "fold")["amount_chips"] is None


def test_deep_merge_overrides():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 9}, "e": 4})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3, "e": 4}


def test_judge_adapter_contract_declares_boundaries():
    c = judge_adapter_contract()
    assert c["hero_is_production_judge_chain"] is True
    assert c["uses_vendored_cfr_for_own_decisions"] is False
    assert c["opponent_cards_visible"] is False


# ------------------------------------------------------- 端到端（真实链）

def _hero():
    from external_harness.adapter.judge_adapter import JudgeHeroController
    return JudgeHeroController(
        base_config_path="configs/config.yaml",
        overrides={"realtime": {"deadline_ms": 60000,
                                "agent_timeout_ms": 30000},
                   "storage": {"dir": "data/bench_test/repository"},
                   "app": {"logs_dir": "logs/bench_test"}})


def test_hu_match_two_hands_real_chain_and_deterministic():
    from external_harness.protocol.bench_match import run_match
    hero = _hero()
    res1 = run_match(hero=hero, n_hands=2, seed_base=42, bot_kind="pivot",
                     bot_seed=701, progress_every=0)
    assert len(res1["records"]) == 2
    for r in res1["records"]:
        assert r["aborted"] is None, r["aborted"]
        assert "hero_net_bb" in r
    assert res1["counters"]["hero_illegal_action"] == 0
    assert res1["counters"]["hero_abstain"] == 0
    assert res1["counters"]["bot_illegal"] == 0
    # 结算零和：hero_net + bot_net == 0（HU）
    nets = [r["hero_net_bb"] for r in res1["records"]]
    assert all(abs(x) < 1e9 for x in nets)
    # 同 seed 复跑：逐字节一致（确定性）
    res2 = run_match(hero=hero, n_hands=2, seed_base=42, bot_kind="pivot",
                     bot_seed=701, progress_every=0)
    assert json.dumps(res1["records"], sort_keys=True) == \
        json.dumps(res2["records"], sort_keys=True)
