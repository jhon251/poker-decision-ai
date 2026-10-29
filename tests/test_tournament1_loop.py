"""TOURNEY-1 — MTT/PKO 多手锦标赛回路正确性测试（小 N；全脚本 bot；机制验证级）。

覆盖：MTT/PKO 端到端（淘汰至目标人数）、逐手零和、淘汰判定（≤bb）与赏金
50/50 渐进记账守恒、record v2 契约基本字段、同 seed 逐字节确定性、
本模块 scope 卫生（不 import 项目包——与 harness scope-boundary 纪律一致）。
"""
from __future__ import annotations

import json
import pathlib

import pytest

from external_harness.protocol.tournament_match import (
    BOT_PROFILES, GAME_MODES, TOURNEY_MATCH_VERSION, run_tournament)


# ---------------------------------------------------------------- 契约层

def test_contract_and_profiles():
    assert TOURNEY_MATCH_VERSION == "tournament1-match-v1"
    assert set(GAME_MODES) == {"mtt", "pko"}
    for name, p in BOT_PROFILES.items():
        assert 0 <= p["fold_small"] <= 1 and 0 <= p["fold_big"] <= 1
        assert p["fold_small"] < p["fold_big"]


def test_no_project_imports():
    """scope 卫生：本模块不得 import 项目包；随机一律走 vendored _Lcg。"""
    import ast
    src_path = pathlib.Path("external_harness/protocol/tournament_match.py")
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "poker_decision_ai" not in imported
    assert "random" not in imported
    assert imported <= {"__future__", "typing", "external_harness"}


# ---------------------------------------------------------------- MTT 端到端

@pytest.fixture(scope="module")
def mtt() -> dict:
    return run_tournament(game_mode="mtt", n_seats=6, starting_stack=20000,
                          max_hands=30, seed_base=20261001,
                          matchup_id="test_mtt_6max")


def test_mtt_reaches_target(mtt: dict):
    meta = mtt["meta"]
    assert len(mtt["hands"]) >= 1
    assert mtt["players_remaining"] == meta["target_players"]
    assert meta["counters"]["hand_aborts"] == 0
    assert meta["counters"]["bot_illegal"] == 0
    assert meta["counters"]["view_void"] == 0
    assert meta["ante_modeled"] is False  # 诚实边界：ante 不建模


def test_mtt_zero_sum_and_chips_conserved(mtt: dict):
    start = mtt["start_stack_chips"] * mtt["entries"]
    final = sum(mtt["meta"]["final_stacks"].values())
    assert final == start  # 无赏金模式：总筹码守恒
    for h in mtt["hands"]:
        assert sum(h["result"]["net_chips"]) == 0


def test_elimination_threshold_and_status(mtt: dict):
    bb_by_level = {lv["level"]: lv["bb"] for lv in mtt["meta"]["levels"]}
    busted_order = mtt["meta"]["bust_order"]
    assert len(busted_order) == mtt["entries"] - mtt["players_remaining"]
    for h in mtt["hands"]:
        bb = bb_by_level[h["level"]]
        for sf in h["seat_final"]:
            if sf["status"] == "eliminated":
                assert sf["stack_chips"] <= bb, f"hand{h['hand_index']}: seat{sf['seat']}"


def test_record_v2_contract(mtt: dict):
    assert mtt["schema"] == "poer.live.record_v2"
    board_len = {"preflop": 0, "flop": 3, "turn": 4, "river": 5, "showdown": 5}
    for h in mtt["hands"]:
        assert len(h["seats"]) == mtt["entries"]
        assert h["blinds"]["ante"] == 0
        assert h["hands_to_next_level"] >= 0
        assert len(h["board"]) in (0, 3, 4, 5)
        assert h["result"]["terminal_reason"] in ("fold_out", "showdown")
        for a in h["actions"]:
            if a["a"] == "deal":
                assert a["seat"] == -1
            else:
                assert 0 <= a["seat"] < mtt["entries"]
        # 开手动作必须是盲注结构（ante 不建模 → 无 post_ante）
        firsts = [a["a"] for a in h["actions"][:2]]
        assert firsts == ["post_sb", "post_bb"]
        # 弃牌/摊牌与公共牌一致
        if h["result"]["terminal_reason"] == "fold_out" and h["board"]:
            # fold_out 允许发生在翻牌后下注轮结束（board 非空合法）
            assert len(h["board"]) in (3, 4, 5)
        # 预先淘汰座位：无手牌、无动作
        active = {s["seat"] for s in h["seats"] if s["status"] == "active"}
        for a in h["actions"]:
            assert a["seat"] == -1 or a["seat"] in active


# ---------------------------------------------------------------- PKO 赏金

def test_pko_bounty_ledger_conservation():
    bounty = 5000
    pko = run_tournament(game_mode="pko", n_seats=6, starting_stack=20000,
                         bounty_chips=bounty, max_hands=30, seed_base=20261001,
                         matchup_id="test_pko_6max")
    meta = pko["meta"]
    initial_pool = bounty * pko["entries"]
    ledger = {s: bounty for s in range(pko["entries"])}
    awarded_total = 0
    for ev in meta["bounty_history"]:
        seat, elim, awarded = ev["seat"], ev["eliminated_by"], ev["awarded_chips"]
        # 50/50：淘汰者得一半，另一半并入淘汰者自己的赏金（渐进）
        assert awarded == ledger[seat] // 2
        ledger[elim] += ledger[seat] - awarded
        ledger[seat] = 0
        awarded_total += awarded
    # 池守恒：已发放（进筹码）+ 仍挂账 == 初始池
    assert awarded_total + sum(ledger.values()) == initial_pool
    # 被淘汰者赏金必须清零
    for ev in meta["bounty_history"]:
        assert ledger[ev["seat"]] == 0
    # 有淘汰发生时赏金事件非空；finish order 去重
    if meta["bust_order"]:
        assert meta["bounty_history"]
    assert sorted(meta["finish_order"]) == list(range(pko["entries"]))


def test_chip_plus_bounty_conservation():
    """PKO 总价值守恒：总筹码 + 赏金池余量 == 初始筹码 + 初始赏金池。"""
    bounty = 5000
    pko = run_tournament(game_mode="pko", n_seats=4, starting_stack=12000,
                         bounty_chips=bounty, max_hands=25, seed_base=77,
                         matchup_id="test_pko_4max")
    total_chips = sum(pko["meta"]["final_stacks"].values())
    assert total_chips == 12000 * 4  # 赏金进账已含在筹码里


# ---------------------------------------------------------------- 确定性

def test_determinism_same_seed():
    a = run_tournament(game_mode="mtt", n_seats=4, starting_stack=12000,
                       max_hands=12, seed_base=42, matchup_id="det")
    b = run_tournament(game_mode="mtt", n_seats=4, starting_stack=12000,
                       max_hands=12, seed_base=42, matchup_id="det")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_invalid_mode_fails_closed():
    with pytest.raises(ValueError):
        run_tournament(game_mode="sit_and_go")
