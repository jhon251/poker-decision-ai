"""M7 / B7-4：mode-dispatch 全图（CASH / MTT / PKO / Mystery）+ 模式互不串线。

覆盖：
  * resolve_mode 全图：cash|mtt|pko|ko|mystery 各自解析；缺失/未知 → unknown
    （UNKNOWN-first，绝不默认成 cash）；
  * capability 矩阵：CASH 禁 ICM/bounty，MTT 允许 ICM 禁 bounty，
    PKO/KO/Mystery 允许 ICM+bounty；
  * sanitize 构造性隔离：CASH 剥离 tournament/bounty，MTT 剥离 bounty；
    缺失的 mode 内上下文**原样保留**（诚实 unavailable，不借用其他 mode 补全）；
  * verify_mode_isolation：对 intel 结果逐 mode 审计，串线即列出违规；
  * 串线负向：把某 mode 的 state 拿去当另一 mode 用，必须被剥离或列为违规。

不修改 strategy_core / CFI / λ / 阈值；本文件只做隔离与确定性断言。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai.strategy_mode_switch import (
    MODE_CAPABILITIES, SEMANTIC_MODES, mode_capability_report, resolve_mode,
    sanitize_state_for_mode, verify_mode_isolation)

_TOURNAMENT = {"payouts": [{"place": 1, "amount": 100.0}],
               "field_size": 100, "players_left": 20}
_BOUNTY_PLAYERS = [{"player_id": "a", "stack": 100.0, "bounty": 5.0},
                   {"player_id": "b", "stack": 100.0}]


def _state(mode: str) -> dict:
    st = {"game_type": mode, "hero": {"stack": 100.0}, "pot": 10.0,
          "players": copy.deepcopy(_BOUNTY_PLAYERS),
          "tournament": copy.deepcopy(_TOURNAMENT),
          "bounty_type": "pko", "bounties": {"a": 5.0}}
    return st


# ── resolve_mode：全图 + UNKNOWN-first ─────────────────────────────────
def test_resolve_mode_covers_every_semantic_mode():
    for mode in SEMANTIC_MODES:
        got, reason = resolve_mode({"game_type": mode})
        assert got == mode and reason == "", mode
    # 大小写/空白归一化（不得因格式差异落到 unknown 或错误模式）
    assert resolve_mode({"game_type": "  CASH "})[0] == "cash"


def test_resolve_mode_missing_or_unknown_never_defaults_to_cash():
    for st, expect in (({}, "game_type missing"),
                       ({"game_type": None}, "game_type missing"),
                       ({"game_type": ""}, "game_type missing"),
                       ({"game_type": "spin"}, "unknown game_type")):
        mode, reason = resolve_mode(st)
        assert mode == "unknown", st
        assert reason.startswith(expect), (st, reason)
        assert mode != "cash"


def test_resolve_mode_is_deterministic_and_pure():
    st = _state("mtt")
    snap = copy.deepcopy(st)
    assert [resolve_mode(st) for _ in range(5)] == [("mtt", "")] * 5
    assert st == snap, "resolve_mode 不得修改 state"


# ── capability 矩阵（构造性隔离契约）───────────────────────────────────
def test_capability_matrix_shape_unchanged():
    assert set(MODE_CAPABILITIES) == set(SEMANTIC_MODES)
    for mode, caps in MODE_CAPABILITIES.items():
        assert set(caps) == {"icm", "payouts", "bounty"}, mode


def test_cash_forbids_icm_and_bounty():
    r = mode_capability_report(_state("cash"), "cash")
    assert r["status"] == "resolved"
    assert r["capabilities"]["icm"] == {"permission": "forbidden",
                                       "status": "forbidden"}
    assert r["capabilities"]["bounty"]["permission"] == "forbidden"
    assert r["capabilities"]["payouts"]["permission"] == "forbidden"


def test_mtt_permits_icm_but_forbids_bounty():
    r = mode_capability_report(_state("mtt"), "mtt")
    assert r["capabilities"]["icm"]["permission"] == "permitted"
    assert r["capabilities"]["bounty"]["permission"] == "forbidden"
    assert r["capabilities"]["bounty"]["status"] == "forbidden"


def test_pko_and_mystery_permit_icm_and_bounty():
    for mode in ("pko", "ko", "mystery"):
        r = mode_capability_report(_state(mode), mode)
        assert r["capabilities"]["icm"]["permission"] == "permitted", mode
        assert r["capabilities"]["bounty"]["permission"] == "permitted", mode
        assert r["capabilities"]["bounty"]["status"] == "enabled", mode


def test_unknown_mode_capability_report_fails_closed():
    r = mode_capability_report(_state("cash"), "spin")
    assert r["status"] == "unknown" and r["capabilities"] == {}
    assert "unknown mode" in r["unknown_reason"]


# ── sanitize：构造性隔离（剥离，而非借用）─────────────────────────────
def test_cash_strips_tournament_and_bounty_context():
    st = _state("cash")
    snap = copy.deepcopy(st)
    clean, stripped = sanitize_state_for_mode(st, "cash")
    for key in ("tournament", "bounty_type", "bounties"):
        assert key not in clean, key
    assert all("bounty" not in p for p in clean["players"])
    assert set(stripped) == {"state.tournament", "state.bounty_type",
                             "state.bounties", "players.bounty"}
    assert st == snap, "sanitize 不得修改输入 state"


def test_mtt_strips_bounty_type_and_player_bounty():
    """MTT：全部 bounty 载体（bounty_type / 顶层 bounties / 玩家级 bounty）
    必须剥离；tournament 保留。"""
    clean, stripped = sanitize_state_for_mode(_state("mtt"), "mtt")
    assert "tournament" in clean                      # MTT 合法上下文保留
    assert "bounty_type" not in clean
    assert "bounties" not in clean
    assert all("bounty" not in p for p in clean["players"])
    assert set(stripped) == {"state.bounty_type", "state.bounties",
                             "players.bounty"}


def test_mtt_must_also_strip_top_level_bounties_table():
    """M8 闭环：MTT 顶层 `bounties`（PKO 语义）必须被剥离，不得串入。"""
    clean, stripped = sanitize_state_for_mode(_state("mtt"), "mtt")
    assert "bounties" not in clean
    assert "bounty_type" not in clean
    assert all("bounty" not in p for p in clean["players"])
    assert set(stripped) == {"state.bounty_type", "state.bounties",
                             "players.bounty"}


def test_pko_keeps_all_tournament_and_bounty_context():
    clean, stripped = sanitize_state_for_mode(_state("pko"), "pko")
    assert stripped == []
    assert clean["bounty_type"] == "pko" and clean["bounties"] == {"a": 5.0}
    assert clean["players"][0]["bounty"] == 5.0


def test_missing_mode_context_is_preserved_not_borrowed():
    """MTT/PKO 缺失 payouts/bounty → 原样保留缺失（不借用别处补全）。"""
    st = {"game_type": "mtt", "players": [{"player_id": "a", "stack": 10.0}]}
    clean, stripped = sanitize_state_for_mode(st, "mtt")
    assert stripped == [] and clean == st
    r = mode_capability_report(clean, "mtt")
    assert r["capabilities"]["icm"]["status"] == "unavailable:missing_payouts"
    assert r["capabilities"]["bounty"]["permission"] == "forbidden"


def test_cash_report_is_unavailable_not_enabled_without_context():
    """CASH 无 tournament/bounty 上下文：forbidden ≠ "缺失"；不得出现 enabled。"""
    r = mode_capability_report({"game_type": "cash", "players": []}, "cash")
    assert all(v["status"] == "forbidden" for v in r["capabilities"].values())


# ── 串线（cross-mode contamination）负向 ───────────────────────────────
def test_pko_state_sanitized_as_cash_loses_bounty_ability():
    """把 PKO state 当 CASH 跑：bounty 必须被剥离，且能力报不可用（不串线）。"""
    clean, _ = sanitize_state_for_mode(_state("pko"), "cash")
    r = mode_capability_report(clean, "cash")
    assert r["capabilities"]["bounty"]["status"] == "forbidden"


def test_mtt_sanitized_state_cannot_reuse_pko_bounty():
    """MTT 剥离 bounty 后，即使原 state 有 bounty，也不得再算出 bounty 能力。"""
    clean, _ = sanitize_state_for_mode(_state("pko"), "mtt")
    assert mode_capability_report(clean, "mtt")["capabilities"]["bounty"][
        "status"] == "forbidden"
    assert all("bounty" not in p for p in clean["players"])


def test_verify_mode_isolation_flags_cash_icm_activation():
    bad = {"available": True, "objective": "chip_ev+icm",
           "provenance": {"icm_available": True}, "candidates": []}
    v = verify_mode_isolation(bad, "cash")
    assert "cash_icm_activated" in v
    assert any(x.startswith("cash_objective:") for x in v)


def test_verify_mode_isolation_flags_mtt_bounty_leak():
    bad = {"available": True, "objective": "chip_ev",
           "provenance": {}, "candidates": [
               {"action": "call", "adjustments": {"bounty_util_bb": 1.5}}]}
    assert verify_mode_isolation(bad, "mtt") == ["mtt_bounty_util:call"]
    # CASH 同样不得出现 bounty 项
    assert verify_mode_isolation(bad, "cash") == ["mtt_bounty_util:call"]


def test_verify_mode_isolation_clean_results_have_no_violations():
    ok = {"available": True, "objective": "chip_ev", "provenance": {},
          "candidates": [{"action": "call", "adjustments": {}}]}
    assert verify_mode_isolation(ok, "cash") == []
    assert verify_mode_isolation(ok, "mtt") == []
    assert verify_mode_isolation(ok, "pko") == []


def test_verify_mode_isolation_unknown_mode_is_a_violation():
    assert verify_mode_isolation({"available": True}, "spin") == [
        "unknown mode 'spin'"]


def test_unavailable_result_is_not_audited_as_leak():
    """不可用结果不产生违规（但也不得被当成"通过"来掩盖缺失）。"""
    assert verify_mode_isolation({"available": False}, "cash") == []
    assert verify_mode_isolation({"available": False}, "mtt") == []


def test_mode_dispatch_graph_is_mutually_exclusive():
    """全图终检：任一 state 的 mode 解析唯一；每 mode 只匹配自己的契约。"""
    seen = {}
    for mode in SEMANTIC_MODES:
        got, _ = resolve_mode(_state(mode))
        assert got == mode
        seen[got] = mode
        clean, _ = sanitize_state_for_mode(_state(mode), mode)
        assert verify_mode_isolation(
            {"available": True, "objective": "chip_ev", "provenance": {},
             "candidates": []}, mode) == []
        assert mode_capability_report(clean, mode)["mode"] == mode
    assert sorted(seen) == sorted(SEMANTIC_MODES)
