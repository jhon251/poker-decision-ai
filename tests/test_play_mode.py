"""EXE-1 — play 对打模式测试（引擎复用/落盘/便携频率索引/CLI 注册）。

纪律：不跑真实 judge 链（重）；advisor 关闭路径 + 便携索引装载路径 +
play_history 契约全部单测覆盖。真实 judge 链的端到端验证在 EXE 干净环境
验收脚本（scripts/verify_product_clean_env.py）完成。
"""
from __future__ import annotations

import itertools
import json

import pytest

from poker_decision_ai.play.history import (
    PLAY_HISTORY_SCHEMA, PlayHistoryWriter, load_session, session_metrics)
from poker_decision_ai.play.match import (
    HumanAction, LINEUP_PRESETS, run_play_session)


def _scripted(*actions):
    seq = itertools.cycle(actions)
    return lambda ctx: next(seq)


# ══════════════════════ play_history 契约 ══════════════════════
def test_play_history_roundtrip(tmp_path):
    h = PlayHistoryWriter(tmp_path, "s1", stake="nl2", label="t",
                          seed_base=7, lineup=[{"seat": 1, "archetype": "x"}],
                          deterministic=True)
    h.write_hand(hand_index=0, deal_seed=99, button=0, hero_hole=["As", "Kd"],
                 board=["2c", "7d", "Th"], street_reached="flop",
                 n_players_dealt=6,
                 hero_decisions=[{"street": "preflop", "action": "call",
                                  "advisor": {"available": True,
                                              "action": "call",
                                              "provenance": {"source": "preflop_chart"}}}],
                 net_bb=1.5, net_bb_raked=1.4, net_bb_raked_full=1.2,
                 rake_chips=10.0, pot_chips=300.0,
                 terminal_reason="showdown", aborted=None, hand_wall_ms=1234.5)
    h.write_session_tail({"summary": {"n_hands": 1}})
    data = load_session(h.path)
    assert data["head"]["schema"] == PLAY_HISTORY_SCHEMA
    assert data["head"]["deterministic_seed"] is True
    assert data["head"]["lineup"][0]["seat"] == 1
    assert len(data["hands"]) == 1
    hand = data["hands"][0]
    assert hand["hero_hole"] == ["As", "Kd"]
    assert hand["net_bb"] == 1.5 and hand["aborted"] is None
    assert hand["hero_decisions"][0]["advisor"]["available"] is True
    assert data["tail"]["summary"]["summary"]["n_hands"] == 1
    m = session_metrics(h.path)
    assert m["n_hands"] == 1 and m["hero_decisions"] == 1
    assert m["advisor_agreement"] == 1.0
    assert m["advisor_solver_anchored_share"] == 1.0


def test_play_history_schema_mismatch(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"schema": "other-v9", "record": "hand"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_session(p)


def test_play_history_aborted_hand_recorded(tmp_path):
    h = PlayHistoryWriter(tmp_path, "s2", stake="nl2", label="t", seed_base=1,
                          lineup=[], deterministic=False)
    h.write_hand(hand_index=0, deal_seed=1, button=0, hero_hole=["Ah", "Ad"],
                 board=[], street_reached="preflop", n_players_dealt=6,
                 hero_decisions=[], net_bb=None, net_bb_raked=None,
                 net_bb_raked_full=None, rake_chips=0.0, pot_chips=30.0,
                 terminal_reason=None, aborted="BOT_ILLEGAL:x",
                 hand_wall_ms=1.0)
    m = session_metrics(h.path)
    assert m["n_hands"] == 1 and m["n_hands_completed"] == 0


# ══════════════════════ 对打回路（advisor 关闭） ══════════════════════
def test_play_session_scripted_fold(tmp_path):
    events = []
    h = PlayHistoryWriter(tmp_path, "s3", stake="nl2", label="t", seed_base=42,
                          lineup=[], deterministic=True)
    summary = run_play_session(
        stake="nl2", n_hands=3, seed_base=42,
        get_human_action=_scripted(HumanAction("fold")),
        on_event=events.append, judge=None, rake=None, history=h,
        session_id="s3")
    types = {e["type"] for e in events}
    assert {"session_start", "hand_start", "street", "hero_turn",
            "decision_result", "hand_end", "session_end"} <= types
    assert summary["n_hands"] == 3 and summary["n_completed"] >= 1
    assert summary["counters"]["hero_decisions"] >= 3
    # 每手都落盘
    data = load_session(h.path)
    assert len(data["hands"]) == 3
    # 确定性：同种子两次会话 deal_seed 一致
    events2 = []
    run_play_session(stake="nl2", n_hands=3, seed_base=42,
                     get_human_action=_scripted(HumanAction("fold")),
                     on_event=events2.append, judge=None, rake=None,
                     history=None, session_id="s4")
    seeds1 = [e["hand_index"] for e in events if e["type"] == "hand_end"]
    assert seeds1 == [0, 1, 2]
    hs1 = [e for e in events if e["type"] == "hand_start"]
    hs2 = [e for e in events2 if e["type"] == "hand_start"]
    assert [e["hero_hole"] for e in hs1] == [e["hero_hole"] for e in hs2]


def test_play_session_lineup_declared():
    events = []
    run_play_session(stake="nl5", n_hands=1, seed_base=5,
                     get_human_action=_scripted(HumanAction("fold")),
                     on_event=events.append, judge=None, rake=None,
                     history=None, session_id="s5")
    ss = next(e for e in events if e["type"] == "session_start")
    assert ss["stake"] == "nl5"
    assert len(ss["lineup"]) == 5
    archs = {s["archetype"] for s in ss["lineup"]}
    assert archs <= {"fish", "recreational", "semi_reg", "reg"}
    for s in ss["lineup"]:
        assert s["bot_kind"] == "pool_bot" and s["bot_seed"] >= 0


def test_lineup_presets_declared():
    # v2 扩展：nl2/nl5 之外新增 nl10/nl25/nl50（画像构成逐级收紧）
    assert set(LINEUP_PRESETS) == {"nl2", "nl5", "nl10", "nl25", "nl50"}
    for stake, preset in LINEUP_PRESETS.items():
        assert len(preset["archetypes"]) == 5, stake
        assert all(1 <= seat <= 5 for seat in preset["archetypes"]), stake


# ══════════════════════ 用户动作校验（非法重试，绝不静默改判） ══════
def test_validate_human_actions():
    from external_harness.engine import fresh_nseat_state
    from external_harness.protocol.cash_match_6max import stake_rules
    from poker_decision_ai.play.match import _validate_human
    st = fresh_nseat_state(seed=1, n_seats=6, starting_stack=2000, button=0,
                           rules=stake_rules("nl2"))
    legal = st.legal_actions()
    assert "fold" in legal and "raise" in legal
    assert _validate_human(st, HumanAction("fold"))["ok"] is True
    v = _validate_human(st, HumanAction("raise", 60))
    assert v["ok"] and v["amount_chips"] == 60
    assert _validate_human(st, HumanAction("raise", 5))["ok"] is False
    assert _validate_human(st, HumanAction("bet", 60))["ok"] is False  # 面对加注无 bet
    assert _validate_human(st, HumanAction("spin"))["ok"] is False


# ══════════════════════ advisor 关闭路径（诚实 unavailable） ═════════
def test_advisor_disabled_honest():
    from poker_decision_ai.play.match import Advisor
    a = Advisor(None)
    a.start(None, hand_index=0, decision_index=0)
    res = a.collect()
    assert res["available"] is False and res["reason"] == "advisor_disabled"


def test_decision_record_carries_advisor_fields(tmp_path):
    """advisor 关闭时 hero_decisions.advisor.available=False 如实落盘。"""
    events = []
    h = PlayHistoryWriter(tmp_path, "s6", stake="nl2", label="t", seed_base=9,
                          lineup=[], deterministic=True)
    run_play_session(stake="nl2", n_hands=1, seed_base=9,
                     get_human_action=_scripted(HumanAction("fold")),
                     on_event=events.append, judge=None, rake=None,
                     history=h, session_id="s6")
    data = load_session(h.path)
    decs = data["hands"][0]["hero_decisions"]
    assert decs, "hero must have at least one decision"
    assert decs[0]["advisor"]["available"] is False
    assert decs[0]["advisor_agree"] is None


# ══════════════════ 便携频率索引（EXE 产品模式） ══════════════════
def test_frequency_prior_portable_fallback(tmp_path, monkeypatch):
    """源语料缺失 + 便携索引在场 → 直接装载 population 桶（provenance 声明）。"""
    from poker_decision_ai.agents.frequency_prior.agent import (
        INDEX_SCHEMA_VERSION, _load_index)
    bucket = {"n": 42, "ts_max": 1.7e9}
    portable = {"schema": INDEX_SCHEMA_VERSION, "level": "NL10:cash:6max",
                "fingerprint": "portable-test",
                "index": {"NL10:cash:6max": {
                    json.dumps(["population", "BU", "flop", False, "check"],
                               ensure_ascii=False): bucket}}}
    (tmp_path / "frequency_prior_portable.json").write_text(
        json.dumps(portable), encoding="utf-8")
    warnings: list = []
    idx, prov = _load_index(
        {"source_dir": str(tmp_path / "no_such_source"), "max_chunks": 5,
         "cache_path": str(tmp_path / "cache.json"),
         "level": "NL10:cash:6max"}, type("C", (), {"config": None})(),
        warnings)
    inner = idx["NL10:cash:6max"]
    assert inner[("population", "BU", "flop", False, "check")] == bucket
    assert prov["portable"] is True
    assert "population" in prov["scope"]


def test_frequency_prior_missing_everything_degrades(tmp_path):
    """源缺失且无便携索引 → 空索引 + SOURCE_UNAVAILABLE（诚实降级不变）。"""
    from poker_decision_ai.agents.frequency_prior.agent import _load_index
    warnings: list = []
    idx, prov = _load_index(
        {"source_dir": str(tmp_path / "none"), "max_chunks": 5,
         "cache_path": str(tmp_path / "c.json"),
         "level": "NL10:cash:6max"},
        type("C", (), {"config": None})(), warnings)
    assert idx == {} and prov == {}
    assert any(w["code"] == "SOURCE_UNAVAILABLE" for w in warnings)


# ══════════════════════ CLI 注册 ══════════════════════
def test_cli_play_subcommand_registered(capsys):
    from poker_decision_ai.cli import main
    with pytest.raises(SystemExit) as ei:
        main(["play", "--help"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "--stake" in out and "--seed" in out and "--no-advisor" in out


# ══════════════════════ 红线：无捕捉/自动操作路径 ══════════════════════
def test_play_module_has_no_capture_or_automation():
    import pathlib
    play_dir = pathlib.Path(__file__).resolve().parent.parent / "poker_decision_ai" / "play"
    banned = ("win32gui", "win32ui", "ctypes.windll", "BitBlt",
              "pyautogui", "mouse_event", "keybd_event", "SendInput",
              "capture_frame", "locate_coinpoker")
    for f in play_dir.glob("*.py"):
        text = f.read_text(encoding="utf-8")
        for b in banned:
            assert b not in text, f"banned symbol {b!r} in {f}"
