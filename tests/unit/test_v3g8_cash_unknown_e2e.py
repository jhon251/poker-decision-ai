"""V3G-8 — CASH UNKNOWN End-to-End 测试（授权矩阵 1–30）。

**不修改**任何既有测试定义；**不触碰**冻结文件。
"""
from __future__ import annotations

import ast
import copy
import json
import uuid
from pathlib import Path

import pytest

from poker_decision_ai.cash_unknown_e2e import (
    EMPTY, PRESENT, REQUIRED_REPORT_KEYS, UNKNOWN, UNKNOWN_E2E_VERSION,
    build_unknown_report)

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"


def st(**over) -> dict:
    b = {
        "game_type": "cash", "street": "flop", "board": ["Ah", "7c", "2c"],
        "pot": 20.0, "to_call": 5.0, "bb": 1.0, "sb": 0.5,
        "min_raise_to": 10.0, "hero_id": "hero", "button_seat": 2,
        "table_size": 3, "synthetic_test_only": True,
        "players": [
            {"player_id": "v1", "seat": 1, "position": "BB", "status": "active",
             "stack": 100.0, "hole_cards": []},
            {"player_id": "hero", "seat": 2, "position": "BTN", "is_hero": True,
             "status": "active", "hole_cards": ["As", "Kd"], "stack": 100.0},
            {"player_id": "v2", "seat": 3, "position": "SB", "status": "active",
             "stack": 100.0, "hole_cards": []},
        ],
        "action_history": [{"street": "flop", "player_id": "v1",
                            "action": "bet", "amount": 5.0}],
    }
    b.update(over)
    return b


def _cls(r, name):
    return r["fields"][name]["classification"]


def _hero_mut(**changes) -> dict:
    s = st()
    s["players"][1].update(changes)
    return s


# ── 1. 完整 CASH 输入 ─────────────────────────────────────────────────
def test_01_complete_cash_input():
    r = build_unknown_report(st())
    assert r["present"] is True and r["complete"] is True
    assert r["complete_with_context"] is True
    assert r["reason"] == ""
    assert r["mode"] == "cash"
    for k in ("table_size", "button_seat", "hero.seat", "hero.hole_cards",
              "hero.stack", "hero.position", "pot", "to_call", "bb", "board",
              "street", "action_history", "current_actor", "min_raise_to"):
        assert _cls(r, k) == PRESENT, k


# ── 2–8. state integrity 缺失 ─────────────────────────────────────────
def test_02_missing_table_size_is_unknown_not_defaulted_six():
    r = build_unknown_report(st(table_size=None))
    assert _cls(r, "table_size") == UNKNOWN
    assert r["fields"]["table_size"]["value"] is None      # 不是 6
    assert r["fields"]["table_size"]["defaulted"] is False
    assert r["complete"] is True                            # 核心齐备
    assert r["complete_with_context"] is False
    assert "table_size" in r["reason"]


def test_03_missing_button_seat_is_unknown_not_defaulted():
    r = build_unknown_report(st(button_seat=None))
    assert _cls(r, "button_seat") == UNKNOWN
    assert r["fields"]["button_seat"]["value"] is None
    assert r["complete_with_context"] is False


def test_04_missing_hero_seat_is_unknown():
    r = build_unknown_report(_hero_mut(seat=None))
    assert _cls(r, "hero.seat") == UNKNOWN


def test_05_missing_hero_hole_cards_is_incomplete():
    r = build_unknown_report(_hero_mut(hole_cards=[]))
    assert _cls(r, "hero.hole_cards") == UNKNOWN
    assert r["complete"] is False
    assert "hero.hole_cards" in r["reason"]


def test_05_null_card_element_is_unknown():
    r = build_unknown_report(_hero_mut(hole_cards=["As", None]))
    assert _cls(r, "hero.hole_cards") == UNKNOWN


def test_06_missing_stack_is_incomplete_not_defaulted():
    r = build_unknown_report(_hero_mut(stack=None))
    assert _cls(r, "hero.stack") == UNKNOWN
    assert r["fields"]["hero.stack"]["value"] is None
    assert r["complete"] is False


def test_07_missing_pot_is_reported_by_integrity():
    s = st()
    s.pop("pot", None)
    r = build_unknown_report(s)
    assert _cls(r, "pot") == UNKNOWN
    assert "pot" in r["incomplete"]
    assert r["complete"] is False


def test_08_missing_to_call_is_reported_by_integrity():
    s = st()
    s.pop("to_call", None)
    r = build_unknown_report(s)
    assert _cls(r, "to_call") == UNKNOWN
    assert "to_call" in r["incomplete"]


# ── 9/10/11. board 缺失 / 冲突 / duplicate card ───────────────────────
def test_09_missing_board_on_postflop_is_unknown_not_invented():
    s = st()
    s.pop("board", None)
    r = build_unknown_report(s)
    assert _cls(r, "board") == UNKNOWN
    assert r["fields"]["board"]["value"] is None           # 不是 []


def test_09_preflop_empty_board_is_empty_not_unknown():
    """preflop 的空 board 是**证据确认的空**（EMPTY），与 UNKNOWN 区分。"""
    r = build_unknown_report(st(street="preflop", board=[],
                                action_history=[]))
    assert _cls(r, "board") == EMPTY
    assert r["evidence"]["empty_fields"] == ["board"]


def test_10_board_street_conflict_is_surfaced():
    r = build_unknown_report(st(street="preflop", board=["Ah", "7c", "2c"]))
    assert any(c.startswith("board_street_mismatch") for c in r["conflicts"])
    assert r["complete"] is False


def test_11_duplicate_card_is_surfaced_as_conflict():
    s = st()
    s["players"][1]["hole_cards"] = ["Ah", "Kd"]           # Ah 同时在 board
    r = build_unknown_report(s)
    assert any(c.startswith("duplicate_card") for c in r["conflicts"])


def test_11_duplicate_seat_is_surfaced():
    s = st()
    s["players"][0]["seat"] = 2
    s["players"][2]["seat"] = 2
    r = build_unknown_report(s)
    assert "duplicate_seat" in r["conflicts"]


# ── 12/13/14. street / action history / current actor ─────────────────
def test_12_unknown_street_is_unknown_not_preflop():
    for bad in (None, "", "mystery_street"):
        s = st()
        s["street"] = bad
        r = build_unknown_report(s)
        assert _cls(r, "street") == UNKNOWN, bad
        assert r["fields"]["street"]["value"] is None
        assert r["complete"] is False


def test_13_missing_action_history_is_unknown_not_empty_list():
    s = st()
    s.pop("action_history", None)
    r = build_unknown_report(s)
    assert _cls(r, "action_history") == UNKNOWN
    assert r["fields"]["action_history"]["value"] is None  # 不是 []


def test_13_empty_action_history_is_present_empty_list():
    r = build_unknown_report(st(action_history=[]))
    # 显式空列表 = 有证据（无历史），据此不产生 actor
    assert _cls(r, "action_history") == PRESENT
    assert _cls(r, "current_actor") == UNKNOWN


def test_14_current_actor_uses_same_rule_and_is_unknown_when_unreliable():
    ok = build_unknown_report(st())
    assert _cls(ok, "current_actor") == PRESENT
    assert ok["fields"]["current_actor"]["value"] == "hero"
    off_street = build_unknown_report(st(action_history=[
        {"street": "turn", "player_id": "v1", "action": "check"}]))
    assert _cls(off_street, "current_actor") == UNKNOWN
    no_seats = build_unknown_report(st(
        players=[{k: v for k, v in p.items() if k != "seat"}
                 for p in st()["players"]]))
    assert _cls(no_seats, "current_actor") == UNKNOWN


# ── 15/16/17. opponent model / range / provenance ─────────────────────
def test_15_unknown_opponent_model_is_unknown_not_fabricated():
    r = build_unknown_report(st())
    assert _cls(r, "opponent_model") == UNKNOWN
    assert r["fields"]["opponent_model"]["value"] is None
    assert "不构造" in r["fields"]["opponent_model"]["evidence"]


def test_16_zero_sample_range_is_unknown_not_fabricated():
    r = build_unknown_report(st())
    assert _cls(r, "range") == UNKNOWN
    assert r["fields"]["range"]["value"] is None


def test_17_missing_range_provenance_is_unknown():
    r = build_unknown_report(st())
    assert _cls(r, "range_provenance") == UNKNOWN
    assert r["fields"]["range_provenance"]["value"] is None


# ── 18/19/20. min_raise_to / action tree / illegal sizing ────────────
def test_18_missing_min_raise_to_is_unknown():
    s = st()
    s.pop("min_raise_to", None)
    r = build_unknown_report(s)
    assert _cls(r, "min_raise_to") == UNKNOWN
    assert r["complete_with_context"] is False


def test_19_absent_layer_marked_not_supplied_not_fabricated():
    r = build_unknown_report(st())
    for k, v in r["layer_presence"].items():
        assert v["supplied"] is False and v["present"] is None
        assert v["reason"] == "not_supplied"


def test_19_supplied_but_not_present_layer_propagates_to_unknown():
    r = build_unknown_report(st(), layers={
        "action_tree": {"present": False, "reason": "unknown_street:None"},
        "range": {"present": True}})
    assert r["layer_presence"]["action_tree"]["supplied"] is True
    assert any(x == "layer:action_tree" for x in r["unknown"])
    assert r["layer_presence"]["range"]["present"] is True


def test_20_sizing_is_never_fabricated_by_this_layer():
    r = build_unknown_report(st())
    assert _cls(r, "sizing") == UNKNOWN
    assert r["fields"]["sizing"]["value"] is None


# ── 21/22. terminal / non-CASH ────────────────────────────────────────
def test_21_terminal_state_still_reports_unknowns_without_filling():
    s = st(street="river", board=["Ah", "7c", "2c", "2d", "3s"],
           to_call=0.0)
    r = build_unknown_report(s)
    assert r["present"] is True
    assert _cls(r, "board") == PRESENT
    assert _cls(r, "future_board") == UNKNOWN               # 河牌后无未来牌


@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin", None])
def test_22_non_cash_fails_closed(mode):
    r = build_unknown_report(st(game_type=mode))
    assert r["present"] is False
    assert "fields" in r and r["fields"] == {}


def test_22_non_dict_fails_closed():
    for bad in (None, [], "cash", 3):
        r = build_unknown_report(bad)
        assert r["present"] is False


def test_22_unknown_never_triggers_tournament_logic():
    """UNKNOWN **不得**自动触发 MTT/PKO/Mystery 逻辑：输出无相关字样。"""
    r = build_unknown_report(st(table_size=None, button_seat=None))
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("icm", "bounty", "satellite", "payout", "tournament"):
        assert banned not in blob, banned


# ── 23/24. 不生成 future/runout，不使用默认值 ─────────────────────────
def test_23_no_future_card_or_runout_is_generated():
    r = build_unknown_report(st())
    fb = r["fields"]["future_board"]
    assert fb["classification"] == UNKNOWN
    assert fb["value"] is None and fb["must_not_invent"] is True
    # 输出中不得出现 runout/未来牌**字段**（注意：保证位名称包含
    # "no_fabricated_future_cards"，属**声明**而非字段，故按字段名精确判定）
    assert "runout" not in r["fields"]
    assert "future_cards" not in r["fields"]
    assert "future_board" in r["fields"] and r["fields"]["future_board"]["value"] is None
    # 任何**牌面相关**字段的值都不得是"未来牌"形态
    for name in ("board", "hero.hole_cards"):
        v = r["fields"][name]["value"]
        if isinstance(v, list):
            assert all(isinstance(x, str) and len(x) == 2 for x in v), name


def test_24_no_field_is_ever_defaulted():
    cases = [st(), st(table_size=None), st(button_seat=None),
             _hero_mut(hole_cards=[]), _hero_mut(stack=None),
             st(action_history=[]), st(street="preflop", board=[])]
    for s in cases:
        r = build_unknown_report(s)
        assert r["guarantees"]["defaulted_fields"] == []
        for name, f in r["fields"].items():
            assert f["defaulted"] is False, name
            if f["classification"] == UNKNOWN:
                assert f["value"] is None, name
    assert build_unknown_report(st())["guarantees"]["no_default_fill"] is True


# ── 25. V3G-1～V3G-7 回归 ─────────────────────────────────────────────
def test_25_v3g1_to_v3g7_still_consistent():
    from poker_decision_ai.cash_action_tree import build_action_tree
    from poker_decision_ai.cash_multistreet import build_multistreet_context
    from poker_decision_ai.cash_opponent_model import build_opponent_model
    from poker_decision_ai.cash_preflop_decision import build_preflop_decision
    from poker_decision_ai.cash_range_model import build_range_model
    s = st(street="preflop", board=[], to_call=2.5, min_raise_to=5.0,
           action_history=[])
    pf = build_preflop_decision(s)
    ms = build_multistreet_context(s, preflop=pf)
    om = build_opponent_model(s, this_hand_actions=[
        {"player_id": "v1", "action": "raise"}])
    rm = build_range_model(s, opponent_model=om)
    at = build_action_tree(s)
    assert all(d["present"] for d in (pf, ms, om, rm, at))
    r = build_unknown_report(s, layers={"preflop": pf, "multistreet": ms,
                                        "opponent_model": om, "range": rm,
                                        "action_tree": at})
    assert r["present"] is True
    assert all(v["supplied"] and v["present"] is True
               for v in r["layer_presence"].values())


# ── 26/27/28. 真实 RealtimeRunner / persistence / 确定性 ──────────────
def _runtime():
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.registry import register_all_agents
    from poker_decision_ai.core.orchestration.runner import PipelineRunner
    from poker_decision_ai.core.orchestration.runtime import Runtime
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    cfg = load_config()
    # BENCH-1 Task0.3：冷启动（首次 import/equity 预热）+ 并行负载下默认
    # 3000ms 死线偶发超时（顺序依赖的脆弱失败）。测试内提高死线——被测
    # 语义（UNKNOWN 报告暴露）与死线无关，不得靠删测试修脆弱。
    cfg.root._data["realtime"] = {**cfg.realtime_config(),
                                  "deadline_ms": 60000,
                                  "agent_timeout_ms": 30000}
    logs = ROOT / ".tmp" / "v3g8_logs" / uuid.uuid4().hex
    logs.mkdir(parents=True, exist_ok=True)
    providers, repos = build_providers(cfg), build_in_memory()
    registry = register_all_agents()
    logger, metrics = EventLogger(str(logs)), MetricsCollector()
    runner = PipelineRunner(registry, cfg, logger, metrics)
    return Runtime(config=cfg, providers=providers, repos=repos,
                   registry=registry, runner=runner, logger=logger,
                   metrics=metrics)


def test_26_real_realtime_decide_exposes_unknown_report():
    """真实 `RealtimeRunner.decide()` 端到端：报告与持久化均携带
    UNKNOWN 报告，且 JSON-safe。"""
    import asyncio
    from tests.conftest import CASH_HAND_TEXT
    from poker_decision_ai.realtime import RealtimeRunner

    rt = _runtime()
    report = asyncio.run(RealtimeRunner(rt).decide(
        {"kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT,
         "hero_id": "Hero"}))
    ur = report.unknown_report
    assert ur is not None, "cash 路径必须产出 UNKNOWN 报告"
    assert ur["present"] is True
    for k in REQUIRED_REPORT_KEYS:
        assert k in ur, k
    assert ur["guarantees"]["defaulted_fields"] == []
    # 报告含 UNKNOWN 时必须是显式 unknown，而不是被填了默认值
    assert isinstance(ur["unknown"], list)
    json.dumps(report.to_dict())


def test_27_persistence_and_report_are_json_safe():
    import asyncio
    from tests.conftest import CASH_HAND_TEXT
    from poker_decision_ai.realtime import RealtimeRunner

    rt = _runtime()
    report = asyncio.run(RealtimeRunner(rt).decide(
        {"kind": "cash_hand", "hand_history_text": CASH_HAND_TEXT,
         "hero_id": "Hero"}))
    d = report.to_dict()
    assert "unknown_report" in d
    json.dumps(d)
    rec = rt.repos.decisions.get(report.request_id)
    assert "unknown_report" in rec
    json.dumps(rec)


def test_28_repeated_runs_are_deterministic():
    s = st(table_size=None)
    first = build_unknown_report(s)
    for _ in range(5):
        assert build_unknown_report(copy.deepcopy(s)) == first
    json.dumps(first)


def test_28_does_not_mutate_input():
    s = st()
    before = copy.deepcopy(s)
    build_unknown_report(s)
    assert s == before


# ── 29/30. UNKNOWN vs EMPTY vs fallback ──────────────────────────────
def test_29_unknown_and_empty_are_distinguished():
    pre = build_unknown_report(st(street="preflop", board=[],
                                  action_history=[]))
    post = build_unknown_report(st(street="flop", board=["Ah", "7c", "2c"]))
    # preflop 空 board = EMPTY（证据确认空）；缺 board 字段 = UNKNOWN
    assert _cls(pre, "board") == EMPTY
    s = st()
    s.pop("board", None)
    assert _cls(build_unknown_report(s), "board") == UNKNOWN
    # 显式空 history = PRESENT（有证据）；缺 history = UNKNOWN
    assert _cls(pre, "action_history") == PRESENT
    s2 = st()
    s2.pop("action_history", None)
    assert _cls(build_unknown_report(s2), "action_history") == UNKNOWN
    assert post["evidence"]["empty_fields"] == []


def test_30_unknown_is_not_silent_fallback():
    """UNKNOWN 不得静默 fallback 到旧默认值：报告只在"有证据"时给值。"""
    r = build_unknown_report(st(table_size=None, button_seat=None,
                                players=[{k: v for k, v in p.items()
                                          if k != "stack"}
                                         for p in st()["players"]]))
    for name in ("table_size", "button_seat", "hero.stack"):
        assert r["fields"][name]["classification"] == UNKNOWN
        assert r["fields"][name]["value"] is None
        assert r["fields"][name]["defaulted"] is False
    # 明确的可观察保证
    g = r["guarantees"]
    assert g["unknown_is_not_empty"] is True
    assert g["unknown_is_not_fallback"] is True
    assert g["no_empty_substitution"] is True


def test_30_integrity_conflicts_are_propagated_verbatim():
    """V3G-1 的既有冲突（含 to_call>hero_stack）原样转述，不修复。"""
    s = st(to_call=500.0, pot=100.0,
           players=[
               {"player_id": "v1", "seat": 1, "position": "BB",
                "status": "active", "stack": 200.0},
               {"player_id": "hero", "seat": 2, "position": "BTN",
                "is_hero": True, "status": "active", "hole_cards": ["As", "Kd"],
                "stack": 30.0},
               {"player_id": "v2", "seat": 3, "position": "SB",
                "status": "active", "stack": 200.0}])
    r = build_unknown_report(s)
    assert any(x.startswith("to_call_exceeds_hero_stack")
               for x in r["conflicts"])
    assert r["complete"] is False


# ── 只读 / 依赖 / random 审计 ─────────────────────────────────────────
def test_module_has_no_strategy_or_random_dependencies():
    src = (PKG / "cash_unknown_e2e.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("random", "research_unfreeze", "vision2", "judge",
                   "strategy_core", "multistreet", "action_tree"):
        assert banned not in joined, banned
    assert set(imported) <= {"__future__", "dataclasses", "typing",
                             "cash_integrity_context",
                             "cash_state_integrity"}, imported
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for banned in ("compute_ev", "evaluate_intel", "positions_for"):
        assert banned not in called, banned


def test_wiring_touches_only_declared_production_files():
    """接线范围审计：只有 `realtime.py`（声明过的旁路）引用本层。"""
    importers = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name == "cash_unknown_e2e.py":
            continue
        src = p.read_text(encoding="utf-8-sig", errors="ignore")
        if "cash_unknown_e2e" in src:
            importers.append(str(p.relative_to(ROOT)))
    # BENCH-1 Task0.1（owner 授权基线迁移）：realtime.py 为 V3G-8 已核准旁路；
    # cash_strategy_consumption_gate.py 为 2026-09 中旬授权阶段新增引用方。
    sanctioned_importers = {
        "poker_decision_ai/cash_strategy_consumption_gate.py",
        "poker_decision_ai/realtime.py",
    }
    normalized = {i.replace(chr(92), '/') for i in importers}
    assert normalized == sanctioned_importers, (
        f"cash_unknown_e2e 引用方与授权基线不符: "
        f"unregistered={sorted(normalized - sanctioned_importers)} "
        f"missing={sorted(sanctioned_importers - normalized)}")


async def test_wiring_does_not_change_judge_output():
    """接线前后 Judge 输出**逐位不变**（本层只在决策后追加 metadata）。"""
    import asyncio
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    cfg = load_config()
    # BENCH-1 Task0.3：冷启动（首次 import/equity 预热）+ 并行负载下默认
    # 3000ms 死线偶发超时（顺序依赖的脆弱失败）。测试内提高死线——被测
    # 语义（UNKNOWN 报告暴露）与死线无关，不得靠删测试修脆弱。
    cfg.root._data["realtime"] = {**cfg.realtime_config(),
                                  "deadline_ms": 60000,
                                  "agent_timeout_ms": 30000}
    logs = ROOT / ".tmp" / "v3g8_logs" / uuid.uuid4().hex
    logs.mkdir(parents=True, exist_ok=True)
    ctx = AgentContext(request_id="v3g8", session_id="v3g8", config=cfg,
                       providers=build_providers(cfg), repos=build_in_memory(),
                       logger=EventLogger(str(logs)),
                       metrics=MetricsCollector())
    s = st(pot=200.0, to_call=50.0, bb=2.0, min_raise_to=100.0)
    eq = {"hero_equity": 0.8, "outcome": {"win": 0.8, "tie": 0.0, "lose": 0.2}}
    inp = {"state": copy.deepcopy(s), "equity": copy.deepcopy(eq)}
    before = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output
    build_unknown_report(s)
    after = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output
    assert set(before) == set(after)
    for k in sorted(before):
        assert before.get(k) == after.get(k), k
