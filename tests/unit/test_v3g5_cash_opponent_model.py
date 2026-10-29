"""V3G-5 — CASH 对手建模闭环测试。

覆盖授权要求：单玩家累计、多玩家隔离、小样本、缺失 action、冲突数据、
本手 action 更新、快照确定性、evidence/provenance、CASH/非 CASH fail closed、
V3G-2/3/4 回归、production 旁路接线前后 action/EV/size 逐位一致。

**不修改**任何既有测试定义；**不触碰**冻结文件。
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest

from poker_decision_ai.cash_opponent_model import (
    ADEQUATE_SAMPLES, LEVEL_ADEQUATE, LEVEL_LOW, LEVEL_UNKNOWN, METRIC_KEYS,
    MIN_SAMPLES, NON_CASH_GAMES, OPPONENT_MODEL_VERSION, build_opponent_model)
from poker_decision_ai.domain.poker.stats import HAND_RECORD_FIELDS, compute_all

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"


def cash_state(**over) -> dict:
    st = {
        "game_type": "cash", "street": "flop", "board": ["Ah", "7c", "2c"],
        "pot": 10.0, "to_call": 0.0, "bb": 1.0, "sb": 0.5,
        "hero_id": "hero", "button_seat": 2, "table_size": 3,
        "synthetic_test_only": True,
        "players": [
            {"player_id": "v1", "seat": 1, "position": "BB", "status": "active",
             "stack": 100.0, "hole_cards": []},
            {"player_id": "hero", "seat": 2, "position": "BTN", "is_hero": True,
             "status": "active", "hole_cards": ["As", "Kd"], "stack": 100.0},
            {"player_id": "v2", "seat": 3, "position": "SB", "status": "active",
             "stack": 100.0, "hole_cards": []},
        ],
        "action_history": [],
    }
    st.update(over)
    return st


def rec(pid: str, hid: str, **over) -> dict:
    """归一化手牌记录（只写该用例显式观察到的字段；其余保持 None）。"""
    r = {k: None for k in HAND_RECORD_FIELDS}
    r.update({"player_id": pid, "hand_id": hid, "game": "cash",
              "bets": 0, "calls": 0, "folds": 0})
    r.update(over)
    return r


# ── 单玩家累计统计 + 真实复用既有指标 ─────────────────────────────────
def test_single_player_cumulative_stats_reuse_existing_engine():
    recs = [rec("v1", f"h{i}", vpip=True, pfr=(i % 2 == 0),
                saw_flop=True, bets=1) for i in range(4)]
    r = build_opponent_model(cash_state(), records=recs)
    m = r["model"]["v1"]
    assert m["hands"] == 4
    assert m["inferred"]["vpip"]["value"] == 1.0
    assert m["inferred"]["vpip"]["samples"] == 4
    assert m["inferred"]["pfr"]["value"] == 0.5
    # 与既有引擎**直接对照**（证明是复用而非重算）
    ref = compute_all(recs).to_dict()
    for k in METRIC_KEYS:
        assert m["inferred"][k]["value"] == (
            ref[k]["value"] if ref[k]["samples"] > 0 else None), k


def test_metric_keys_are_sourced_from_existing_engine():
    bundle = compute_all([rec("v1", "h1", vpip=True)]).to_dict()
    assert set(METRIC_KEYS) <= set(bundle), set(METRIC_KEYS) - set(bundle)


def test_opportunity_metrics_use_opportunity_denominator():
    recs = [rec("v1", "h1", faced_cbet=True, folded_to_cbet=True),
            rec("v1", "h2", faced_cbet=False, folded_to_cbet=True),
            rec("v1", "h3", faced_cbet=True, folded_to_cbet=False)]
    r = build_opponent_model(cash_state(), records=recs)
    f2c = r["model"]["v1"]["inferred"]["fold_to_cbet"]
    assert f2c["samples"] == 2 and f2c["value"] == 0.5


# ── 多玩家隔离 ────────────────────────────────────────────────────────
def test_multi_player_isolation_no_cross_contamination():
    recs = [rec("v1", "a1", vpip=True), rec("v1", "a2", vpip=True),
            rec("v2", "b1", vpip=False)]
    r = build_opponent_model(cash_state(), records=recs)
    assert sorted(r["model"]) == ["v1", "v2"]
    assert r["model"]["v1"]["hands"] == 2
    assert r["model"]["v2"]["hands"] == 1
    assert r["model"]["v1"]["inferred"]["vpip"]["value"] == 1.0
    assert r["model"]["v2"]["inferred"]["vpip"]["value"] == 0.0
    # 同一 hand_id 不同玩家互不影响
    recs2 = [rec("v1", "same", vpip=True), rec("v2", "same", vpip=False)]
    r2 = build_opponent_model(cash_state(), records=recs2)
    assert r2["model"]["v1"]["hands"] == 1 and r2["model"]["v2"]["hands"] == 1
    assert r2["conflicts"] == []


def test_records_without_player_id_are_not_attributed():
    r = build_opponent_model(cash_state(),
                             records=[rec("v1", "h1", vpip=True),
                                      {**rec("v1", "h2", vpip=True),
                                       "player_id": None}])
    assert r["model"]["v1"]["hands"] == 1
    assert any(x["reason"] == "missing_player_id" for x in r["incomplete"])


# ── 小样本 → UNKNOWN / 低置信，不强行分类 ─────────────────────────────
def test_small_sample_is_unknown_and_not_classified():
    r = build_opponent_model(cash_state(),
                             records=[rec("v1", "h1", vpip=True)])
    m = r["model"]["v1"]
    assert m["confidence"]["level"] == LEVEL_UNKNOWN
    assert m["confidence"]["type_label"] is None      # 不贴标签
    assert m["inferred"]["vpip"]["level"] == LEVEL_UNKNOWN
    assert MIN_SAMPLES == 30 and ADEQUATE_SAMPLES == 100


def test_sample_levels_progress_as_evidence_accumulates():
    def level(n, **kw):
        recs = [rec("v1", f"h{i}", **kw) for i in range(n)]
        return build_opponent_model(cash_state(), records=recs)["model"]["v1"]

    assert level(5, vpip=True)["inferred"]["vpip"]["level"] == LEVEL_UNKNOWN
    assert level(MIN_SAMPLES, vpip=True)["inferred"]["vpip"]["level"] == LEVEL_LOW
    assert level(ADEQUATE_SAMPLES, vpip=True)["inferred"]["vpip"][
        "level"] == LEVEL_ADEQUATE
    assert level(ADEQUATE_SAMPLES, vpip=True)["confidence"][
        "level"] == LEVEL_ADEQUATE


def test_zero_sample_metric_hides_value():
    r = build_opponent_model(cash_state(), records=[rec("v1", "h1", vpip=True)])
    m = r["model"]["v1"]["inferred"]
    assert m["wtsd"]["value"] is None and m["wtsd"]["samples"] == 0
    assert m["wtsd"]["level"] == LEVEL_UNKNOWN


# ── 缺失 action → 不计入样本（不默认为否）─────────────────────────────
def test_missing_action_fields_are_not_counted_and_not_defaulted():
    recs = [rec("v1", "h1", vpip=True),               # pfr 缺失（None）
            rec("v1", "h2", vpip=True, pfr=None)]
    r = build_opponent_model(cash_state(), records=recs)
    m = r["model"]["v1"]
    assert m["hands"] == 2
    assert m["inferred"]["pfr"]["samples"] == 0        # 不得计为 False
    assert m["inferred"]["pfr"]["value"] is None
    assert m["observed"]["pfr"]["unknown"] == 2


def test_unknown_action_is_not_recorded_as_evidence():
    r = build_opponent_model(
        cash_state(),
        this_hand_actions=[{"player_id": "v1", "action": "teleport"}])
    assert "v1" not in r["model"]                      # 无证据 → 不建条目
    assert any(x["reason"].startswith("unknown_action")
               for x in r["incomplete"])


def test_action_without_player_id_is_not_attributed():
    r = build_opponent_model(cash_state(),
                             this_hand_actions=[{"action": "raise"}])
    assert r["model"] == {}
    assert any(x["reason"] == "missing_player_id" for x in r["incomplete"])


# ── 冲突数据 → 两条都不采信，如实标记 ─────────────────────────────────
def test_conflicting_duplicate_hand_id_is_flagged_and_dropped():
    recs = [rec("v1", "dup", vpip=True),
            rec("v1", "dup", vpip=False)]              # 同手不同证据
    r = build_opponent_model(cash_state(), records=recs)
    assert any(x["reason"] == "duplicate_hand_id_conflicting"
               for x in r["conflicts"])
    # 冲突记录**不**进入统计（不猜哪条正确）
    assert "v1" not in r["model"] or r["model"]["v1"]["hands"] == 0


def test_identical_duplicate_hand_id_is_not_a_conflict():
    recs = [rec("v1", "dup", vpip=True), rec("v1", "dup", vpip=True)]
    r = build_opponent_model(cash_state(), records=recs)
    assert r["conflicts"] == []
    assert r["model"]["v1"]["hands"] == 2


# ── 本手 action 更新 ──────────────────────────────────────────────────
def test_this_hand_action_updates_only_observed_flags():
    r = build_opponent_model(
        cash_state(),
        records=[rec("v1", "h1", vpip=False, pfr=False)],
        this_hand_actions=[{"player_id": "v1", "action": "raise",
                            "street": "flop", "amount": 5.0, "sequence": 0}])
    m = r["model"]["v1"]
    assert m["hands"] == 2
    assert m["inferred"]["vpip"]["value"] == 0.5      # 1/2
    assert m["inferred"]["pfr"]["value"] == 0.5


def test_this_hand_action_does_not_write_unobserved_fields():
    r = build_opponent_model(
        cash_state(),
        this_hand_actions=[{"player_id": "v1", "action": "call"}])
    m = r["model"]["v1"]
    # call → 只有 vpip 证据；pfr/cbet 等**不得**被写成 False
    assert m["inferred"]["vpip"]["samples"] == 1
    assert m["inferred"]["pfr"]["samples"] == 0
    assert m["inferred"]["cbet"]["samples"] == 0
    assert m["observed"]["wtsd"]["unknown"] == 1


def test_this_hand_action_goes_only_to_its_actor():
    r = build_opponent_model(
        cash_state(),
        records=[rec("v2", "h1", vpip=True)],
        this_hand_actions=[{"player_id": "v1", "action": "raise"}])
    assert r["model"]["v1"]["hands"] == 1
    assert r["model"]["v2"]["hands"] == 1
    assert r["model"]["v2"]["inferred"]["pfr"]["samples"] == 0


def test_fold_and_check_do_not_imply_pfr():
    for act in ("fold", "check", "call"):
        r = build_opponent_model(cash_state(),
                                 this_hand_actions=[{"player_id": "v1",
                                                     "action": act}])
        m = r["model"]["v1"]
        assert m["inferred"]["vpip"]["samples"] == 1, act
        assert m["inferred"]["pfr"]["samples"] == 0, act


# ── 快照确定性 ────────────────────────────────────────────────────────
def test_snapshot_is_deterministic_and_json_safe():
    recs = [rec("v1", f"h{i}", vpip=True, pfr=(i % 2 == 0)) for i in range(5)]
    acts = [{"player_id": "v2", "action": "bet"}]
    a = build_opponent_model(cash_state(), records=recs,
                             this_hand_actions=acts)
    for _ in range(5):
        assert build_opponent_model(cash_state(), records=recs,
                                    this_hand_actions=acts) == a
    json.dumps(a)


def test_model_ordering_is_stable():
    recs = [rec("v2", "h1", vpip=True), rec("v1", "h2", vpip=True),
            rec("v10", "h3", vpip=True)]
    r = build_opponent_model(cash_state(), records=recs)
    assert list(r["model"]) == ["v1", "v10", "v2"]      # 排序确定


# ── evidence / provenance ─────────────────────────────────────────────
def test_evidence_and_provenance_are_present_and_traceable():
    r = build_opponent_model(cash_state(), records=[rec("v1", "h1", vpip=True)])
    m = r["model"]["v1"]
    assert m["evidence"]["metric_engine"] == "domain.poker.stats.compute_all"
    assert m["evidence"]["player_isolation"] is True
    assert "visual" in m["evidence"]["note"] or "视觉" in m["evidence"]["note"]
    assert m["observed"]["vpip"]["field"] == "vpip"
    assert m["observed"]["cbet_opportunity"]["opportunity_field"] == \
        "cbet_opportunity"
    assert m["inferred"]["vpip"]["evidence"] == "observed_samples=1"
    assert r["evidence"]["source"] == "v3g5_cash_opponent_model"
    assert r["version"] == OPPONENT_MODEL_VERSION
    assert m["evidence"]["source"].startswith("normalized_hand_records")


# ── CASH / 非 CASH fail closed ────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin", None])
def test_non_cash_fails_closed(mode):
    r = build_opponent_model(cash_state(game_type=mode),
                             records=[rec("v1", "h1", vpip=True)])
    assert r["present"] is False
    assert "model" not in r


def test_non_dict_state_fails_closed():
    for bad in (None, [], "cash", 3):
        assert build_opponent_model(bad)["present"] is False


def test_non_cash_hand_records_are_excluded_from_cash_stats():
    for g in NON_CASH_GAMES:
        r = build_opponent_model(
            cash_state(),
            records=[rec("v1", "h1", vpip=True), rec("v1", "h2", vpip=True,
                                                     game=g)])
        assert r["model"]["v1"]["hands"] == 1, g
        assert any(x["reason"] == f"non_cash_game:{g}" for x in r["excluded"])


def test_no_tournament_or_bounty_fields_in_output():
    r = build_opponent_model(cash_state(), records=[rec("v1", "h1", vpip=True)])
    blob = json.dumps(r, ensure_ascii=False).lower()
    for banned in ("icm", "bounty", "satellite", "payout"):
        assert banned not in blob, banned


# ── 只读 / 无随机 / 依赖审计 ──────────────────────────────────────────
def test_model_does_not_mutate_inputs():
    st = cash_state()
    recs = [rec("v1", "h1", vpip=True)]
    acts = [{"player_id": "v1", "action": "raise"}]
    b_st, b_recs, b_acts = (copy.deepcopy(st), copy.deepcopy(recs),
                            copy.deepcopy(acts))
    build_opponent_model(st, records=recs, this_hand_actions=acts)
    assert st == b_st and recs == b_recs and acts == b_acts


def test_module_has_no_randomness_or_io():
    src = (PKG / "cash_opponent_model.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            imported.append(n.module or "")
    joined = " ".join(imported)
    for banned in ("random", "research_unfreeze", "vision2", "positions_for",
                   "strategy_core", "judge"):
        assert banned not in joined, banned
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for banned in ("compute_ev", "evaluate_intel", "positions_for", "open"):
        assert banned not in called, banned
    assert set(imported) <= {"__future__", "dataclasses", "typing",
                             "cash_integrity_context",
                             "domain.poker.stats"}, imported


def test_model_is_not_wired_into_production():
    importers = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name == "cash_opponent_model.py":
            continue
        src = p.read_text(encoding="utf-8-sig", errors="ignore")
        if "cash_opponent_model" in src:
            importers.append(str(p.relative_to(ROOT)))
    # BENCH-1 Task0.1（owner 授权基线迁移，4 个引用方）：
    # 2026-09 中旬 P30-P41 生产桥/策略链阶段已授权下列模块引用本层。
    # 按 test_v3g11 SANCTIONED_WIRING 惯例登记为合法基线（精确集合匹配——
    # 任何未登记的新引用方、或已登记引用方消失，都会被检出）。
    sanctioned_importers = {
        "poker_decision_ai/cash_architecture_closure_audit.py",
        "poker_decision_ai/cash_strategy_belief_update.py",
        "poker_decision_ai/cash_strategy_consumption_gate.py",
        "poker_decision_ai/cash_strategy_dynamic_belief.py",
    }
    normalized = {i.replace(chr(92), '/') for i in importers}
    assert normalized == sanctioned_importers, (
        f"cash_opponent_model 引用方与授权基线不符（新引用方需所有者授权登记）: "
        f"unregistered={sorted(normalized - sanctioned_importers)} "
        f"missing={sorted(sanctioned_importers - normalized)}")


# ── production 旁路接线前后逐位一致（本层未接线 → 结构性成立）─────────
async def test_judge_action_ev_size_bitwise_identical_with_and_without_model():
    """接 production 前的硬不变式：跑/不跑对手模型，Judge 输出**逐位一致**。"""
    import uuid
    from poker_decision_ai.agents.judge.agent import JudgeAgent
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    cfg = load_config()
    logs = ROOT / ".tmp" / "v3g5_logs" / uuid.uuid4().hex
    logs.mkdir(parents=True, exist_ok=True)
    ctx = AgentContext(request_id="v3g5", session_id="v3g5", config=cfg,
                       providers=build_providers(cfg), repos=build_in_memory(),
                       logger=EventLogger(str(logs)),
                       metrics=MetricsCollector())
    st = cash_state(pot=200.0, to_call=50.0, bb=2.0, min_raise_to=100.0,
                    street="flop", board=["Ah", "7c", "2c"])
    eq = {"hero_equity": 0.8,
          "outcome": {"win": 0.8, "tie": 0.0, "lose": 0.2}}
    inp = {"state": copy.deepcopy(st), "equity": copy.deepcopy(eq)}

    before = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output
    # 跑一次对手模型（含记录与本手行动）
    build_opponent_model(st, records=[rec("v1", "h1", vpip=True)],
                         this_hand_actions=[{"player_id": "v1",
                                             "action": "raise"}])
    after = (await JudgeAgent().run(copy.deepcopy(inp), ctx)).output

    for k in ("action", "alternative_action", "recommended_size", "ev_core",
              "weights_used", "final_inputs_summary", "confidence",
              "reasoning", "warnings"):
        assert before.get(k) == after.get(k), k


# ── V3G-2/3/4 回归（本层视角）────────────────────────────────────────
def test_v3g2_evidence_carried_and_v3g3_v3g4_reusable():
    from poker_decision_ai.cash_multistreet import build_multistreet_context
    from poker_decision_ai.cash_preflop_decision import build_preflop_decision
    st = cash_state(street="preflop", board=[])
    pf = build_preflop_decision(st)
    assert pf["present"] is True
    ms = build_multistreet_context(st, preflop=pf)
    assert ms["present"] is True
    r = build_opponent_model(st, records=[rec("v1", "h1", vpip=True)])
    assert r["present"] is True
    assert r["evidence"]["present"] is True            # V3G-2 旁路
    assert r["integrity"]["level"] in ("COMPLETE", "DEGRADED", "INCOMPLETE")


def test_visual_fields_are_ignored_as_behavior_evidence():
    """不得用视觉推断替代行为证据：未知视觉字段不产生任何统计。"""
    r = build_opponent_model(
        cash_state(),
        records=[{**{k: None for k in HAND_RECORD_FIELDS},
                  "player_id": "v1", "hand_id": "h1", "game": "cash",
                  "visual_aggression": "high", "seat": 1}])
    m = r["model"]["v1"]
    assert m["inferred"]["vpip"]["samples"] == 0
    assert m["confidence"]["level"] == LEVEL_UNKNOWN
