"""V3F-P54 — CASH Decision-Grade Validation Integration 测试。

覆盖：PHH 解析正确性（cbr/cc/街切换/底池/栈）；AI 座位隔离（数量+
事件排除）；P52 evaluate_dataset decision_grade 覆盖；split 无泄漏；
确定性重复；legality（观察动作 ∈ 严格推导合法集）；UNKNOWN 传播；
标签分离（observed_human_action / pluribus_ai_action / solver_label /
model_recommendation）；策略数学冻结（mtime 间接由回归保证）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from poker_decision_ai.pluribus_decision_adapter import (
    AI_SEAT,
    PARQUET_DIR,
    PluribusDecisionAdapter,
    extract_decision_points,
    iter_pluribus_rows,
    legal_action_types,
    manifest,
    parse_hand_row,
)
from poker_decision_ai.pluribus_validation import (
    build_evidence_events,
    build_indexes,
    split_hands,
    validate,
)

HAVE_PARQUET = PARQUET_DIR.exists() and any(PARQUET_DIR.glob("train-*.parquet"))


def _load_hands(limit: int) -> list[dict]:
    out = []
    for row in iter_pluribus_rows(limit=limit):
        p = parse_hand_row(row)
        if p:
            out.append(p)
    return out


# ------------------------------------------------------------ 解析
@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_parse_preflop_and_flop_accounting():
    rows = list(iter_pluribus_rows(limit=1))
    p = parse_hand_row(rows[0])
    assert p is not None
    assert len(p["players"]) == 6
    assert len(p["hole_cards"]) == 6                 # 全部 6 座发牌
    assert p["bb"] == 100.0 and p["sb"] == 50.0
    assert all(s == 10000.0 for s in p["stacks"])
    # flop 首个动作 pot = preflop 总投入（盲注+cbr+call）
    flop = next(s for s in p["streets"] if s["street"] == "flop")
    assert flop["actions"][0]["pot_before"] == 600.0  # 50+100+225+225
    # 本街 committed 重置：flop bet 金额 = cbr 值本身
    bets = [a for a in flop["actions"] if a["action"] == "bet"]
    assert bets and bets[0]["amount"] == bets[0]["to"]


@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_legal_action_types_strict():
    # to_call==0 → fold/check/bet（无下注时无 raise 可言——严格）
    assert legal_action_types(0.0, 100.0) == ["fold", "check", "bet"]
    # to_call>0 → fold/call/raise（无 check/bet）
    la = legal_action_types(10.0, 100.0)
    assert la == ["fold", "call", "raise"]
    assert "check" not in la and "bet" not in la
    assert legal_action_types(10.0, 10.0) == ["fold", "call"]   # 栈<=to_call
    assert legal_action_types(0.0, 0.0) is None                 # 栈耗尽


# ------------------------------------------------------------ AI 座位隔离
@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_ai_seat_excluded_from_decision_points():
    hands = _load_hands(10)
    for h in hands:
        dps = extract_decision_points(h)
        assert all(d["player"] != AI_SEAT for d in dps)
        # AI 座位 hole cards 仍在（作对手），但其动作只出现在对手历史
        assert AI_SEAT in h["hole_cards"]


@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_evidence_events_exclude_ai():
    hands = _load_hands(10)
    for h in hands:
        for ev in build_evidence_events(h):
            assert ev["player"] != AI_SEAT


# ------------------------------------------------------------ P52 集成
@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_evaluate_dataset_decision_grade():
    """P52 evaluate_dataset：decision_grade 覆盖率 + 字段 verdict +
    AI 排除计数（契约不放宽）。"""
    adapter = PluribusDecisionAdapter(limit=30)
    rep = adapter.evaluate_dataset(limit=0)
    # 内部一致：每 parsed 手 5 个人类座位记录（个别手可能解析跳过）
    assert rep["n_records"] == 5 * rep["ai_seat_excluded_hands"]
    assert rep["ai_seat_excluded_hands"] + rep["hands_skipped"] == 30
    # 少数手人类无行动 → 诚实落入 behavior_grade（非 decision_grade）
    assert rep["dataset_verdict_ratio"].get("decision_grade", 0) >= 0.95
    # 字段口径：结构字段 ~100%；行动字段 ≥0.95（无行动记录诚实缺省）
    for f in ("hero_hole_cards", "pot_effective_stack", "position_button",
              "game_mode", "board_by_street"):
        assert rep["field_usable_ratio"][f] >= 0.99, f
    for f in ("action_sequence_sizes", "observed_hero_action"):
        assert rep["field_usable_ratio"][f] >= 0.95, f
    assert rep["dataset_meta"]["ground_truth_claim"].startswith("human seats")


# ------------------------------------------------------------ split/泄漏
@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_split_no_leakage_deterministic():
    hands = _load_hands(60)
    a = split_hands(hands)
    b = split_hands(hands)
    assert ([h["hand_id"] for h in a[0]] == [h["hand_id"] for h in b[0]])
    ev_ids = {h["hand_id"] for h in a[0]}
    eval_ids = {h["hand_id"] for h in a[1]}
    assert ev_ids and eval_ids and not (ev_ids & eval_ids)


# ------------------------------------------------------------ 主验证
@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_validate_labels_legality_and_determinism():
    hands = _load_hands(40)
    rep = validate(hands, decision_hand_limit=6, equity_iterations=600)
    assert rep["split"]["leakage_free"] is True
    assert rep["labels"]["observed_human_action"] > 0
    assert rep["labels"]["pluribus_ai_action_excluded"] > 0
    assert rep["labels"]["model_recommendation"] > 0
    assert rep["legality"]["illegal"] == 0               # 观察动作全合法
    rows = rep["decision_ev"]["rows_sample"] if "rows_sample" in rep["decision_ev"] else []
    assert all(d["legal"] for d in rows)
    # 确定性：双跑逐位一致
    rep2 = validate(hands, decision_hand_limit=6, equity_iterations=600)
    assert rep == rep2


@pytest.mark.skipif(not HAVE_PARQUET, reason="raw parquet not present")
def test_unknown_propagation_declared():
    """UNKNOWN 计数字段存在且为非负整数（诚实传播口径）。"""
    hands = _load_hands(20)
    rep = validate(hands, decision_hand_limit=4, equity_iterations=400)
    for k, v in rep["unknown"].items():
        assert isinstance(v, int) and v >= 0


def test_manifest_label_discipline():
    m = manifest()
    ld = m["label_discipline"]
    assert "real_observed_action" in ld["observed_human_action"] or \
        "真实" in ld["observed_human_action"]
    assert "solver_label" in ld["pluribus_ai_action"]
    assert "模型" in ld["model_recommendation"] or \
        "compute_ev" in ld["model_recommendation"]
    assert m["honesty"]["research_play"]
    assert "NOT VALIDATABLE" in m["honesty"]["mtt"]
    assert "NOT VALIDATABLE" in m["honesty"]["pko"]
