"""Phase 4（数据就绪）测试：Canonical Schema / Validator / 时间切分 / Dataset 工具。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_decision_ai.dataset_tool import DatasetQualifier
from poker_decision_ai.domain.poker.canonical import (
    CANONICAL_HAND_SCHEMA,
    CanonicalHand,
    canonical_to_vision_observation,
    parsed_hand_to_canonical,
)
from poker_decision_ai.domain.poker.hand_history import parse_hand_history
from poker_decision_ai.domain.poker.validator import validate_hand
from poker_decision_ai.experiment_splits import arm_request, arm_spec, as_of, time_split

ROOT = Path(__file__).parent.parent.parent
FULL = (ROOT / "examples/hands/cash_full_hand.txt").read_text(encoding="utf-8")
SHOWDOWN = (ROOT / "examples/hands/cash_showdown_hand.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------- Canonical
def test_canonical_schema_is_json_schema():
    assert CANONICAL_HAND_SCHEMA["type"] == "object"
    for f in ("hand_id", "game_mode", "game_type", "stakes", "players",
              "hero_id", "hole_cards", "preflop_actions", "flop", "flop_actions",
              "turn", "turn_actions", "river", "river_actions", "pot_sizes",
              "bet_sizes", "all_in", "folds", "showdown", "winners"):
        assert f in CANONICAL_HAND_SCHEMA["properties"], f


def test_parsed_to_canonical_missing_fields_null_not_guessed():
    hand = parsed_hand_to_canonical(parse_hand_history(FULL))
    assert hand.game_mode == "cash" and hand.game_type == "cash"
    assert hand.stakes is None            # canonical 不猜级别（由 State 层推导）
    assert hand.started_at == "2025-01-12T12:00:00"    # 头部时间已被解析
    assert all(p.stack_end is None for p in hand.players)   # 原始无终局筹码 → null
    assert hand.hero_id == "Hero" and hand.hole_cards == ["As", "Kc"]
    assert len(hand.preflop_actions) == 9      # 2 posts + 7 行动
    assert hand.flop == ["Qh", "7d", "2c"]
    assert hand.total_pot == 24.55
    assert hand.winners[0]["player_id"] == "Bob"


def test_showdown_canonical():
    hand = parsed_hand_to_canonical(parse_hand_history(SHOWDOWN))
    assert hand.showdown["happened"] is True
    assert len(hand.showdown["shows"]) == 2


def test_canonical_to_vision_observation():
    hand = parsed_hand_to_canonical(parse_hand_history(FULL))
    obs = canonical_to_vision_observation(hand, "flop")
    assert obs is not None
    assert obs["street"] == "flop" and obs["board"] == ["Qh", "7d", "2c"]
    assert obs["to_call"] is not None
    # 无 hero 手牌 → None（不猜）
    hand2 = parsed_hand_to_canonical(parse_hand_history(FULL))
    hand2.hole_cards = []
    assert canonical_to_vision_observation(hand2, "flop") is None


# ---------------------------------------------------------------- Validator
def test_validator_valid_hand():
    hand = parsed_hand_to_canonical(parse_hand_history(FULL))
    report = validate_hand(hand)
    assert report.status in ("valid", "warning")
    assert all(c.status != "invalid" for c in report.checks)


def test_validator_detects_duplicate_board():
    hand = parsed_hand_to_canonical(parse_hand_history(FULL))
    hand.flop = ["Qh", "Qh", "2c"]
    report = validate_hand(hand)
    assert report.status == "invalid"
    assert any(c.name == "2_board_sane" and c.status == "invalid" for c in report.checks)


def test_validator_detects_fold_then_act():
    hand = parsed_hand_to_canonical(parse_hand_history(FULL))
    from poker_decision_ai.domain.poker.canonical import CanonicalAction

    hand.preflop_actions.append(CanonicalAction(
        street="preflop", seq=99, player_id="Eve", action="call", amount=1.0))
    report = validate_hand(hand)
    assert report.status == "invalid"
    assert any("弃牌后仍行动" in c.detail for c in report.checks if c.status == "invalid")


def test_validator_missing_fields_warning():
    hand = parsed_hand_to_canonical(parse_hand_history(FULL))
    report = validate_hand(hand)
    c10 = next(c for c in report.checks if c.name == "10_missing_fields")
    assert c10.status == "warning"
    assert "stack_end" in c10.detail      # 原始数据无终局筹码 → 如实报告缺失


# ---------------------------------------------------------------- 切分与泄漏
def test_time_split_excludes_missing_ts():
    hands = [{"started_at": f"2025-01-{d:02d}T12:00:00"} for d in range(1, 11)]
    hands.append({"no_time": True})
    split = time_split(hands, train_frac=0.7, val_frac=0.15)
    assert split.counts() == {"train": 7, "validation": 1, "test": 2,
                              "excluded_no_timestamp": 1}
    # 时间顺序：test 的时间晚于 train
    assert split.test[0]["started_at"] > split.train[-1]["started_at"]


def test_as_of_point_in_time():
    hands = [{"started_at": f"2025-01-{d:02d}T12:00:00", "v": d} for d in range(1, 6)]
    visible = as_of(hands, "2025-01-03T00:00:00")
    assert [h["v"] for h in visible] == [1, 2]


def test_arm_specs():
    assert arm_spec("A")["flags"]["use_individual"] is False
    assert arm_spec("E")["flags"]["use_solver"] is True
    with pytest.raises(ValueError):
        arm_spec("Z")
    req = arm_request({"kind": "cash_hand"}, "C")
    assert req["arm"] == "C" and req["arm_flags"]["use_llm"] is True


# ---------------------------------------------------------------- Dataset 工具
def test_dataset_qualifier_end_to_end(tmp_path):
    from scripts.generate_synthetic_hands import make_hand

    src = tmp_path / "hand_history"
    src.mkdir()
    for i in range(8):
        (src / f"hand_{i}.txt").write_text(make_hand(i), encoding="utf-8")
    qualifier = DatasetQualifier(base_dir=str(tmp_path / "data"),
                                 docs_dir=str(tmp_path / "docs"))
    report = qualifier.qualify(str(src), source="test_synthetic", license_info="internal")
    assert report.hands == 8
    assert report.by_mode.get("cash") == 8
    assert report.valid + report.warning + report.invalid == 8
    assert report.invalid == 0, "合成手不应 invalid（防底池/行动记账回归）"
    # 分层落盘
    assert len(list((tmp_path / "data/normalized").glob("*.json"))) == 8
    assert (tmp_path / "data/manifest.json").exists()
    assert len(list((tmp_path / "data/raw").glob("*.txt"))) == 8
    # 玩家统计（12 个名字池，Hero 恒有 8 手）
    assert report.player_stats["Hero"]["hands"] == 8
    assert "vpip" in report.player_stats["Hero"]
    # 日期范围（合成手带时间戳）
    assert report.date_range["n_with_timestamp"] == 8
    # manifest 文档生成（写入隔离的 docs 目录，不污染仓库 docs/）
    assert (tmp_path / "docs" / "DATASET_MANIFEST.md").exists()


def test_dataset_qualifier_rejects_bad_file(tmp_path):
    src = tmp_path / "hand_history"
    src.mkdir()
    (src / "bad.txt").write_text("not a hand history at all", encoding="utf-8")
    qualifier = DatasetQualifier(base_dir=str(tmp_path / "data"),
                                 docs_dir=str(tmp_path / "docs"))
    report = qualifier.qualify(str(src), source="t", license_info="t")
    assert report.hands == 0
    assert report.files_failed == 1
