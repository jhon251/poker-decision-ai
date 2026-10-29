"""Vision Agent V2（Phase V0）测试：schema 无默认填值 / 15 条 validator /
assembly 装配 / position 确定性推导 / synthetic_test_only / 红线。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2 import (  # noqa: E402
    NormalizedVisionState, RawAction, RawSeat, RawVisionState,
    UncertaintyEntry, assemble, to_dict, validate_raw)
from poker_decision_ai.vision2 import fixtures as F  # noqa: E402


# ---------------------------------------------------------------- schema：无默认填值
def test_raw_fields_default_null():
    r = RawVisionState()
    # source / synthetic_test_only 是元数据默认（非观测事实）；其余观测字段必须 null
    for name, val in r.__dataclass_fields__.items():
        if name in ("source", "synthetic_test_only"):
            continue
        assert getattr(r, name) is None, f"字段 {name} 默认应为 None（禁止填 0/[]/''）"


def test_raw_seat_fields_default_null():
    s = RawSeat()
    for name in ("seat", "occupied", "anonymous_player_id", "stack",
                 "current_bet", "folded", "all_in", "in_hand", "visible_action"):
        assert getattr(s, name) is None


def test_to_dict_preserves_null():
    d = to_dict(RawVisionState())
    assert d["hero_hole_cards"] is None
    assert d["board"] is None
    assert d["pot"] is None
    assert d["seats"] is None
    assert d["action_history"] is None


def test_normalized_shape_contract():
    out = assemble(F.good_6max_flop()).to_dict()
    assert set(out) == {"source", "synthetic_test_only", "validation",
                        "uncertainty", "game", "hero", "players", "board",
                        "pot", "action_history"}
    assert set(out["validation"]) == {"valid", "errors", "warnings"}


def test_no_decision_fields_in_output():
    out = assemble(F.good_6max_flop()).to_dict()
    for bad in ("recommended_action", "action", "ev", "roi", "profitability",
                "exploit", "raise", "bet_size"):
        assert bad not in out and bad not in out["hero"] and bad not in out["game"]


# ---------------------------------------------------------------- validator：15 条
def test_good_fixture_valid():
    errs, unc, _ = validate_raw(F.good_6max_flop())
    assert errs == []


def test_duplicate_hero_board_card_invalid():
    errs, _, _ = validate_raw(F.duplicate_card())
    assert any(e["code"] == "duplicate_card" for e in errs)


def test_duplicate_within_hero_cards():
    r = F.good_6max_flop()
    r.hero_hole_cards = ["As", "As"]
    errs, _, _ = validate_raw(r)
    assert any(e["code"] == "duplicate_card" for e in errs)


def test_negative_pot_invalid():
    errs, _, _ = validate_raw(F.negative_pot())
    assert any(e["code"] == "pot_negative" for e in errs)


def test_negative_stack_invalid():
    r = F.good_6max_flop()
    r.seats[0].stack = -1.0
    errs, _, _ = validate_raw(r)
    assert any(e["code"] == "stack_negative" for e in errs)


def test_bet_exceeds_stack_invalid():
    errs, _, _ = validate_raw(F.bet_exceeds_stack())
    assert any(e["code"] == "hero_bet_exceeds_stack" for e in errs)


def test_street_regression_invalid():
    errs, _, _ = validate_raw(F.street_regression())
    assert any(e["code"] == "street_regression" for e in errs)


def test_folded_later_action_invalid():
    errs, _, _ = validate_raw(F.folded_then_action())
    assert any(e["code"] == "folded_later_action" for e in errs)


def test_no_hero_uncertain_not_invalid():
    errs, unc, _ = validate_raw(F.no_hero())
    assert not any(e["code"] == "hero_seat_not_found" for e in errs)
    assert any(u.field == "hero.seat" for u in unc)


def test_multiple_hero_invalid():
    errs, _, _ = validate_raw(F.multiple_hero())
    assert any(e["code"] in ("multiple_hero", "duplicate_seat") for e in errs)


def test_flop_length_invalid():
    r = F.good_6max_flop()
    r.board = {"flop": ["Qh", "7d"], "turn": None, "river": None}
    errs, _, _ = validate_raw(r)
    assert any(e["code"] == "flop_length" for e in errs)


def test_board_on_preflop_invalid():
    r = F.preflop_minimal()
    r.board = {"flop": ["Qh", "7d", "2c"], "turn": None, "river": None}
    errs, _, _ = validate_raw(r)
    assert any(e["code"] == "board_on_preflop" for e in errs)


def test_action_order_regression_invalid():
    r = F.good_6max_flop()
    r.action_history = [
        RawAction(player="p1", action="call", street="preflop", order=3),
        RawAction(player="p2", action="call", street="preflop", order=1),
    ]
    errs, _, _ = validate_raw(r)
    assert any(e["code"] == "action_order_regression" for e in errs)


def test_invalid_action_enum():
    r = F.good_6max_flop()
    r.action_history = [RawAction(player="p1", action="bluff",
                                  street="preflop", order=1)]
    errs, _, _ = validate_raw(r)
    assert any(e["code"] == "invalid_action" for e in errs)


def test_button_not_found_invalid():
    r = F.good_6max_flop()
    r.button_seat = 99
    errs, _, _ = validate_raw(r)
    assert any(e["code"] == "button_seat_not_found" for e in errs)


def test_null_fields_produce_uncertainty_not_errors():
    r = F.good_6max_flop()
    r.pot = None
    r.hero_hole_cards = None
    errs, unc, _ = validate_raw(r)
    assert not any(e["code"] in ("pot_negative", "hero_cards_length") for e in errs)
    assert any(u.field == "pot" for u in unc)


# ---------------------------------------------------------------- assembly / position
def _frozen_positions(seats, button_seat):
    from poker_decision_ai.domain.poker.models import positions_for
    players = [{"seat": s["seat"], "name": s.get("anonymous_player_id")
                or f"p{s['seat']}", "stack": s.get("stack") or 0.0}
               for s in seats if s.get("seat") is not None]
    return positions_for(players, button_seat)


def test_position_from_deterministic_derivation():
    out = assemble(F.good_6max_flop())
    expected = _frozen_positions(out.players, 2)
    assert out.hero["position"] == expected[3]
    for p in out.players:
        assert p["position"] == expected[p["seat"]]


def test_button_missing_positions_all_null():
    out = assemble(F.button_missing())
    assert out.hero["position"] is None
    assert all(p["position"] is None for p in out.players)
    assert any(u["field"] == "button_seat" for u in out.uncertainty)


def test_uncertain_hero_card_propagates():
    out = assemble(F.hero_card_uncertain())
    assert out.validation.valid is True                    # 不确定 ≠ invalid
    assert out.hero["hole_cards"] == ["As", None]          # 不猜第二张
    assert any(u["field"] == "hero.hole_cards[1]"
               and u["status"] == "uncertain" for u in out.uncertainty)


def test_invalid_fixture_valid_flag_false():
    out = assemble(F.duplicate_card())
    assert out.validation.valid is False
    assert any(e["code"] == "duplicate_card" for e in out.validation.errors)


def test_action_history_passthrough():
    out = assemble(F.good_6max_flop())
    assert len(out.action_history) == 3
    assert out.action_history[0]["action"] == "fold"
    assert out.action_history[1]["order"] == 2


def test_assemble_does_not_mutate_raw():
    raw = F.good_6max_flop()
    snapshot = to_dict(raw)
    assemble(raw)
    assert to_dict(raw) == snapshot


# ---------------------------------------------------------------- fixtures / 红线
def test_all_fixtures_synthetic_test_only():
    for name, fn in F.ALL_FIXTURES.items():
        r = fn()
        assert r.synthetic_test_only is True, name


def test_fixture_registry_complete():
    assert len(F.ALL_FIXTURES) >= 11


def test_vision2_no_ocr_cv_ml_imports():
    import poker_decision_ai.vision2 as v2
    pkg = Path(v2.__file__).parent
    for p in pkg.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        checks = ("cv2", "pytesseract", "easyocr", "paddleocr",
                  "torchvision", "sklearn", "torch", "PIL")
        if p.name == "capture.py":   # V2N 只读截图 API（Pillow ImageGrab）合法
            checks = tuple(b for b in checks if b != "PIL")
        for bad in checks:
            assert bad not in src, f"{p} 引用了禁止库 {bad}"


def test_vision2_no_phase7_8_imports():
    import poker_decision_ai.vision2 as v2
    pkg = Path(v2.__file__).parent
    for p in pkg.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        for bad in ("phase7", "phase8", "phase6", "phase9"):
            assert bad not in src, f"{p} 引用了冻结阶段 {bad}"


def test_hero_cards_length_must_be_two_when_present():
    r = F.good_6max_flop()
    r.hero_hole_cards = ["As"]
    errs, _, _ = validate_raw(r)
    assert any(e["code"] == "hero_cards_length" for e in errs)
