"""Vision V2K — 状态组装与单元解析测试。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.assets import UNIT_CHARS  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.backends import (  # noqa: E402
    _parse_gold_value)
from poker_decision_ai.vision2.pixel.realclient.state import (  # noqa: E402
    assemble_visual_state, street_from_board)
from poker_decision_ai.vision2.realtime.recognition import (  # noqa: E402
    OCRResult, OCRStatus)


def test_parse_gold_value_thousands():
    assert _parse_gold_value("361,860") == (361860, None)


def test_parse_gold_value_units():
    assert _parse_gold_value("1.75M") == (1750000.0, "M")
    assert _parse_gold_value("1.03K") == (1030.0, "K")
    assert _parse_gold_value("2B") == (2000000000.0, "B")


def test_parse_gold_value_unparseable():
    assert _parse_gold_value("1.5") == (None, None)     # 无单位小数 → 不猜
    assert _parse_gold_value("abc") == (None, None)
    assert _parse_gold_value("") == (None, None)


def test_unit_chars_present():
    assert set(UNIT_CHARS) == {"M", "K", "B"}


def test_street_from_board_deterministic():
    assert street_from_board(None) is None
    assert street_from_board([]) is None
    assert street_from_board(["a", "b"]) is None          # 2 张 → 不确定
    assert street_from_board(["a", "b", "c"]) == "flop"
    assert street_from_board(["a", "b", "c", "d"]) == "turn"
    assert street_from_board(["a", "b", "c", "d", "e"]) == "river"


def _res(text, value, unit):
    return OCRResult(value=value, raw_text=text, confidence=0.9,
                     status=OCRStatus.OK, error=None,
                     synthetic_test_only=False, extra={"unit": unit})


def test_assemble_null_discipline_and_provenance():
    st = {0: _res("157,530", 157530, None), 3: _res("361,860", 361860, None)}
    state = assemble_visual_state(
        1, "t", stacks=st, bets={}, folded={}, all_in={},
        board_cards=None, button_present=False, button_bbox=None,
        hero_cards=None)
    norm = state["normalized"]
    assert norm.pot == {} or norm.pot.get("value") is None   # pot 恒 null
    # seats：未识别座位 stack=null
    players = norm.players
    seats = {p.get("seat"): p for p in players}
    assert seats.get(0)["stack"] == 157530
    assert seats.get(1)["stack"] is None
    # provenance 可追溯
    prov = {p["field"]: p for p in state["provenance"]}
    assert prov["player_0.stack"]["value"] == 157530
    assert prov["player_0.stack"]["evidence_frame_ids"] == [1]
    assert prov["player_0.stack"]["roi"] == "player_0_stack_roi"


def test_assemble_position_not_computed():
    """button_seat 未知 → position 不产生（视觉层不猜 position）。"""
    state = assemble_visual_state(
        1, "t", stacks={}, bets={}, folded={}, all_in={},
        board_cards=None, button_present=False, button_bbox=None,
        hero_cards=None)
    for p in state["normalized"].players:
        assert "position" not in p or p.get("position") is None


def test_state_module_redlines():
    src = (Path(__file__).parent.parent
           / "poker_decision_ai" / "vision2" / "pixel" / "realclient"
           / "state.py").read_text(encoding="utf-8")
    banned = ("pyautogui", "pynput", "recommend", "exploit", "gto",
              "bluff", "villain", "cv2", "torch", "sklearn")
    for bad in banned:
        assert bad not in src, f"含禁止 token {bad}"
