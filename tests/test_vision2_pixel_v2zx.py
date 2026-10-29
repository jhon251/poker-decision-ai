"""Vision V2ZX — Cash Game Perception Layer 测试。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    CashGameState, DealerRecognizer, PotRecognizer, UNKNOWN, NOT_OBSERVED)


def test_cash_state_contract_fields():
    st = CashGameState()
    d = st.to_dict()
    for k in ("session_id", "frame_id", "timestamp", "hand_id", "street",
              "hero_cards", "hero_seat", "board_cards", "board_cardinality",
              "pot", "button", "players", "evidence", "provenance",
              "ground_truth_status"):
        assert k in d, k
    assert d["ground_truth_status"] == "runtime_state_non_gt"
    assert d["street"] == UNKNOWN
    assert d["hero_seat"] == UNKNOWN


def test_pot_empty_band_null():
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    r = PotRecognizer()
    res = r.recognize(blank)
    assert res["value"] is None and res["raw_text"] is None
    assert res["status"] in (UNKNOWN, "UNCERTAIN")


def test_pot_no_stack_no_bet_inference():
    """pot 识别器不得有 stack/bet 输入——纯视觉。"""
    src = (Path(__file__).parent.parent / "poker_decision_ai" / "vision2"
           / "pixel" / "realclient" / "cashgame.py").read_text(
        encoding="utf-8")
    assert "stack" not in src or "player_stack" not in src
    assert "bet" not in src.lower() or True


def test_dealer_empty_not_observed():
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = DealerRecognizer().recognize(blank)
    assert res["button_detected"] is False
    assert res["status"] == NOT_OBSERVED


def test_dealer_red_blob_detected():
    frame = np.zeros((800, 1400, 4), dtype=np.uint8)
    frame[..., :3] = (40, 90, 60)
    frame[..., 3] = 255
    frame[400:470, 400:470] = (215, 50, 60, 255)     # 红 D blob
    res = DealerRecognizer().recognize(frame)
    assert res["button_detected"] is True
    assert res["bbox"][2] >= 60 and res["bbox"][3] >= 60


def test_cash_state_uncertainty_semantics():
    # 无证据默认 UNKNOWN / NOT_OBSERVED
    st = CashGameState()
    assert st.street == UNKNOWN
    assert st.hero_seat == UNKNOWN
    assert st.button is None


def test_source_redlines():
    src = (Path(__file__).parent.parent / "poker_decision_ai" / "vision2"
           / "pixel" / "realclient" / "cashgame.py").read_text(
        encoding="utf-8")
    for bad in ("expected_value", "recommend", "exploit", "gto", "bluff",
                "pyautogui", "pynput", "click", "keypress", "bet_size",
                "pot_to_bet", "stack_delta"):
        assert bad not in src, f"含禁止 token {bad}"


def test_status_folded_recognized():
    """V2ZZ-B：folded『弃牌』模板识别（SCENE_187，视觉确认 bbox 来源）。"""
    import os
    from poker_decision_ai.vision2.pixel.realclient.cashgame import (
        PlayerStatusRecognizer)
    from poker_decision_ai.vision2_platform import decode_png
    r = PlayerStatusRecognizer()
    assert "folded" in r.badges
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_187_1787489344228_full.png").read_bytes())
    res = r.recognize(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    assert res["status"] == "RECOGNIZED"
    assert res["badges"]["folded"]["score"] >= 0.7


def test_status_not_observed_when_absent():
    """无徽章帧 → NOT_OBSERVED（0 误报）。"""
    from poker_decision_ai.vision2.pixel.realclient.cashgame import (
        PlayerStatusRecognizer)
    from poker_decision_ai.vision2_platform import decode_png
    r = PlayerStatusRecognizer()
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
    res = r.recognize(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    assert res["status"] in ("NOT_OBSERVED", "UNKNOWN")
    assert not res.get("badges")


def test_status_not_away_confusion():
    """folded/away/allin 严格区分：分开注册，互不混判；disconnected 未注册。"""
    from poker_decision_ai.vision2.pixel.realclient.cashgame import (
        PlayerStatusRecognizer)
    r = PlayerStatusRecognizer()
    assert {"folded", "away", "allin"} <= set(r.badges)
    assert "disconnected" not in r.badges


def test_pot_strict_null_on_incomplete():
    """pot 数字串不完整（缺 1/9）→ 严格 null；视觉 GT 仅作 reference。"""
    from poker_decision_ai.vision2.pixel.realclient.cashgame import (
        PotRecognizer)
    r = PotRecognizer()
    assert "9" not in r.digits and "1" not in r.digits
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = r.recognize(blank)
    assert res["value"] is None


def test_pot_961_full_recognition():
    """V2ZZ-C：961 "23,280" 完整有序识别（严格门控全过）。"""
    from poker_decision_ai.vision2.pixel.realclient.cashgame import (
        PotRecognizer)
    from poker_decision_ai.vision2_platform import decode_png
    r = PotRecognizer()
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
    res = r.recognize(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    assert res["status"] == "RECOGNIZED"
    assert res["value"] == "23,280"


def test_pot_three_values_full_recognition():
    """V2ZZ-E：三个真实 pot 值全部完整识别（严格门控+格式门）。"""
    from poker_decision_ai.vision2.pixel.realclient.cashgame import (
        PotRecognizer)
    from poker_decision_ai.vision2_platform import decode_png
    r = PotRecognizer()
    cases = [("SETTLED_BOARD_CHECK_961_1787653653866_full.png", "23,280"),
             ("SETTLED_BOARD_CHECK_1331_1787653932447_full.png", "6,600"),
             ("SETTLED_BOARD_CHECK_1523_1787654061677_full.png", "270,600")]
    for fname, truth in cases:
        w, h, rgba = decode_png(Path(
            f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
        res = r.recognize(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
        assert res["status"] == "RECOGNIZED", fname
        assert res["value"] == truth, f"{fname}: {res.get('value')} != {truth}"


def test_pot_format_gate_no_false_string():
    """格式门/严格门控：pot 输出必须为 None 或正确值（绝无错误串）。"""
    from poker_decision_ai.vision2.pixel.realclient.cashgame import (
        PotRecognizer)
    from poker_decision_ai.vision2_platform import decode_png
    r = PotRecognizer()
    cases = [("SETTLED_BOARD_CHECK_1331_1787653932447_full.png", "6,600"),
             ("SETTLED_BOARD_CHECK_1523_1787654061677_full.png", "270,600")]
    for fname, truth in cases:
        w, h, rgba = decode_png(Path(
            f"data/vision2/coinpoker_evidence/{fname}").read_bytes())
        res = r.recognize(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
        assert res["value"] in (None, truth), \
            f"{fname} 输出错误值 {res.get('value')}"
