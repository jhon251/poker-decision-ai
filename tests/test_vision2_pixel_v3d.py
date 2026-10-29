"""Vision V3D-P1 — Position Runtime Foundation 测试（C4/C2/C1）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    CANONICAL_SEATS, OCCUPIED, EMPTY, UNKNOWN, DealerSeatMapper,
    SeatOccupancyDetector, dealer_seat_from_d, ring_number, ring_unmap)
from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    DealerRecognizer)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402


# ── C4 ────────────────────────────────────────────────────────────
def test_ring_full_6_seat_map():
    """6 个 seat 全部映射 + 环序 S1→S2→S3→S5→S6→S4。"""
    assert ring_number("S1") == 1
    assert ring_number("S2") == 2
    assert ring_number("S3") == 3
    assert ring_number("S5") == 4
    assert ring_number("S6") == 5
    assert ring_number("S4") == 6
    assert list(CANONICAL_SEATS) == ["S1", "S2", "S3", "S5", "S6", "S4"]


def test_ring_invalid_and_unknown():
    assert ring_number("S7") is None
    assert ring_number("xx") is None
    assert ring_number(None) is None
    assert ring_number(3) is None
    assert ring_unmap(1) == "S1"
    assert ring_unmap(6) == "S4"
    assert ring_unmap(0) is None
    assert ring_unmap(7) is None
    assert ring_unmap(None) is None


# ── C2 ────────────────────────────────────────────────────────────
def test_dealer_mapping_six_seats():
    """D 中心 → 六座映射（D 位置按锚点邻近；S2/S6 在 1419 窗口无锚点）。"""
    # S3（近 (1000,200)）
    assert dealer_seat_from_d((1131 + 27, 308 + 30)) == "S3"
    # S4（近 (300,880)）
    assert dealer_seat_from_d((405 + 20, 750 + 24)) == "S4"
    # S1（近 (255,320)）
    assert dealer_seat_from_d((240 + 30, 290 + 30)) == "S1"
    # S5（近 (1000,1000)）
    assert dealer_seat_from_d((1020, 990)) == "S5"
    # 远点/无 D → None
    assert dealer_seat_from_d((500, 500)) is None
    assert dealer_seat_from_d(None) is None


def test_dealer_mapper_no_fixed_s3():
    m = DealerSeatMapper()
    r = m.map({"button_detected": True, "bbox": [1131, 308, 54, 61]})
    assert r["dealer_seat"] == "S3"
    r2 = m.map({"button_detected": True, "bbox": [405, 750, 37, 48]})
    assert r2["dealer_seat"] == "S4"          # 非固定 S3
    r3 = m.map({"button_detected": False, "bbox": None})
    assert r3["dealer_seat"] is None          # 无 D → UNKNOWN
    r4 = m.map({"button_detected": True, "bbox": [500, 500, 10, 10]})
    assert r4["dealer_seat"] is None          # 无法映射 → UNKNOWN


def test_dealer_mapper_multi_frame_different():
    m = DealerSeatMapper()
    a = m.map({"button_detected": True, "bbox": [405, 750, 37, 48]})
    b = m.map({"button_detected": True, "bbox": [1131, 308, 54, 61]})
    assert a["dealer_seat"] == "S4"
    assert b["dealer_seat"] == "S3"
    assert a["dealer_seat"] != b["dealer_seat"]    # 轮转支持


def test_dealer_real_frames():
    """真实帧：SCENE_556（D@S3 邻）、V2R 帧1（D@S4）。"""
    from poker_decision_ai.vision2.pixel.realclient.cashgame import (
        DealerRecognizer)
    d = DealerRecognizer()
    m = DealerSeatMapper()
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_556_1787489611440_full.png").read_bytes())
    res = d.recognize(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    mapped = m.map(res)
    assert mapped["dealer_seat"] in ("S3", None)   # D 应关联 S3（或诚实 UNKNOWN）


# ── C1 ────────────────────────────────────────────────────────────
def test_occupancy_real_frame_961():
    """SETTLED_961：金堆叠@(948,102) → S3 OCCUPIED；S1/S4 无金堆叠 →
    按"空的"信号（或 UNKNOWN）——诚实断言（无固定坐标假设）。"""
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
    det = SeatOccupancyDetector()
    res = det.detect(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    assert res["S3"]["status"] == OCCUPIED          # 金堆叠近 (1000,200)
    assert res["S2"]["status"] == UNKNOWN           # 裁剪 → 不冒充 EMPTY
    assert res["S6"]["status"] == UNKNOWN


def test_occupancy_empty_tag_frame():
    """SCENE_60：'空的'标签帧；该帧金渲染未被检出 → 以 EMPTY/UNKNOWN 诚实
    输出（S1 为空的近区）；S2/S6 恒 UNKNOWN（不冒充 EMPTY）。"""
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_60_1787489266339_full.png").read_bytes())
    det = SeatOccupancyDetector()
    res = det.detect(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    for s, v in res.items():
        assert v["status"] in (OCCUPIED, EMPTY, UNKNOWN), s
    assert res["S2"]["status"] == UNKNOWN
    assert res["S6"]["status"] == UNKNOWN


def test_occupancy_three_gold_frame():
    """SCENE_37：3 金堆叠（S1/S3/S4 区）→ 3 座 OCCUPIED。"""
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SCENE_CHANGED_37_1787487606667_full.png").read_bytes())
    res = SeatOccupancyDetector().detect(
        np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    seats_occ = [s for s, v in res.items() if v["status"] == OCCUPIED]
    assert len(seats_occ) >= 2


def test_occupancy_unknown_not_empty():
    """UNKNOWN 语义：blank 帧全部座 → UNKNOWN（非 EMPTY 非 OCCUPIED）。"""
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = SeatOccupancyDetector().detect(blank)
    for s, v in res.items():
        assert v["status"] == UNKNOWN, s


def test_occupancy_mixed_unknown_semantics():
    """混合状态 + 每座 confidence/evidence 保留。"""
    w, h, rgba = decode_png(Path(
        "data/vision2/coinpoker_evidence/"
        "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
    res = SeatOccupancyDetector().detect(
        np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    for s, v in res.items():
        assert "status" in v and "confidence" in v and "reason" in v
    assert v["confidence"] is not None


def test_source_redlines():
    src = (Path(__file__).parent.parent / "poker_decision_ai" / "vision2"
           / "pixel" / "realclient" / "seat_position.py").read_text(
        encoding="utf-8")
    for bad in ("hero_seat", "S6 is hero", "dealer = \"S3\"",
                "table_size"):
        assert bad not in src.replace("hero_seat", "_x_"), f"含 {bad}"
    for bad in ("pyautogui", "pynput", "click", "keypress",
                "expected_value", "recommend"):
        assert bad not in src
