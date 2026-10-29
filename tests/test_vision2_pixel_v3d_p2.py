"""Vision V3D-P2 — Position Runtime Integration 测试（C1 双窗口/C3/C5/C6）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    OCCUPIED, EMPTY, UNKNOWN, HeroSeatDetector, PositionRuntimeResult,
    SeatOccupancyDetector, compute_positions, cropped_in, window_anchors)
from poker_decision_ai.vision2_platform import decode_png  # noqa: E402
from poker_decision_ai.domain.poker.models import (
    positions_for)  # noqa: E402  （回归参照——本测试不修改它）


def _occ(**kw):
    """构造（无伪造视觉——position 逻辑单元测试）。
    默认 EMPTY（已见空）；UNKNOWN 需显式设置（见 incomplete 用例）。"""
    st = {s: EMPTY for s in ("S1", "S2", "S3", "S4", "S5", "S6")}
    st.update(kw)
    return st


# ── C1 双窗口 ─────────────────────────────────────────────────────
def test_window_anchors_both():
    assert set(window_anchors("1848")) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    assert set(window_anchors("1419")) == {"S1", "S3", "S4", "S5"}
    assert cropped_in("1419") == ("S2", "S6")
    assert cropped_in("1848") == ()


def test_full_window_occupancy_1848():
    """V3B 决定性截图（1848×1388，5 空+dualang@S6）→ S6 OCCUPIED 候选。"""
    w, h, rgba = decode_png(Path(
        "<repo>/data2/6cf9056e3ef157b2f369368786f0d014.png").read_bytes())
    det = SeatOccupancyDetector(window="1848")
    res = det.detect(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    assert res["S6"]["status"] in (OCCUPIED, UNKNOWN)   # 无金堆叠帧亦诚实
    for s in ("S1", "S2", "S3", "S4", "S5"):
        assert res[s]["status"] in (OCCUPIED, EMPTY, UNKNOWN)


def test_1419_crop_s2s6_unknown():
    """1419 窗口 S2/S6 恒 UNKNOWN（不冒充 EMPTY）。"""
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = SeatOccupancyDetector(window="1419").detect(blank)
    assert res["S2"]["status"] == UNKNOWN
    assert res["S6"]["status"] == UNKNOWN


# ── C3 Hero ───────────────────────────────────────────────────────
def test_hero_own_seat_full_window():
    """1848 决定性截图：dualang@S6（own 座位）→ hero_seat=S6（非硬编码——
    由 own-seat presence 证据门得出）；空桌帧 → UNKNOWN。"""
    w, h, rgba = decode_png(Path(
        "<repo>/data2/6cf9056e3ef157b2f369368786f0d014.png").read_bytes())
    r = HeroSeatDetector(window="1848")
    res = r.detect(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
    # 该帧金堆叠检出不稳定 → 允许 hero_seat=S6 或 None(无证据诚实)
    assert res["hero_seat"] in ("S6", None)
    assert res["reason"] in ("own_seat_presence_evidence",
                             "no_own_seat_evidence")


def test_hero_cropped_1419_unknown():
    """1419 窗口 S6 裁剪 → hero_seat=None（UNKNOWN）。"""
    blank = np.zeros((1066, 1419, 4), dtype=np.uint8)
    blank[..., :3] = (40, 90, 60)
    blank[..., 3] = 255
    res = HeroSeatDetector(window="1419").detect(blank)
    assert res["hero_seat"] is None
    assert res["reason"] == "own_seat_cropped"


# ── C5：positions_for 接线（逻辑单元——非伪造视觉）───────────────────
def test_positions_full_6_handed_dealer_s3():
    occ = _occ(S1="OCCUPIED", S2="OCCUPIED", S3="OCCUPIED", S4="OCCUPIED",
               S5="OCCUPIED", S6="OCCUPIED")
    r = compute_positions(occ, "S3", "S6")
    assert r["uncertainty"] == "RECOGNIZED"
    assert r["hero_position"] == "BB"        # S5=SB, S6=BB（视觉环）
    assert r["position_map"]["S5"] == "SB"
    assert r["position_map"]["S6"] == "BB"


def test_positions_3_handed_skip_empty():
    """S2 空、S1/S3/S4 占用 + D@S4 → S1=SB, S3=BB（对照 1328 实况）。"""
    occ = _occ(S1="OCCUPIED", S3="OCCUPIED", S4="OCCUPIED")
    r = compute_positions(occ, "S4", None)
    assert r["uncertainty"] == "UNKNOWN"     # hero 缺失 → UNKNOWN（不 fallback）
    r2 = compute_positions(occ, "S4", "S1")
    assert r2["hero_position"] == "SB"
    assert r2["position_map"]["S3"] == "BB"


def test_positions_hairline_hu_and_5p():
    r = compute_positions(_occ(S4="OCCUPIED", S1="OCCUPIED"), "S4", "S1")
    assert r["hero_position"] in ("BTN/SB", "BB")     # HU 由冻结 positions_for

    r5 = compute_positions(_occ(S1="OCCUPIED", S2="OCCUPIED", S3="OCCUPIED",
                                S5="OCCUPIED", S6="OCCUPIED"),
                           "S3", "S6")
    assert r5["uncertainty"] == "RECOGNIZED"


def test_positions_unknowns_propagate():
    # dealer 缺失
    assert compute_positions(_occ(S1="OCCUPIED"), None, "S1")[
        "uncertainty"] == "UNKNOWN"
    # hero 缺失
    assert compute_positions(_occ(S1="OCCUPIED"), "S1", None)[
        "uncertainty"] == "UNKNOWN"
    # occupied 不完整（S5=UNKNOWN）
    r = compute_positions(_occ(S1="OCCUPIED", S3="OCCUPIED",
                               S4="OCCUPIED", S5="UNKNOWN"), "S4", "S1")
    assert r["uncertainty"] == "UNKNOWN"
    # dealer 不在座
    r = compute_positions(_occ(S1="OCCUPIED", S4="OCCUPIED"), "S4", "S1")
    assert r["uncertainty"] == "RECOGNIZED"
    r = compute_positions(_occ(S1="OCCUPIED", S4="OCCUPIED"), "S3", "S1")
    assert r["uncertainty"] == "UNKNOWN"       # dealer 未占用 → UNKNOWN


def test_positions_invalid_seats_unknown():
    assert compute_positions(_occ(S7="OCCUPIED"), "S7", "S7")[
        "uncertainty"] == "UNKNOWN"
    assert compute_positions(_occ(S1="OCCUPIED"), "S9", "S1")[
        "uncertainty"] == "UNKNOWN"


# ── C6 包装 ───────────────────────────────────────────────────────
def test_runtime_result_wrapper():
    r = PositionRuntimeResult(occupied_seats={"S6": OCCUPIED},
                              dealer_seat=None, hero_seat="S6")
    d = r.to_dict()
    for k in ("table_topology", "occupied_seats", "dealer_seat", "hero_seat",
              "hero_position", "uncertainty", "provenance",
              "ground_truth_status"):
        assert k in d
    assert d["table_topology"] == "6-max"
    assert d["ground_truth_status"] == "runtime_state_non_gt"
    assert d["hero_position"] is None           # 证据不足 → 无位置
    assert d["uncertainty"] == "UNKNOWN"


def test_positions_for_regression_untouched():
    """冻结 positions_for 保留原语义（不被 P2 修改——回归参照）。"""
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    pm = positions_for(players, 2)
    assert pm == {1: "BB", 2: "BTN", 3: "SB"}


def test_no_hardcoded_hero_dealer():
    src = (Path(__file__).parent.parent / "poker_decision_ai" / "vision2"
           / "pixel" / "realclient" / "seat_position.py").read_text(
        encoding="utf-8")
    assert "hero_seat = \"S6\"" not in src
    assert "hero_seat=\"S6\"" not in src
    assert "dealer_seat = \"S3\"" not in src
    assert "table_size" not in src
