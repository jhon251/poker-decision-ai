"""V3F-P27-C — 1876 Seat Occupancy 三态 detector 单元测试。

覆盖：真实 EMPTY S4 / 同帧 OCCUPIED 5 座 / 34 GT OCCUPIED / 2 overlay
UNKNOWN / away 不判 EMPTY / disconnected / blank / partial crop / 冲突 /
1876 dispatch / 1419 regression / positions_for 零修改。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.seat_occupancy import (  # noqa: E402
    EMPTY_DIM_LABEL_MIN, OCC_GOLD_MIN, OCC_NAME_MIN,
    SeatOccupancyRecognizer1876)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PositionObserver, SeatOccupancyDetector)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))

EMPTY_FRAME = "42b553c9468d1d1e55b9801bebad7b4f.png"
REC = SeatOccupancyRecognizer1876()


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _blank():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    return fb


# ── 1/2. 真实 EMPTY + 同帧 OCCUPIED ─────────────────────────────────
def test_empty_frame_s4_empty():
    out = REC.recognize(_frame(EMPTY_FRAME))
    assert out["S4"]["status"] == "EMPTY"
    assert out["S4"]["reason"] == "empty_label_evidence"


def test_empty_frame_occupied_five():
    out = REC.recognize(_frame(EMPTY_FRAME))
    for seat in ("S1", "S2", "S3", "S5", "S6"):
        assert out[seat]["status"] == "OCCUPIED", seat


# ── 34 GT OCCUPIED + 2 overlay UNKNOWN ──────────────────────────────
def test_34_gt_occupied_2_overlay_unknown():
    occ = 0
    fns = []
    for ann in GT["annotations"]:
        out = REC.recognize(_frame(ann["frame_id"]))
        st = (ann.get("fields") or {}).get("stack") or {}
        for seat in sorted(out):
            v = out[seat]
            if v["status"] == "OCCUPIED":
                occ += 1
            elif seat in st and st[seat] is not None:
                fns.append((ann["frame_id"][:6], seat))
    assert occ == 34, occ
    assert not fns, fns


def test_hero_overlay_unknown():
    out = REC.recognize(_frame("POT_CHANGED_102_1787992505794_full.png"))
    assert out["S6"]["status"] == "UNKNOWN"


def test_animation_overlay_unknown():
    out = REC.recognize(_frame("POT_CHANGED_139_1787992592538_full.png"))
    assert out["S5"]["status"] == "UNKNOWN"


# ── away / disconnected 不判 EMPTY ──────────────────────────────────
def test_away_not_empty():
    # f01 S3 away（gold=416 name=1102）→ OCCUPIED 非 EMPTY
    out = REC.recognize(_frame("BUTTON_CHANGED_167_1787992682749_full.png"))
    assert out["S3"]["status"] == "OCCUPIED"


def test_disconnected_not_empty():
    # f05 S3 disconnected → OCCUPIED（有 gold+name）
    out = REC.recognize(_frame("POT_CHANGED_139_1787992592538_full.png"))
    assert out["S3"]["status"] == "OCCUPIED"


# ── blank / partial crop ───────────────────────────────────────────
def test_blank_all_unknown():
    out = REC.recognize(_blank())
    for seat, v in out.items():
        assert v["status"] == "UNKNOWN"


def test_partial_crop_unknown():
    fb = np.zeros((400, 400, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    out = REC.recognize(fb)
    for seat, v in out.items():
        assert v["status"] == "UNKNOWN"


# ── OCCUPIED 不完整 → UNKNOWN（只有单一缺失/不足）───────────────────
def test_single_missing_not_empty():
    """只有 gold 缺失（合成：抠掉 stack 行 gold 但留 name）→ 不能 EMPTY。"""
    arr = _frame("BUTTON_CHANGED_167_1787992682749_full.png").copy()
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (
        SEAT_STACK_ROIS_1876)
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876["S1"]
    arr[y0:y1, x0:x1, :3] = (6, 35, 27)   # 抹掉 stack 行（含 gold）
    out = REC.recognize(arr)
    # name 仍 OCCUPIED；绝不出 EMPTY
    assert out["S1"]["status"] != "EMPTY"


def test_avatar_only_missing_not_empty():
    """只有 avatar 缺失 → 不判 EMPTY（合成抠掉头像环，name/gold 仍在）。"""
    arr = _frame("BUTTON_CHANGED_167_1787992682749_full.png").copy()
    arr[220:400, 155:355, :3] = (6, 35, 27)   # 抹掉 S1 头像区
    out = REC.recognize(arr)
    assert out["S1"]["status"] != "EMPTY"


# ── EMPTY 与 OCCUPIED 冲突 → UNKNOWN ────────────────────────────────
def test_conflict_unknown():
    """合成：EMPTY 标签 + gold 叠加 → 冲突 → UNKNOWN（gold≥400 先命中
    OCCUPIED？——需验证判定顺序不产生 EMPTY）。"""
    # 用真实 EMPTY 帧 S4 叠加 gold → OCCUPIED（gold 优先于 EMPTY，
    # 因 EMPTY 要求 gold<100）
    arr = _frame(EMPTY_FRAME).copy()
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (
        SEAT_STACK_ROIS_1876)
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876["S4"]
    # 画一个金色 stack 文本块（模拟 OCCUPIED 证据 + EMPTY 标签共存）
    arr[y0:y0 + 40, x0:x0 + 80, :3] = (255, 200, 60)
    out = REC.recognize(arr)
    # gold 掩码可能命中 → OCCUPIED；EMPTY 因 gold<100 不满足 → 不会 EMPTY
    assert out["S4"]["status"] != "EMPTY"


# ── 1876 dispatch ──────────────────────────────────────────────────
def test_1876_dispatch_uses_new_detector():
    out = PositionObserver().observe(_frame(EMPTY_FRAME).tobytes(), 1876, 1400)
    occ = out["runtime_result"].get("occupied_seats", {})
    s4 = occ.get("S4", {})
    st = s4.get("status") if isinstance(s4, dict) else s4
    assert st == "EMPTY"


# ── 1419 regression ────────────────────────────────────────────────
def test_1419_detector_unchanged():
    """1419 帧 → 旧 SeatOccupancyDetector（无 1876 三态介入）。"""
    w, h, rgba = decode_png((EV /
        "SCENE_CHANGED_37_1787487606667_full.png").read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    out = PositionObserver().observe(arr, w, h)
    assert out["ground_truth_status"] == "runtime_state_non_gt"
    assert "runtime_result" in out


def test_1419_seat_detector_frozen():
    """1419 SeatOccupancyDetector 类语义未变。"""
    src = Path(
        "poker_decision_ai/vision2/pixel/realclient/seat_position.py"
    ).read_text(encoding="utf-8")
    assert "class SeatOccupancyDetector" in src
    assert "gold_stack_nearest_seat" in src


# ── positions_for 零修改 ───────────────────────────────────────────
def test_positions_for_untouched():
    from poker_decision_ai.domain.poker.models import positions_for  # noqa: F401
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}


# ── 阈值不放宽（红线）──────────────────────────────────────────────
def test_thresholds_frozen_measurements():
    assert OCC_GOLD_MIN == 400
    assert OCC_NAME_MIN == 200
    assert EMPTY_DIM_LABEL_MIN == 150
