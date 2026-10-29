"""Vision V2I — RealClientLayoutProfile 几何校准测试。

覆盖：碰撞=0、provenance 完整、normalized 坐标、逐座覆盖（未测量=null）、
pot 与各对象零碰撞、源码红线。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.layout import (  # noqa: E402
    RealClientLayoutProfile, ROIEntry)


def _profile():
    return RealClientLayoutProfile.measured()


def test_collision_zero():
    rep = _profile().collision_report()
    assert rep["collision_count"] == 0, rep["collisions"]
    # 唯一允许的 Z 序覆盖：call 面板叠牌面下缘（UI 事实）
    assert all(set(p[:2]) == {"hero_cards", "action_call"}
               for p in rep["z_order_overlays"])


def test_provenance_fields_present():
    p = _profile()
    d = p.to_dict()["rois"]["pot"]
    for k in ("semantic_name", "coordinate_system", "bbox", "source",
              "calibration_version", "confidence", "evidence_frame_id"):
        assert k in d, k
    assert d["bbox"]["normalized"]["x"] < 1.0          # 归一化存在
    assert 0.0 <= d["bbox"]["normalized"]["width"] <= 1.0


def test_pot_disjoint_from_all_objects():
    p = _profile()
    pot = p.entries["pot"].bbox
    for n, e in p.entries.items():
        if n in ("pot", "table") or n.startswith("player_"):
            continue
        if e.bbox is None:
            continue
        overlap = not (pot.x + pot.width <= e.bbox.x or
                       e.bbox.x + e.bbox.width <= pot.x or
                       pot.y + pot.height <= e.bbox.y or
                       e.bbox.y + e.bbox.height <= pot.y)
        assert not overlap, f"pot 与 {n} 重叠"


def test_seat_coverage_honest():
    cov = _profile().seat_coverage()
    assert cov["total_seats"] == 6
    assert cov["measured_seats"] == 1
    assert cov["measured"] == ["player_0"]
    assert "player_1" in cov["unmeasured"]          # 未测量 → 如实 null


def test_unmeasured_seat_null_bbox():
    p = _profile()
    assert p.entries["player_1"].bbox is None
    assert p.entries["player_1"].confidence is None
    d = p.entries["player_1"].to_dict()
    assert d["bbox"] is None


def test_roi_entry_null_bbox_dict():
    e = ROIEntry("x", "client", None, "unmeasured", 1, None)
    assert e.to_dict()["bbox"] is None


def test_layout_source_redlines():
    src = (Path(__file__).parent.parent
           / "poker_decision_ai" / "vision2" / "pixel" / "realclient"
           / "layout.py").read_text(encoding="utf-8")
    banned = ("pyautogui", "pynput", "selenium", "playwright",
              "expected_value", "recommend", "exploit", "gto",
              "bluff", "villain", "cv2", "pytesseract", "torch", "sklearn")
    for bad in banned:
        assert bad not in src, f"含禁止 token {bad}"
