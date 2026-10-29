"""Vision V2D — 角色级语义标注测试。

角色级（role/role_identified）由 ROI provenance 确定性推导；
字符级（semantic_label）保持 null——值识别后端仍恒 UNAVAILABLE。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.fonts import GLYPHS_5X7, _render  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientCardRecognitionBackend, RealClientDigitOCRBackend,
    RealClientTemplate, RealClientTemplateLibrary, ROLE_TABLE,
    assign_role_labels, role_for)
from poker_decision_ai.vision2.realtime.recognition import OCRStatus  # noqa: E402


def _asset(tid, kind, roi, label=None, status="unlabeled_candidate"):
    return RealClientTemplate(
        template_id=tid, kind=kind, semantic_label=label,
        source_frame="shot.png", source_roi=roi, client="test",
        crop_bbox={"x": 0.0, "y": 0.0, "width": 5.0, "height": 7.0},
        ground_truth_status=status)


# ------------------------------------------------------------------ 角色表
def test_role_table_mapping():
    assert role_for("digit_glyph", "pot") == "pot_digit_role"
    assert role_for("card_slot", "hero_cards") == "hero_card_slot_role"
    assert role_for("card_glyph", "board") == "board_card_glyph_role"
    assert role_for("disc_candidate", "table_meta") == "button_disc_role"
    assert role_for("region_snapshot", "board") == "board_region_role"
    assert role_for("digit_glyph", "unknown_roi") is None
    assert role_for("weird_kind", "pot") is None


def test_assign_role_labels_semantics():
    lib = RealClientTemplateLibrary(client="test")
    lib.add(_asset("d1", "digit_glyph", "pot"))
    lib.add(_asset("c1", "card_glyph", "hero_cards"))
    lib.add(_asset("u1", "digit_glyph", "unknown_roi"))
    n = assign_role_labels(lib)
    assert n == 2
    d1 = lib.get("d1")
    assert d1.role == "pot_digit_role"
    assert d1.ground_truth_status == "role_identified"
    assert d1.label_source == "roi_provenance"
    assert d1.semantic_label is None                 # 字符级保持 null
    assert d1.is_labeled is False                    # role 不开启值识别
    assert lib.get("u1").role is None
    assert lib.get("u1").ground_truth_status == "unlabeled_candidate"


def test_asset_validation_role_status():
    with pytest.raises(ValueError):
        RealClientTemplate(template_id="x", role="")
    a = RealClientTemplate(template_id="x", role="pot_digit_role",
                           ground_truth_status="role_identified")
    assert a.role == "pot_digit_role"


def test_manifest_v2_roundtrip_role(tmp_path):
    lib = RealClientTemplateLibrary(client="test", version=2)
    lib.add(_asset("d1", "digit_glyph", "pot"))
    assign_role_labels(lib)
    path = tmp_path / "manifest.json"
    lib.to_json(str(path))
    lib2 = RealClientTemplateLibrary.from_json(str(path))
    a = lib2.get("d1")
    assert a.role == "pot_digit_role"
    assert a.ground_truth_status == "role_identified"
    assert a.asset_version == 2


# ------------------------------------------------------------------ 后端角色状态
def test_backend_role_status_honest():
    lib = RealClientTemplateLibrary(client="test")
    lib.add(_asset("d1", "digit_glyph", "pot"))
    assign_role_labels(lib)
    backend = RealClientDigitOCRBackend(lib)
    assert backend.available() is False
    st = backend.role_status()
    assert st["known_roles"] == {"pot_digit_role": 1}
    assert st["character_labels_pending"] is True
    res = backend.recognize(None, "pot", "number")   # 不可用 → UNAVAILABLE
    # crop None → FAILED 优先；改用真实 crop 验证 UNAVAILABLE
    from poker_decision_ai.vision2.realtime.image import ImageData
    res2 = backend.recognize(ImageData.from_rgba_bytes(
        8, 8, bytes([0, 0, 0, 255]) * 64), "pot", "number")
    assert res2.status is OCRStatus.UNAVAILABLE


def test_card_backend_role_status_after_roles():
    lib = RealClientTemplateLibrary(client="test")
    lib.add(_asset("g1", "card_glyph", "hero_cards"))
    lib.add(_asset("g2", "card_glyph", "board"))
    assign_role_labels(lib)
    backend = RealClientCardRecognitionBackend(lib)
    st = backend.role_status()
    assert st["available"] is False
    assert st["character_labels_pending"] is True
    assert "hero_card_glyph_role" in st["known_roles"]


# ------------------------------------------------------------------ 工作清单
def test_worklist_generation(tmp_path):
    from collections import defaultdict
    from poker_decision_ai.vision2.pixel.realclient import role_for as _rf
    lib = RealClientTemplateLibrary(client="test", version=2)
    for i in range(6):
        lib.add(_asset(f"d{i}", "digit_glyph", "pot"))
    assign_role_labels(lib)
    clusters = defaultdict(list)
    for a in lib.assets.values():
        if a.cluster_id is not None:
            clusters.setdefault(a.cluster_id, []).append(a)
    # 手动模拟聚簇（无聚类时为单资产簇）
    for a in lib.assets.values():
        a.cluster_id = 0
    work = [{
        "cluster_id": a.cluster_id, "candidate_count": len(lib.assets),
        "roles": [a.role], "source_frames": [a.source_frame],
        "png_refs": [a.png_ref or ""], "status": "awaiting_visual_label"}
        for a in list(lib.assets.values())[:1]]
    assert work[0]["roles"] == ["pot_digit_role"]
    blob = json.dumps(work)
    assert "awaiting_visual_label" in blob
    assert "pot_digit_role" in blob
