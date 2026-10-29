"""Vision V2E — 最小标注工具链测试。

覆盖：角色词表、簇级标注应用（校验/拒绝/覆盖保护/审计字段）、标注资产
流入 labeled_templates 与后端、null 纪律、脚本红线扫描。

真实截图未标注期间：任何标注 API 不得猜测；semantic_label 无证据保持 null。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.fonts import GLYPHS_5X7, _render  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    ROLE_VOCAB, RealClientCardRecognitionBackend, RealClientDigitOCRBackend,
    RealClientTemplate, RealClientTemplateLibrary, apply_cluster_labels,
    label_vocab_for)
from poker_decision_ai.vision2.realtime.recognition import OCRStatus  # noqa: E402


def _asset(tid, kind, roi, cluster, label=None, status="unlabeled_candidate"):
    return RealClientTemplate(
        template_id=tid, kind=kind, semantic_label=label,
        source_frame="shot.png", source_roi=roi, client="test",
        crop_bbox={"x": 0.0, "y": 0.0, "width": 5.0, "height": 7.0},
        ground_truth_status=status, cluster_id=cluster)


def _role_library():
    lib = RealClientTemplateLibrary(client="test")
    for i in range(3):
        lib.add(_asset(f"p{i}", "digit_glyph", "pot", 7))
    for i in range(2):
        lib.add(_asset(f"c{i}", "card_glyph", "hero_cards", 8))
    lib.add(_asset("u1", "digit_glyph", "stacks", None))
    from poker_decision_ai.vision2.pixel.realclient import assign_role_labels
    assign_role_labels(lib)
    return lib


# ------------------------------------------------------------------ 词表
def test_role_vocab_complete():
    from poker_decision_ai.vision2.pixel.realclient.roles import ROLE_TABLE
    # ROLE_VOCAB 的每个键必须来自 ROLE_TABLE 的已知角色；词表非空
    table_roles = set(ROLE_TABLE.values())
    for role in ROLE_VOCAB:
        assert role in table_roles, role
        assert ROLE_VOCAB[role]


def test_slot_role_not_labelable():
    lib = _role_library()
    lib.add(_asset("s1", "card_slot", "hero_cards", 30))
    from poker_decision_ai.vision2.pixel.realclient import assign_role_labels
    assign_role_labels(lib)
    report = apply_cluster_labels(lib, {30: "A"}, label_source="human_review")
    assert report.applied_assets == 0
    assert any(r["reason"].startswith("role_not_labelable")
               for r in report.rejected)


def test_label_vocab_for():
    assert "0" in label_vocab_for("pot_digit_role")
    assert "A" in label_vocab_for("hero_card_glyph_role")
    assert "s" in label_vocab_for("board_card_glyph_role")
    assert label_vocab_for("unknown_role") is None


# ------------------------------------------------------------------ 应用
def test_apply_valid_cluster_label():
    lib = _role_library()
    report = apply_cluster_labels(lib, {7: "4"}, label_source="human_review",
                                  confidence=0.95)
    assert report.applied_clusters == [7]
    assert report.applied_assets == 3
    for a in lib.by_kind("digit_glyph"):
        if a.cluster_id == 7:
            assert a.semantic_label == "4"
            assert a.ground_truth_status == "visually_labeled"
            assert a.label_source == "human_review"
            assert a.confidence == 0.95


def test_apply_invalid_label_rejected():
    lib = _role_library()
    report = apply_cluster_labels(lib, {7: "X"}, label_source="human_review")
    assert report.applied_assets == 0
    assert any(r["reason"].startswith("label_not_in_vocab")
               for r in report.rejected)
    assert all(a.semantic_label is None for a in lib.assets.values())


def test_apply_unknown_role_cluster_rejected():
    lib = _role_library()
    report = apply_cluster_labels(lib, {9: "4"}, label_source="human_review")
    assert report.applied_assets == 0
    assert any(r["reason"] == "cluster_role_unknown" for r in report.rejected)


def test_apply_overwrite_protected_and_force():
    lib = _role_library()
    apply_cluster_labels(lib, {7: "4"}, label_source="human_review")
    r2 = apply_cluster_labels(lib, {7: "5"}, label_source="human_review")
    assert r2.applied_assets == 0
    assert any(r["reason"] == "already_labeled_without_force"
               for r in r2.rejected)
    r3 = apply_cluster_labels(lib, {7: "5"}, label_source="human_review",
                              force=True)
    assert r3.applied_assets == 3
    assert lib.get("p0").semantic_label == "5"


def test_apply_does_not_touch_other_clusters():
    lib = _role_library()
    apply_cluster_labels(lib, {7: "4"}, label_source="human_review")
    for a in lib.assets.values():
        if a.cluster_id != 7:
            assert a.semantic_label is None        # null 纪律


def test_report_to_dict():
    lib = _role_library()
    report = apply_cluster_labels(lib, {7: "4"}, label_source="human")
    d = report.to_dict()
    assert set(d) == {"applied_clusters", "applied_assets", "rejected"}
    assert d["applied_assets"] == 3


# ------------------------------------------------------------------ 流入识别
def test_labeled_cluster_flows_to_backend():
    from poker_decision_ai.vision2.pixel.realclient import assign_role_labels
    lib = _role_library()
    for ch in "0123456789.,:":
        lib.add(_asset(f"tpl_{ch}", "digit_glyph", "pot", 100 + ord(ch)),
                _render(GLYPHS_5X7[ch], 1))
    assign_role_labels(lib)                    # 新模板资产补角色
    apply_cluster_labels(
        lib, {100 + ord(ch): ch for ch in "0123456789.,:"},
        label_source="test")
    backend = RealClientDigitOCRBackend(lib)
    assert backend.available() is True
    parts = [_render(GLYPHS_5X7[ch], 1) for ch in "42.5"]
    w = sum(p.width for p in parts) + 4 * (len(parts) + 1)
    h = max(p.height for p in parts) + 8
    buf = bytearray(bytes((0, 0, 0, 255)) * (w * h))
    ox = 4
    for p in parts:
        for y in range(p.height):
            st = ((y + 4) * w + ox) * 4
            buf[st:st + p.width * 4] = p.data[y * p.width * 4:(y + 1) * p.width * 4]
        ox += p.width + 4
    from poker_decision_ai.vision2.realtime.image import ImageData
    res = backend.recognize(ImageData.from_rgba_bytes(w, h, bytes(buf)),
                            "table_meta", "number")
    assert res.status is OCRStatus.OK and res.value == 42.5


def test_unlabeled_library_backend_still_unavailable():
    lib = _role_library()
    backend = RealClientDigitOCRBackend(lib)
    assert backend.available() is False
    assert backend.unavailable_reason()


# ------------------------------------------------------------------ 红线
def test_v2e_scripts_redlines():
    scripts = ("phase_v2e_label_assets.py", "phase_v2e_contact_sheet.py",
               "phase_v2e_eval_recognition.py")
    banned = ("pyautogui", "pynput", "selenium", "playwright", "pywinauto",
              "win32", "SendInput", "mouse_event", "keybd_event",
              "SetCursorPos", "PostMessage", "ctypes",
              "expected_value", "recommend", "exploit", "gto",
              "profitability", "equity", "villain", "bluff",
              "phase6", "phase7", "phase8", "phase9",
              "cv2", "pytesseract", "PIL", "torch", "sklearn")
    base = Path(__file__).parent.parent / "scripts"
    for name in scripts:
        src = (base / name).read_text(encoding="utf-8")
        for bad in banned:
            assert bad not in src, f"{name} 含禁止 token {bad}"
