"""Vision V2C — 真实客户端视觉域适配测试。

覆盖：资产 provenance/null 纪律/标注门控、库版本化与 JSON 往返、
无监督聚类、真实截图候选提取（本机截图存在时）、识别后端严格门控
（无标注 → UNAVAILABLE；test 标注资产 → 机制验证，绝不冒充真实能力）、
V1 流水线接线、源码红线。

synthetic/test 资产仅用于机制测试（source="test"），真实识别效果必须
来自真实截图资产的已标注集（本阶段当前无标注者 → 恒 UNAVAILABLE）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.fonts import GLYPHS_5X7, _render  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    DIGITS, KINDS, RANKS, SUITS, RealClientCardRecognitionBackend,
    RealClientDigitOCRBackend, RealClientTemplate, RealClientTemplateLibrary,
    assign_clusters, extract_candidates)
from poker_decision_ai.vision2.realtime import (  # noqa: E402
    Rect, RealtimeVisionConfig, RealtimeVisionPipeline)
from poker_decision_ai.vision2.realtime.capture import (  # noqa: E402
    CapturedFrame, FixtureCaptureProvider)
from poker_decision_ai.vision2.realtime.coords import GeometryContext  # noqa: E402
from poker_decision_ai.vision2.realtime.frame_manager import Frame  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2.realtime.recognition import (  # noqa: E402
    OCRStatus, UnavailableCardBackend)
from poker_decision_ai.vision2.realtime.roi import (  # noqa: E402
    ROIManager, default_roi_specs)


# ------------------------------------------------------------------ helpers
def make_asset(template_id, kind, label=None, status="unlabeled_candidate",
               source="test", roi="pot"):
    return RealClientTemplate(
        template_id=template_id, kind=kind, semantic_label=label,
        source_frame=source, source_roi=roi, client="test",
        crop_bbox={"x": 0.0, "y": 0.0, "width": 5.0, "height": 7.0},
        ground_truth_status=status, label_source=source)


def digit_composite(text, scale=3, spacing=4):
    parts = [_render(GLYPHS_5X7[ch], scale) for ch in text]
    w = sum(p.width for p in parts) + spacing * (len(parts) + 1)
    h = max(p.height for p in parts) + spacing * 2
    buf = bytearray(bytes((0, 0, 0, 255)) * (w * h))
    ox = spacing
    for p in parts:
        for y in range(p.height):
            start = ((y + spacing) * w + ox) * 4
            buf[start:start + p.width * 4] = \
                p.data[y * p.width * 4:(y + 1) * p.width * 4]
        ox += p.width + spacing
    return ImageData.from_rgba_bytes(w, h, bytes(buf))


def _library_with_digits(labels, scale=1, status="test"):
    lib = RealClientTemplateLibrary(client="test")
    for i, (ch, img) in enumerate(labels.items()):
        lib.add(make_asset(f"d_{ch}", "digit_glyph", ch, status=status,
                           source="test"), img)
    return lib


# ------------------------------------------------------------------ 资产
def test_asset_defaults_null():
    a = RealClientTemplate(template_id="x")
    assert a.semantic_label is None
    assert a.confidence is None
    assert a.label_source is None
    assert a.cluster_id is None
    assert a.png_ref is None
    assert a.ground_truth_status == "unlabeled_candidate"


def test_asset_provenance_fields():
    a = make_asset("a1", "digit_glyph", "4", "visually_labeled", "shot1.png")
    d = a.to_dict()
    for k in ("template_id", "kind", "semantic_label", "source_frame",
              "source_roi", "client", "crop_bbox", "asset_version",
              "confidence", "ground_truth_status", "label_source",
              "cluster_id", "png_ref"):
        assert k in d, k


def test_asset_validation_rejects_bad():
    with pytest.raises(ValueError):
        RealClientTemplate(template_id="x", kind="bogus")
    with pytest.raises(ValueError):
        RealClientTemplate(template_id="x",
                           ground_truth_status="guessed")
    with pytest.raises(ValueError):
        RealClientTemplate(template_id="x", kind="card_glyph",
                           semantic_label="X")          # 非法 rank/suit
    with pytest.raises(ValueError):
        RealClientTemplate(template_id="x", kind="digit_glyph",
                           semantic_label="Z")
    with pytest.raises(ValueError):
        RealClientTemplate(template_id="", kind="digit_glyph")


def test_asset_is_labeled_semantics():
    assert make_asset("a", "digit_glyph", None).is_labeled is False
    assert make_asset("a", "digit_glyph", "4",
                      "unlabeled_candidate").is_labeled is False
    assert make_asset("a", "digit_glyph", "4",
                      "visually_labeled").is_labeled is True
    assert make_asset("a", "digit_glyph", "4", "verified").is_labeled is True
    assert make_asset("a", "digit_glyph", "4", "test").is_labeled is True


def test_library_version_and_by_kind():
    lib = RealClientTemplateLibrary(client="c1", version=3)
    lib.add(make_asset("a1", "digit_glyph"))
    assert lib.get("a1").asset_version == 3
    assert len(lib.by_kind("digit_glyph")) == 1
    assert lib.by_kind("card_glyph") == []


def test_labeled_templates_only_labeled_and_vocab():
    img4 = _render(GLYPHS_5X7["4"], 1)
    img5 = _render(GLYPHS_5X7["5"], 1)
    lib = RealClientTemplateLibrary(client="test")
    lib.add(make_asset("a4", "digit_glyph", "4", "visually_labeled"), img4)
    lib.add(make_asset("a5", "digit_glyph", "5", "verified"), img5)
    lib.add(make_asset("au", "digit_glyph", None), img4)     # 未标注
    tpl = lib.labeled_templates("digit_glyph", DIGITS)
    assert set(tpl) == {"4", "5"}                            # 未标注被排除
    assert lib.labeled_templates("card_glyph", RANKS + SUITS) == {}


def test_library_json_roundtrip(tmp_path):
    img = _render(GLYPHS_5X7["7"], 1)
    lib = RealClientTemplateLibrary(client="c2", version=2)
    lib.add(make_asset("a7", "digit_glyph", "7", "visually_labeled",
                       "s1.png"), img)
    path = tmp_path / "manifest.json"
    lib.to_json(str(path))
    lib2 = RealClientTemplateLibrary.from_json(str(path),
                                               lambda tid: img)
    a = lib2.get("a7")
    assert a.semantic_label == "7" and a.asset_version == 2
    assert lib2.labeled_templates("digit_glyph", DIGITS)["7"] is img


# ------------------------------------------------------------------ 聚类
def test_cluster_identical_merge_different_split():
    img_a = _render(GLYPHS_5X7["4"], 1)
    img_b = _render(GLYPHS_5X7["9"], 1)
    lib = RealClientTemplateLibrary(client="test")
    for tid, img in (("a1", img_a), ("a2", img_a), ("b1", img_b)):
        lib.add(make_asset(tid, "digit_glyph"), img)
    assets = [lib.get(t) for t in ("a1", "a2", "b1")]
    assignment = assign_clusters(assets, lib.images)
    assert assignment["a1"] == assignment["a2"]           # 同形同簇
    assert assignment["a1"] != assignment["b1"]           # 异形异簇
    assert len(set(assignment.values())) == 2


def test_cluster_kind_separation():
    img = _render(GLYPHS_5X7["4"], 1)
    lib = RealClientTemplateLibrary(client="test")
    lib.add(make_asset("d1", "digit_glyph"), img)
    lib.add(make_asset("c1", "card_glyph"), img)
    assignment = assign_clusters([lib.get("d1"), lib.get("c1")], lib.images)
    assert assignment["d1"] != assignment["c1"]           # 跨 kind 不合并


# ------------------------------------------------------------------ 候选提取（真实截图存在时）
def test_extract_real_screenshots_candidates():
    png_dir = Path(__file__).parent.parent / "picture data"
    if not png_dir.exists():
        pytest.skip("真实截图目录不可用")
    import poker_decision_ai.vision2_platform.png_decode as _pd
    from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile
    p = sorted(png_dir.glob("*.png"))[0]
    w, h, rgba = _pd.decode_png(p.read_bytes())
    geo = GeometryContext(screen_size=(float(w), float(h)), window_rect=None,
                          client_offset=(0.0, 0.0),
                          client_size_physical=(float(w), float(h)),
                          dpi_scale=1.0)
    cap = CapturedFrame(image=ImageData.from_rgba_bytes(w, h, rgba),
                        captured_at=0.0, geometry=geo,
                        source=f"real:{p.name}", synthetic_test_only=False)
    frame = Frame.from_capture(cap, 1)
    prof = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
    mgr = ROIManager(prof, default_roi_specs())
    rois, _ = mgr.resolve()
    crops, _ = mgr.extract_all(frame.image, rois)
    assets = extract_candidates(1, crops, "unknown_client_1848x1388",
                                source_frame=p.name)
    kinds = {a.kind for a in assets}
    assert "digit_glyph" in kinds and "card_slot" in kinds
    assert all(a.ground_truth_status == "unlabeled_candidate"
               for a in assets)
    assert all(a.semantic_label is None for a in assets)   # 绝不标记为已识别
    assert all(a.source_frame == p.name for a in assets)
    ids = [a.template_id for a in assets]
    assert len(ids) == len(set(ids))                       # 确定性唯一 id


# ------------------------------------------------------------------ 后端门控
def test_digit_backend_unlabeled_unavailable():
    lib = RealClientTemplateLibrary(client="real")
    lib.add(make_asset("u1", "digit_glyph", None, source="shot1.png"))
    backend = RealClientDigitOCRBackend(lib)
    assert backend.available() is False
    res = backend.recognize(digit_composite("4"), "pot", "number")
    assert res.status is OCRStatus.UNAVAILABLE
    assert res.value is None


def test_digit_backend_labeled_mechanism():
    tpls = {ch: _render(GLYPHS_5X7[ch], 1) for ch in DIGITS}
    lib = _library_with_digits(tpls)
    backend = RealClientDigitOCRBackend(lib)
    assert backend.available() is True
    # 非金色 ROI → 灰度 V2B 路径（pot/bet_area/stacks 走金色路径；此处用 table_meta）
    res = backend.recognize(digit_composite("42.5", scale=1), "table_meta",
                            "number")
    assert res.status is OCRStatus.OK
    assert res.value == 42.5
    assert res.extra["templates_source"] == "real_client_labeled"


def test_card_backend_unlabeled_unavailable():
    lib = RealClientTemplateLibrary(client="real")
    lib.add(make_asset("u1", "card_glyph", None, source="shot1.png"))
    backend = RealClientCardRecognitionBackend(lib)
    assert backend.available() is False
    assert backend.unavailable_reason()


def test_card_backend_labeled_mechanism():
    from poker_decision_ai.vision2.pixel.fonts import (builtin_rank_templates,
                                                       builtin_suit_templates)
    from poker_decision_ai.vision2.pixel import render_test_card
    card = render_test_card("A", "s", scale=1)
    rank_img = builtin_rank_templates()["A"]      # 规范形（亮字黑底）
    suit_img = builtin_suit_templates()["s"]
    lib = RealClientTemplateLibrary(client="test")
    lib.add(make_asset("rank_A", "card_glyph", "A", "test", roi="hero_cards"),
            rank_img)
    lib.add(make_asset("suit_s", "card_glyph", "s", "test", roi="hero_cards"),
            suit_img)
    backend = RealClientCardRecognitionBackend(lib)
    assert backend.available() is True
    two = bytearray(bytes((0, 0, 0, 255)) * (80 * 56))
    for y in range(56):
        two[(y * 80) * 4:(y * 80 + 40) * 4] = \
            card.data[y * 40 * 4:(y + 1) * 40 * 4]
        two[(y * 80 + 40) * 4:(y * 80 + 80) * 4] = \
            card.data[y * 40 * 4:(y + 1) * 40 * 4]
    res = backend.recognize(ImageData.from_rgba_bytes(80, 56, bytes(two)),
                            "hero_cards")
    assert res.status.value == "OK"
    assert [(c.rank, c.suit) for c in res.cards] == [("A", "s"), ("A", "s")]


# ------------------------------------------------------------------ V1 流水线接线
def test_v1_pipeline_unlabeled_backend_degrades():
    """未标注真实资产 → UNAVAILABLE → 字段 null + 关键阻断（诚实）。"""
    img = ImageData.from_rgba_bytes(400, 300, bytes((0, 0, 0, 255)) * (400 * 300))
    geo = GeometryContext(screen_size=(400, 300), window_rect=None,
                          client_size_physical=(400, 300))
    frame = Frame.from_capture(CapturedFrame(
        img, 0.0, geo, source="t", synthetic_test_only=True), 1)
    lib = RealClientTemplateLibrary(client="real")
    pipe = RealtimeVisionPipeline(
        config=RealtimeVisionConfig(fps=10),
        capture_provider=FixtureCaptureProvider([]),
        ocr=RealClientDigitOCRBackend(lib),
        card=RealClientCardRecognitionBackend(lib))
    obs = pipe.process(frame)
    assert obs.normalized.pot["pot"] is None
    assert obs.blocked is True                            # pot 关键字段


# ------------------------------------------------------------------ 红线
def test_realclient_source_redlines():
    pkg = Path(__file__).parent.parent / "poker_decision_ai" \
        / "vision2" / "pixel" / "realclient"
    banned = ("pyautogui", "pynput", "selenium", "playwright", "pywinauto",
              "win32", "SendInput", "mouse_event", "keybd_event",
              "SetCursorPos", "PostMessage", "ctypes",
              "expected_value", "recommend", "exploit", "gto",
              "profitability", "equity", "villain", "bluff",
              "phase6", "phase7", "phase8", "phase9",
              "cv2", "pytesseract", "torch", "sklearn", "easyocr",
              "paddleocr", "torchvision")
    # capture.py 的合法只读截图 API（Pillow ImageGrab）——PIL 检测是该模块
    # 的职责（V2N 允许只读截图，禁止窗口控制）；其余模块仍禁 PIL。
    for p in pkg.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        checks = banned
        if p.name == "capture.py":
            checks = tuple(b for b in banned if b not in ("PIL", "ctypes"))
        for bad in checks:
            assert bad not in src, f"{p} 含禁止 token {bad}"
    cap = (pkg / "capture.py").read_text(encoding="utf-8")
    assert "PIL" in cap and "ImageGrab" in cap   # 只读截图 API 存在
    assert "pyautogui" not in cap and "click" not in cap
