"""Vision V2B — 视觉对象识别层测试（几何/分割/模板/数字字形/卡牌模板）。

所有识别走确定性 builtin_synthetic 模板库；未知 → null + uncertainty；
不涉及任何行动/意图/决策推断；synthetic 与 real 严格区分。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel import (  # noqa: E402
    CardTemplateRecognitionBackend, DigitRecognitionStatus,
    DigitTemplateOCRBackend, PixelEngineConfig, PixelVisionEngine,
    PixelEventKind, TemplateEntry, TemplateLibrary, builtin_card_library,
    builtin_digit_templates, card_candidate_slots, classify_components,
    connected_components, dominant_color, edge_density, match_library,
    otsu_threshold, recognize_card_slot, recognize_digits, render_test_card,
    segment_cells, segment_components, segment_glyphs, threshold_mask)
from poker_decision_ai.vision2.pixel import geometry as px_geometry  # noqa: E402
from poker_decision_ai.vision2.pixel.fonts import (  # noqa: E402
    GLYPHS_5X7, _render)
from poker_decision_ai.vision2.realtime import (  # noqa: E402
    RecognitionStatus, Rect, RealtimeVisionConfig, RealtimeVisionPipeline)
from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile  # noqa: E402
from poker_decision_ai.vision2.realtime.capture import (  # noqa: E402
    CapturedFrame, FixtureCaptureProvider)
from poker_decision_ai.vision2.realtime.coords import GeometryContext  # noqa: E402
from poker_decision_ai.vision2.realtime.frame_manager import Frame  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2.realtime.recognition import (  # noqa: E402
    OCRStatus, UnavailableCardBackend, UnavailableOCRBackend)
from poker_decision_ai.vision2.realtime.roi import (  # noqa: E402
    ROIManager, ROIName, default_roi_specs)


# ------------------------------------------------------------------ helpers
def solid(w, h, r, g, b) -> ImageData:
    return ImageData.from_rgba_bytes(w, h, bytes((r, g, b, 255)) * (w * h))


def digit_composite(text: str, scale: int = 4, spacing: int = 4,
                    fg: int = 255, bg: int = 0) -> ImageData:
    """builtin 字体渲染数字串（含 '.'）。"""
    parts = [_render(GLYPHS_5X7[ch], scale, fg=fg, bg=bg) for ch in text]
    w = sum(p.width for p in parts) + spacing * (len(parts) + 1)
    h = max(p.height for p in parts) + spacing * 2
    buf = bytearray(bytes((bg, bg, bg, 255)) * (w * h))
    ox = spacing
    for p in parts:
        for y in range(p.height):
            start = ((y + spacing) * w + ox) * 4
            buf[start:start + p.width * 4] = \
                p.data[y * p.width * 4:(y + 1) * p.width * 4]
        ox += p.width + spacing
    return ImageData.from_rgba_bytes(w, h, bytes(buf))


def gray_of(img: ImageData):
    from poker_decision_ai.vision2.realtime.image import to_gray
    return to_gray(img)


# ------------------------------------------------------------------ 几何
def test_otsu_bimodal():
    img = solid(20, 20, 0, 0, 0)
    buf = bytearray(img.data)
    for i in range(0, 100 * 4, 4):
        buf[i:i + 4] = bytes((255, 255, 255, 255))
    gray = gray_of(ImageData.from_rgba_bytes(20, 20, bytes(buf)))
    t = otsu_threshold(gray)
    assert 0 <= t < 255
    mask = threshold_mask(gray)
    assert sum(1 for row in mask for v in row if v) == 100   # 正确分离两簇


def test_threshold_mask_polarity_both():
    dark_bg = digit_composite("4", scale=2)          # 亮字形黑底
    mask1 = threshold_mask(gray_of(dark_bg))
    assert any(any(row) for row in mask1)
    light_bg = digit_composite("4", scale=2, fg=10, bg=240)   # 深字形浅底
    mask2 = threshold_mask(gray_of(light_bg))
    assert any(any(row) for row in mask2)
    # 深字浅底的 mask 应与亮字黑底结构一致（前景=字形）
    assert sum(1 for r in mask1 for v in r if v) == \
        sum(1 for r in mask2 for v in r if v)


def test_threshold_mask_explicit_invert():
    img = digit_composite("8", scale=2)
    m1 = threshold_mask(gray_of(img), invert=False)
    m2 = threshold_mask(gray_of(img), invert=True)
    for y in range(len(m1)):
        for x in range(len(m1[0])):
            assert m1[y][x] != m2[y][x]


def test_connected_components_diagonal_merged():
    # 对角相邻的两个像素在 8 连通下应属于同一组件
    mask = [[False] * 6 for _ in range(6)]
    mask[1][1] = mask[2][2] = True
    comps = connected_components(mask, min_area=1)
    assert len(comps) == 1
    assert comps[0].area == 2


def test_component_stats():
    mask = [[False] * 6 for _ in range(6)]
    for x in range(1, 4):
        for y in range(1, 4):
            mask[y][x] = True
    comps = connected_components(mask, min_area=1)
    c = comps[0]
    assert c.area == 9
    assert c.bbox.to_dict() == {"x": 1.0, "y": 1.0, "width": 3.0, "height": 3.0}
    assert c.fill_ratio == 1.0 and c.rect_score == 1.0
    assert c.circle_score == 1.0                     # 上限截断（方形不可能超过圆）


def test_edge_density_solid_vs_checker():
    a = gray_of(solid(12, 12, 40, 40, 40))
    assert edge_density(a) == 0.0
    buf = bytearray()
    for y in range(12):
        for x in range(12):
            v = 255 if ((x // 2 + y // 2) % 2 == 0) else 0   # 2px 棋盘
            buf += bytes((v, v, v, 255))
    b = gray_of(ImageData.from_rgba_bytes(12, 12, bytes(buf)))
    assert edge_density(b) > 0.5


def test_geometry_cross_path():
    if px_geometry._np is None:
        pytest.skip("numpy 不可用")
    img = digit_composite("42.5", scale=2)
    gray = gray_of(img)
    t = otsu_threshold(gray)
    mask = threshold_mask(gray)
    comps = connected_components(mask)
    ed = edge_density(gray)
    flag = px_geometry._np
    try:
        px_geometry._np = None
        assert otsu_threshold(gray) == t
        mask2 = threshold_mask(gray)
        assert mask2 == mask
        comps2 = connected_components(mask2)
        assert [(c.area, c.bbox.to_dict()) for c in comps2] == \
            [(c.area, c.bbox.to_dict()) for c in comps]
        assert edge_density(gray) == pytest.approx(ed, abs=1e-9)
    finally:
        px_geometry._np = flag


# ------------------------------------------------------------------ 分割
def test_segment_and_classify():
    card = render_test_card("A", "s", scale=2)
    comps = segment_components(gray_of(card))
    out = classify_components(comps, gray_of(card))
    assert "card_candidates" in out and "component_count" in out
    for entry in out["card_candidates"]:
        assert entry["confidence"] is None          # 未校准不编造


# ------------------------------------------------------------------ 字体/模板
def test_builtin_templates_dims():
    for ch, tpl in builtin_digit_templates().items():
        assert (tpl.width, tpl.height) == (5, 7), ch


def test_render_scale():
    t1 = _render(GLYPHS_5X7["4"], 1)
    t4 = _render(GLYPHS_5X7["4"], 4)
    assert (t4.width, t4.height) == (20, 28)


def test_template_library_hint_and_dict():
    lib = TemplateLibrary([
        TemplateEntry("d1", _render(GLYPHS_5X7["1"], 1), roi_hint="pot"),
        TemplateEntry("d2", _render(GLYPHS_5X7["2"], 1)),
    ])
    assert len(lib.by_hint("pot")) == 2
    assert len(lib.by_hint("stacks")) == 1
    d = lib.to_dict()
    assert d["count"] == 2 and d["entries"][0]["source"] == "builtin_synthetic"


def test_match_library_finds_and_absent():
    lib = TemplateLibrary([
        TemplateEntry("d4", _render(GLYPHS_5X7["4"], 1), min_score=0.7),
    ])
    img = digit_composite("4", scale=1)
    matches = match_library(img, lib)
    assert matches and matches[0].template_id == "d4"
    assert matches[0].score > 0.9
    img2 = digit_composite("9", scale=1)
    assert not match_library(img2, lib)


# ------------------------------------------------------------------ 数字
def test_digit_roundtrip_decimal():
    img = digit_composite("42.5", scale=4)
    rec = recognize_digits(img, builtin_digit_templates())
    assert rec.status is DigitRecognitionStatus.OK
    assert rec.text == "42.5" and rec.value == 42.5
    assert rec.confidence == pytest.approx(1.0, abs=0.05)


def test_digit_roundtrip_integer():
    img = digit_composite("425", scale=3)
    rec = recognize_digits(img, builtin_digit_templates())
    assert rec.text == "425" and rec.value == 425.0


def test_digit_roundtrip_scale_one():
    img = digit_composite("7", scale=1)
    rec = recognize_digits(img, builtin_digit_templates())
    assert rec.text == "7"


def test_digit_no_templates_unavailable():
    img = digit_composite("5")
    rec = recognize_digits(img, {})
    assert rec.status is DigitRecognitionStatus.UNAVAILABLE
    assert rec.text is None


def test_digit_empty_crop_uncertain():
    rec = recognize_digits(solid(30, 30, 0, 0, 0), builtin_digit_templates())
    assert rec.status is DigitRecognitionStatus.UNCERTAIN
    assert rec.text is None and rec.value is None


def test_digit_noise_uncertain():
    random.seed(9)
    noise = ImageData.from_rgba_bytes(
        60, 30, bytes(random.randrange(256) for _ in range(60 * 30 * 4)))
    rec = recognize_digits(noise, builtin_digit_templates())
    assert rec.status is DigitRecognitionStatus.UNCERTAIN
    assert rec.text is None                      # 不猜


def test_digit_inverted_polarity():
    img = digit_composite("18", scale=3, fg=10, bg=245)   # 深字形浅底
    rec = recognize_digits(img, builtin_digit_templates())
    assert rec.status is DigitRecognitionStatus.OK
    assert rec.text == "18"


def test_digit_ocr_backend_interface():
    backend = DigitTemplateOCRBackend()
    img = digit_composite("42.5", scale=3)
    res = backend.recognize(img, "pot", "number")
    assert res.status is OCRStatus.OK
    assert res.value == 42.5 and res.raw_text == "42.5"
    assert res.synthetic_test_only is False
    assert res.extra["templates_source"] == "builtin_synthetic"
    bad = backend.recognize(img, "pot", "text")
    assert bad.status is OCRStatus.FAILED
    missing = backend.recognize(None, "pot", "number")
    assert missing.status is OCRStatus.FAILED


# ------------------------------------------------------------------ 卡牌
@pytest.mark.parametrize("rank,suit", [("A", "s"), ("K", "d"), ("T", "c"),
                                       ("2", "h"), ("Q", "s"), ("8", "c"),
                                       ("9", "h"), ("J", "d"), ("3", "c"),
                                       ("7", "s")])
def test_card_slot_roundtrip(rank, suit):
    r, s, conf, _ = recognize_card_slot(
        render_test_card(rank, suit, scale=3), builtin_card_library())
    assert (r, s) == (rank, suit)
    assert conf == pytest.approx(1.0, abs=0.05)


def test_card_backend_hero_two_slots():
    backend = CardTemplateRecognitionBackend()
    card = render_test_card("A", "s", scale=3)
    two = bytearray(bytes((0, 0, 0, 255)) * (80 * 56))
    for y in range(56):
        two[(y * 80) * 4:(y * 80 + 40) * 4] = card.data[y * 40 * 4:(y + 1) * 40 * 4]
        two[(y * 80 + 40) * 4:(y * 80 + 80) * 4] = card.data[y * 40 * 4:(y + 1) * 40 * 4]
    res = backend.recognize(ImageData.from_rgba_bytes(80, 56, bytes(two)),
                            "hero_cards")
    assert res.status.value == "OK"
    assert [(c.rank, c.suit) for c in res.cards] == [("A", "s"), ("A", "s")]
    assert res.cards[0].bbox is not None           # ROI 内归一化 bbox
    assert res.synthetic_test_only is False


def test_card_backend_hero_too_small():
    backend = CardTemplateRecognitionBackend()
    res = backend.recognize(solid(8, 8, 200, 200, 200), "hero_cards")
    assert res.status.value == "UNCERTAIN"


def test_card_backend_noise_uncertain():
    random.seed(4)
    noise = ImageData.from_rgba_bytes(
        80, 56, bytes(random.randrange(256) for _ in range(80 * 56 * 4)))
    backend = CardTemplateRecognitionBackend()
    res = backend.recognize(noise, "hero_cards")
    assert res.status.value == "UNCERTAIN"
    assert all(c.rank is None for c in res.cards)


def test_card_backend_board_no_components_uncertain():
    backend = CardTemplateRecognitionBackend()
    res = backend.recognize(solid(200, 80, 20, 60, 20), "board")
    assert res.status.value == "UNCERTAIN"          # 0 候选 ≠ 空板
    assert res.cards is None


def test_card_board_three_cards_order():
    backend = CardTemplateRecognitionBackend()
    a = render_test_card("A", "s", scale=2)
    k = render_test_card("K", "d", scale=2)
    q = render_test_card("Q", "c", scale=2)
    w = 3 * a.width + 20
    h = a.height + 10
    buf = bytearray(bytes((0, 0, 0, 255)) * (w * h))
    for i, card in enumerate((a, k, q)):
        ox = 5 + i * (card.width + 5)
        for y in range(card.height):
            start = ((y + 5) * w + ox) * 4
            buf[start:start + card.width * 4] = \
                card.data[y * card.width * 4:(y + 1) * card.width * 4]
    img = ImageData.from_rgba_bytes(w, h, bytes(buf))
    res = backend.recognize(img, "board")
    assert res.status.value == "OK"
    assert [(c.rank, c.suit) for c in res.cards] == \
        [("A", "s"), ("K", "d"), ("Q", "c")]


def test_card_candidate_slots_empty():
    assert card_candidate_slots(solid(100, 50, 10, 10, 10)) == []


# ------------------------------------------------------------------ 引擎 V2B 扩展
def _engine_on_crops(frames_and_crops):
    engine = PixelVisionEngine(PixelEngineConfig())
    results = []
    for frame, rois, crops in frames_and_crops:
        results.append(engine.process(frame, rois, crops))
    return results


def test_engine_geometry_features_filled():
    digits = digit_composite("42.5", scale=2)
    img = _compose_image(200, 100, [(digits, 56, 20)])
    geo = GeometryContext(screen_size=(200, 100), window_rect=None,
                          client_size_physical=(200, 100))
    cap = CapturedFrame(image=img, captured_at=0.0, geometry=geo,
                        source="test", synthetic_test_only=True)
    frame = Frame.from_capture(cap, 1)
    prof = bootstrap_profile("t", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
    mgr = ROIManager(prof, default_roi_specs())
    rois, missing = mgr.resolve()
    crops, cm = mgr.extract_all(frame.image, rois)
    missing.update(cm)
    engine = PixelVisionEngine()
    r = engine.process(frame, rois, crops, missing)
    obs = r.observations["pot"]
    assert obs.geometry_features is not None
    assert "component_count" in obs.geometry_features
    assert obs.geometry_features["edge_density"] is not None
    assert obs.segmentation is not None


def test_engine_shape_changed_event():
    a = solid(60, 40, 0, 0, 0)
    b = solid(60, 40, 0, 0, 0)
    buf = bytearray(b.data)
    for y in range(5, 25):
        for x in range(10, 20):
            i = (y * 60 + x) * 4
            buf[i:i + 4] = bytes((255, 255, 255, 255))
    b2 = ImageData.from_rgba_bytes(60, 40, bytes(buf))
    geo = GeometryContext(screen_size=(60, 40), window_rect=None,
                          client_size_physical=(60, 40))
    f1 = Frame.from_capture(CapturedFrame(a, 0.0, geo, source="t",
                                          synthetic_test_only=True), 1)
    f2 = Frame.from_capture(CapturedFrame(b2, 1.0, geo, source="t",
                                          synthetic_test_only=True), 2)
    prof = bootstrap_profile("t", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
    mgr = ROIManager(prof, default_roi_specs())
    rois, _ = mgr.resolve()
    c1, _ = mgr.extract_all(f1.image, rois)
    c2, _ = mgr.extract_all(f2.image, rois)
    engine = PixelVisionEngine()
    engine.process(f1, rois, c1)
    r2 = engine.process(f2, rois, c2)
    kinds = {e.kind for e in r2.events}
    assert PixelEventKind.SHAPE_CHANGED in kinds
    assert PixelEventKind.PIXEL_CHANGED in kinds


def test_engine_template_matched_event():
    digits = digit_composite("4", scale=1)
    img = _compose_image(200, 250, [(digits, 72, 46)])
    geo = GeometryContext(screen_size=(200, 250), window_rect=None,
                          client_size_physical=(200, 250))
    frame = Frame.from_capture(CapturedFrame(img, 0.0, geo, source="t",
                                             synthetic_test_only=True), 1)
    prof = bootstrap_profile("t", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
    mgr = ROIManager(prof, default_roi_specs())
    rois, _ = mgr.resolve()
    crops, _ = mgr.extract_all(frame.image, rois)
    lib = TemplateLibrary([
        TemplateEntry("digit_4", _render(GLYPHS_5X7["4"], 1), roi_hint="pot"),
    ])
    engine = PixelVisionEngine(PixelEngineConfig(template_library=lib))
    r = engine.process(frame, rois, crops)
    kinds = {e.kind for e in r.events}
    assert PixelEventKind.TEMPLATE_MATCHED in kinds
    obs = r.observations["pot"]
    assert obs.template_id == "digit_4"
    assert obs.template_score is not None


def test_engine_unchanged_no_shape_event():
    img = digit_composite("42.5", scale=2)
    geo = GeometryContext(screen_size=(img.width, img.height), window_rect=None,
                          client_size_physical=(img.width, img.height))
    f1 = Frame.from_capture(CapturedFrame(img, 0.0, geo, source="t",
                                          synthetic_test_only=True), 1)
    f2 = Frame.from_capture(CapturedFrame(img, 1.0, geo, source="t",
                                          synthetic_test_only=True), 2)
    prof = bootstrap_profile("t", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
    mgr = ROIManager(prof, default_roi_specs())
    rois, _ = mgr.resolve()
    c1, _ = mgr.extract_all(f1.image, rois)
    c2, _ = mgr.extract_all(f2.image, rois)
    engine = PixelVisionEngine()
    engine.process(f1, rois, c1)
    r2 = engine.process(f2, rois, c2)
    assert not any(e.kind is PixelEventKind.SHAPE_CHANGED for e in r2.events)
    assert r2.events == ()


# ------------------------------------------------------------------ V1 流水线集成（真实确定性识别路径）
def _frame_with_roi_content(img, roi_name, spec_offset, frame_id):
    """构造单帧：bootstrap 布局 + 指定 ROI 放内容。"""
    geo = GeometryContext(screen_size=(img.width, img.height), window_rect=None,
                          client_size_physical=(img.width, img.height))
    cap = CapturedFrame(image=img, captured_at=float(frame_id), geometry=geo,
                        source="test_frame", synthetic_test_only=True)
    return Frame.from_capture(cap, frame_id), geo


def _compose_image(w, h, placements):
    """placements: [(ImageData, x, y)] → 组合图。"""
    buf = bytearray(bytes((0, 0, 0, 255)) * (w * h))
    for img, ox, oy in placements:
        for y in range(img.height):
            start = ((oy + y) * w + ox) * 4
            buf[start:start + img.width * 4] = \
                img.data[y * img.width * 4:(y + 1) * img.width * 4]
    return ImageData.from_rgba_bytes(w, h, bytes(buf))


def test_v1_pipeline_digit_recognition_end_to_end():
    """DigitTemplateOCRBackend 接入 V1 流水线：真实像素数字 → pot 字段。"""
    digits = digit_composite("42.5", scale=2)
    img = _compose_image(400, 300, [(digits, 144, 56)])   # bootstrap pot 区
    frame, geo = _frame_with_roi_content(img, None, None, 1)
    pipe = RealtimeVisionPipeline(
        config=RealtimeVisionConfig(fps=10),
        capture_provider=FixtureCaptureProvider([]),
        ocr=DigitTemplateOCRBackend(),
        card=UnavailableCardBackend())
    # 直接把帧塞进 pipeline 处理（不经 provider）
    obs = pipe.process(frame)
    assert obs.normalized.pot["pot"] == 42.5            # 数字真实识别
    prov = obs.provenance.get("pot")
    assert prov and prov["source_layer"] == "ocr"
    assert prov["confidence"] == pytest.approx(1.0, abs=0.05)


def test_v1_pipeline_card_recognition_end_to_end():
    """CardTemplateRecognitionBackend 接入 V1 流水线：真实像素牌 → hole_cards。"""
    a = render_test_card("A", "s", scale=1)
    k = render_test_card("K", "s", scale=1)
    img = _compose_image(600, 500, [(a, 250, 375), (k, 310, 375)])
    frame, geo = _frame_with_roi_content(img, None, None, 1)
    specs = dict(default_roi_specs())
    from poker_decision_ai.vision2.realtime.roi import ROISpec
    specs[ROIName.HERO_CARDS] = ROISpec(
        ROIName.HERO_CARDS, "table", Rect(0.4, 0.75, 0.2, 0.15),
        required=True, read_layer="cards")
    pipe = RealtimeVisionPipeline(
        config=RealtimeVisionConfig(fps=10),
        capture_provider=FixtureCaptureProvider([]),
        ocr=UnavailableOCRBackend(),
        card=CardTemplateRecognitionBackend(),
        roi_specs=specs)
    obs = pipe.process(frame)
    assert obs.normalized.hero["hole_cards"] == ["As", "Ks"]
    assert obs.provenance["hero.hole_cards"]["source_layer"] == \
        "card_recognition"


def test_v1_pipeline_template_backend_degrades_on_nonmatching():
    """模板后端遇不匹配输入 → null + uncertainty（不猜）。"""
    img = _compose_image(400, 300, [(solid(80, 40, 90, 90, 90), 112, 30)])
    frame, geo = _frame_with_roi_content(img, None, None, 1)
    pipe = RealtimeVisionPipeline(
        config=RealtimeVisionConfig(fps=10),
        capture_provider=FixtureCaptureProvider([]),
        ocr=DigitTemplateOCRBackend(),
        card=UnavailableCardBackend())
    obs = pipe.process(frame)
    assert obs.normalized.pot["pot"] is None          # 不匹配 → null
    assert obs.blocked is True                        # pot 关键字段


# ------------------------------------------------------------------ 红线
def test_v2b_source_redlines():
    pkg = Path(__file__).parent.parent / "poker_decision_ai" \
        / "vision2" / "pixel"
    banned = ("pyautogui", "pynput", "selenium", "playwright", "pywinauto",
              "win32", "SendInput", "mouse_event", "keybd_event",
              "SetCursorPos", "PostMessage", "ctypes",
              "expected_value", "recommend", "exploit", "gto",
              "profitability", "equity", "villain", "bluff",
              "phase6", "phase7", "phase8", "phase9",
              "cv2", "pytesseract", "PIL", "torch", "sklearn", "easyocr",
              "paddleocr", "torchvision")
    for p in pkg.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        checks = banned
        if p.name == "capture.py":   # V2O 只读 BitBlt：PIL/ctypes 合法
            checks = tuple(b for b in banned if b not in ("PIL", "ctypes"))
        for bad in checks:
            assert bad not in src, f"{p} 含禁止 token {bad}"


def test_v2b_events_factual_only():
    img = digit_composite("42.5", scale=2)
    geo = GeometryContext(screen_size=(img.width, img.height), window_rect=None,
                          client_size_physical=(img.width, img.height))
    frame = Frame.from_capture(CapturedFrame(img, 0.0, geo, source="t",
                                             synthetic_test_only=True), 1)
    prof = bootstrap_profile("t", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
    mgr = ROIManager(prof, default_roi_specs())
    rois, _ = mgr.resolve()
    crops, _ = mgr.extract_all(frame.image, rois)
    lib = TemplateLibrary([
        TemplateEntry("d4", _render(GLYPHS_5X7["4"], 1), roi_hint="pot")])
    engine = PixelVisionEngine(PixelEngineConfig(template_library=lib))
    r = engine.process(frame, rois, crops)
    serialized = json.dumps(r.to_dict())
    for bad in ("villain", "bluff", "tight", "should call", "下注", "弃牌"):
        assert bad not in serialized.lower()


# ------------------------------------------------------------------ 基准 / 真实截图
def test_object_benchmark_stages():
    from poker_decision_ai.vision2.pixel.benchmark import run_object_benchmark
    res = run_object_benchmark(iterations=3, warmup=1, input_kind="synthetic")
    for stage in ("otsu_threshold", "threshold_mask", "connected_components",
                  "edge_density", "match_template_x2", "digit_recognize",
                  "card_recognize_slot", "engine_total_v2b"):
        assert stage in res.stages, stage
        assert res.stages[stage]["count"] >= 3


def test_real_png_digit_card_honest_status():
    png_dir = Path(__file__).parent.parent / "picture data"
    if not png_dir.exists():
        pytest.skip("真实截图目录不可用")
    import poker_decision_ai.vision2_platform.png_decode as _pd
    files = sorted(png_dir.glob("*.png"))[:2]
    digit_backend = DigitTemplateOCRBackend()
    card_backend = CardTemplateRecognitionBackend()
    for p in files:
        w, h, rgba = _pd.decode_png(p.read_bytes())
        img = ImageData.from_rgba_bytes(w, h, rgba)
        geo = GeometryContext(screen_size=(float(w), float(h)),
                              window_rect=None, client_offset=(0.0, 0.0),
                              client_size_physical=(float(w), float(h)),
                              dpi_scale=1.0)
        prof = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
        mgr = ROIManager(prof, default_roi_specs())
        rois, _ = mgr.resolve()
        crops, _ = mgr.extract_all(img, rois)
        pot = crops.get("pot")
        if pot is not None:
            rec = recognize_digits(pot, builtin_digit_templates())
            assert rec.status in (DigitRecognitionStatus.OK,
                                  DigitRecognitionStatus.UNCERTAIN)
            if rec.status is DigitRecognitionStatus.UNCERTAIN:
                assert rec.text is None              # 未命中不猜
        hero = crops.get("hero_cards")
        if hero is not None:
            res = card_backend.recognize(hero, "hero_cards")
            assert res.synthetic_test_only is False
            if res.status.value == "UNCERTAIN":
                assert all(c.rank is None for c in res.cards)
