"""Vision Agent V2 Real-Time 感知层（Phase V1）测试。

覆盖：坐标链 / DPI / 窗口移动 / anchor 校准 / ROI 提取与缺失 / frame
dedup 与变化检测 / OCR 仅识别变化 ROI / 卡牌输出 schema / 多模态兜底
触发与限频 / temporal diff / 新手牌 / 街道推进 / 7 个 failure 态 /
uncertainty 传播 / position 确定性推导 / V0 回归 / 匿名身份 / 无决策
字段 / 无自动操作源码红线。

所有测试确定性：synthetic fixture + 注入时钟，无真实屏幕/模型调用。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.realtime import (  # noqa: E402
    ALLOWED_FPS, CoordinateError, CoordinateSystem, CoordinateTransformer,
    FrameManager, FrameStatus, GeometryContext, Point, Rect, ROIManager,
    ROIName, RecognitionStatus, RealtimeVisionConfig,
    RealtimeVisionPipeline, UnavailableMultimodalBackend,
    byte_equality_ratio, bootstrap_profile, calibrate, detect_anchor_drift,
    dhash, pixel_diff_ratio)
from poker_decision_ai.vision2.realtime import image as rt_image  # noqa: E402
from poker_decision_ai.vision2.realtime import (  # noqa: E402
    FixtureCaptureProvider, ImageData, ncc_template_match)
from poker_decision_ai.vision2.realtime.fallback import (  # noqa: E402
    FallbackPolicyConfig, SyntheticMultimodalBackend)
from poker_decision_ai.vision2.realtime.fixtures import (  # noqa: E402
    DEFAULT_ROI_SPECS, duplicate_card_flop_scene, flop_scene, preflop_scene,
    river_scene, scenes_new_hand, scenes_street_progression, turn_scene)
from poker_decision_ai.vision2.realtime.recognition import (  # noqa: E402
    OCRStatus, SyntheticSceneCardBackend, SyntheticSceneOCRBackend,
    UnavailableOCRBackend)


# ------------------------------------------------------------------ helpers
def make_pipeline(scenes, *, fps=10.0, ocr=None, card=None, mm=None,
                  config=None, templates_override=None, **kw):
    provider = FixtureCaptureProvider(
        [s.render(now=float(i)) for i, s in enumerate(scenes)])
    cur = {"s": scenes[0]}

    def get_scene():
        return cur["s"]

    ocr = ocr if ocr is not None else SyntheticSceneOCRBackend(get_scene)
    card = card if card is not None else SyntheticSceneCardBackend(get_scene)
    mm = mm if mm is not None else SyntheticMultimodalBackend(get_scene)
    if templates_override is None:
        templates = {"table": scenes[0].anchor_template("table")}
    else:
        templates = templates_override
    sizes = {"table": (scenes[0].anchors_client["table"].width,
                       scenes[0].anchors_client["table"].height)}
    pipe = RealtimeVisionPipeline(
        config=config or RealtimeVisionConfig(fps=fps),
        capture_provider=provider, ocr=ocr, card=card, multimodal=mm,
        roi_specs=dict(DEFAULT_ROI_SPECS), anchor_templates=templates,
        anchor_sizes=sizes, **kw)
    return pipe, cur


def step_all(pipe, cur, scenes, t0=0.0):
    out = []
    for i, s in enumerate(scenes):
        cur["s"] = s
        out.append(pipe.step(now=t0 + float(i)))
    return out


ANCHOR_TABLE = {"table": Rect(40.0, 40.0, 1200.0, 690.0)}


def calibrated_manager(geometry=None):
    from poker_decision_ai.vision2.realtime.anchors import calibrate as _cal
    from poker_decision_ai.vision2.realtime import CalibrationProfile
    geo = geometry or preflop_scene().geometry
    det = {"table": (ANCHOR_TABLE["table"], 1.0)}
    prof = _cal("t", geo, det)
    return ROIManager(prof, dict(DEFAULT_ROI_SPECS))


# ------------------------------------------------------------------ 坐标层
def test_coord_chain_screen_to_normalized():
    geo = preflop_scene().geometry
    tr = CoordinateTransformer(geo)
    p = Point(124.0, 84.0)                       # screen
    pw = tr.screen_to_window(p)
    assert (pw.x, pw.y) == (24.0, 34.0)
    pc = tr.window_to_client(pw)
    assert (pc.x, pc.y) == (24.0, 4.0)
    pn = tr.client_to_normalized(pc)
    assert abs(pn.x - 24.0 / 1280.0) < 1e-12
    assert abs(pn.y - 4.0 / 770.0) < 1e-12
    # 反向链
    back = tr.transform(pn, CoordinateSystem.NORMALIZED, CoordinateSystem.SCREEN)
    assert abs(back.x - p.x) < 1e-9 and abs(back.y - p.y) < 1e-9


def test_rect_scaling_per_axis():
    geo = preflop_scene().geometry
    tr = CoordinateTransformer(geo)
    rn = Rect(0.25, 0.5, 0.1, 0.2)
    rc = tr.transform(rn, CoordinateSystem.NORMALIZED, CoordinateSystem.CLIENT)
    assert abs(rc.x - 0.25 * 1280) < 1e-9
    assert abs(rc.y - 0.5 * 770) < 1e-9
    assert abs(rc.width - 0.1 * 1280) < 1e-9
    assert abs(rc.height - 0.2 * 770) < 1e-9        # 高度必须按高度缩放


def test_roundtrip_all_pairs():
    geo = preflop_scene().geometry
    tr = CoordinateTransformer(geo)
    points = [Point(124.0, 84.0), Point(640.0, 400.0), Point(0.0, 0.0)]
    systems = list(CoordinateSystem)
    for p in points:
        for a in systems:
            for b in systems:
                q = tr.transform(p, a, b)
                back = tr.transform(q, b, a)
                assert abs(back.x - p.x) < 1e-9, (p, a, b)
                assert abs(back.y - p.y) < 1e-9, (p, a, b)


def test_resolution_change_same_normalized_roi():
    geo1 = GeometryContext(screen_size=(1920, 1080), window_rect=None,
                           client_size_physical=(1280, 770))
    geo2 = GeometryContext(screen_size=(2560, 1440), window_rect=None,
                           client_size_physical=(1920, 1080))
    rn = Rect(0.3, 0.4, 0.2, 0.1)
    r1 = CoordinateTransformer(geo1).transform(
        rn, CoordinateSystem.NORMALIZED, CoordinateSystem.CLIENT)
    r2 = CoordinateTransformer(geo2).transform(
        rn, CoordinateSystem.NORMALIZED, CoordinateSystem.CLIENT)
    assert (r1.width, r1.height) == (0.2 * 1280, 0.1 * 770)
    assert (r2.width, r2.height) == (0.2 * 1920, 0.1 * 1080)
    n1 = CoordinateTransformer(geo1).transform(
        r1, CoordinateSystem.CLIENT, CoordinateSystem.NORMALIZED)
    n2 = CoordinateTransformer(geo2).transform(
        r2, CoordinateSystem.CLIENT, CoordinateSystem.NORMALIZED)
    assert n1.to_dict() == n2.to_dict() == rn.to_dict()


def test_dpi_scale_normalized_invariance():
    geo1 = GeometryContext(screen_size=(1920, 1080), window_rect=None,
                           client_size_physical=(1280, 770), dpi_scale=1.0)
    geo2 = GeometryContext(screen_size=(1920, 1080), window_rect=None,
                           client_size_physical=(1280, 770), dpi_scale=1.25)
    pn = Point(0.5, 0.5)
    p1 = CoordinateTransformer(geo1).normalized_to_client(pn)
    p2 = CoordinateTransformer(geo2).normalized_to_client(pn)
    assert (p1.x, p1.y) == (640.0, 385.0)
    assert (p2.x, p2.y) == (640.0, 385.0)          # normalized 对 DPI 不变
    l1 = CoordinateTransformer(geo1).client_to_logical(p1)
    l2 = CoordinateTransformer(geo2).client_to_logical(p2)
    assert (l1.x, l1.y) == (640.0, 385.0)
    assert (l2.x, l2.y) == (512.0, 308.0)          # 逻辑px 随 DPI 换算


def test_window_movement_shifts_screen_coords_only():
    geo1 = preflop_scene().geometry
    geo2 = preflop_scene().with_geometry(window_rect=Rect(300, 200, 1280, 800)).geometry
    p = Point(0.4, 0.4)
    c1 = CoordinateTransformer(geo1).normalized_to_client(p)
    c2 = CoordinateTransformer(geo2).normalized_to_client(p)
    assert (c1.x, c1.y) == (c2.x, c2.y)            # client 不变
    s1 = CoordinateTransformer(geo1).transform(c1, CoordinateSystem.CLIENT,
                                               CoordinateSystem.SCREEN)
    s2 = CoordinateTransformer(geo2).transform(c2, CoordinateSystem.CLIENT,
                                               CoordinateSystem.SCREEN)
    assert s2.x - s1.x == 200.0 and s2.y - s1.y == 150.0


def test_client_offset_title_bar():
    geo = preflop_scene().geometry                   # client_offset=(0,30)
    tr = CoordinateTransformer(geo)
    pc = tr.window_to_client(Point(5.0, 35.0))
    assert (pc.x, pc.y) == (5.0, 5.0)


def test_geometry_validation_rejects_invalid():
    with pytest.raises(CoordinateError):
        GeometryContext(dpi_scale=0.0)
    with pytest.raises(CoordinateError):
        GeometryContext(client_size_physical=(0.0, 100.0))
    with pytest.raises(CoordinateError):
        GeometryContext(screen_size=(100.0, 100.0),
                        window_rect=Rect(0, 0, 200, 200),
                        client_size_physical=(200, 200))


# ------------------------------------------------------------------ anchor / 校准
def test_calibrate_normalizes_anchors():
    geo = preflop_scene().geometry
    prof = calibrate("p1", geo, {"table": (ANCHOR_TABLE["table"], 0.98)})
    a = prof.anchor("table")
    assert a.source == "calibrated"
    assert a.confidence == pytest.approx(0.98)
    assert a.rect_norm.width == pytest.approx(1200.0 / 1280.0)
    assert a.rect_norm.height == pytest.approx(690.0 / 770.0)


def test_bootstrap_low_confidence_seed():
    geo = preflop_scene().geometry
    prof = bootstrap_profile("b", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
    a = prof.anchor("table")
    assert a.source == "bootstrap" and a.confidence == 0.3


def test_anchor_drift_window_move():
    geo1 = preflop_scene().geometry
    geo2 = preflop_scene().with_geometry(window_rect=Rect(300, 200, 1280, 800)).geometry
    prof = bootstrap_profile("b", geo1, {"table": (0.0, 0.0, 1.0, 1.0)})
    assert detect_anchor_drift(prof, geo1) is False
    assert detect_anchor_drift(prof, geo2) is True


def test_locate_anchor_template_match():
    frame = preflop_scene().render(now=0.0)
    tpl = preflop_scene().anchor_template("table")
    found = ncc_template_match(frame.image, tpl, score_threshold=0.9)
    assert found is not None
    x, y, score = found
    assert (x, y) == (40, 40) and score > 0.95


def test_pipeline_layout_unknown_when_anchor_not_found():
    scene = preflop_scene()
    random.seed(7)
    noise = ImageData.from_rgba_bytes(
        10, 10, bytes(random.randrange(256) for _ in range(400)))
    pipe, cur = make_pipeline([scene], templates_override={"table": noise})
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.status is RecognitionStatus.LAYOUT_UNKNOWN
    assert obs.blocked is True and obs.consumable() is False


def test_pipeline_uses_bootstrap_layout_without_anchor_templates():
    """无 anchor 模板时，首帧必须使用 bootstrap 种子布局（不判 LAYOUT_UNKNOWN）。"""
    scene = preflop_scene()
    pipe, cur = make_pipeline([scene], templates_override={})
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.status is not RecognitionStatus.LAYOUT_UNKNOWN
    assert pipe.profile.anchor("table").source == "bootstrap"
    assert obs.normalized is not None            # bootstrap 布局下链路继续
    assert obs.normalized.game["street"] == "preflop"


# ------------------------------------------------------------------ ROI
def test_roi_resolve_anchor_relative():
    mgr = calibrated_manager()
    rois, missing = mgr.resolve()
    assert missing == {}
    pot = rois[ROIName.POT]
    # anchor(40,40,1200,690) + offset(0.28,0.10,0.44,0.10)（anchor 尺寸相对）
    assert pot.x == pytest.approx(40 + 0.28 * 1200)
    assert pot.y == pytest.approx(40 + 0.10 * 690)
    assert pot.width == pytest.approx(0.44 * 1200)
    assert pot.height == pytest.approx(0.10 * 690)


def test_roi_contract_fields_present():
    mgr = calibrated_manager()
    rois, _ = mgr.resolve()
    d = rois[ROIName.BOARD].to_dict()
    for k in ("semantic_name", "coordinate_system", "x", "y", "width",
              "height", "confidence", "source", "calibration_version"):
        assert k in d, k
    assert d["coordinate_system"] == "client"


def test_roi_extraction_crop_pixels():
    # 手工小图：左上角 2x2 红块，其余蓝
    w, h = 8, 8
    buf = bytearray(bytes((0, 0, 255, 255)) * (w * h))
    for y in range(2):
        for x in range(2):
            i = (y * w + x) * 4
            buf[i:i + 4] = bytes((255, 0, 0, 255))
    img = ImageData.from_rgba_bytes(w, h, bytes(buf))
    crop = img.crop(Rect(0, 0, 2, 2))
    assert crop is not None
    assert crop.rgba_bytes() == bytes((255, 0, 0, 255)) * 4


def test_roi_missing_out_of_bounds():
    mgr = calibrated_manager()
    from poker_decision_ai.vision2.realtime.roi import ROISpec
    specs = dict(DEFAULT_ROI_SPECS)
    specs[ROIName.BOARD] = ROISpec(ROIName.BOARD, "table",
                                   Rect(1.5, 0.3, 0.2, 0.2),
                                   required=True, read_layer="cards")
    rois, missing = ROIManager(mgr.profile, specs).resolve()
    assert ROIName.BOARD in missing
    assert "out_of_bounds" in missing[ROIName.BOARD]


def test_roi_missing_anchor_missing():
    mgr = calibrated_manager()
    from poker_decision_ai.vision2.realtime.anchors import CalibrationProfile
    from poker_decision_ai.vision2.realtime.roi import ROISpec
    prof = CalibrationProfile(profile_id="x", geometry=mgr.profile.geometry,
                              anchors=(), source="calibrated")
    specs = dict(DEFAULT_ROI_SPECS)
    specs[ROIName.POT] = ROISpec(ROIName.POT, "ghost_anchor",
                                 Rect(0.1, 0.1, 0.2, 0.2),
                                 required=True, read_layer="number")
    rois, missing = ROIManager(prof, specs).resolve()
    assert ROIName.POT in missing
    assert "anchor_missing" in missing[ROIName.POT]


# ------------------------------------------------------------------ Frame Manager
def test_fps_validation_only_allowed_values():
    for fps in ALLOWED_FPS:
        FrameManager(FixtureCaptureProvider([]), fps=fps)
    for bad in (0.5, 3, 7, 20):
        with pytest.raises(ValueError):
            FrameManager(FixtureCaptureProvider([]), fps=bad)


def test_default_fps_is_two():
    fm = FrameManager(FixtureCaptureProvider([]))
    assert fm.fps == 2.0


def test_rate_limit_between_polls():
    prov = FixtureCaptureProvider([preflop_scene().render(now=0.0)],
                                  cycle=True)
    fm = FrameManager(prov, fps=10)                 # interval 0.1
    r1 = fm.poll(now=0.0)
    assert r1.status is FrameStatus.FRESH
    assert fm.poll(now=0.05).status is FrameStatus.WAIT_RATE_LIMIT
    assert fm.poll(now=0.1).status is FrameStatus.DUPLICATE


def test_frame_dedup_identical():
    scene = preflop_scene()
    prov = FixtureCaptureProvider([scene.render(now=0.0),
                                   scene.render(now=0.5)])
    fm = FrameManager(prov, fps=10)
    assert fm.poll(now=0.0).status is FrameStatus.FRESH
    r2 = fm.poll(now=0.5)
    assert r2.status is FrameStatus.DUPLICATE
    assert r2.frame.deduplicated is True


def test_frame_change_detection_fresh():
    a, b = preflop_scene(), preflop_scene(pot=4.0)
    prov = FixtureCaptureProvider([a.render(now=0.0), b.render(now=0.5)])
    fm = FrameManager(prov, fps=10)
    fm.poll(now=0.0)
    r2 = fm.poll(now=0.5)
    assert r2.status is FrameStatus.FRESH
    assert r2.frame.deduplicated is False


def test_frame_stale_when_desktop_unchanged_long():
    scene = preflop_scene()
    prov = FixtureCaptureProvider([scene.render(now=0.0),
                                   scene.render(now=0.5),
                                   scene.render(now=1.0)], cycle=True)
    fm = FrameManager(prov, fps=10, stale_after_s=3.0)
    assert fm.poll(now=0.0).status is FrameStatus.FRESH
    assert fm.poll(now=0.5).status is FrameStatus.DUPLICATE
    r3 = fm.poll(now=4.0)
    assert r3.status is FrameStatus.FRAME_STALE


def test_capture_failed_propagates():
    prov = FixtureCaptureProvider([preflop_scene().render(now=0.0)],
                                  fail_after=1)
    fm = FrameManager(prov, fps=10)
    assert fm.poll(now=0.0).status is FrameStatus.FRESH
    r2 = fm.poll(now=1.0)
    assert r2.status is FrameStatus.CAPTURE_FAILED
    assert "fixture_injected_failure" in r2.error


# ------------------------------------------------------------------ 识别层
def test_ocr_only_changed_rois():
    scenes = [preflop_scene(), preflop_scene(pot=4.0)]
    log = []
    pipe, cur = make_pipeline(scenes, ocr=SyntheticSceneOCRBackend(
        lambda: cur["s"], call_log=log))
    step_all(pipe, cur, scenes)
    pot_calls = [c for c in log if c[0] == "pot"]
    assert len(pot_calls) == 2                  # pot 每帧都变化 → 都识别
    players_calls = [c for c in log if c[0] == "players"]
    assert len(players_calls) == 1              # 未变化 → 仅首帧识别


def test_ocr_result_schema():
    scene = preflop_scene()
    backend = SyntheticSceneOCRBackend(lambda: scene)
    res = backend.recognize(None, "pot", "number")
    assert res.status is OCRStatus.OK
    assert res.value == 1.5
    assert res.confidence is not None
    assert res.synthetic_test_only is True


FALLBACK_OFF = RealtimeVisionConfig(
    fps=10, fallback=FallbackPolicyConfig(enabled=False))


def test_ocr_unavailable_blocks_critical_pot():
    scene = preflop_scene()
    pipe, cur = make_pipeline([scene], ocr=UnavailableOCRBackend(),
                              config=FALLBACK_OFF)
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.blocked is True
    assert obs.normalized.pot["pot"] is None    # null，不是 0
    fields = [u["field"] for u in obs.normalized.uncertainty]
    assert "pot" in fields


def test_card_recognition_output_schema():
    scene = preflop_scene()
    backend = SyntheticSceneCardBackend(lambda: scene)
    res = backend.recognize(None, "hero_cards")
    assert res.status.value == "OK"
    c = res.cards[0]
    d = c.to_dict()
    for k in ("rank", "suit", "bbox", "confidence", "source"):
        assert k in d
    assert c.rank == "A" and c.suit == "s"
    assert c.bbox is not None


def test_card_uncertain_degrades_to_none():
    scene = preflop_scene()
    pipe, cur = make_pipeline(
        [scene], config=FALLBACK_OFF,
        card=SyntheticSceneCardBackend(lambda: cur["s"],
                                       uncertain_rois={"hero_cards"}))
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.normalized.hero["hole_cards"] is None
    assert obs.blocked is True                    # hero.hole_cards 关键字段


def test_malformed_ocr_number_treated_as_uncertain():
    class MalformedOCR(SyntheticSceneOCRBackend):
        def recognize(self, crop=None, roi_name="", kind="number"):
            res = super().recognize(crop, roi_name, kind)
            if roi_name == "pot":
                res.value = "banana"
            return res

    scene = preflop_scene()
    pipe, cur = make_pipeline([scene], ocr=MalformedOCR(lambda: cur["s"]),
                              config=FALLBACK_OFF)
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.normalized.pot["pot"] is None      # 不猜、不崩溃
    assert obs.normalized.validation.valid is True
    assert obs.blocked is True                    # pot 缺失 = 关键 uncertain


# ------------------------------------------------------------------ 多模态兜底
def _conflict_scenes(pots=(10.0, 12.0, 14.0, 16.0)):
    return [duplicate_card_flop_scene(pot=p) for p in pots]


def test_fallback_triggers_on_new_hand():
    mm_log = []
    scenes = scenes_new_hand()
    cfg = RealtimeVisionConfig(
        fps=10, fallback=FallbackPolicyConfig(min_interval_s=0.0,
                                              max_calls_per_hand=5))
    pipe, cur = make_pipeline(
        scenes, config=cfg,
        mm=SyntheticMultimodalBackend(lambda: cur["s"], call_log=mm_log))
    step_all(pipe, cur, scenes)
    assert mm_log.count("multimodal_called") >= 1


def test_fallback_triggers_on_state_conflict():
    mm_log = []
    cfg = RealtimeVisionConfig(
        fps=10, fallback=FallbackPolicyConfig(min_interval_s=0.0,
                                              max_calls_per_hand=5))
    pipe, cur = make_pipeline(
        [duplicate_card_flop_scene()], config=cfg,
        mm=SyntheticMultimodalBackend(lambda: cur["s"], call_log=mm_log))
    cur["s"] = duplicate_card_flop_scene()
    obs = pipe.step(now=0.0)
    assert obs.status is RecognitionStatus.STATE_CONFLICT
    assert mm_log.count("multimodal_called") == 1


def test_fallback_triggers_on_layout_unknown():
    mm_log = []
    random.seed(11)
    noise = ImageData.from_rgba_bytes(
        10, 10, bytes(random.randrange(256) for _ in range(400)))
    scene = preflop_scene()
    cfg = RealtimeVisionConfig(
        fps=10, fallback=FallbackPolicyConfig(min_interval_s=0.0,
                                              max_calls_per_hand=5))
    pipe, cur = make_pipeline(
        [scene], config=cfg, templates_override={"table": noise},
        mm=SyntheticMultimodalBackend(lambda: cur["s"], call_log=mm_log))
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.status is RecognitionStatus.LAYOUT_UNKNOWN
    assert mm_log.count("multimodal_called") == 1


def test_fallback_triggers_on_ocr_failure_streak():
    mm_log = []
    scenes = [preflop_scene(pot=p) for p in (1.5, 4.0, 10.0)]
    cfg = RealtimeVisionConfig(
        fps=10, ocr_failure_streak_trigger=3,
        fallback=FallbackPolicyConfig(min_interval_s=0.0,
                                      max_calls_per_hand=5))
    pipe, cur = make_pipeline(
        scenes, config=cfg,
        ocr=SyntheticSceneOCRBackend(lambda: cur["s"], fail_next=30),
        mm=SyntheticMultimodalBackend(lambda: cur["s"], call_log=mm_log))
    step_all(pipe, cur, scenes)
    assert pipe.ocr_fail_streak >= 3
    assert mm_log.count("multimodal_called") >= 1


def test_fallback_not_every_frame_cooldown():
    mm_log = []
    scenes = _conflict_scenes()
    cfg = RealtimeVisionConfig(
        fps=10, fallback=FallbackPolicyConfig(min_interval_s=1.0,
                                              max_calls_per_hand=10))
    pipe, cur = make_pipeline(
        scenes, config=cfg,
        mm=SyntheticMultimodalBackend(lambda: cur["s"], call_log=mm_log))
    cur["s"] = scenes[0]
    pipe.step(now=0.0)                          # 触发 → 调用
    cur["s"] = scenes[1]
    pipe.step(now=0.2)                          # 冷却中 → 拒绝
    cur["s"] = scenes[2]
    pipe.step(now=1.2)                          # 冷却结束 → 调用
    assert mm_log.count("multimodal_called") == 2
    assert pipe.gate.stats()["calls_total"] == 2


def test_fallback_budget_per_hand():
    scenes = _conflict_scenes()
    cfg = RealtimeVisionConfig(
        fps=10, fallback=FallbackPolicyConfig(min_interval_s=0.0,
                                              max_calls_per_hand=2))
    pipe, cur = make_pipeline(scenes, config=cfg)
    step_all(pipe, cur, scenes, t0=0.0)
    assert pipe.stats["fallback_calls"] == 2
    assert pipe.stats["fallback_denied"] >= 1


def test_fallback_unavailable_degrades_uncertainty():
    scene = duplicate_card_flop_scene()
    cfg = RealtimeVisionConfig(
        fps=10, fallback=FallbackPolicyConfig(min_interval_s=0.0,
                                              max_calls_per_hand=5))
    pipe, cur = make_pipeline(scenes=[scene], config=cfg, mm=None)
    pipe.multimodal = UnavailableMultimodalBackend()
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.status is RecognitionStatus.STATE_CONFLICT
    fields = [u["field"] for u in obs.normalized.uncertainty]
    assert "vision.multimodal" in fields


def test_fallback_output_still_validated():
    scene = duplicate_card_flop_scene()
    cfg = RealtimeVisionConfig(
        fps=10, fallback=FallbackPolicyConfig(min_interval_s=0.0,
                                              max_calls_per_hand=5))
    pipe, cur = make_pipeline(
        [scene], config=cfg,
        mm=SyntheticMultimodalBackend(lambda: cur["s"],
                                      override_fields={
                                          "hero_hole_cards": ["As", "Kd"]}))
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    # 兜底输出仍须过 validator：冲突不消失
    assert obs.status is RecognitionStatus.STATE_CONFLICT
    assert obs.blocked is True


# ------------------------------------------------------------------ temporal
def test_temporal_pot_delta_no_actor_inference():
    scenes = [flop_scene(pot=p) for p in (10.0, 14.0, 18.0)]
    pipe, cur = make_pipeline(scenes)
    obs = step_all(pipe, cur, scenes)
    t2 = obs[1].transition
    assert t2 is not None
    pot_d = next(d for d in t2["deltas"] if d["field"] == "pot")
    assert pot_d["kind"] == "increased" and pot_d["delta"] == 4.0
    assert "description" not in pot_d            # delta 本身无行动者归因
    events = t2["events"]
    pot_ev = next(e for e in events if e["kind"] == "POT_CHANGED")
    assert pot_ev["description"] == "pot increased +4.0 (10.0 -> 14.0)"
    serialized = json.dumps(obs[1].to_dict())
    for bad in ("villain", "Villain", "对手下注", "villain bet"):
        assert bad not in serialized


def test_new_hand_detection():
    pipe, cur = make_pipeline(scenes_new_hand())
    obs = step_all(pipe, cur, scenes_new_hand())
    kinds = [e.kind.value for e in obs[1].events]
    assert "NEW_HAND" in kinds


def test_flop_appeared_event():
    pipe, cur = make_pipeline([preflop_scene(), flop_scene()])
    obs = step_all(pipe, cur, [preflop_scene(), flop_scene()])
    kinds = [e.kind.value for e in obs[1].events]
    assert "FLOP_APPEARED" in kinds


def test_turn_and_river_events():
    scenes = [preflop_scene(), flop_scene(), turn_scene(), river_scene()]
    pipe, cur = make_pipeline(scenes)
    obs = step_all(pipe, cur, scenes)
    assert "TURN_APPEARED" in [e.kind.value for e in obs[2].events]
    assert "RIVER_APPEARED" in [e.kind.value for e in obs[3].events]


def test_action_area_changed_event():
    pipe, cur = make_pipeline([preflop_scene(), flop_scene()])
    obs = step_all(pipe, cur, [preflop_scene(), flop_scene()])
    assert "ACTION_AREA_CHANGED" in [e.kind.value for e in obs[1].events]


def test_transition_evidence_frame_ids():
    pipe, cur = make_pipeline([preflop_scene(), flop_scene()])
    obs = step_all(pipe, cur, [preflop_scene(), flop_scene()])
    t = obs[1].transition
    assert t["previous_frame_id"] == 1 and t["new_frame_id"] == 2
    assert t["evidence_frame_ids"] == [1, 2]


def test_pot_decrease_unexplained_triggers():
    mm_log = []
    scenes = [preflop_scene(pot=4.0), preflop_scene(pot=2.0)]
    cfg = RealtimeVisionConfig(
        fps=10, fallback=FallbackPolicyConfig(min_interval_s=0.0,
                                              max_calls_per_hand=5))
    pipe, cur = make_pipeline(
        scenes, config=cfg,
        mm=SyntheticMultimodalBackend(lambda: cur["s"], call_log=mm_log))
    obs = step_all(pipe, cur, scenes)
    assert any(e.description.startswith("unexplained:")
               for e in obs[1].events)
    assert mm_log.count("multimodal_called") >= 1


def test_first_frame_no_transition():
    pipe, cur = make_pipeline([preflop_scene()])
    cur["s"] = preflop_scene()
    obs = pipe.step(now=0.0)
    assert obs.transition is None and obs.events == ()


# ------------------------------------------------------------------ 状态 / 阻断
def test_state_conflict_hard_block():
    scene = duplicate_card_flop_scene()
    pipe, cur = make_pipeline([scene])
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.status is RecognitionStatus.STATE_CONFLICT
    assert obs.blocked is True
    assert obs.consumable() is False
    assert obs.normalized.validation.valid is False


def test_capture_failed_blocked():
    scene = preflop_scene()
    pipe, cur = make_pipeline([scene])
    pipe.frame_manager.provider = FixtureCaptureProvider(
        [scene.render(now=0.0)], fail_after=1)
    cur["s"] = scene
    pipe.step(now=0.0)
    obs = pipe.step(now=1.0)
    assert obs.status is RecognitionStatus.CAPTURE_FAILED
    assert obs.blocked is True


def test_frame_stale_blocked():
    scene = preflop_scene()
    pipe, cur = make_pipeline([scene])
    pipe.frame_manager.provider = FixtureCaptureProvider(
        [scene.render(now=0.0), scene.render(now=0.5)], cycle=True)
    cur["s"] = scene
    pipe.step(now=0.0)
    pipe.step(now=0.5)
    obs = pipe.step(now=4.0)
    assert obs.status is RecognitionStatus.FRAME_STALE
    assert obs.blocked is True


def test_critical_uncertainty_blocks_non_critical_does_not():
    scene = preflop_scene()
    pipe, cur = make_pipeline(
        [scene], ocr=SyntheticSceneOCRBackend(lambda: cur["s"],
                                              uncertain_rois={"stacks"}))
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.status is RecognitionStatus.RECOGNITION_UNCERTAIN
    assert obs.blocked is False                      # stacks 非关键
    assert obs.consumable() is True


def test_nulls_never_default_filled():
    scene = preflop_scene()
    pipe, cur = make_pipeline([scene], ocr=UnavailableOCRBackend(),
                              config=FALLBACK_OFF)
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    n = obs.normalized
    assert n.pot["pot"] is None                      # 不是 0
    assert n.hero["seat"] is None                    # 不是 0/空串
    assert n.board["flop"] is None                   # 不是 []
    assert n.hero["hole_cards"] == ["As", "Kd"]      # 卡牌后端仍可用


# ------------------------------------------------------------------ position / 身份 / 输出
def test_position_deterministic_derivation():
    from poker_decision_ai.domain.poker.models import positions_for
    scene = flop_scene()
    pipe, cur = make_pipeline([scene])
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    players = [{"seat": p["seat"], "name": p["anonymous_player_id"],
                "stack": p["stack"] or 0.0}
               for p in obs.normalized.players if p["seat"] is not None]
    expected = positions_for(players, 2)
    assert obs.normalized.hero["position"] == expected[3]
    for p in obs.normalized.players:
        assert p["position"] == expected[p["seat"]]


def test_button_missing_position_null_and_blocked():
    scene = preflop_scene()
    from dataclasses import replace as _replace
    scene = _replace(scene, button_seat=None)
    pipe, cur = make_pipeline([scene])
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.normalized.hero["position"] is None
    assert all(p["position"] is None for p in obs.normalized.players)
    assert any(u["field"] == "button_seat"
               for u in obs.normalized.uncertainty)
    assert obs.blocked is True                      # button 关键字段


def test_anonymous_player_ids():
    scene = flop_scene()
    pipe, cur = make_pipeline([scene])
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    for p in obs.normalized.players:
        assert p["anonymous_player_id"] == f"player_{p['seat']}"


def test_output_shape_and_no_decision_fields():
    scene = flop_scene()
    pipe, cur = make_pipeline([scene])
    cur["s"] = scene
    d = pipe.step(now=0.0).to_dict()
    assert set(d) == {"status", "blocked", "failure_reason", "unchanged",
                      "waiting", "timestamp", "frame_id",
                      "synthetic_test_only", "normalized", "provenance",
                      "transition", "events"}
    for bad in ("ev", "decision", "recommendation", "action", "roi",
                "bet_size", "raise", "expected_value", "exploit",
                "profitability"):
        assert bad not in d, bad


def test_provenance_fields_have_source():
    scene = flop_scene()
    pipe, cur = make_pipeline([scene])
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    for fname in ("pot", "hero.hole_cards", "board", "game.street",
                  "hero.seat", "button_seat", "game.blinds", "players"):
        assert fname in obs.provenance, fname
        assert obs.provenance[fname]["source_layer"]
        assert obs.provenance[fname]["frame_id"] is not None


def test_v0_contract_regression_through_pipeline():
    scene = flop_scene()
    pipe, cur = make_pipeline([scene])
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    d = obs.normalized.to_dict()
    assert set(d) == {"source", "synthetic_test_only", "validation",
                      "uncertainty", "game", "hero", "players", "board",
                      "pot", "action_history"}
    assert set(d["validation"]) == {"valid", "errors", "warnings"}
    assert d["validation"]["valid"] is True


def test_fixture_scenes_synthetic_test_only():
    for s in (preflop_scene(), flop_scene(), turn_scene(), river_scene(),
              duplicate_card_flop_scene()):
        f = s.render(now=0.0)
        assert f.synthetic_test_only is True


# ------------------------------------------------------------------ 集成
def test_full_flop_pipeline_ok():
    scene = flop_scene()
    pipe, cur = make_pipeline([scene])
    cur["s"] = scene
    obs = pipe.step(now=0.0)
    assert obs.status is RecognitionStatus.RECOGNITION_OK
    assert obs.blocked is False and obs.consumable() is True
    assert obs.normalized.game["street"] == "flop"
    assert obs.normalized.pot["pot"] == 10.0


def test_duplicate_frame_reuses_observation():
    scene = preflop_scene()
    pipe, cur = make_pipeline([scene])
    prov = FixtureCaptureProvider([scene.render(now=0.0),
                                   scene.render(now=0.5)])
    pipe.frame_manager.provider = prov
    cur["s"] = scene
    o1 = pipe.step(now=0.0)
    o2 = pipe.step(now=0.5)
    assert o2.unchanged is True and o2.frame_id == 2
    assert o2.normalized.pot == o1.normalized.pot
    assert o2.transition is None


def test_window_moved_recalibrates():
    s1 = preflop_scene()
    s2 = preflop_scene("moved").with_geometry(
        window_rect=Rect(300, 200, 1280, 800))
    pipe, cur = make_pipeline([s1, s2])
    obs = step_all(pipe, cur, [s1, s2])
    assert obs[0].status is RecognitionStatus.RECOGNITION_OK
    assert obs[1].status is RecognitionStatus.RECOGNITION_OK
    assert pipe.profile.calibration_version >= 2


def test_continuous_run_max_frames():
    class AdvancingClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            self.t += 0.2
            return self.t

    scenes = scenes_street_progression()
    provider = FixtureCaptureProvider(
        [s.render(now=float(i)) for i, s in enumerate(scenes * 5)], cycle=True)
    cur = {"s": scenes[0]}
    pipe = RealtimeVisionPipeline(
        config=RealtimeVisionConfig(fps=10), capture_provider=provider,
        ocr=SyntheticSceneOCRBackend(lambda: cur["s"]),
        card=SyntheticSceneCardBackend(lambda: cur["s"]),
        multimodal=SyntheticMultimodalBackend(lambda: cur["s"]),
        roi_specs=dict(DEFAULT_ROI_SPECS),
        anchor_templates={"table": scenes[0].anchor_template("table")},
        anchor_sizes={"table": (scenes[0].anchors_client["table"].width,
                                scenes[0].anchors_client["table"].height)},
        clock=AdvancingClock())
    out = pipe.run(max_frames=5, sleep_s=0.0)
    assert len(out) == 5
    assert all(o.status is RecognitionStatus.RECOGNITION_OK for o in out)
    assert pipe.frame_manager.stats.fresh >= 4


# ------------------------------------------------------------------ image / cheap 检测
def test_image_dhash_and_diff_semantics():
    # 小图：dhash 语义（相同 → 相等；结构不同 → 不等）。
    # 注意：dhash 是结构哈希，纯色图无梯度 → 全部为 0；用条纹结构测试。
    def stripes(vertical: bool) -> bytes:
        out = bytearray()
        for y in range(32):
            for x in range(32):
                v = 255 if ((x if vertical else y) % 8 < 4) else 0
                out += bytes((v, v, v, 255))
        return bytes(out)

    img_a = ImageData.from_rgba_bytes(32, 32, stripes(vertical=True))
    img_b = ImageData.from_rgba_bytes(32, 32, stripes(vertical=False))
    assert dhash(img_a) == dhash(img_a)
    assert dhash(img_a) != dhash(img_b)
    # 场景级：完全相同 → diff 0；pot 变化 → diff > 0（整帧 + 字节相等率）
    a = preflop_scene().render(now=0.0).image
    b = preflop_scene(pot=4.0).render(now=0.0).image
    assert pixel_diff_ratio(a, a) == 0.0
    assert pixel_diff_ratio(a, b) > 0.0
    assert byte_equality_ratio(a.rgba_bytes(), a.rgba_bytes()) == 1.0
    assert byte_equality_ratio(a.rgba_bytes(), b.rgba_bytes()) < 1.0


def test_cross_path_consistency_pure_python_vs_numpy():
    if rt_image._np is None:
        pytest.skip("numpy 不可用，无需 cross-path")
    random.seed(42)
    w, h = 32, 24
    data = bytes(random.randrange(256) for _ in range(w * h * 4))
    img = ImageData.from_rgba_bytes(w, h, data)
    h1 = dhash(img)
    gray = img.crop(Rect(0, 0, w, h))
    h2 = dhash(gray)
    np_flag = rt_image._np
    try:
        rt_image._np = None                          # 强制纯 Python 路径
        assert dhash(img) == h1
        assert dhash(gray) == h2
        a = preflop_scene().render(now=0.0).image.crop(Rect(0, 0, 64, 64))
        b = preflop_scene(pot=4.0).render(now=0.0).image.crop(Rect(0, 0, 64, 64))
        pure_ratio = pixel_diff_ratio(a, b)
    finally:
        rt_image._np = np_flag
    a = preflop_scene().render(now=0.0).image.crop(Rect(0, 0, 64, 64))
    b = preflop_scene(pot=4.0).render(now=0.0).image.crop(Rect(0, 0, 64, 64))
    np_ratio = pixel_diff_ratio(a, b)
    assert pure_ratio == pytest.approx(np_ratio, rel=1e-6)


def test_ncc_finds_known_pattern_position():
    scene = preflop_scene()
    frame = scene.render(now=0.0)
    tpl = scene.anchor_template("table")
    found = ncc_template_match(frame.image, tpl, score_threshold=0.9)
    assert found is not None and found[0] == 40 and found[1] == 40


# ------------------------------------------------------------------ benchmark
def test_percentile_math():
    from poker_decision_ai.vision2.realtime.benchmark import percentile
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0
    assert percentile([], 50) == 0.0
    assert percentile([7.0], 99) == 7.0


def test_benchmark_fast_path_stages():
    from poker_decision_ai.vision2.realtime.benchmark import (
        run_fast_path_benchmark)
    res = run_fast_path_benchmark(iterations=5, warmup=2)
    for stage in ("capture", "layout", "roi_extraction", "cheap_detection",
                  "ocr", "card_recognition", "validation", "temporal",
                  "total_fast_path"):
        assert stage in res.stages, stage
        assert res.stages[stage]["count"] > 0
    assert res.stages["multimodal_fallback"]["count"] == 0
    assert res.synthetic_input is True


def test_benchmark_fallback_stages():
    from poker_decision_ai.vision2.realtime.benchmark import (
        run_fallback_benchmark)
    res = run_fallback_benchmark(iterations=4, warmup=2)
    assert res.stages["multimodal_fallback"]["count"] >= 4


# ------------------------------------------------------------------ 源码红线
_REALTIME_PKGS = (
    Path(__file__).parent.parent / "poker_decision_ai" / "vision2" / "realtime",
    Path(__file__).parent.parent / "poker_decision_ai" / "vision2_platform",
)


def _sources():
    for pkg in _REALTIME_PKGS:
        for p in sorted(pkg.rglob("*.py")):
            yield p


def test_realtime_no_automation_tokens():
    banned = ("pyautogui", "pynput", "selenium", "playwright", "pywinauto",
              "win32", "SendInput", "mouse_event", "keybd_event",
              "SetCursorPos", "PostMessage", "ctypes")
    for p in _sources():
        src = p.read_text(encoding="utf-8")
        for bad in banned:
            assert bad not in src, f"{p} 含禁止自动化 token {bad}"


def test_realtime_no_decision_tokens():
    banned = ("expected_value", "recommend", "exploit", "gto",
              "profitability", "equity")
    for p in _sources():
        src = p.read_text(encoding="utf-8")
        for bad in banned:
            assert bad not in src, f"{p} 含决策层 token {bad}"


def test_realtime_no_frozen_phase_links():
    banned = ("phase6", "phase7", "phase8", "phase9")
    for p in _sources():
        src = p.read_text(encoding="utf-8")
        for bad in banned:
            assert bad not in src, f"{p} 含冻结阶段引用 {bad}"


def test_platform_capture_readonly_only():
    p = (Path(__file__).parent.parent / "poker_decision_ai"
         / "vision2_platform" / "capture_backends.py")
    src = p.read_text(encoding="utf-8")
    assert "ImageGrab" in src or "mss" in src
    for bad in ("pyautogui", "pynput", "SendInput", "SetCursorPos",
                "mouse_event", "keybd_event", "PostMessage"):
        assert bad not in src


def test_pipeline_has_no_control_or_decision_methods():
    scene = preflop_scene()
    pipe, _ = make_pipeline([scene])
    for name in dir(pipe):
        assert "click" not in name.lower()
        assert "bet_api" not in name.lower()
        assert "keyboard" not in name.lower() and "mouse" not in name.lower()
