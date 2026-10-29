"""Vision V2A — Pixel Core 测试。

覆盖：RawPixelObservation null 纪律 / PixelEvent 白名单 / 变化掩码与
包围盒 / RGB-HSV 特征与主色 / 时序状态机（appear/disappear/stable/
transient/moved）/ 引擎行为 / 与 V1 流水线组合（V0 路径不受影响）/
证据帧无未来信息 / synthetic 与 real 区分 / 源码红线扫描 / 基准 /
真实截图离线（本机截图存在时）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel import (  # noqa: E402
    PixelEngineConfig, PixelEvent, PixelEventKind, PixelFrameResult,
    PixelTemporalConfig, PixelTemporalTracker, PixelVisionEngine,
    PixelVisionPipeline, RawPixelObservation, changed_bbox,
    changed_pixel_count, changed_pixel_mask, changed_pixel_ratio,
    color_distance, dominant_color, hsv_from_rgb, hsv_stats, rgb_stats)
from poker_decision_ai.vision2.pixel import stats as px_stats  # noqa: E402
from poker_decision_ai.vision2.realtime import (  # noqa: E402
    CalibrationProfile, RecognitionStatus, Rect, RealtimeVisionConfig,
    RealtimeVisionPipeline)
from poker_decision_ai.vision2.realtime.anchors import calibrate  # noqa: E402
from poker_decision_ai.vision2.realtime.fixtures import (  # noqa: E402
    DEFAULT_ROI_SPECS, flop_scene, preflop_scene)
from poker_decision_ai.vision2.realtime.frame_manager import Frame  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2.realtime.recognition import (  # noqa: E402
    SyntheticSceneCardBackend, SyntheticSceneOCRBackend)
from poker_decision_ai.vision2.realtime.fallback import (  # noqa: E402
    SyntheticMultimodalBackend)
from poker_decision_ai.vision2.realtime.roi import ROIManager, ROIName  # noqa: E402

TABLE_RECT = {"table": (Rect(40.0, 40.0, 1200.0, 690.0), 1.0)}


# ------------------------------------------------------------------ helpers
def scene_frame(scene, fid: int) -> Frame:
    return Frame.from_capture(scene.render(now=float(fid)), fid)


def layout_for(frame):
    prof = calibrate("t", frame.geometry, TABLE_RECT)
    return prof, ROIManager(prof, dict(DEFAULT_ROI_SPECS))


def crops_for(frame, mgr):
    rois, missing = mgr.resolve()
    crops, cm = mgr.extract_all(frame.image, rois)
    missing.update(cm)
    return rois, crops, missing


def solid(w, h, r, g, b) -> ImageData:
    return ImageData.from_rgba_bytes(w, h, bytes((r, g, b, 255)) * (w * h))


# ------------------------------------------------------------------ 契约
def test_raw_pixel_observation_defaults_null():
    o = RawPixelObservation()
    for name, val in o.__dataclass_fields__.items():
        assert getattr(o, name) is None, name


def test_pixel_event_kind_enum_exactly_ten():
    kinds = {k.value for k in PixelEventKind}
    assert kinds == {"PIXEL_CHANGED", "ROI_APPEARED", "ROI_DISAPPEARED",
                     "COLOR_CHANGED", "SHAPE_CHANGED", "TEMPLATE_MATCHED",
                     "TEXT_REGION_CHANGED", "CARD_REGION_CHANGED",
                     "STACK_REGION_CHANGED", "POT_REGION_CHANGED"}


def test_pixel_event_to_dict_none_fields():
    e = PixelEvent(kind=PixelEventKind.PIXEL_CHANGED)
    d = e.to_dict()
    assert d["kind"] == "PIXEL_CHANGED"
    assert d["roi"] is None and d["evidence_frame_ids"] is None
    assert d["pixel_change_ratio"] is None


def test_pixel_frame_result_to_dict():
    r = PixelFrameResult(frame_id=3)
    d = r.to_dict()
    assert d["frame_id"] == 3 and d["observations"] == {}
    assert d["events"] == []


# ------------------------------------------------------------------ 颜色/统计
def test_hsv_from_rgb_primary_colors():
    h, s, v = hsv_from_rgb(255, 0, 0)
    assert (h, s, v) == pytest.approx((0.0, 1.0, 1.0))
    h, s, v = hsv_from_rgb(0, 255, 0)
    assert (h, s, v) == pytest.approx((120.0, 1.0, 1.0))
    h, s, v = hsv_from_rgb(0, 0, 255)
    assert (h, s, v) == pytest.approx((240.0, 1.0, 1.0))


def test_hsv_from_rgb_gray_saturation_zero():
    h, s, v = hsv_from_rgb(128, 128, 128)
    assert s == 0.0 and v == pytest.approx(128 / 255)


def test_rgb_stats_solid():
    img = solid(4, 4, 10, 20, 30)
    st = rgb_stats(img)
    assert st["mean_rgb"] == [10.0, 20.0, 30.0]
    assert st["std_rgb"] == [0.0, 0.0, 0.0]


def test_rgb_stats_gray_image_none():
    img = ImageData.from_gray_bytes(4, 4, bytes([40]) * 16)
    assert rgb_stats(img) is None


def test_hsv_stats_solid_red():
    st = hsv_stats(solid(4, 4, 255, 0, 0))
    assert st["mean_hsv"][0] == pytest.approx(0.0, abs=1e-9)
    assert st["mean_hsv"][1] == pytest.approx(1.0)
    assert st["mean_hsv"][2] == pytest.approx(1.0)


def test_dominant_color_solid_ratio_one():
    dom = dominant_color(solid(8, 8, 200, 100, 50))
    assert dom["dominant_ratio"] == 1.0
    # 量化桶中心（step=64）：200→[192..255] 中心 224；100→[64..127] 中心 96；
    # 50→[0..63] 中心 32
    assert dom["dominant_rgb"] == [224, 96, 32]


def test_dominant_color_two_colors_majority():
    img = solid(6, 6, 255, 0, 0)
    buf = bytearray(img.data)
    for i in range(0, 2 * 6 * 4, 4):          # 前 2 行改蓝
        buf[i:i + 4] = bytes((0, 0, 255, 255))
    dom = dominant_color(ImageData.from_rgba_bytes(6, 6, bytes(buf)))
    assert dom["dominant_rgb"] == [224, 32, 32]      # 红桶仍多数（4 行）
    assert dom["dominant_ratio"] == pytest.approx(4 / 6)


# ------------------------------------------------------------------ 变化掩码
def test_changed_pixel_mask_identical_all_false():
    a = solid(5, 5, 1, 2, 3)
    b = solid(5, 5, 1, 2, 3)
    mask = changed_pixel_mask(a, b)
    assert changed_pixel_ratio(a, b) == 0.0
    assert changed_pixel_count(a, b) == 0
    assert changed_bbox(a, b) is None


def test_changed_bbox_localized_change():
    w, h = 32, 24
    buf = bytearray(bytes((10, 10, 10, 255)) * (w * h))
    for y in range(20, 23):
        for x in range(10, 14):
            i = (y * w + x) * 4
            buf[i:i + 4] = bytes((250, 250, 250, 255))
    changed = ImageData.from_rgba_bytes(w, h, bytes(buf))
    base = solid(w, h, 10, 10, 10)
    rect = changed_bbox(base, changed)
    assert rect is not None
    assert rect.to_dict() == {"x": 10.0, "y": 20.0, "width": 4.0, "height": 3.0}
    assert changed_pixel_count(base, changed) == 12


def test_changed_below_threshold_ignored():
    a = solid(4, 4, 100, 100, 100)
    buf = bytearray(a.data)
    buf[0:4] = bytes((105, 100, 100, 255))     # 通道差 5 < 阈值 12
    b = ImageData.from_rgba_bytes(4, 4, bytes(buf))
    assert changed_pixel_ratio(a, b) == 0.0
    assert changed_bbox(a, b) is None


def test_geometry_mismatch_returns_none():
    a = solid(4, 4, 1, 2, 3)
    b = solid(6, 6, 1, 2, 3)
    assert changed_pixel_mask(a, b) is None
    assert changed_pixel_ratio(a, b) is None
    assert changed_bbox(a, b) is None
    assert changed_pixel_count(a, b) is None


def test_cross_path_numpy_vs_pure_python():
    if px_stats._np is None:
        pytest.skip("numpy 不可用")
    w, h = 20, 16
    buf_a = bytearray()
    buf_b = bytearray()
    for i in range(w * h):
        v = (i * 13) % 256
        buf_a += bytes((v, (v + 40) % 256, (v + 90) % 256, 255))
        buf_b += bytes((v if i > 40 else 255 - v,
                        (v + 40) % 256 if i > 40 else v,
                        (v + 90) % 256, 255))
    a = ImageData.from_rgba_bytes(w, h, bytes(buf_a))
    b = ImageData.from_rgba_bytes(w, h, bytes(buf_b))
    np_ratio = changed_pixel_ratio(a, b)
    np_bbox = changed_bbox(a, b)
    np_dom = dominant_color(a)
    np_rgb = rgb_stats(a)
    flag = px_stats._np
    try:
        px_stats._np = None
        pure_ratio = changed_pixel_ratio(a, b)
        pure_bbox = changed_bbox(a, b)
        pure_dom = dominant_color(a)
        pure_rgb = rgb_stats(a)
    finally:
        px_stats._np = flag
    assert pure_ratio == pytest.approx(np_ratio, abs=1e-12)
    assert pure_bbox.to_dict() == np_bbox.to_dict()
    assert pure_dom == np_dom
    for k in ("mean_rgb", "std_rgb"):
        assert pure_rgb[k] == pytest.approx(np_rgb[k], abs=1e-9)


def test_color_distance():
    assert color_distance({"dominant_rgb": [0, 0, 0]},
                          {"dominant_rgb": [0, 0, 0]}) == 0.0
    assert color_distance({"dominant_rgb": [0, 0, 0]},
                          {"dominant_rgb": [10, 0, 0]}) == 10.0
    assert color_distance({}, {"dominant_rgb": [0, 0, 0]}) == float("inf")


# ------------------------------------------------------------------ 时序状态机
def _tracker():
    return PixelTemporalTracker(PixelTemporalConfig(
        confirm_frames=2, stable_frames=3, transient_window_frames=3,
        moved_centroid_px=4.0))


def test_tracker_initial_state():
    tr = _tracker()
    feats, events = tr.update(1, {"pot": True}, {"pot": False}, {"pot": None})
    assert events == []
    assert feats["pot"]["state"] == "present"


def test_tracker_disappearance_confirmed():
    tr = _tracker()
    tr.update(1, {"pot": True}, {"pot": False}, {"pot": None})
    _, e1 = tr.update(2, {"pot": False}, {"pot": False}, {"pot": None})
    assert e1 == []
    feats, e2 = tr.update(3, {"pot": False}, {"pot": False}, {"pot": None})
    assert [e.kind for e in e2] == [PixelEventKind.ROI_DISAPPEARED]
    assert e2[0].evidence_frame_ids == (2, 3)
    assert feats["pot"]["state"] == "absent"


def test_tracker_appearance_confirmed():
    tr = _tracker()
    tr.update(1, {"pot": False}, {"pot": False}, {"pot": None})
    tr.update(2, {"pot": False}, {"pot": False}, {"pot": None})
    tr.update(3, {"pot": False}, {"pot": False}, {"pot": None})
    feats, e1 = tr.update(4, {"pot": True}, {"pot": False}, {"pot": None})
    assert e1 == []                                   # 首现未确认
    assert feats["pot"]["state"] == "appeared"
    _, e2 = tr.update(5, {"pot": True}, {"pot": False}, {"pot": None})
    assert [e.kind for e in e2] == [PixelEventKind.ROI_APPEARED]
    assert e2[0].evidence_frame_ids == (4, 5)


def test_tracker_single_frame_blink_no_events_transient():
    tr = _tracker()
    tr.update(1, {"pot": True}, {"pot": False}, {"pot": None})
    _, e1 = tr.update(2, {"pot": False}, {"pot": False}, {"pot": None})
    feats, e2 = tr.update(3, {"pot": True}, {"pot": False}, {"pot": None})
    assert e1 == [] and e2 == []                    # 闪烁不产生语义事件
    assert feats["pot"]["state"] == "transient"
    assert feats["pot"]["last_absent_gap"] == 1


def test_tracker_stable_after_n_frames():
    tr = _tracker()
    tr.update(1, {"pot": True}, {"pot": False}, {"pot": None})
    feats, _ = tr.update(2, {"pot": True}, {"pot": False}, {"pot": None})
    assert feats["pot"]["state"] is None            # 2 帧未变化但不足阈值
    feats, _ = tr.update(3, {"pot": True}, {"pot": False}, {"pot": None})
    assert feats["pot"]["state"] == "stable"
    assert feats["pot"]["stable_frames"] == 3


def test_tracker_moved_state():
    tr = _tracker()
    tr.update(1, {"pot": True}, {"pot": True}, {"pot": Rect(0, 0, 10, 10)})
    feats, _ = tr.update(2, {"pot": True}, {"pot": True},
                         {"pot": Rect(30, 0, 10, 10)})
    assert feats["pot"]["state"] == "moved"
    assert feats["pot"]["moved"] == pytest.approx(30.0)


def test_tracker_reset():
    tr = _tracker()
    tr.update(1, {"pot": True}, {"pot": False}, {"pot": None})
    tr.reset()
    feats, _ = tr.update(2, {"pot": True}, {"pot": False}, {"pot": None})
    assert feats["pot"]["state"] == "present"
    assert feats["pot"]["presence_run"] == 1


# ------------------------------------------------------------------ 引擎
def _engine_frames(scenes):
    frames = [scene_frame(s, i + 1) for i, s in enumerate(scenes)]
    prof, mgr = layout_for(frames[0])
    rois, crops, missing = crops_for(frames[0], mgr)
    return frames, rois, crops, missing


def test_engine_first_frame_nulls_and_uncertainty():
    scene = preflop_scene()
    frame = scene_frame(scene, 1)
    prof, mgr = layout_for(frame)
    rois, crops, missing = crops_for(frame, mgr)
    engine = PixelVisionEngine()
    r = engine.process(frame, rois, crops, missing)
    assert r.events == ()
    pot = r.observations["pot"]
    assert pot.pixel_change_ratio is None          # 首帧无基准 → null 不是 0
    assert pot.changed_bbox is None
    assert pot.uncertainty and any(
        u["field"] == "pixel_change_ratio" and u["status"] == "missing"
        for u in pot.uncertainty)
    assert pot.evidence_frame_ids == (1,)
    assert pot.color_features["dominant_ratio"] is not None


def test_engine_changed_frame_events():
    scenes = [preflop_scene(pot=1.5), preflop_scene(pot=4.0)]
    frames, rois, crops, missing = _engine_frames(scenes)
    engine = PixelVisionEngine()
    engine.process(frames[0], rois, crops, missing)
    prof, mgr = layout_for(frames[1])
    _, crops2, missing2 = crops_for(frames[1], mgr)
    r = engine.process(frames[1], rois, crops2, missing2)
    kinds = [e.kind for e in r.events]
    assert PixelEventKind.PIXEL_CHANGED in kinds
    assert PixelEventKind.POT_REGION_CHANGED in kinds
    pot = r.observations["pot"]
    assert pot.pixel_change_ratio is not None and pot.pixel_change_ratio > 0
    assert pot.changed_bbox is not None
    assert pot.evidence_frame_ids == (1, 2)


def test_engine_unchanged_frame_no_events_stable():
    scenes = [preflop_scene(), preflop_scene(), preflop_scene()]
    frames, rois, crops, missing = _engine_frames(scenes)
    engine = PixelVisionEngine()
    engine.process(frames[0], rois, crops, missing)
    prof, mgr = layout_for(frames[1])
    _, crops2, missing2 = crops_for(frames[1], mgr)
    r2 = engine.process(frames[1], rois, crops2, missing2)
    assert r2.events == ()
    _, crops3, missing3 = crops_for(frames[2], mgr)
    r3 = engine.process(frames[2], rois, crops3, missing3)
    assert r3.events == ()
    pot = r3.observations["pot"]
    assert pot.pixel_change_ratio == 0.0           # 计算过 → 0.0 合法（真实测量）
    assert pot.temporal_features["state"] == "stable"


def test_engine_missing_roi_uncertainty():
    frame = scene_frame(preflop_scene(), 1)
    prof, mgr = layout_for(frame)
    rois, crops, missing = crops_for(frame, mgr)
    crops.pop("stacks")
    missing["stacks"] = "synthetic_removed"
    engine = PixelVisionEngine()
    r = engine.process(frame, rois, crops, missing)
    obs = r.observations["stacks"]
    assert obs.color_features is None
    assert any(u["field"] == "roi_crop" and u["reason"] == "synthetic_removed"
               for u in obs.uncertainty)


def test_engine_region_event_mapping():
    from poker_decision_ai.vision2.realtime.fixtures import turn_scene
    # board 3→4 张（CARD 区变化）+ action 文字变化（TEXT 区变化）
    s1 = flop_scene(pot=10.0)
    s2 = turn_scene(pot=14.0).with_truth(ROIName.ACTION_AREA,
                                         text="Call 4.0", actor_seat=3)
    scenes = [s1, s2]
    frames, rois, crops, missing = _engine_frames(scenes)
    engine = PixelVisionEngine()
    engine.process(frames[0], rois, crops, missing)
    prof, mgr = layout_for(frames[1])
    _, crops2, missing2 = crops_for(frames[1], mgr)
    r = engine.process(frames[1], rois, crops2, missing2)
    kinds = {e.kind for e in r.events}
    assert PixelEventKind.CARD_REGION_CHANGED in kinds     # board 变化
    assert PixelEventKind.TEXT_REGION_CHANGED in kinds     # action_area 变化


def test_engine_color_changed_threshold():
    scenes = [preflop_scene(pot=1.5), preflop_scene(pot=4.0)]
    frames, rois, crops, missing = _engine_frames(scenes)
    cfg = PixelEngineConfig(color_change_threshold=10.0)
    engine = PixelVisionEngine(cfg)
    engine.process(frames[0], rois, crops, missing)
    prof, mgr = layout_for(frames[1])
    _, crops2, missing2 = crops_for(frames[1], mgr)
    r = engine.process(frames[1], rois, crops2, missing2)
    assert any(e.kind is PixelEventKind.COLOR_CHANGED for e in r.events)
    # 阈值极高 → 不产生 COLOR_CHANGED（其余事件不受影响）
    engine2 = PixelVisionEngine(PixelEngineConfig(color_change_threshold=9999.0))
    engine2.process(frames[0], rois, crops, missing)
    r2 = engine2.process(frames[1], rois, crops2, missing2)
    assert not any(e.kind is PixelEventKind.COLOR_CHANGED for e in r2.events)


def test_engine_evidence_no_future_frames():
    scenes = [preflop_scene(pot=p) for p in (1.5, 4.0, 10.0, 14.0)]
    frames, rois, crops, missing = _engine_frames(scenes)
    engine = PixelVisionEngine()
    for i, f in enumerate(frames):
        prof, mgr = layout_for(f)
        _, c, m = crops_for(f, mgr)
        r = engine.process(f, rois, c, m)
        for e in r.events:
            assert e.frame_id == f.frame_id
            if e.evidence_frame_ids:
                assert max(e.evidence_frame_ids) <= f.frame_id
        for obs in r.observations.values():
            if obs.evidence_frame_ids:
                assert max(obs.evidence_frame_ids) <= f.frame_id


def test_engine_synthetic_vs_real_flag():
    scene = preflop_scene()
    frame = scene_frame(scene, 1)
    prof, mgr = layout_for(frame)
    rois, crops, missing = crops_for(frame, mgr)
    engine = PixelVisionEngine()
    r = engine.process(frame, rois, crops, missing)
    assert r.synthetic_test_only is True
    assert all(o.synthetic_test_only is True
               for o in r.observations.values())
    # real 帧（synthetic_test_only=False）
    from poker_decision_ai.vision2.realtime.capture import CapturedFrame
    from poker_decision_ai.vision2.realtime.coords import GeometryContext
    geo = GeometryContext(screen_size=(100.0, 100.0), window_rect=None,
                          client_offset=(0.0, 0.0),
                          client_size_physical=(100.0, 100.0), dpi_scale=1.0)
    real_cap = CapturedFrame(image=solid(100, 100, 5, 5, 5), captured_at=0.0,
                             geometry=geo, source="real_test",
                             synthetic_test_only=False)
    real_frame = Frame.from_capture(real_cap, 2)
    engine.reset()
    # 全幅 ROI 布局（真实截图风格 bootstrap）
    from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile
    prof2 = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
    from poker_decision_ai.vision2.realtime.roi import default_roi_specs
    mgr2 = ROIManager(prof2, default_roi_specs())
    rois2, missing2 = mgr2.resolve()
    crops2, cm2 = mgr2.extract_all(real_frame.image, rois2)
    missing2.update(cm2)
    r2 = engine.process(real_frame, rois2, crops2, missing2)
    assert r2.synthetic_test_only is False


def test_engine_reset():
    scenes = [preflop_scene(pot=1.5), preflop_scene(pot=4.0)]
    frames, rois, crops, missing = _engine_frames(scenes)
    engine = PixelVisionEngine()
    engine.process(frames[0], rois, crops, missing)
    engine.reset()
    prof, mgr = layout_for(frames[1])
    _, c2, m2 = crops_for(frames[1], mgr)
    r = engine.process(frames[1], rois, c2, m2)
    pot = r.observations["pot"]
    assert pot.pixel_change_ratio is None          # reset 后又是首帧
    assert any(u["field"] == "pixel_change_ratio"
               for u in pot.uncertainty)


def test_engine_reappearance_uses_last_present_baseline():
    scenes = [preflop_scene(pot=1.5), preflop_scene(pot=4.0)]
    frames, rois, crops, missing = _engine_frames(scenes)
    engine = PixelVisionEngine()
    engine.process(frames[0], rois, crops, missing)
    # 帧 2：pot ROI 缺失
    prof, mgr = layout_for(frames[1])
    _, c2, m2 = crops_for(frames[1], mgr)
    c2.pop("pot")
    m2["pot"] = "synthetic_removed"
    r2 = engine.process(frames[1], rois, c2, m2)
    assert r2.observations["pot"].color_features is None
    # 帧 3：pot 重现（不同内容）→ 与最后在场基线比较，非 null
    frame3 = scene_frame(preflop_scene(pot=9.0), 3)
    _, c3, m3 = crops_for(frame3, mgr)
    r3 = engine.process(frame3, rois, c3, m3)
    pot = r3.observations["pot"]
    assert pot.pixel_change_ratio is not None
    assert pot.pixel_change_ratio > 0.0
    assert pot.temporal_features["state"] == "transient"   # 缺席 1 帧即回


# ------------------------------------------------------------------ 集成（V1 组合，V0 路径不动）
def _make_v1_pipeline(scenes):
    from poker_decision_ai.vision2.realtime.capture import FixtureCaptureProvider
    provider = FixtureCaptureProvider(
        [s.render(now=float(i)) for i, s in enumerate(scenes)])
    cur = {"s": scenes[0]}
    pipe = RealtimeVisionPipeline(
        config=RealtimeVisionConfig(fps=10), capture_provider=provider,
        ocr=SyntheticSceneOCRBackend(lambda: cur["s"]),
        card=SyntheticSceneCardBackend(lambda: cur["s"]),
        multimodal=SyntheticMultimodalBackend(lambda: cur["s"]),
        roi_specs=dict(DEFAULT_ROI_SPECS),
        anchor_templates={"table": scenes[0].anchor_template("table")},
        anchor_sizes={"table": (scenes[0].anchors_client["table"].width,
                                scenes[0].anchors_client["table"].height)})
    return pipe, cur


def test_integration_pixel_pipeline_synthetic():
    scenes = [preflop_scene(), flop_scene()]
    pipe, cur = _make_v1_pipeline(scenes)
    combo = PixelVisionPipeline(pipe)
    results = []
    for i, s in enumerate(scenes):
        cur["s"] = s
        results.append(combo.step(now=float(i)))
    assert all(r.vision is not None for r in results)
    assert results[0].vision.status is RecognitionStatus.RECOGNITION_OK
    assert results[0].pixel is not None
    assert set(results[0].pixel.observations) == {
        "hero_cards", "board", "pot", "action_area", "bet_area", "stacks",
        "players", "table_meta"}
    # 帧 2 有像素事件（board/action 变化）
    assert any(e.kind is PixelEventKind.CARD_REGION_CHANGED
               for e in results[1].pixel.events)


def test_integration_pixel_does_not_alter_vision_output():
    scenes = [preflop_scene(), flop_scene()]
    pipe1, cur1 = _make_v1_pipeline(scenes)
    pipe2, cur2 = _make_v1_pipeline(scenes)
    combo = PixelVisionPipeline(pipe2)
    for i, s in enumerate(scenes):
        cur1["s"] = s
        cur2["s"] = s
        o1 = pipe1.step(now=float(i))
        o2 = combo.step(now=float(i)).vision
        assert o1.to_dict() == o2.to_dict()          # V0 路径完全一致
        assert o2.normalized.validation.valid is True


# ------------------------------------------------------------------ 事件无推断 + 红线
def test_event_descriptions_factual_only():
    scenes = [flop_scene(pot=10.0), flop_scene(pot=12.0)]
    frames, rois, crops, missing = _engine_frames(scenes)
    engine = PixelVisionEngine()
    engine.process(frames[0], rois, crops, missing)
    prof, mgr = layout_for(frames[1])
    _, c2, m2 = crops_for(frames[1], mgr)
    r = engine.process(frames[1], rois, c2, m2)
    serialized = json.dumps(r.to_dict())
    for bad in ("villain", "bluff", "tight", "should call", "bet 4",
                "下注", "弃牌"):
        assert bad not in serialized.lower(), bad


def test_pixel_package_source_redlines():
    pkg = Path(__file__).parent.parent / "poker_decision_ai" \
        / "vision2" / "pixel"
    automation = ("pyautogui", "pynput", "selenium", "playwright",
                  "pywinauto", "win32", "SendInput", "mouse_event",
                  "keybd_event", "SetCursorPos", "PostMessage", "ctypes")
    decision = ("expected_value", "recommend", "exploit", "gto",
                "profitability", "equity")
    inference = ("villain", "bluff")
    frozen = ("phase6", "phase7", "phase8", "phase9")
    libs = ("cv2", "pytesseract", "PIL", "torch", "sklearn", "easyocr",
            "paddleocr", "torchvision")
    for p in pkg.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        checks = automation + decision + inference + frozen + libs
        if p.name == "capture.py":   # V2O 只读 BitBlt：PIL/ctypes 合法
            checks = tuple(b for b in checks
                           if b not in ("PIL", "ctypes"))
        for bad in checks:
            assert bad not in src, f"{p} 含禁止 token {bad}"


# ------------------------------------------------------------------ 基准
def test_pixel_benchmark_synthetic_stages():
    from poker_decision_ai.vision2.pixel.benchmark import run_pixel_benchmark
    res = run_pixel_benchmark(iterations=5, warmup=2, input_kind="synthetic")
    for stage in ("changed_mask", "changed_bbox", "changed_ratio",
                  "rgb_stats", "hsv_stats", "dominant_color", "engine_total"):
        assert stage in res.stages, stage
        assert res.stages[stage]["count"] >= 5
    assert res.environment["synthetic_input"] is True


def test_pixel_benchmark_real_if_screenshots_available():
    png_dir = Path(__file__).parent.parent / "picture data"
    if not png_dir.exists() or len(list(png_dir.glob("*.png"))) < 2:
        pytest.skip("真实截图目录不可用")
    from poker_decision_ai.vision2.pixel.benchmark import run_pixel_benchmark
    res = run_pixel_benchmark(iterations=3, warmup=1, input_kind="real")
    assert res.environment["synthetic_input"] is False
    assert res.stages["engine_total"]["count"] >= 3


# ------------------------------------------------------------------ 真实截图离线（本机截图存在时）
def test_real_png_pixel_engine_offline():
    png_dir = Path(__file__).parent.parent / "picture data"
    if not png_dir.exists():
        pytest.skip("真实截图目录不可用")
    import poker_decision_ai.vision2_platform.png_decode as _pd
    from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile
    from poker_decision_ai.vision2.realtime.capture import CapturedFrame
    from poker_decision_ai.vision2.realtime.coords import GeometryContext
    from poker_decision_ai.vision2.realtime.roi import default_roi_specs
    files = sorted(png_dir.glob("*.png"))[:3]
    engine = PixelVisionEngine()
    for i, p in enumerate(files):
        w, h, rgba = _pd.decode_png(p.read_bytes())
        geo = GeometryContext(screen_size=(float(w), float(h)),
                              window_rect=None, client_offset=(0.0, 0.0),
                              client_size_physical=(float(w), float(h)),
                              dpi_scale=1.0)
        cap = CapturedFrame(image=ImageData.from_rgba_bytes(w, h, rgba),
                            captured_at=float(i), geometry=geo,
                            source=f"real:{p.name}", synthetic_test_only=False)
        frame = Frame.from_capture(cap, i + 1)
        prof = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
        mgr = ROIManager(prof, default_roi_specs())
        rois, missing = mgr.resolve()
        crops, cm = mgr.extract_all(frame.image, rois)
        missing.update(cm)
        r = engine.process(frame, rois, crops, missing)
        assert r.synthetic_test_only is False
        if i >= 1:
            assert r.observations["pot"].pixel_change_ratio is not None
        for e in r.events:
            if e.evidence_frame_ids:
                assert max(e.evidence_frame_ids) <= frame.frame_id
