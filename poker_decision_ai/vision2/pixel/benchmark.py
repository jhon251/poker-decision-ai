"""V2A 像素核心基准（P50/P95/P99 实测；不编造）。

- 输入：synthetic 场景循环（确定性）或真实截图文件循环（如可用）；
- 阶段：changed_mask / changed_bbox / changed_ratio / rgb_stats /
  hsv_stats / dominant_color / engine_total；
- 数字来自实测样本；真实截图解码耗时单独报告（不并入阶段）。
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from ..realtime.benchmark import percentile
from ..realtime.coords import Rect
from ..realtime.fixtures import DEFAULT_ROI_SPECS, preflop_scene
from ..realtime.pipeline import StageTimer
from .engine import PixelEngineConfig, PixelVisionEngine
from .stats import (changed_bbox, changed_pixel_mask, changed_pixel_ratio,
                    dominant_color, hsv_stats, rgb_stats)


@dataclass
class PixelBenchmarkResult:
    stages: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    input_kind: str = "synthetic"

    def to_dict(self) -> dict:
        return {"stages": self.stages, "environment": self.environment,
                "input_kind": self.input_kind}

    def to_json(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    def render_table(self) -> str:
        def fmt(v, width=10, digits=3):
            if v is None:
                return "n/a".rjust(width)
            return f"{v:>{width}.{digits}f}"

        lines = [f"{'stage':<22}{'count':>7}{'P50 ms':>10}{'P95 ms':>10}"
                 f"{'P99 ms':>10}{'mean ms':>10}"]
        for s in self.stages.values():
            lines.append(f"{s['name']:<22}{s['count']:>7}"
                         f"{fmt(s['p50_ms'])}{fmt(s['p95_ms'])}"
                         f"{fmt(s['p99_ms'])}{fmt(s['mean_ms'])}")
        return "\n".join(lines)


def _ms(name: str, samples: list) -> dict:
    s = sorted(samples)
    if not s:
        return {"name": name, "count": 0, "p50_ms": None, "p95_ms": None,
                "p99_ms": None, "mean_ms": None}
    return {"name": name, "count": len(s),
            "p50_ms": percentile(s, 50) * 1000.0,
            "p95_ms": percentile(s, 95) * 1000.0,
            "p99_ms": percentile(s, 99) * 1000.0,
            "mean_ms": sum(s) / len(s) * 1000.0}


def _build_engine_frames(kind: str):
    """构造 (engine, frames, rois, crops0, crops1, synthetic)。"""
    from ..realtime.anchors import calibrate
    from ..realtime.capture import CapturedFrame
    from ..realtime.coords import GeometryContext
    from ..realtime.frame_manager import Frame
    from ..realtime.image import ImageData
    from ..realtime.roi import ROIManager

    if kind == "synthetic":
        scenes = [preflop_scene("s0", pot=1.5), preflop_scene("s1", pot=4.0)]
        captured = [s.render(now=float(i)) for i, s in enumerate(scenes)]
        synthetic = True
        det = {"table": (Rect(40.0, 40.0, 1200.0, 690.0), 1.0)}
    else:
        from pathlib import Path
        import poker_decision_ai.vision2_platform.png_decode as _pd
        png_dir = Path(__file__).resolve().parents[3] / "picture data"
        files = sorted(png_dir.glob("*.png"))[:2]
        if len(files) < 2:
            raise FileNotFoundError(f"真实截图不足 2 张: {png_dir}")
        captured = []
        for i, p in enumerate(files):
            w, h, rgba = _pd.decode_png(p.read_bytes())
            geo = GeometryContext(screen_size=(float(w), float(h)),
                                  window_rect=None, client_offset=(0.0, 0.0),
                                  client_size_physical=(float(w), float(h)),
                                  dpi_scale=1.0)
            captured.append(CapturedFrame(
                image=ImageData.from_rgba_bytes(w, h, rgba),
                captured_at=float(i), geometry=geo,
                source=f"real:{p.name}", synthetic_test_only=False))
        synthetic = False
        det = {"table": (Rect(0.0, 0.0, float(captured[0].image.width),
                              float(captured[0].image.height)), 1.0)}

    frames = [Frame.from_capture(c, i + 1) for i, c in enumerate(captured)]
    prof = calibrate("bench", captured[0].geometry, det)
    rois, _ = ROIManager(prof, dict(DEFAULT_ROI_SPECS)).resolve()
    mgr = ROIManager(prof, dict(DEFAULT_ROI_SPECS))
    crops0, _ = mgr.extract_all(frames[0].image, rois)
    crops1, _ = mgr.extract_all(frames[1].image, rois)
    engine = PixelVisionEngine(PixelEngineConfig())
    return engine, frames, rois, crops0, crops1, synthetic


def run_pixel_benchmark(iterations: int = 200, warmup: int = 10,
                        input_kind: str = "synthetic") -> PixelBenchmarkResult:
    engine, frames, rois, crops0, crops1, synthetic = _build_engine_frames(
        input_kind)
    pot0, pot1 = crops0["pot"], crops1["pot"]
    timers: dict = {}

    for _ in range(max(warmup, 1)):
        with StageTimer(timers, "changed_mask"):
            changed_pixel_mask(pot0, pot1)
    for _ in range(iterations):
        with StageTimer(timers, "changed_mask"):
            changed_pixel_mask(pot0, pot1)
        with StageTimer(timers, "changed_bbox"):
            changed_bbox(pot0, pot1)
        with StageTimer(timers, "changed_ratio"):
            changed_pixel_ratio(pot0, pot1)
        with StageTimer(timers, "rgb_stats"):
            rgb_stats(pot0)
        with StageTimer(timers, "hsv_stats"):
            hsv_stats(pot0)
        with StageTimer(timers, "dominant_color"):
            dominant_color(pot0)

    engine.reset()
    for i in range(max(warmup, 1)):
        engine.process(frames[i % 2], rois,
                       crops0 if i % 2 == 0 else crops1)
    engine.reset()
    for i in range(iterations):
        with StageTimer(timers, "engine_total"):
            engine.process(frames[i % 2], rois,
                           crops0 if i % 2 == 0 else crops1)

    stages = {name: _ms(name, samples) for name, samples in timers.items()}
    env = {"python": sys.version.split()[0], "input_kind": input_kind,
           "synthetic_input": synthetic, "iterations": iterations,
           "warmup": warmup}
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        env["numpy"] = None
    env["note"] = ("synthetic 场景为确定性 fixture；real 输入为 "
                   "picture data 下真实截图（只读，不修改文件）")
    return PixelBenchmarkResult(stages=stages, environment=env,
                                input_kind=input_kind)


def run_object_benchmark(iterations: int = 100, warmup: int = 5,
                         input_kind: str = "synthetic") -> PixelBenchmarkResult:
    """V2B 对象层基准：几何/分割/模板/数字/卡牌 + 引擎全量（几何开启）。"""
    from ..realtime.image import to_gray
    from .geometry import connected_components, edge_density, otsu_threshold, \
        threshold_mask
    from .fonts import builtin_digit_templates
    from .templates import TemplateEntry, TemplateLibrary, match_library
    from .digit import recognize_digits
    from .cards import builtin_card_library, recognize_card_slot, \
        render_test_card

    engine, frames, rois, crops0, crops1, synthetic = _build_engine_frames(
        input_kind)
    pot0 = crops0["pot"]
    gray = to_gray(pot0)
    mask = threshold_mask(gray)

    digit_tpls = builtin_digit_templates()
    small_lib = TemplateLibrary([
        TemplateEntry(f"digit_{ch}", img, roi_hint="pot")
        for ch, img in list(digit_tpls.items())[:2]])
    card_lib = builtin_card_library()
    card_img = render_test_card("A", "s", scale=3)
    obj_engine = PixelVisionEngine(PixelEngineConfig())

    timers: dict = {}
    for _ in range(max(warmup, 1)):
        otsu_threshold(gray)
    for _ in range(iterations):
        with StageTimer(timers, "otsu_threshold"):
            otsu_threshold(gray)
        with StageTimer(timers, "threshold_mask"):
            threshold_mask(gray)
        with StageTimer(timers, "connected_components"):
            connected_components(mask)
        with StageTimer(timers, "edge_density"):
            edge_density(gray)
        with StageTimer(timers, "match_template_x2"):
            match_library(pot0, small_lib, roi_hint="pot")
        with StageTimer(timers, "digit_recognize"):
            recognize_digits(pot0, digit_tpls)
        with StageTimer(timers, "card_recognize_slot"):
            recognize_card_slot(card_img, card_lib)

    obj_engine.reset()
    for i in range(max(warmup, 1)):
        obj_engine.process(frames[i % 2], rois,
                           crops0 if i % 2 == 0 else crops1)
    obj_engine.reset()
    for i in range(iterations):
        with StageTimer(timers, "engine_total_v2b"):
            obj_engine.process(frames[i % 2], rois,
                               crops0 if i % 2 == 0 else crops1)

    stages = {name: _ms(name, samples) for name, samples in timers.items()}
    env = {"python": sys.version.split()[0], "input_kind": input_kind,
           "synthetic_input": synthetic, "iterations": iterations,
           "warmup": warmup,
           "note": ("geometry/segmentation/template/digit/card 均为确定性"
                    "本地计算（builtin_synthetic 模板库），无远程调用")}
    return PixelBenchmarkResult(stages=stages, environment=env,
                                input_kind=input_kind)
