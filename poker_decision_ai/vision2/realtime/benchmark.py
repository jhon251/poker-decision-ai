"""Benchmark harness：分阶段 P50/P95/P99 实测（不编造数字）。

- 全部输入为 synthetic fixture（确定性场景序列）；真实截图/OCR/多模态
  路径的 latency 未经实测，报告必须明确标注；
- 统计口径：线性插值分位数；P50/P95/P99 只来自实测样本；
- fast path 与 multimodal fallback 分别测量（fallback 用冲突场景强制触发）。
"""
from __future__ import annotations

import json
import platform as _platform
import sys
from dataclasses import dataclass, field

from .capture import FixtureCaptureProvider
from .fallback import FallbackPolicyConfig
from .fixtures import (DEFAULT_ROI_SPECS, duplicate_card_flop_scene,
                       scenes_street_progression)
from .pipeline import RealtimeVisionConfig, RealtimeVisionPipeline


def percentile(sorted_samples: list, p: float) -> float:
    """线性插值分位数（确定性；空样本 → 0.0）。"""
    if not sorted_samples:
        return 0.0
    n = len(sorted_samples)
    if n == 1:
        return float(sorted_samples[0])
    pos = (n - 1) * (p / 100.0)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_samples[lo]) * (1.0 - frac) \
        + float(sorted_samples[hi]) * frac


@dataclass
class StageStats:
    name: str = ""
    count: int = 0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    total_s: float = 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "count": self.count, "p50_ms": self.p50,
                "p95_ms": self.p95, "p99_ms": self.p99, "mean_ms": self.mean,
                "min_ms": self.minimum, "max_ms": self.maximum,
                "total_s": self.total_s}


@dataclass
class BenchmarkResult:
    stages: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    synthetic_input: bool = True

    def to_dict(self) -> dict:
        return {"stages": self.stages, "config": self.config,
                "environment": self.environment,
                "synthetic_input": self.synthetic_input}

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
                 f"{'P99 ms':>10}{'mean ms':>10}{'total s':>10}"]
        for s in self.stages.values():
            lines.append(f"{s['name']:<22}{s['count']:>7}"
                         f"{fmt(s['p50_ms'])}{fmt(s['p95_ms'])}"
                         f"{fmt(s['p99_ms'])}{fmt(s['mean_ms'])}"
                         f"{fmt(s['total_s'])}")
        return "\n".join(lines)


def _ms(samples: list) -> StageStats:
    s = sorted(samples)
    return StageStats(count=len(s),
                      p50=percentile(s, 50) * 1000.0,
                      p95=percentile(s, 95) * 1000.0,
                      p99=percentile(s, 99) * 1000.0,
                      mean=(sum(s) / len(s) * 1000.0) if s else 0.0,
                      minimum=(s[0] * 1000.0) if s else 0.0,
                      maximum=(s[-1] * 1000.0) if s else 0.0,
                      total_s=sum(s))


def _build_pipeline(scenes, *, fps: float, cycles: int,
                    fallback_min_interval_s: float | None,
                    fallback_budget: int) -> RealtimeVisionPipeline:
    """确定性 synthetic 流水线：后端 scene 与 provider 帧精确同步。

    fallback_min_interval_s=None → fallback 完全禁用（快路径基准）。
    """
    from .fallback import SyntheticMultimodalBackend
    from .recognition import (SyntheticSceneCardBackend,
                              SyntheticSceneOCRBackend)

    all_scenes = scenes * cycles
    provider = FixtureCaptureProvider(
        [s.render(now=float(i), source="benchmark")
         for i, s in enumerate(all_scenes)], cycle=True)

    def get_scene():
        return all_scenes[provider.index % len(all_scenes)]

    ocr = SyntheticSceneOCRBackend(get_scene)
    cards = SyntheticSceneCardBackend(get_scene)
    mm = SyntheticMultimodalBackend(get_scene)
    anchor_templates = {"table": scenes[0].anchor_template("table")}
    anchor_sizes = {"table": (scenes[0].anchors_client["table"].width,
                              scenes[0].anchors_client["table"].height)}
    fallback_enabled = fallback_min_interval_s is not None
    config = RealtimeVisionConfig(
        fps=fps,
        fallback=FallbackPolicyConfig(
            enabled=fallback_enabled,
            min_interval_s=(fallback_min_interval_s or 0.0),
            max_calls_per_hand=fallback_budget))
    return RealtimeVisionPipeline(
        config=config, capture_provider=provider, ocr=ocr, card=cards,
        multimodal=mm, roi_specs=dict(DEFAULT_ROI_SPECS),
        anchor_templates=anchor_templates, anchor_sizes=anchor_sizes)


def run_fast_path_benchmark(iterations: int = 200, warmup: int = 10,
                            fps: float = 10.0) -> BenchmarkResult:
    """synthetic 快路径分阶段实测（无兜底调用）。"""
    scenes = scenes_street_progression()
    pipe = _build_pipeline(scenes, fps=fps, cycles=40,
                           fallback_min_interval_s=None, fallback_budget=1)
    t0 = float(warmup + iterations + 10)
    for i in range(warmup):
        pipe.step(now=t0 + float(i))
    pipe.timers.clear()
    for i in range(iterations):
        pipe.step(now=t0 + float(warmup + i))
    return _result(pipe, iterations, warmup, fps, scenes[0],
                   mode="fast_path")


def run_fallback_benchmark(iterations: int = 60, warmup: int = 5,
                           fps: float = 10.0) -> BenchmarkResult:
    """synthetic 冲突场景 → 每帧 STATE_CONFLICT → 强制兜底调用实测。"""
    scenes = [duplicate_card_flop_scene(pot=p) for p in (10.0, 12.0, 14.0, 16.0)]
    pipe = _build_pipeline(scenes, fps=fps, cycles=100,
                           fallback_min_interval_s=0.0, fallback_budget=100)
    t0 = float(warmup + iterations + 10)
    for i in range(warmup):
        pipe.step(now=t0 + float(i))
    pipe.timers.clear()
    for i in range(iterations):
        pipe.step(now=t0 + float(warmup + i))
    return _result(pipe, iterations, warmup, fps, scenes[0],
                   mode="fallback_forced")


def _result(pipe, iterations, warmup, fps, scene, mode) -> BenchmarkResult:
    stages = {}
    for name, samples in pipe.timers.items():
        st = _ms(samples)
        st.name = name
        stages[name] = st.to_dict()
    if "multimodal_fallback" not in stages:
        stages["multimodal_fallback"] = {
            "name": "multimodal_fallback", "count": 0, "p50_ms": None,
            "p95_ms": None, "p99_ms": None, "mean_ms": None,
            "min_ms": None, "max_ms": None, "total_s": 0.0,
            "note": "not exercised (fast path: no remote calls)"}
    env = {
        "python": sys.version.split()[0],
        "platform": _platform.platform(),
        "synthetic_input": True,
        "mode": mode,
        "iterations": iterations,
        "warmup": warmup,
        "frame_size": (int(scene.geometry.client_size_physical[0]),
                       int(scene.geometry.client_size_physical[1])),
    }
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except Exception:
        env["numpy"] = None
    config = {
        "fps": fps, "dedup_threshold": pipe.config.dedup_threshold,
        "roi_change_threshold": pipe.config.roi_change_threshold,
        "anchor_match_threshold": pipe.config.anchor_match_threshold,
        "fallback_min_interval_s": pipe.config.fallback.min_interval_s,
        "note": "synthetic inputs only; real capture/OCR/multimodal latency "
                "is unmeasured and must not be inferred from these numbers",
    }
    return BenchmarkResult(stages=stages, config=config, environment=env,
                           synthetic_input=True)


def run_benchmark(iterations: int = 200, warmup: int = 10,
                  fps: float = 10.0) -> BenchmarkResult:
    """兼容入口：快路径 + 兜底两段实测合并。"""
    fast = run_fast_path_benchmark(iterations, warmup, fps)
    fb = run_fallback_benchmark(max(20, iterations // 4), max(3, warmup // 2),
                                fps)
    stages = dict(fast.stages)
    for name, st in fb.stages.items():
        if name in ("capture", "layout", "roi_extraction", "cheap_detection",
                    "validation", "temporal", "total_fast_path"):
            continue
        stages[f"{name}"] = st
    fast.stages = stages
    fast.environment["fallback_mode_note"] = (
        "multimodal_fallback stage measured on forced-conflict synthetic "
        "scenes (separate run)")
    return fast
