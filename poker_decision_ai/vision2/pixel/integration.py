"""与 V1 流水线的组合（不修改任何 V0/V1 文件）。

PixelVisionPipeline = RealtimeVisionPipeline（V0 校验路径原样运行）
                    + PixelVisionEngine（同 Frame/ROI/Calibration 类型，并行消费）。

像素层输出独立于 NormalizedVisionState：不改变、不绕过、不注入 V0 契约；
V0 validator 路径的行为与像素层无关（测试固化）。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..realtime.pipeline import RealtimeVisionObservation
from ..realtime.roi import ROIManager
from .engine import PixelEngineConfig, PixelVisionEngine
from .events import PixelFrameResult


@dataclass
class PixelStepResult:
    vision: RealtimeVisionObservation | None = None
    pixel: PixelFrameResult | None = None

    def to_dict(self) -> dict:
        return {
            "vision": None if self.vision is None else self.vision.to_dict(),
            "pixel": None if self.pixel is None else self.pixel.to_dict(),
        }


class PixelVisionPipeline:
    """组合 V1 感知流水线与 V2A 像素引擎（纯组合，零 V1 改动）。"""

    def __init__(self, pipeline, engine: PixelVisionEngine | None = None,
                 config: PixelEngineConfig | None = None,
                 roi_specs: dict | None = None):
        self.pipeline = pipeline
        self.engine = engine or PixelVisionEngine(config)
        self.roi_specs = dict(roi_specs or pipeline.roi_specs)

    def step(self, now: float | None = None) -> PixelStepResult:
        obs = self.pipeline.step(now)
        frame = self.pipeline.frame_manager.last_frame
        if frame is None or self.pipeline.profile is None:
            return PixelStepResult(vision=obs, pixel=None)
        roiman = ROIManager(self.pipeline.profile, self.roi_specs)
        rois, missing = roiman.resolve()
        crops, crop_missing = roiman.extract_all(frame.image, rois)
        for k, v in crop_missing.items():
            missing[k] = v
        pixel = self.engine.process(frame, rois, crops, missing)
        return PixelStepResult(vision=obs, pixel=pixel)

    def run(self, *, stop_event=None, max_frames: int | None = None,
            sleep_s: float | None = None) -> list:
        """持续轮询（组合两层的观测序列）。"""
        import time as _sleep_time
        out = []
        delay = (self.pipeline.frame_manager.interval
                 if sleep_s is None else sleep_s)
        while stop_event is None or not stop_event.is_set():
            if max_frames is not None and len(out) >= max_frames:
                break
            out.append(self.step())
            _sleep_time.sleep(delay)
        return out
