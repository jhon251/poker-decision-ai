"""Vision Agent V2 — Real-Time 感知层（Phase V1）。

职责（纯感知，冻结边界）：
  Screen Capture(只读) → Frame Manager → Coordinate/Anchor Layer →
  ROI Extraction → Cheap Change Detection → OCR → Card Recognition →
  Multimodal Fallback(低频) → Temporal Tracking → V0 Validation →
  NormalizedVisionState。

红线：
  - 不实现任何自动操作（点击/键盘/浏览器/客户端控制/投注 API）；
  - 不输出任何决策字段；不接入历史统计验证链；
  - STATE_CONFLICT 与关键 uncertainty 硬阻断下游；
  - 未知值 = null + uncertainty（禁止填 0/[]/""）；
  - position 只由冻结 positions_for 确定性推导；
  - 所有 synthetic fixture/后端恒 synthetic_test_only=True，不得冒充真实能力；
  - 多模态兜底绝不每帧调用（事件触发 + cooldown + 预算）。
"""
from __future__ import annotations

from .capture import (CaptureResult, CaptureStatus, CapturedFrame,
                      FixtureCaptureProvider, ScreenCaptureProvider)
from .coords import (CoordinateError, CoordinateSystem, CoordinateTransformer,
                     GeometryContext, Point, Rect)
from .anchors import (Anchor, CalibrationProfile, bootstrap_profile, calibrate,
                      detect_anchor_drift)
from .roi import ROI, ROIManager, ROIName, ROISpec, default_roi_specs
from .image import (ImageData, byte_equality_ratio, dhash, dhash_hex,
                    hamming_distance, ncc_template_match, pixel_diff_ratio)
from .frame_manager import (ALLOWED_FPS, DEFAULT_FPS, Frame, FrameManager,
                            FrameResult, FrameStatus)
from .recognition import (CardRecognitionBackend, CardRecognitionResult,
                          CardRecognitionStatus, DetectedCard, OCRBackend,
                          OCRResult, OCRStatus, SyntheticSceneCardBackend,
                          SyntheticSceneOCRBackend, UnavailableCardBackend,
                          UnavailableOCRBackend)
from .fallback import (FallbackCall, FallbackGate, FallbackPolicyConfig,
                       FallbackTrigger, MultimodalBackend, MultimodalResult,
                       MultimodalStatus, SyntheticMultimodalBackend,
                       UnavailableMultimodalBackend)
from .temporal import (EventKind, FieldDelta, ObservationSnapshot,
                       StateTransition, TemporalEvent, TemporalTracker)
from .status import RecognitionStatus
from .pipeline import (RealtimeVisionConfig, RealtimeVisionObservation,
                       RealtimeVisionPipeline, parse_action_text)

__all__ = [
    # capture / frame
    "CaptureResult", "CaptureStatus", "CapturedFrame",
    "FixtureCaptureProvider", "ScreenCaptureProvider",
    "ALLOWED_FPS", "DEFAULT_FPS", "Frame", "FrameManager", "FrameResult",
    "FrameStatus",
    # coords / anchors / roi
    "CoordinateError", "CoordinateSystem", "CoordinateTransformer",
    "GeometryContext", "Point", "Rect",
    "Anchor", "CalibrationProfile", "bootstrap_profile", "calibrate",
    "detect_anchor_drift",
    "ROI", "ROIManager", "ROIName", "ROISpec", "default_roi_specs",
    # image
    "ImageData", "byte_equality_ratio", "dhash", "dhash_hex",
    "hamming_distance", "ncc_template_match", "pixel_diff_ratio",
    # recognition / fallback
    "CardRecognitionBackend", "CardRecognitionResult", "CardRecognitionStatus",
    "DetectedCard", "OCRBackend", "OCRResult", "OCRStatus",
    "SyntheticSceneCardBackend", "SyntheticSceneOCRBackend",
    "UnavailableCardBackend", "UnavailableOCRBackend",
    "FallbackCall", "FallbackGate", "FallbackPolicyConfig", "FallbackTrigger",
    "MultimodalBackend", "MultimodalResult", "MultimodalStatus",
    "SyntheticMultimodalBackend", "UnavailableMultimodalBackend",
    # temporal / pipeline
    "EventKind", "FieldDelta", "ObservationSnapshot", "StateTransition",
    "TemporalEvent", "TemporalTracker", "RecognitionStatus",
    "RealtimeVisionConfig", "RealtimeVisionObservation",
    "RealtimeVisionPipeline", "parse_action_text",
]
