"""Vision Agent V2 — Pixel Core + Object Recognition（Phase V2A/V2B）。

职责边界：
  - V2A 像素事实：变化比率/包围盒、RGB/HSV 特征、主色、ROI 时序状态、
    PixelEvent / RawPixelObservation、证据帧；
  - V2B 对象层：几何特征（连通域/边缘）、分割候选、模板库匹配、
    数字字形识别、确定性卡牌模板识别（实现 V1 后端接口）；
  - 明确不实现：行动语义、意图推断、行动者归因、决策输出、历史统计链、
    自动操作；
  - 未知 = null；事件携带 evidence_frame_ids；禁止引用未来帧；
  - synthetic 与 real 严格区分；builtin 模板 source 恒标注
    builtin_synthetic，未标注验证不宣称识别成功；
  - 不修改任何 V0/V1 文件（仅 import 复用其类型）。
"""
from __future__ import annotations

from .events import (ALL_PIXEL_EVENT_KINDS, PixelEvent, PixelEventKind,
                     PixelFrameResult, RawPixelObservation)
from .stats import (changed_bbox, changed_pixel_count, changed_pixel_mask,
                    changed_pixel_ratio, color_distance, dominant_color,
                    hsv_from_rgb, hsv_stats, rgb_stats)
from .temporal import PixelTemporalConfig, PixelTemporalTracker
from .geometry import (Component, connected_components, edge_density,
                       otsu_threshold, threshold_mask)
from .segmentation import classify_components, segment_components
from .templates import (TemplateEntry, TemplateLibrary, TemplateMatch,
                        best_match, foreground_is_dark, invert_image,
                        match_library, polarity_normalized)
from .fonts import (builtin_card_back, builtin_digit_templates,
                    builtin_rank_templates, builtin_suit_templates)
from .digit import (DigitRecognition, DigitRecognitionStatus,
                    DigitTemplateOCRBackend, GlyphMatch, normalize_glyph,
                    recognize_digits, segment_cells, segment_glyphs,
                    segment_runs)
from .cards import (CardTemplateLibrary, CardTemplateRecognitionBackend,
                    builtin_card_library, card_candidate_slots,
                    recognize_card_slot, render_test_card)
from .engine import PixelEngineConfig, PixelVisionEngine
from .integration import PixelStepResult, PixelVisionPipeline

__all__ = [
    "ALL_PIXEL_EVENT_KINDS", "PixelEvent", "PixelEventKind",
    "PixelFrameResult", "RawPixelObservation",
    "changed_bbox", "changed_pixel_count", "changed_pixel_mask",
    "changed_pixel_ratio", "color_distance", "dominant_color",
    "hsv_from_rgb", "hsv_stats", "rgb_stats",
    "PixelTemporalConfig", "PixelTemporalTracker",
    "Component", "connected_components", "edge_density",
    "otsu_threshold", "threshold_mask",
    "classify_components", "segment_components",
    "TemplateEntry", "TemplateLibrary", "TemplateMatch", "best_match",
    "match_library", "polarity_normalized", "invert_image",
    "foreground_is_dark",
    "builtin_card_back", "builtin_digit_templates",
    "builtin_rank_templates", "builtin_suit_templates",
    "DigitRecognition", "DigitRecognitionStatus", "DigitTemplateOCRBackend",
    "GlyphMatch", "normalize_glyph", "recognize_digits", "segment_glyphs",
    "segment_cells", "segment_runs",
    "CardTemplateLibrary", "CardTemplateRecognitionBackend",
    "builtin_card_library", "card_candidate_slots", "recognize_card_slot",
    "render_test_card",
    "PixelEngineConfig", "PixelVisionEngine",
    "PixelStepResult", "PixelVisionPipeline",
]
