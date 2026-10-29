"""Vision Agent V2 — 离线数据契约与验证层（Phase V0）。

本包只负责：RawVisionState → 15 条确定性 validator → NormalizedVisionState。
红线：无 OCR/CV/ML；无真实截图识别；不接 7A–8C；不做决策；缺失字段一律
null + uncertain，禁止默认填 0/[]/""；validator 冲突显式暴露，不静默修复；
position 由冻结 `positions_for` 确定性推导，视觉层不得猜。

V0 阶段所有 fixture 标注 `synthetic_test_only=True`，不得冒充真实视觉能力。
"""
from __future__ import annotations

from .schemas import (
    RawAction,
    RawSeat,
    RawVisionState,
    UncertaintyEntry,
    NormalizedVisionState,
    ValidationResult,
    to_dict,
)
from .validator import validate_raw
from .assembly import assemble

__all__ = [
    "RawAction", "RawSeat", "RawVisionState", "UncertaintyEntry",
    "NormalizedVisionState", "ValidationResult", "to_dict",
    "validate_raw", "assemble",
]
