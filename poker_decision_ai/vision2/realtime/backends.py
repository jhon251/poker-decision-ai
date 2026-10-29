"""平台后端工厂：按需加载 vision2_platform 的可选真实后端（可用性门控）。

真实后端（系统截图 / tesseract / 多模态 LLM）依赖可选第三方库与
vision-capable 模型；不可用时返回 None + 原因（诚实降级，不伪造）。
本模块只做惰性加载，不在 vision2 包内引入任何第三方图像/OCR 依赖。
"""
from __future__ import annotations

import importlib


def platform_capture(*, dpi_scale: float = 1.0):
    """系统截图 provider（只读截图）；不可用 → (None, reason)。"""
    try:
        mod = importlib.import_module(
            "poker_decision_ai.vision2_platform.capture_backends")
        provider = mod.SystemScreenCaptureProvider(dpi_scale=dpi_scale)
        if provider.available():
            return provider, ""
        return None, provider.unavailable_reason()
    except Exception as e:  # pragma: no cover - 环境缺失
        return None, f"platform capture unavailable: {e}"


def platform_ocr():
    """tesseract 数字/文字 OCR；不可用 → (None, reason)。"""
    try:
        mod = importlib.import_module(
            "poker_decision_ai.vision2_platform.ocr_backends")
        backend = mod.TesseractNumberOCRBackend()
        if backend.available():
            return backend, ""
        return None, backend.unavailable_reason()
    except Exception as e:  # pragma: no cover - 环境缺失
        return None, f"platform OCR unavailable: {e}"


def platform_multimodal():
    """多模态 LLM 兜底（复用核心 provider，需 vision-capable 模型）；
    不可用 → (None, reason)。"""
    try:
        mod = importlib.import_module(
            "poker_decision_ai.vision2_platform.multimodal_backends")
        backend = mod.build_llm_multimodal_backend()
        if backend is not None and backend.available():
            return backend, ""
        reason = backend.unavailable_reason() if backend is not None \
            else "LLM multimodal backend not buildable"
        return None, reason
    except Exception as e:  # pragma: no cover - 环境缺失
        return None, f"platform multimodal unavailable: {e}"


def platform_status() -> dict:
    """三后端可用性总览（配置/演示用）。"""
    cap, cap_reason = platform_capture()
    ocr, ocr_reason = platform_ocr()
    mm, mm_reason = platform_multimodal()
    return {
        "capture": {"available": cap is not None, "reason": cap_reason},
        "ocr": {"available": ocr is not None, "reason": ocr_reason},
        "multimodal": {"available": mm is not None, "reason": mm_reason},
    }
