"""vision2_platform —— 真实设备后端（可选依赖，可用性门控）。

与 vision2/realtime 的关系：realtime 包是纯逻辑层（零第三方图像/OCR
依赖，兼容 V0 红线扫描）；本包承载需要真实设备的适配器：
  - 只读系统截图（Pillow ImageGrab / mss，均纯截图、无控制注入）；
  - tesseract 数字/文字 OCR；
  - 多模态 LLM 兜底（复用核心 MultimodalLLMVisionProvider）。

红线：
  - 截图只读：绝不导入/调用鼠标、键盘、点击、投注、客户端控制 API；
  - 全部后端可用性门控：依赖缺失/未配置 → available()=False → 诚实降级；
  - 不在此实现任何识别逻辑之外的决策能力。
"""
from __future__ import annotations

from .png_encode import encode_rgba_png
from .png_decode import decode_png

__all__ = ["encode_rgba_png", "decode_png"]
