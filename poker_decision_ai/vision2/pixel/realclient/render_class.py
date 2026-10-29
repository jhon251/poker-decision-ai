"""V3F-P15 — Render-class geometry gate（render_class.py）。

规范 render class 按帧几何 (w,h) 键控：
- "1280": 1280×779   （bet 胶囊类）
- "1419": 1419×1066  （经典窗口，125% DPI 裁剪视图）
- "1848": 1848×1388  （V2E/V2H 时期完整窗口）
- "1876": 1876×1400  （当前对局类，100% Windows 缩放）

window_for_frame(w, h) → 规范 class 字符串；未知几何 → None（绝不猜测）。

设计红线（P15）：
- 派发依据 = 合法 render geometry (w,h) 本身，非 frame id / scene id /
  expected value / historical frame；
- 未知几何 → None（消费者据此返回 EVIDENCE_LIMITED，绝不 fallback 到
  1419/默认几何——防止跨 render class ROI 污染）；
- 本模块零视觉逻辑（纯几何键控，供所有观察者/识别器复用）。
"""
from __future__ import annotations

#: 规范 render class 键控表（几何 → class）
WINDOW_BY_GEOMETRY = {
    (1280, 779): "1280",
    (1419, 1066): "1419",
    (1848, 1388): "1848",
    (1876, 1400): "1876",
}

#: 显式隔离标注状态名（消费者/识别器对未校准 class 的统一返回）
EVIDENCE_LIMITED = "EVIDENCE_LIMITED"


def window_for_frame(w: int, h: int) -> str | None:
    """帧几何 (w,h) → 规范 render class；未知几何 → None。"""
    return WINDOW_BY_GEOMETRY.get((int(w), int(h)))
