"""V3F-P22-C — 1876 Dealer Button 双D模板检测器（最小 adapter）。

- 识别对象 = dealer 按钮内的 **D 字母结构**（白D 或 暗D 双渲染态）；
- 模板来源（GT provenance）：
  - D_white：帧 POT_CHANGED_139（dealer=S6）白 D 字母；
  - D_dark：帧 POT_CHANGED_102（dealer=S5）红圆内暗 D（镂空）；
- 流程：render_class=1876 → 座锚点有界双D NCC → 最佳 match → 座关联 → seat；
- 红线：颜色仅辅助（白/暗掩码），D 字母为主导；不恢复红阈值；
  无 frame-id/scene-id 特判 / previous / cache / 扑克规则推导；
- UNKNOWN-first：D 匹配分数 < 阈值 或 无座关联 → None（UNKNOWN）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ...realtime.image import to_gray  # noqa: F401  (unused, kept for parity)
from .seat_position import SEAT_ANCHORS_1848

ASSET_DIR = "data/vision2/coinpoker_assets_v1876dealer"
#: D 匹配阈值（真实按钮 0.738-0.891；无按钮区应 < 此值）
D_SCORE_MIN = 0.6
#: 座关联最大距离
ASSOC_MAX = 400.0
#: 搜索半径（座锚点周围）
SEARCH_RADIUS = 350


def _load_template(name: str) -> np.ndarray | None:
    """装载 D 模板（二进制化 float）。"""
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    p = Path(ASSET_DIR) / name
    if not p.exists():
        return None
    w, h, rgba = decode_png(p.read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    return (arr[..., 0] > 128).astype(np.float64)


#: 模板缓存（init 一次）
_WHITE = _load_template("d_white.png")
_DARK = _load_template("d_dark.png")


def _ncc_bounded(g: np.ndarray, t: np.ndarray, cx: int, cy: int,
                 radius: int = SEARCH_RADIUS) -> tuple:
    """(cx,cy) ±radius 有界 NCC（stride 2）。返回 (score, (x,y))。

    数值等价加速（语义零改动）：以积分图（SAT）求每窗 Σ/Σ²，FFT 求
    Σ wins·t0 ——与逐窗 mean/去均值/点积同一公式（浮点求和机制不同，
    分数 ε 级差异；阈值/掩码/座位关联不变）。等价性由 46 帧证据回归
    （scripts/p33_vision_regression_dump.py 前后 diff）+ 全量测试验证。
    """
    if t is None:
        return -1.0, None
    th, tw = t.shape
    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(g.shape[1], cx + radius), min(g.shape[0], cy + radius)
    sub = g[y0:y1, x0:x1]
    if sub.shape[0] < th or sub.shape[1] < tw:
        return -1.0, None
    n = float(th * tw)
    sat = np.zeros((sub.shape[0] + 1, sub.shape[1] + 1), dtype=np.float64)
    sat[1:, 1:] = sub.cumsum(0).cumsum(1)
    sat2 = np.zeros((sub.shape[0] + 1, sub.shape[1] + 1), dtype=np.float64)
    sat2[1:, 1:] = (sub * sub).cumsum(0).cumsum(1)

    def _box(table, ys, xs, hh, ww):
        return (table[ys + hh, xs + ww] - table[ys, xs + ww]
                - table[ys + hh, xs] + table[ys, xs])

    ys = np.arange(0, sub.shape[0] - th + 1, 2)
    xs = np.arange(0, sub.shape[1] - tw + 1, 2)
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    s1 = _box(sat, Y, X, th, tw)
    s2 = _box(sat2, Y, X, th, tw)
    wm = s1 / n
    wn = np.sqrt(np.maximum(s2 - 2.0 * wm * s1 + n * wm * wm, 0.0))
    t0 = t - t.mean()
    tn = float(np.sqrt((t0 * t0).sum()))
    from numpy.fft import irfft2, rfft2
    fh, fw = sub.shape[0], sub.shape[1]
    corr = irfft2(rfft2(sub, s=(fh, fw)) * np.conj(rfft2(t0, s=(fh, fw))),
                  s=(fh, fw))
    num = corr[Y, X] - wm * t0.sum()
    scores = np.abs(num / np.maximum(wn * tn, 1e-9))
    idx = np.unravel_index(np.argmax(scores), scores.shape)
    return float(scores[idx]), (x0 + int(xs[idx[1]]), y0 + int(ys[idx[0]]))


def detect_dealer_button_1876(frame: np.ndarray) -> dict:
    """1876 帧 → dealer button 检测 → {dealer_seat, evidence}。

    未知/证据不足 → dealer_seat=None（UNKNOWN，不猜测）。
    """
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = (l > 150).astype(np.float64)
    redm = ((r > 40) & (g < 60) & (b < 60) & (r > g * 2)).astype(np.uint8)
    dark = ((l < 120) & (~(redm > 0))).astype(np.float64)
    best = None                      # (score, (x,y), kind, seat)
    for seat, (ax, ay) in SEAT_ANCHORS_1848.items():
        for mask, tpl, kind in ((white, _WHITE, "white_d"),
                                (dark, _DARK, "dark_d")):
            if tpl is None:
                continue
            sc, pos_ = _ncc_bounded(mask, tpl, ax, ay)
            if pos_ is None or sc < D_SCORE_MIN:
                continue
            d = float(((pos_[0] - ax) ** 2 + (pos_[1] - ay) ** 2) ** 0.5)
            if d <= ASSOC_MAX and (best is None or sc > best[0]):
                best = (sc, pos_, kind, seat)
    if best is None:
        return {"dealer_seat": None, "confidence": 0.0,
                "reason": "no_d_letter_match", "evidence": {
                    "method": "dual_d_ncc", "render_class": "1876"}}
    sc, pos_, kind, seat = best
    return {"dealer_seat": seat, "confidence": round(sc, 3),
            "reason": "d_letter_nearest_seat", "evidence": {
                "method": "dual_d_ncc", "render_class": "1876",
                "kind": kind, "d_pos": list(pos_), "score": round(sc, 3)}}
