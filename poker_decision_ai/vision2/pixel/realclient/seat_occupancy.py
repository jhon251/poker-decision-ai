"""V3F-P27-C — 1876 Seat Occupancy 三态检测器（OCCUPIED / EMPTY / UNKNOWN）。

严格基于 P27-A/B.0 冻结视觉证据：

OCCUPIED（P27-A 对象级三件套，overlay 允许证据不足）：
- gold stack 文本（背景相对判据，复用 P26-B 冻结原语，DIM 鲁棒）
  或 name 灰白字（NORMAL 鲁棒）；
- 阈值：gold_n ≥ 400（"0"值 416/454 之上，动画伪 gold 376 之下）
  或 name ≥ 200（正常占用 685+ 之上，动画 150 之下）——测量分离，非调低。

EMPTY（P27-B.0 组合证据，禁止单一缺失信号）：
- 全部满足才输出 EMPTY：
  1. gold 缺失（gold_n < 100）
  2. name 缺失（name < 50）
  3. avatar/牌面缺失（maroon < 200）
  4. dim「空的」文本结构（锚点区 dim 灰字 ≥150px 且 列 run ≥12px）
  —— away「暂离」（纯白 253+ 有 gold+name）→ OCCUPIED；
  —— hero overlay（红牌 maroon 28710）→ 非 EMPTY；
  —— 动画 overlay（dim 文本 + maroon 26395）→ 非 EMPTY。

UNKNOWN（其余全部）：证据不足/裁剪/overlay/冲突/只有单一缺失信号。

红线：绝不 panel-missing→EMPTY / gold-missing→EMPTY / avatar-missing→EMPTY；
不用 Stack/Bet/Status/Dealer/Hero 反推；无 previous/cache；无视觉 LLM；
1419 路径零改动；UNKNOWN-first。
"""
from __future__ import annotations

import numpy as np

from .coinpoker_stack import SEAT_STACK_ROIS_1876, stack_gold_1876

#: 每座锚点（1876 → 1848 锚点，P27-A 沿用）
SEAT_ANCHORS_1876 = {
    "S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
    "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160),
}

#: OCCUPIED 阈值（P27-A 实测：gold "0"=416/454, 动画伪 gold=376；
#: name 正常占用 ≥685, 动画=150）
OCC_GOLD_MIN = 400
OCC_NAME_MIN = 200

#: EMPTY 组合阈值（P27-B.0 实测：S4 空 gold=0/name=0/maroon=0/
#: dim 字=284/run=20；hero overlay maroon=28710；动画 maroon=26395）
EMPTY_GOLD_MAX = 100
EMPTY_NAME_MAX = 50
EMPTY_MAROON_MAX = 200
EMPTY_DIM_LABEL_MIN = 150
EMPTY_DIM_RUN_MIN = 12


def _name_ink(frame: np.ndarray, seat: str) -> int:
    """name 行灰白字量（stack ROI 上 52px；luma>110 近中性）。"""
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876[seat]
    if y0 - 52 < 0 or x1 > frame.shape[1] or y1 > frame.shape[0]:
        return -1
    ln = frame[y0 - 52:y0 - 6, x0:x1]
    l = ln[..., 0] * 0.299 + ln[..., 1] * 0.587 + ln[..., 2] * 0.114
    r = ln[..., 0].astype(np.int16); g = ln[..., 1].astype(np.int16)
    b = ln[..., 2].astype(np.int16)
    return int(((l > 110) & (np.abs(r - g) < 28)
                & (np.abs(g - b) < 28)).sum())


def _gold_n(frame: np.ndarray, seat: str) -> int:
    """stack 行 gold 像素量（背景相对判据，P26-B 冻结原语）。"""
    x0, y0, x1, y1 = SEAT_STACK_ROIS_1876[seat]
    if x1 > frame.shape[1] or y1 > frame.shape[0] or y0 < 0 or x0 < 0:
        return -1
    roi = frame[y0:y1, x0:x1]
    m, _ = stack_gold_1876(roi)
    return int(m.sum())


def _maroon_n(frame: np.ndarray, seat: str) -> int:
    """头像/牌面 maroon 环像素量（锚点 ±100/±90）。"""
    ax, ay = SEAT_ANCHORS_1876[seat]
    x0, y0 = max(0, ax - 100), max(0, ay - 90)
    x1, y1 = min(frame.shape[1], ax + 100), min(frame.shape[0], ay + 90)
    sub = frame[y0:y1, x0:x1]
    r = sub[..., 0].astype(np.int16); g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    return int(((r > g + 20) & (r > b + 10) & (r > 40) & (r < 150)).sum())


def _dim_label_sig(frame: np.ndarray, seat: str) -> tuple:
    """锚点 ±70 区 dim「空的」文本：灰字量 + 最大列 run。

    「空的」= dim 灰（luma 80-190 近中性，非纯白——区别于 away 暂离纯白）；
    列 run ≥12px = 文本块结构（区别于散斑）。
    """
    ax, ay = SEAT_ANCHORS_1876[seat]
    x0, y0 = max(0, ax - 70), max(0, ay - 50)
    x1, y1 = min(frame.shape[1], ax + 70), min(frame.shape[0], ay + 60)
    roi = frame[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    dim = ((l >= 80) & (l <= 190) & (np.abs(r - g) < 25)
           & (np.abs(g - b) < 25))
    dim_n = int(dim.sum())
    cold = dim.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v >= 4 and run is None:
            run = [i, i]
        elif v >= 4 and run is not None:
            run[1] = i
        elif run is not None:
            runs.append(run)
            run = None
    if run is not None:
        runs.append(run)
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= 3:
            merged[-1][1] = rr[1]
        else:
            merged.append(list(rr))
    max_run = max((b - a + 1 for a, b in merged), default=0)
    return dim_n, max_run


class SeatOccupancyRecognizer1876:
    """1876 帧 → 每座 OCCUPIED / EMPTY / UNKNOWN（UNKNOWN-first）。"""

    name = "coinpoker_seat_occupancy_v3f_p27c_1876"

    def _seat(self, frame: np.ndarray, seat: str) -> dict:
        gold = _gold_n(frame, seat)
        name = _name_ink(frame, seat)
        maroon = _maroon_n(frame, seat)
        dim_n, dim_run = _dim_label_sig(frame, seat)
        evidence = {"gold_n": gold, "name_ink": name, "maroon_n": maroon,
                    "dim_label_n": dim_n, "dim_label_run": dim_run}
        # 裁剪/出界 → UNKNOWN
        if gold < 0 or name < 0:
            return {"status": "UNKNOWN", "reason": "roi_out_of_frame",
                    "evidence": evidence}
        # OCCUPIED：gold 或 name（overlay 允许不足）
        if gold >= OCC_GOLD_MIN or name >= OCC_NAME_MIN:
            return {"status": "OCCUPIED", "reason": "seat_box_evidence",
                    "evidence": evidence}
        # EMPTY：四条件全满足（禁止单一缺失信号）
        if (gold < EMPTY_GOLD_MAX and name < EMPTY_NAME_MAX
                and maroon < EMPTY_MAROON_MAX
                and dim_n >= EMPTY_DIM_LABEL_MIN
                and dim_run >= EMPTY_DIM_RUN_MIN):
            return {"status": "EMPTY", "reason": "empty_label_evidence",
                    "evidence": evidence}
        return {"status": "UNKNOWN", "reason": "insufficient_evidence",
                "evidence": evidence}

    def recognize(self, frame: np.ndarray) -> dict:
        return {seat: self._seat(frame, seat)
                for seat in sorted(SEAT_STACK_ROIS_1876)}
