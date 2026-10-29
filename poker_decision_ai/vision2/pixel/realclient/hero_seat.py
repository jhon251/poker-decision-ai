"""V3F-P28-B — 1876 Hero own-seat detector（UNKNOWN-first）。

严格基于 P28-A 冻结视觉事实：本地玩家座牌盒左侧的「表情反应 pill 簇」
= 💬 白气泡 + 😎 金色表情按钮 双件组合，是客户端专属 UI（非英雄座无）。

判据（P28-B 实测分离）：
- 💬 白气泡：相对亮度 l>bg×1.6 近中性（NORMAL/DIM/extreme 同判据）
- 😎 gold-face：相对亮度 l>max(60, bg×2.0) 且 r≥g×0.9 且 b<r+40
- 阈值：white ≥ 2500 ∧ gold ≥ 1000（实测 hero 7/7 white 5307-7339 /
  gold 1560-5984；非英雄 white ≤896 / gold ≤664——3-6× 裕量，非调低）

UNKNOWN-first：0 命中 → None；>1 命中 → None（冲突）；ROI 出界 → None。
禁止：默认 S6 / dealer/position/occupied/HeroCards/stack 反推 /
table_size=6 / previous/cache / 视觉 LLM。
"""
from __future__ import annotations

import numpy as np

#: 每座锚点（1876 → 1848 锚点，P28-A 沿用）
SEAT_ANCHORS_1876 = {
    "S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
    "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160),
}

#: pill ROI 相对锚点偏移（P28-A 实测：hero pill bbox (600,1223,175,74)
#: 于 S6 锚 (945,1160) 左下；x∈[-350,-160], y∈[+40,+150]）
BUBBLE_ROI_DX = (-350, -160)
BUBBLE_ROI_DY = (40, 150)

#: 阈值（P28-B 实测分离；不降）
BUBBLE_WHITE_MIN = 2500
BUBBLE_GOLD_MIN = 1000

def _white_sig(roi: np.ndarray) -> int:
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    if bg < 2:
        return 0
    m = ((l > bg * 1.6) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40))
    return int(m.sum())


def _goldface_sig(roi: np.ndarray) -> int:
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    if bg < 2:
        return 0
    thr = max(60.0, bg * 2.0)
    m = ((l > thr) & (r >= g * 0.9) & (b < r + 40))
    return int(m.sum())


class HeroSeatDetector1876:
    """1876 帧 → hero_seat（own-seat pill 证据；UNKNOWN-first）。"""

    name = "coinpoker_hero_seat_v3f_p28b_1876"

    def _seat_sig(self, frame: np.ndarray, seat: str) -> dict:
        ax, ay = SEAT_ANCHORS_1876[seat]
        x0, y0 = max(0, ax + BUBBLE_ROI_DX[0]), ay + BUBBLE_ROI_DY[0]
        x1, y1 = ax + BUBBLE_ROI_DX[1], ay + BUBBLE_ROI_DY[1]
        if x1 > frame.shape[1] or y1 > frame.shape[0] or y0 < 0:
            return {"white": -1, "gold": -1, "out_of_frame": True}
        roi = frame[y0:y1, x0:x1]
        return {"white": _white_sig(roi), "gold": _goldface_sig(roi),
                "out_of_frame": False}

    def detect(self, frame: np.ndarray) -> dict:
        hits = []
        evidence = {}
        for seat in SEAT_ANCHORS_1876:
            s = self._seat_sig(frame, seat)
            evidence[seat] = {"white": s["white"], "gold": s["gold"]}
            if s["out_of_frame"]:
                continue
            if s["white"] >= BUBBLE_WHITE_MIN and s["gold"] >= BUBBLE_GOLD_MIN:
                hits.append(seat)
        if len(hits) == 0:
            return {"hero_seat": None, "confidence": 0.0,
                    "reason": "no_own_seat_pill", "evidence": evidence}
        if len(hits) > 1:
            return {"hero_seat": None, "confidence": 0.0,
                    "reason": "own_seat_conflict", "evidence": evidence,
                    "hits": hits}
        return {"hero_seat": hits[0], "confidence": 0.9,
                "reason": "own_seat_pill_evidence", "evidence": evidence}
