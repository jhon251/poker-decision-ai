"""V3F-P13-C — Bet Runtime 消费者（BetObserver）。

独立模块（不在 seat_position.py——尊重 P8 红「Stack 不引用 Bet」的文件隔离，
同时保持观察者同型模式）。

- observe(bytes, w, h) → 冻结 BetRecognizer（只读消费——不二次 OCR/NCC、
  不修改 threshold/segmentation/glyph extraction、不调 Stack/Pot 修正）；
- render-class：按帧几何 (w,h) 选择 ROI/识别类——(1280,779) → 1280 类
  （BET_ROIS_1280，recognized）；其它 → 1419 类（evidence-limited → 恒
  UNKNOWN no_bet_scale_glyphs）。选择依据 = 合法 render geometry，非
  frame id / scene / expected value / historical frame；
- 座 ownership = ROI 几何键（1280: S_TL/S_TR 顶左/顶右座下注区——P13-B
  实测校准）；无法归属 → UNKNOWN（不按 stack/pot/status/action 反推）；
- 无缓存/previous/fixture/hardcode；UNKNOWN-first（no_bet_capsule 仅表示
  当前帧胶囊视觉证据缺失——不把 recognizer failure 判成 bet=0）。
"""
from __future__ import annotations

import numpy as np


class BetObserver:
    """采集管线逐帧消费者（与其它 observer 同型，纯消费者）。"""

    name = "coinpoker_bet_runtime_observer"

    def __init__(self):
        self._recognizers = {}
        self.last = None
        self.count = 0
        self.recognized = 0
        self.unknown = 0

    def _recognizer_for(self, w: int, h: int):
        key = (int(w), int(h))
        if key not in self._recognizers:
            from .coinpoker_bet import BetRecognizer
            if key == (1280, 779):
                window = "1280"
            elif key == (1876, 1400):
                window = "1876"      # V3F-P24-C：暗+亮双路径
            else:
                window = "1419"
            self._recognizers[key] = BetRecognizer(window=window)
        return self._recognizers[key]

    def observe(self, frame_rgba: bytes | np.ndarray, w: int, h: int) -> dict:
        frame = (np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
                 if isinstance(frame_rgba, bytes) else frame_rgba)
        rec = self._recognizer_for(w, h)
        res = rec.recognize(frame)
        seats = {zone: {"bet": r.get("bet"), "raw_text": r.get("raw_text"),
                        "status": r.get("status"), "reason": r.get("reason")}
                 for zone, r in res.items()}
        self.last = {
            "seats": seats,
            "window": "1280" if (int(w), int(h)) == (1280, 779)
            else ("1876" if (int(w), int(h)) == (1876, 1400) else "1419"),
            "ground_truth_status": "runtime_state_non_gt",
        }
        self.count += 1
        if any(v["status"] == "RECOGNIZED" for v in seats.values()):
            self.recognized += 1
        else:
            self.unknown += 1
        return self.last

    def summary(self) -> dict:
        return {"frames": self.count, "recognized": self.recognized,
                "unknown_frames": self.unknown,
                "last_seats": (self.last or {}).get("seats"),
                "last_window": (self.last or {}).get("window"),
                "ground_truth_status": "runtime_state_non_gt"}
