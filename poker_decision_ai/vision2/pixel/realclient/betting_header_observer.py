"""V3F-P30-2B — BettingHeader Observer（1876 runtime 消费者）。

与其它 observer 同型（纯消费者）：observe(bytes, w, h) → 冻结
BettingHeaderRecognizer1876 → header {sb, bb, ante, status}；summary() 供
snapshot 聚合。无 cache/previous/fixture；render-class 仅 1876（1419/1848
无 header 检测——视觉对象只存在于 1876 完整窗口，诚实 UNKNOWN）。
"""
from __future__ import annotations

import numpy as np


class BettingHeaderObserver:
    """采集管线逐帧消费者（与其它 observer 同型，纯消费者）。"""

    name = "coinpoker_betting_header_runtime_observer"

    def __init__(self):
        from .coinpoker_betting_header import BettingHeaderRecognizer1876
        self.recognizer = BettingHeaderRecognizer1876()
        self.last = None
        self.count = 0
        self.recognized = 0
        self.unknown = 0

    def observe(self, frame_rgba: bytes | np.ndarray, w: int, h: int) -> dict:
        frame = (np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
                 if isinstance(frame_rgba, bytes) else frame_rgba)
        if (int(w), int(h)) != (1876, 1400):
            self.last = {"sb": None, "bb": None, "ante": None,
                         "status": "UNKNOWN",
                         "reason": "unsupported_render_class",
                         "window": f"{int(w)}x{int(h)}",
                         "ground_truth_status": "runtime_state_non_gt"}
            self.count += 1
            self.unknown += 1
            return self.last
        res = self.recognizer.recognize(frame)
        self.last = {
            "sb": res.get("sb"), "bb": res.get("bb"),
            "ante": res.get("ante"), "status": res.get("status"),
            "reason": res.get("reason"),
            "confidence": res.get("confidence"),
            "window": "1876",
            "ground_truth_status": "runtime_state_non_gt",
        }
        self.count += 1
        if res.get("status") == "RECOGNIZED":
            self.recognized += 1
        else:
            self.unknown += 1
        return self.last

    def summary(self) -> dict:
        return {"frames": self.count, "recognized": self.recognized,
                "unknown_frames": self.unknown,
                "last_sb": (self.last or {}).get("sb"),
                "last_bb": (self.last or {}).get("bb"),
                "last_ante": (self.last or {}).get("ante"),
                "last_status": (self.last or {}).get("status"),
                "last_window": (self.last or {}).get("window"),
                "ground_truth_status": "runtime_state_non_gt"}
