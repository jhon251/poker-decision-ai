"""V3F-P30-2C-A — ToCall Observer（1876 runtime 消费者）。

与其它 observer 同型（纯消费者）：observe(bytes, w, h) → 冻结
ToCallRecognizer1876 → {to_call, status}；summary() 供 snapshot 聚合。
无 cache/previous/fixture；render-class 仅 1876（1419/1848 无动作面板
视觉对象——诚实 UNKNOWN）。
"""
from __future__ import annotations

import numpy as np


class ToCallObserver:
    """采集管线逐帧消费者（与其它 observer 同型，纯消费者）。"""

    name = "coinpoker_to_call_runtime_observer"

    def __init__(self):
        from .coinpoker_to_call import ToCallRecognizer1876
        self.recognizer = ToCallRecognizer1876()
        self.last = None
        self.count = 0
        self.recognized = 0
        self.unknown = 0

    def observe(self, frame_rgba: bytes | np.ndarray, w: int, h: int) -> dict:
        frame = (np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
                 if isinstance(frame_rgba, bytes) else frame_rgba)
        if (int(w), int(h)) != (1876, 1400):
            self.last = {"to_call": None, "status": "UNKNOWN",
                         "reason": "unsupported_render_class",
                         "window": f"{int(w)}x{int(h)}",
                         "ground_truth_status": "runtime_state_non_gt"}
            self.count += 1
            self.unknown += 1
            return self.last
        res = self.recognizer.recognize(frame)
        self.last = {
            "to_call": res.get("to_call"), "status": res.get("status"),
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
                "last_to_call": (self.last or {}).get("to_call"),
                "last_status": (self.last or {}).get("status"),
                "last_window": (self.last or {}).get("window"),
                "ground_truth_status": "runtime_state_non_gt"}
