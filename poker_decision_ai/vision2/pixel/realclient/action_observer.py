"""V3F-P13-ACTION — Runtime 消费者（ActionObserver）。

独立模块（与 bet_observer.py 同型，纯消费者；不污染 Stack/Pot/Bet/Status）。

- observe(bytes, w, h) → 冻结 ActionRecognizer（只读消费——零新增视觉逻辑）：
  fold / raise / call_or_check / UNKNOWN / NOT_OBSERVED / EVIDENCE_LIMITED
  原样透传；
- 当前帧 only：无 previous/cache/history；**按钮消失绝不推断"已执行动作"**
  （"按钮可用"≠"已执行"——Observer 只报告当前帧按钮可用性）；
- Call/Check 恒为 call_or_check（识别器诚实能力——不得映射成其中之一）；
- provenance 明确：recognizer 名 + render class + 帧几何 + 当前视觉结果。
"""
from __future__ import annotations

import numpy as np


class ActionObserver:
    """采集管线逐帧消费者（与其它 observer 同型，纯消费者）。"""

    name = "coinpoker_action_runtime_observer"

    def __init__(self):
        from .coinpoker_action import ActionRecognizer
        self.recognizer = ActionRecognizer(window="1876")
        self.last = None
        self.count = 0
        self.recognized = 0
        self.not_observed = 0
        self.unknown = 0
        self.evidence_limited = 0

    def observe(self, frame_rgba: bytes | np.ndarray, w: int, h: int) -> dict:
        frame = (np.frombuffer(frame_rgba, dtype=np.uint8).reshape(h, w, 4)
                 if isinstance(frame_rgba, bytes) else frame_rgba)
        res = self.recognizer.recognize(frame)
        cls = "1876x1400" if (int(w), int(h)) == (1876, 1400) else "other"
        self.last = {
            "actions": res.get("actions"),
            "status": res.get("status"),
            "reason": res.get("reason"),
            "callcheck_note": res.get("callcheck_note"),
            "window": f"{int(w)}x{int(h)}",
            "provenance": {
                "recognizer": self.recognizer.name,
                "render_class": cls,
                "frame_geometry": [int(w), int(h)],
            },
            "ground_truth_status": "runtime_state_non_gt",
        }
        self.count += 1
        st = res.get("status")
        if st == "RECOGNIZED":
            self.recognized += 1
        elif st == "NOT_OBSERVED":
            self.not_observed += 1
        elif st == "EVIDENCE_LIMITED":
            self.evidence_limited += 1
        else:
            self.unknown += 1
        return self.last

    def summary(self) -> dict:
        return {
            "frames": self.count, "recognized": self.recognized,
            "not_observed": self.not_observed, "unknown": self.unknown,
            "evidence_limited": self.evidence_limited,
            "last_actions": (self.last or {}).get("actions"),
            "last_status": (self.last or {}).get("status"),
            "last_reason": (self.last or {}).get("reason"),
            "last_window": (self.last or {}).get("window"),
            "provenance": (self.last or {}).get("provenance"),
            "ground_truth_status": "runtime_state_non_gt",
        }
