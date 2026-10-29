"""V3F-P13-ACTION FOUNDATION — ActionRecognizer（1876×1400 渲染类）。

按钮行状态识别（当前帧）：
- 三分区几何 + 颜色门（阈值来自 PRE-AUDIT 实测直方图/分区统计，非猜测）：
  fold=红 · raise=橙红 · callcheck=深青；
- **Call vs Check 不可视觉区分**（逐帧白字量分布重叠——PRE-AUDIT 实测：
  check 0-2653 / call 0-2879 严重交叠）→ 按 UNKNOWN-first 纪律，callcheck
  类目合并输出为 "call_or_check"，绝不强行拆分；
- UNKNOWN-first：无按钮行 → NOT_OBSERVED；其它渲染类（1419/1280——已证按钮
  不可见）→ EVIDENCE_LIMITED；
- 无 OCR / 无中文文字语义 / 无 frame 标识·时间戳·用户标注作 runtime 条件 /
  无状态缓存 / 无跨字段推导；
- All-in 不单列（属 raise 家族——本识别器只报告按钮可用性，不报告已执行动作）。
"""
from __future__ import annotations

import numpy as np

#: 1876×1400 类按钮分区（用户截图实测 + 第5/6轮 110 按钮帧逐帧验证）
ACTION_WINDOW = (1876, 1400)
ACTION_BUTTON_ROIS = {
    "fold": (1116, 1263, 1354, 1392),      # 盖牌（红）
    "callcheck": (1354, 1263, 1629, 1392),  # 跟注/过牌（深青，互斥；不可细分）
    "raise": (1629, 1263, 1860, 1392),     # 加注（橙红）
}

#: 实测门（PRE-AUDIT：fold 红 24960-24984 / raise 橙 22054-22897 或 0 /
#: cc teal 9451-12066）
FOLD_RED_MIN = 12000
RAISE_ORANGE_MIN = 10000
CC_TEAL_MIN = 6000


def _region_counts(arr: np.ndarray, roi: tuple) -> dict:
    x0, y0, x1, y1 = roi
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    sub = (slice(y0, y1), slice(x0, x1))
    lum = r * 0.299 + g * 0.587 + b * 0.114
    red = ((r > g + 50) & (r > b + 50) & (r > 110))[sub].sum()
    orange = ((r > 140) & (g > 80) & (g < 190) & (b < 100))[sub].sum()
    teal = ((g > r + 15) & (g >= b - 12) & (g > 50) & (g <= 135))[sub].sum()
    white = ((lum > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
             )[sub].sum()
    return {"red": int(red), "orange": int(orange), "teal": int(teal),
            "white": int(white)}


class ActionRecognizer:
    """当前帧 → 按钮行可用性 {fold, raise, callcheck: call_or_check|None}。"""

    name = "coinpoker_action_foundation_v1"

    def __init__(self, window: str = "1876"):
        self.window = window

    def recognize(self, frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        if (w, h) != ACTION_WINDOW:
            return {"status": "EVIDENCE_LIMITED",
                    "reason": "action_ui_not_visible_in_this_render_class",
                    "window": f"{w}x{h}",
                    "actions": {"fold": None, "raise": None,
                                "callcheck": None},
                    "ground_truth_status": "runtime_state_non_gt"}
        cnt = {k: _region_counts(frame, roi)
               for k, roi in ACTION_BUTTON_ROIS.items()}
        fold_present = cnt["fold"]["red"] >= FOLD_RED_MIN
        if not fold_present:
            return {"status": "NOT_OBSERVED",
                    "reason": "action_row_not_visible",
                    "window": "1876",
                    "actions": {"fold": False, "raise": None,
                                "callcheck": None},
                    "counts": cnt,
                    "ground_truth_status": "runtime_state_non_gt"}
        raise_present = cnt["raise"]["orange"] >= RAISE_ORANGE_MIN
        cc_present = cnt["callcheck"]["teal"] >= CC_TEAL_MIN
        return {
            "status": "RECOGNIZED",
            "reason": "action_row_visible",
            "window": "1876",
            "actions": {"fold": True, "raise": raise_present,
                        "callcheck": "call_or_check" if cc_present else
                        "UNKNOWN"},
            "callcheck_note": "call_and_check_share_button_visuals; "
                              "class_split_is_not_visually_reliable",
            "counts": cnt,
            "bboxes": dict(ACTION_BUTTON_ROIS),
            "ground_truth_status": "runtime_state_non_gt",
        }
