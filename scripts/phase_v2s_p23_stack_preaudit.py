"""P23 — Stack 1876 PRE-AUDIT：6 座 stack 文本精确位置 + 颜色 + 当前识别基线。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    StackRecognizer)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))

#: 候选 per-seat ROI（P21 图像布局估算：顶部座锚点附近；S4/S5 底侧 y900-1100；
#: S6 底中 y1180-1350）
ROIS_1876 = {
    "S1": (60, 180, 420, 420),
    "S2": (720, 140, 340, 300),
    "S3": (1450, 140, 420, 420),
    "S4": (80, 920, 350, 180),
    "S5": (1520, 900, 360, 200),
    "S6": (780, 1180, 260, 170),
}


def _gold_tight(frame, region):
    """区域内饱和金橙文本 tight bbox + 颜色。"""
    x0, y0, x1, y1 = region
    roi = frame[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    gold = (r > 100) & (g > 70) & (r >= g) & (g >= b + 25) & (l > 60)
    ys, xs = np.where(gold)
    if len(ys) < 30:
        return None, 0
    # 颜色
    idx = np.where(gold)
    return (x0 + xs.min(), y0 + ys.min(), xs.max() - xs.min() + 1,
            ys.max() - ys.min() + 1), len(ys)


def main() -> int:
    rec = StackRecognizer(window="1848")
    total = 0
    tp = 0
    for ann in GT["annotations"]:
        st = (ann.get("fields") or {}).get("stack") or {}
        if not st:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        for seat, gt_val in st.items():
            total += 1
            bb, _ = _gold_tight(arr, ROIS_1876[seat])
            res = rec.recognize_in_region(arr, ROIS_1876[seat])
            pred = res.get("value")
            hit = (pred == gt_val)
            if hit:
                tp += 1
            if not hit:
                print(f"MISS {fid[:26]} {seat} GT={gt_val} pred={pred} "
                      f"raw={res.get('raw_text')} goldbb={bb}")
    print(f"\n基线: TP={tp}/{total} = {tp/total*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
