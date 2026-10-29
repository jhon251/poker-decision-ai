"""P22-A — Pot 1876 精确测量（P21 视觉确认后的像素实测）。

测量 6 个 GT 帧的 pot：数字 bbox / 字高 / 颜色 / 中文标签分离。
终止 P15/P16-B/P19-C 的 (760,80,1060,170) 错误 ROI。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def _tight_bbox(mask):
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1))


def main() -> int:
    for ann in GT["annotations"]:
        pv = (ann.get("fields") or {}).get("pot")
        if pv is None:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        # pot 区域（P21 视觉：y 470-545, x 700-1100）
        roi = arr[470:545, 700:1100]
        r = roi[..., 0].astype(np.int16)
        g = roi[..., 1].astype(np.int16)
        b = roi[..., 2].astype(np.int16)
        l = r * 0.299 + g * 0.587 + b * 0.114
        bright = l > 120
        gold = (r > 110) & (g > 80) & (r >= g) & (g >= b + 25)
        print(f"\n{fid}  GT pot={pv}")
        # 亮像素整体 bbox + 颜色
        bb = _tight_bbox(bright)
        if bb:
            bx, by, bw, bh = bb
            px = roi[by:by + bh, bx:bx + bw].reshape(-1, 4)
            px = px[px[:, 0] > 100]
            print(f"  亮bbox abs=({700+bx},{470+by},{bw},{bh}) "
                  f"颜色R={px[:,0].mean():.0f} G={px[:,1].mean():.0f} "
                  f"B={px[:,2].mean():.0f}")
        # 逐列投影（找字符段）
        cols = bright.sum(axis=0)
        runs, run = [], None
        for i, v in enumerate(cols):
            if v >= 2 and run is None:
                run = i
            elif v < 2 and run is not None:
                runs.append((run, i))
                run = None
        if run is not None:
            runs.append((run, len(cols)))
        print(f"  亮字符段: {[(700+s, 700+e, e-s) for s,e in runs]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
