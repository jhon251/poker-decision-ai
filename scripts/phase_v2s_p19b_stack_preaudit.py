"""P19-B — Stack 1876 PRE-AUDIT：33 GT 验证 + 暗金掩码读取对照。"""
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

#: 候选 per-seat stack 文本区域（顶部座锚点附近；底部座底部带 y1180-1390）
REGIONS_1876 = {
    "S1": (80, 290, 480, 510),
    "S2": (740, 170, 1060, 330),
    "S3": (1470, 170, 1840, 510),
    "S4": (0, 1180, 310, 1390),
    "S5": (1420, 1180, 1870, 1390),
    "S6": (470, 1180, 1080, 1390),
}


def _dark_gold_mask(self, frame):
    """P18-C 实测暗金：r>110, g>80, r>=g, g>=b+25（替代 _gold_mask）。"""
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    return ((r > 110) & (g > 80) & (r >= g) & (g >= b + 25)).astype(np.uint8)


def main() -> int:
    rec = StackRecognizer(window="1848")
    # 用暗金掩码替换（READ-ONLY 测试）
    rec._gold_mask = _dark_gold_mask.__get__(rec, type(rec))
    ok = 0
    total = 0
    per_seat = {}
    for ann in GT["annotations"]:
        st = (ann.get("fields") or {}).get("stack") or {}
        if not st:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        for seat, gt_val in st.items():
            total += 1
            region = REGIONS_1876[seat]
            res = rec.recognize_in_region(arr, region)
            pred = res.get("value")
            hit = (pred == gt_val)
            if hit:
                ok += 1
            per_seat.setdefault(seat, [0, 0])
            per_seat[seat][0] += 1
            per_seat[seat][1] += 1 if hit else 0
            if not hit:
                print(f"  MISS {fid} {seat} GT={gt_val} pred={pred} "
                      f"raw={res.get('raw_text')}")
    print(f"\n总: {ok}/{total} = {ok/total*100:.1f}%")
    for seat, (n, hh) in sorted(per_seat.items()):
        print(f"  {seat}: {hh}/{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
