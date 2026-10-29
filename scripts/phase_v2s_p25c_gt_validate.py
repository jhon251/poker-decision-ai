"""P25-C — 22 GT Status 验证：TP/FP/FN + confusion + exact-match。"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    StatusObserver)

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))

# 建 per-frame seat->status
frame_seats = {}
for ann in GT["annotations"]:
    st = (ann.get("fields") or {}).get("status") or {}
    frame_seats[ann["frame_id"]] = {k: v for k, v in st.items()
                                    if v is not None}

UNA = "UNKNOWN"   # GT 未标注 = 无状态视觉（负例）
pos = neg = 0
conf = Counter()
exact = 0
total = 0
fps = []
fns = []
obs = StatusObserver()
for ann in GT["annotations"]:
    fid = ann["frame_id"]
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    out = obs.observe(arr, w, h)
    gts = frame_seats.get(fid, {})
    det = {seat: v["status"] for seat, v in out["seats"].items()
           if v["status"] != "UNKNOWN"}
    print(f"\n{fid} window={out['window']} status={out['status']}")
    for seat in sorted(set(list(gts) + list(det))):
        gt = gts.get(seat, UNA)
        r = detected = det.get(seat, UNA)
        total += 1
        if gt == detected:
            exact += 1
            conf[(gt, detected)] += 1
        else:
            if gt != UNA and detected == UNA:
                fns.append((fid, seat, gt))
                conf[(gt, "UNKNOWN")] += 1
            elif gt == UNA and detected != UNA:
                fps.append((fid, seat, detected))
                conf[("UNKNOWN", detected)] += 1
            else:
                conf[(gt, detected)] += 1
                fps.append((fid, seat, f"{gt}->{detected}"))
        print(f"  {seat}: GT={gt:13s} detected={detected}")
    pos += sum(1 for v in gts.values() if v)
    neg += sum(1 for s in range(6) if f"S{s+1}" not in gts)

print(f"\n=== 22 GT 评估 ===")
print(f"GT 标注座数 = {pos}  （folded=?, away=?, disc=?, allin=?）")
gtc = Counter()
for gts in frame_seats.values():
    gtc.update(gts.values())
print(f"GT 分类计数: {dict(gtc)}")
print(f"未标注座格（负例） = {neg}")
print(f"exact-match = {exact}/{total}")
print(f"FP = {len(fps)}: {fps}")
print(f"FN = {len(fns)}: {fns}")
print(f"confusion:")
for k, v in sorted(conf.items()):
    print(f"   {k}: {v}")

# 参考：单帧 fold-detector-only（teal）
print(f"\n=== 仅为参考：所有已识别座 ===")
recognized = sum(1 for fr, gts in frame_seats.items() for s in gts)
print("recognized GT 座 =", recognized)
