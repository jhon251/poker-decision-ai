"""P26-B — 34 GT 验证（生产 StackRecognizer1876）+ 负例 + 未标注座格。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    StackRecognizer1876)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    StackObserver)

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


rec = StackRecognizer1876()
print("模板实例:", {k: len(v) for k, v in rec._prep.items()})

exact = 0
total = 0
fps, fns = [], []
print(f"\n{'seat':4s} {'value':>8s} {'detected':>10s} {'status':14s} "
      f"{'OK':4s}")
for ann in GT["annotations"]:
    fid = ann["frame_id"]
    arr = load(fid)
    out = rec.recognize(arr)
    st = (ann.get("fields") or {}).get("stack") or {}
    for seat, value in st.items():
        if value is None:
            continue
        got = out[seat]
        ok = got["status"] == "RECOGNIZED" and got["stack"] == value
        total += 1
        exact += int(ok)
        if not ok:
            (fns if got["status"] != "RECOGNIZED" else
             fps).append((fid[:6], seat, value, got["stack"], got["reason"]))
        print(f"{seat:4s} {value:>8d} {str(got['stack']):>10s} "
              f"{got['status']:14s} {'OK' if ok else 'XX'}")

print(f"\n=== exact = {exact}/{total} FP={len(fps)} FN={len(fns)} ===")
for f in fps:
    print("  FP:", f)
for f in fns:
    print("  FN:", f)

# 未标注座格（每帧 GT 无 stack 的座）→ 必须 UNKNOWN
print("\n=== 未标注座格（负例）===")
all_ok = True
for ann in GT["annotations"]:
    fid = ann["frame_id"]
    arr = load(fid)
    out = rec.recognize(arr)
    st = (ann.get("fields") or {}).get("stack") or {}
    for seat in sorted(out):
        if seat in st:
            continue
        got = out[seat]
        flag = got["status"] != "RECOGNIZED"
        all_ok &= flag
        print(f"  {fid[:6]} {seat}: {got['status']} ({got['reason']}) "
              f"flag={'OK' if flag else 'FP!'}")
print(f"未标注座格全部非 RECOGNIZED: {all_ok}")

# Observer dispatch
print("\n=== StackObserver dispatch ===")
ann0 = GT["annotations"][0]
arr = load(ann0["frame_id"])
obs_out = StackObserver().observe(arr, *arr.shape[1::-1])
print(f"window={obs_out['window']} "
      f"S1={obs_out['seats']['S1']['stack']} "
      f"status={obs_out['seats']['S1']['status']}")

# 1419
w, h, rgba = decode_png((EV /
    "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
obs_1419 = StackObserver().observe(rgba, w, h)
print(f"1419 window={obs_1419['window']} statuses="
      f"{[v['status'] for v in obs_1419['seats'].values()]}")
