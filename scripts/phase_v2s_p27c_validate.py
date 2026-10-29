"""P27-C — 三态 detector 验证：EMPTY 帧 + 6 GT 帧 + dispatch。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_occupancy import (  # noqa: E402
    SeatOccupancyRecognizer1876)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PositionObserver)

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))
REC = SeatOccupancyRecognizer1876()


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


# 1) EMPTY 帧：S4=EMPTY, 其余 OCCUPIED
print("=== EMPTY 帧 (42b) ===")
arr = load("42b553c9468d1d1e55b9801bebad7b4f.png")
out = REC.recognize(arr)
for seat in sorted(out):
    v = out[seat]
    print(f"  {seat}: {v['status']:10s} ({v['reason']}) "
          f"gold={v['evidence']['gold_n']} name={v['evidence']['name_ink']} "
          f"maroon={v['evidence']['maroon_n']} dim={v['evidence']['dim_label_n']}"
          f"/{v['evidence']['dim_label_run']}")
assert out["S4"]["status"] == "EMPTY"
for seat in ("S1", "S2", "S3", "S5", "S6"):
    assert out[seat]["status"] == "OCCUPIED", seat

# 2) 6 GT 帧：34 占用 OCCUPIED；2 overlay UNKNOWN
print("\n=== 6 GT 帧 ===")
exact_occ = 0
fps, fns = [], []
for ann in GT["annotations"]:
    a = load(ann["frame_id"])
    o = REC.recognize(a)
    st = (ann.get("fields") or {}).get("stack") or {}
    for seat in sorted(o):
        v = o[seat]
        gt = "occupied"
        # 记录 34 个有 stack 标注的座（overlay 座 GT 无 stack 值）
        has_stack = seat in st and st[seat] is not None
        print(f"  {ann['frame_id'][:6]} {seat}: {v['status']:10s} "
              f"(gold={v['evidence']['gold_n']} name={v['evidence']['name_ink']}"
              f" maroon={v['evidence']['maroon_n']} dim={v['evidence']['dim_label_n']})")
        if v["status"] == "OCCUPIED":
            exact_occ += 1
        else:
            if has_stack:
                fns.append((ann["frame_id"][:6], seat))
            else:
                print(f"      -> non-occ, no GT stack (overlay) OK")
print(f"\nOCCUPIED 命中={exact_occ}（应=36 占用 - 2 overlay=34）")
print(f"FN={fns}")

# 3) dispatch
print("\n=== PositionObserver dispatch ===")
w, h, rgba = decode_png((EV / "42b553c9468d1d1e55b9801bebad7b4f.png")
                        .read_bytes())
obs = PositionObserver().observe(rgba, w, h)
occ = obs["runtime_result"].get("occupied_seats", {})
print("occupied_seats:", {s: (v.get("status") if isinstance(v, dict) else v)
                          for s, v in occ.items()})
