"""P27-C — 三态签名精确定量：
1) 「空的」dim label 在锚点区的检测（空帧 S4 vs 全 41 占用座格 vs 负例）
2) OCCUPIED 三件套（name+gold+avatar）阈值
3) ghost ring 检测可行性
判据设计用——只在测量中取值，不实施生产。"""
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

ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}
STACK = {
    "S1": (165, 413, 360, 485), "S2": (845, 298, 1035, 370),
    "S3": (1520, 413, 1715, 485), "S4": (125, 978, 320, 1050),
    "S5": (1560, 978, 1745, 1050), "S6": (838, 1236, 1040, 1308),
}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def label_sig(arr, seat):
    """「空的」dim gray label 在锚点 ±70 区检测。"""
    ax, ay = ANCH[seat]
    x0, y0 = max(0, ax - 70), max(0, ay - 50)
    x1, y1 = min(arr.shape[1], ax + 70), min(arr.shape[0], ay + 60)
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    dim = ((l >= 80) & (l <= 190) & (np.abs(r - g) < 25)
           & (np.abs(g - b) < 25))
    return int(dim.sum())


def occ_sig(arr, seat):
    """OCCUPIED 三件套：name ink + gold + avatar maroon。"""
    x0, y0, x1, y1 = STACK[seat]
    # name 行（stack 上 52px）
    ln = arr[max(0, y0 - 52):y0 - 6, x0:x1]
    ln2 = ln[..., 0] * 0.299 + ln[..., 1] * 0.587 + ln[..., 2] * 0.114
    rn = ln[..., 0].astype(np.int16); gn = ln[..., 1].astype(np.int16)
    bn = ln[..., 2].astype(np.int16)
    name_ink = int(((ln2 > 110) & (np.abs(rn - gn) < 28)
                    & (np.abs(gn - bn) < 28)).sum())
    # gold
    rs = arr[y0:y1, x0:x1]
    rg = rs[..., 0].astype(np.int16); gg = rs[..., 1].astype(np.int16)
    bg2 = rs[..., 2].astype(np.int16)
    ls = rg * 0.299 + gg * 0.587 + bg2 * 0.114
    mdn = np.median(ls)
    gold = int(((rg > bg2 + 30) & (gg >= bg2 + 18) & (rg >= gg * 0.9)
                & (ls > mdn * 1.6)).sum())
    # avatar maroon（锚点 ±100）
    ax, ay = ANCH[seat]
    sub = arr[ay - 90:ay + 90, ax - 100:ax + 100]
    sr = sub[..., 0].astype(np.int16); sg = sub[..., 1].astype(np.int16)
    sb = sub[..., 2].astype(np.int16)
    maroon = int(((sr > sg + 20) & (sr > sb + 10) & (sr > 40)
                  & (sr < 150)).sum())
    return name_ink, gold, maroon


# 1) EMPTY 帧：S4 空 vs 其余占用
print("=== EMPTY 帧（用户 42b）===")
arr = load("42b553c9468d1d1e55b9801bebad7b4f.png")
for seat in sorted(ANCH):
    n, g, m = occ_sig(arr, seat)
    lab = label_sig(arr, seat)
    print(f"  {seat}: label={lab:4d} name={n:5d} gold={g:5d} maroon={m:6d}")

# 2) 6 GT 帧 36 占用座格：label 假阳性检查
print("\n=== 6 GT 帧 36 占用座格的 dim-label 假阳 ===")
labs = []
for ann in GT["annotations"]:
    a = load(ann["frame_id"])
    for seat in sorted(ANCH):
        labs.append(label_sig(a, seat))
labs = np.array(labs)
print(f"dim label 分布: min={labs.min()} p50={np.median(labs)} "
      f"p90={np.percentile(labs,90)} max={labs.max()}")
print(f">=150 的座格: {int((labs >= 150).sum())}/36")

# 3) EMPTY 帧 S4 label 精确
s4lab = label_sig(arr, "S4")
print(f"\nEMPTY S4 dim label = {s4lab}")
