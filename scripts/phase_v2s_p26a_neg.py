"""P26-A — 负例空间关系：stack ROI 之外 / 邻近的 gold 正对象核查。
1) 六 GT 帧全帧 gold 大 blob（>2000px）清单 vs stack ROI 重叠检查
2) name/bet/status/卡背 的 gold-relation 测试
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

STACK_ROI_1876 = {
    "S1": (165, 413, 360, 485),
    "S2": (845, 298, 1035, 370),
    "S3": (1520, 413, 1715, 485),
    "S4": (125, 978, 320, 1050),
    "S5": (1560, 978, 1745, 1050),
    "S6": (838, 1236, 1040, 1308),
}


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def gold_rel(roi):
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = np.median(l)
    return ((r > b + 30) & (g >= b + 18) & (r >= g * 0.9)
            & (l > bg * 1.6))


def big_blobs(mask):
    """全帧 gold 掩码大 blob（>500px）→ bbox。"""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(cells) < 500:
                continue
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            out.append((len(cells), min(xs), min(ys),
                        max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
    return out


def overlap(blob, roi):
    _, bx, by, bw, bh = blob
    x0, y0, x1, y1 = roi
    ox = max(0, min(bx + bw, x1) - max(bx, x0))
    oy = max(0, min(by + bh, y1) - max(by, y0))
    return ox * oy


for ann in GT["annotations"]:
    fid = ann["frame_id"]
    arr = load(fid)
    m = gold_rel(arr)
    blobs = big_blobs(m)
    print(f"\n=== {fid[:7]} 全帧 big gold blobs (area>=500) ===")
    for blob in blobs:
        a, bx, by, bw, bh = blob
        # 与各 stack ROI 重叠
        ovs = [(s, overlap(blob, r)) for s, r in STACK_ROI_1876.items()
               if overlap(blob, r) > 0]
        print(f"  ({bx},{by},{bw},{bh}) area={a} ov={ovs}")

# 负例行：区域内非 stack 文本是否产生 gold（name/bet/tag）
print("\n=== 负例位置 gold_rel 计数（应 ~0）===")
checks = [
    ("BUTTON_CHANGED_167_1787992682749_full.png", "S1 name", (190, 340, 440, 400)),
    ("BUTTON_CHANGED_167_1787992682749_full.png", "S5 tag", (1531, 891, 103, 40)),
    ("BUTTON_CHANGED_167_1787992682749_full.png", "S1 底池label", (138, 325, 103, 41)),
    ("POT_CHANGED_12_1787995290794_full.png", "S5 过牌", (1350, 886, 130, 48)),
    ("POT_CHANGED_12_1787995290794_full.png", "S2 bet", (680, 390, 120, 40)),
]
for fid, tag, roi in checks:
    arr = load(fid)
    x0, y0, x1, y1 = roi
    m = gold_rel(arr[y0:y1, x0:x1])
    print(f"  {fid[:6]} {tag}: gold_n={int(m.sum())}")
