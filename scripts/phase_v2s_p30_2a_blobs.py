"""P30-2A — 全帧蓝 SB 徽章 / 橙底池徽章 扫描（颜色对象定位）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p30_2a_crops")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def color_blobs(arr, mask, tag, min_n=200):
    """连通域找 blob bbox。"""
    from collections import deque
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            q = deque([(sy, sx)]); seen[sy, sx] = True
            cells = []
            while q:
                cy, cx = q.popleft()
                cells.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            if len(cells) < min_n:
                continue
            xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
            out.append((min(xs), min(ys), max(xs) - min(xs) + 1,
                        max(ys) - min(ys) + 1, len(cells)))
    out.sort(key=lambda b: -b[4])
    for b in out[:6]:
        print(f"[{tag}] blob=({b[0]},{b[1]},{b[2]},{b[3]}) n={b[4]}")
    return out


arr = load("8bf82f134651db0add626be0304a2ff1.png")
r = arr[..., 0].astype(np.int16); g = arr[..., 1].astype(np.int16)
b = arr[..., 2].astype(np.int16)
# 蓝色（SB 徽章）：b 主导
blue = (b > 150) & (b > r + 40) & (b > g + 20)
print("=== 8bf8 blue blobs ===")
color_blobs(arr, blue, "blue")
# 橙色（底池徽章）：r 主导 g 中
orange = (r > 180) & (g > 90) & (g < 160) & (b < 100)
print("=== 8bf8 orange blobs ===")
color_blobs(arr, orange, "orange")
