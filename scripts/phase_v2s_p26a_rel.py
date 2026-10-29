"""P26-A — 相对判据（亮度归一）下 34 GT 全分割验证。

gold_rel = (r > b + 30) & (g >= b + 18) & (r >= g*0.9) & (l > 背景*1.6)
背景 = ROI 中部区域中位 luma。逐 GT：组件数 vs 期望；统计。
"""
from __future__ import annotations
import collections
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))

STACK_ROWS_1876 = {
    "S1": (165, 410, 375, 485),
    "S2": (845, 295, 1035, 375),
    "S3": (1560, 410, 1680, 485),
    "S4": (108, 975, 345, 1055),
    "S5": (1555, 975, 1725, 1050),
    "S6": (835, 1225, 1055, 1315),
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
            & (l > bg * 1.6)), float(bg)


def comps_of(band2d, min_area=3):
    h, w = band2d.shape
    mask = band2d.astype(bool)
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
            if len(cells) < min_area:
                continue
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            out.append((min(xs), min(ys), max(xs) - min(xs) + 1,
                        max(ys) - min(ys) + 1, len(cells)))
    return sorted(out)


def render_expected(value):
    if value == 0:
        return "0"
    if value >= 1000000:
        return f"{value/1e6:.2f}M"
    return f"{value:,}"


cases = []
for ann in GT["annotations"]:
    fid = ann["frame_id"]
    st = (ann.get("fields") or {}).get("stack") or {}
    for seat, v in st.items():
        if v is not None:
            cases.append((fid, seat, v))

stat_h, stat_w, stat_gap = [], [], []
ncomp_dist, exp_dist = collections.Counter(), collections.Counter()
perfect = 0
fails = []
rows = []
for fid, seat, value in cases:
    arr = load(fid)
    x0, y0, x1, y1 = STACK_ROWS_1876[seat]
    roi = arr[y0:y1, x0:x1]
    m, bg = gold_rel(roi)
    comps = comps_of(m)
    exp = render_expected(value)
    n = len(comps)
    ok = n == len(exp)
    perfect += int(ok)
    if not ok:
        fails.append((seat, fid[:6], value, n, len(exp)))
    ws = [c[2] for c in comps]; hs = [c[3] for c in comps]
    gaps = [comps[i][0] - (comps[i-1][0] + comps[i-1][2] - 1) - 1
            for i in range(1, n)]
    stat_h.extend(hs); stat_w.extend(ws); stat_gap.extend(gaps)
    ncomp_dist[n] += 1
    exp_dist[len(exp)] += 1
    rows.append((seat, fid[:6], value, n, len(exp), ws, hs, bg))

print(f"{'seat':4s} {'fid':7s} {'value':>8s} {'n':>3s} {'exp':>4s}  bg  w / h")
for seat, fid, value, n, ne, ws, hs, bg in rows:
    mark = "ok" if n == ne else "FAIL"
    print(f"{seat:4s} {fid:7s} {value:>8d} {n:3d} {ne:4d} {bg:5.1f}  {mark} "
          f"{','.join(str(x) for x in ws)} / {','.join(str(x) for x in hs)}")

print(f"\n=== 统计 ===")
print(f"perfect: {perfect}/{len(cases)}")
print(f"fails: {fails}")
print(f"组件高: min={min(stat_h)} p50={np.median(stat_h)} max={max(stat_h)}")
print(f"组件宽: min={min(stat_w)} p25={np.percentile(stat_w,25):.0f} "
      f"p50={np.median(stat_w)} p75={np.percentile(stat_w,75):.0f} "
      f"max={max(stat_w)}")
if stat_gap:
    print(f"间隙: min={min(stat_gap)} p50={np.median(stat_gap)} "
          f"max={max(stat_gap)}")
print(f"ncomp分布: {dict(ncomp_dist)}")
print(f"exp分布: {dict(exp_dist)}")
