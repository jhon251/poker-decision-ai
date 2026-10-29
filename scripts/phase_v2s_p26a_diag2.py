"""P26-A — 修正 ROI 版深分割诊断（34 GT）。

关键事实（本诊断阶段新测得，替代错误的 P23 bbox）：
- P23 的每座 stack bbox 系统性偏移（S6 真实行 y≈1245-1295 而 P23 给 1307-1337；
  S1 真实行 y≈429-463 而 P23 给 340-369——P23 采样命中的是橙色「底池」label）。
- 真实 stack 行（gold 数字行）从 34 个 GT 的 gold blob 分析确立。

ROI = 真实 stack 行 ± 余量；统计组件数/宽/高/间隙与期望字符数。
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

#: 真实 stack 行 ROI（blob 分析确立；x 每座留 ±40 余量，y 行高 ±15）
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


def gold_mask(frame, x0, y0, x1, y1):
    roi = frame[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    return ((r > 100) & (g > 70) & (r >= g)
            & (g >= b + 25) & (r > b + 40)).astype(np.uint8)


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


def render_expected(value: int) -> str:
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
detail = []
for fid, seat, value in cases:
    arr = load(fid)
    x0, y0, x1, y1 = STACK_ROWS_1876[seat]
    gold = gold_mask(arr, x0, y0, x1, y1)
    comps = comps_of(gold)
    if not comps:
        detail.append((seat, fid[:6], value, "NO-COMP", None, None))
        continue
    exp = render_expected(value)
    n = len(comps)
    ws = [c[2] for c in comps]; hs = [c[3] for c in comps]
    gaps = [comps[i][0] - (comps[i-1][0] + comps[i-1][2] - 1) - 1
            for i in range(1, len(comps))]
    stat_h.extend(hs); stat_w.extend(ws); stat_gap.extend(gaps)
    ncomp_dist[n] += 1; exp_dist[len(exp)] += 1
    ok = n == len(exp)
    perfect += int(ok)
    detail.append((seat, fid[:6], value, "ok" if ok else
                   (f"n{n}!=e{len(exp)}" if n >= len(exp) else f"MERGE{n}<e{len(exp)}"),
                   ws, hs))

print(f"{'seat':4s} {'fid':7s} {'value':>8s} {'状态':14s} w / h")
for seat, fid, value, status, ws, hs in detail:
    if ws is None:
        print(f"{seat:4s} {fid:7s} {value:>8d} NO-COMP")
        continue
    print(f"{seat:4s} {fid:7s} {value:>8d} {status:14s} "
          f"{','.join(str(x) for x in ws)} / {','.join(str(x) for x in hs)}")

print(f"\n=== 统计 ===")
print(f"组件数 vs 期望: {dict(ncomp_dist)} vs {dict(exp_dist)}")
print(f"perfect(n==exp): {perfect}/{len(cases)}")
print(f"glyph h: n={len(stat_h)} min={min(stat_h)} "
      f"p25={np.percentile(stat_h,25):.0f} p50={np.median(stat_h)} "
      f"p75={np.percentile(stat_h,75):.0f} max={max(stat_h)}")
print(f"glyph w: n={len(stat_w)} min={min(stat_w)} "
      f"p25={np.percentile(stat_w,25):.0f} p50={np.median(stat_w)} "
      f"p75={np.percentile(stat_w,75):.0f} max={max(stat_w)}")
if stat_gap:
    print(f"gap: n={len(stat_gap)} min={min(stat_gap)} "
          f"p25={np.percentile(stat_gap,25):.0f} p50={np.median(stat_gap)} "
          f"p75={np.percentile(stat_gap,75):.0f} max={max(stat_gap)}")
# 逗号/点/M 几何
narrow = [w for w in stat_w if w <= 10]
print(f"窄组件(w<=10, 逗号/点/1): n={len(narrow)}")
