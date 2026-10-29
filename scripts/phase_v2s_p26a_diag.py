"""P26-A — 34 GT Stack 深分割诊断（READ-ONLY，零生产修改）。

对每个 GT (frame, seat, value)：
1. 扩展 ROI（P23 实测 bbox ± 余量）→ 暗金掩码 → 行带投影 → 最优带（金量最大）
2. 带内 connected components（8连通）→ 每组件 bbox/宽/高/间隙
3. 与期望渲染字符数比对：粘连（组件数 < 期望）/ 断裂（组件数 > 期望）
4. 统计：glyph 高/宽分布、间隙分布、每座 bbox、亮度变体
期望渲染：>=1M → "X.XXM"(4)；<1M → 千位逗号格式 len；0 → "0"(1)。
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

#: P23 实测 每座 stack 文本 bbox（帧01 测量）
STACK_BBOX_1876 = {
    "S1": (172, 340, 324, 369),
    "S2": (790, 284, 952, 314),
    "S3": (1611, 354, 1631, 381),
    "S4": (162, 1023, 334, 1050),
    "S5": (1590, 1003, 1702, 1029),
    "S6": (851, 1307, 993, 1337),
}
PAD = (60, 25, 45)          # x 左右 60, 上 25, 下 45


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def gold_mask(frame: np.ndarray, x0, y0, x1, y1):
    roi = frame[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    return ((r > 100) & (g > 70) & (r >= g)
            & (g >= b + 25) & (r > b + 40)).astype(np.uint8)


def row_bands(gold, min_ink=3, min_h=6, merge_gap=2):
    rows = gold.sum(axis=1)
    runs, run = [], None
    for i, v in enumerate(rows):
        if v >= min_ink and run is None:
            run = [i, i]
        elif v >= min_ink and run is not None:
            run[1] = i
        elif run is not None:
            runs.append(run); run = None
    if run:
        runs.append(run)
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    return [(a, b) for (a, b) in merged if b - a + 1 >= min_h]


def comps_of(band2d):
    """8-连通组件 → [(x0,y0,w,h,area)]（min_area=3）。"""
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
            if len(cells) < 3:
                continue
            xs = [c[0] for c in cells]; ys = [c[1] for c in cells]
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
            cases.append((fid, seat, v, ann))

print(f"GT cases = {len(cases)}")
stats = {"h": [], "w": [], "gap": [], "ncomp": [], "expected": [],
         "merged": [], "broken": [], "verdict_good": []}
rows = []
for fid, seat, value, ann in cases:
    arr = load(fid)
    bx0, by0, bx1, by1 = STACK_BBOX_1876[seat]
    x0, y0 = max(0, bx0 - PAD[0]), max(0, by0 - PAD[1])
    x1, y1 = min(arr.shape[1], bx1 + PAD[0]), min(arr.shape[0], by1 + PAD[2])
    gold = gold_mask(arr, x0, y0, x1, y1)
    bands = row_bands(gold)
    if not bands:
        rows.append((seat, fid, value, "NO-BAND", None))
        continue
    # 最优带 = 金量最大
    best = max(bands, key=lambda b: int(gold[b[0]:b[1] + 1].sum()))
    band = gold[best[0]:best[1] + 1]
    comps = comps_of(band)
    inner = band
    exp = render_expected(value)
    exp_n = len(exp)
    cw = [c[2] for c in comps]
    ch = [c[3] for c in comps]
    gaps = []
    for i in range(1, len(comps)):
        gaps.append(comps[i][0] - (comps[i - 1][0] + comps[i - 1][2] - 1) - 1)
    ncomp = len(comps)
    merged = ncomp < exp_n
    broken = (ncomp > exp_n * 1.6) or (ncomp > exp_n and max(ch or [0]) < 6)
    verdict = ("OK" if ncomp == exp_n else
               ("MERGE" if merged else ("BROKEN" if broken else "MIXED")))
    stats["h"].extend(ch); stats["w"].extend(cw); stats["gap"].extend(gaps)
    stats["ncomp"].append(ncomp); stats["expected"].append(exp_n)
    stats["merged"].append(int(merged)); stats["broken"].append(int(broken))
    stats["verdict_good"].append(int(ncomp == exp_n))
    # 座几何：栈带 bbox（相对扩展 ROI）
    ys2 = np.where(band.sum(axis=1) > 0)[0]
    xs2 = np.where(band.sum(axis=0) > 0)[0]
    bb = (x0 + int(xs2.min()), y0 + best[0] + int(ys2.min()),
          int(xs2.max() - xs2.min() + 1), int(ys2.max() - ys2.min() + 1))
    br = (arr.shape[1], arr.shape[0])
    rows.append((seat, fid, value, verdict, {
        "exp": exp, "n": ncomp, "bbox": bb, "w": cw[:8], "h": ch[:8],
        "gaps": gaps[:8], "gold_n": int(gold.sum())}))

print(f"\n{'seat':4s} {'fid':6s} {'value':>8s} {'verdict':7s}  n/exp  bbox  w x h / gaps")
for seat, fid, value, verdict, info in rows:
    if info is None:
        print(f"{seat:4s} {fid[:6]:6s} {value:>8d} NO-BAND")
        continue
    # bbox 但取 id 短名
    bb = info["bbox"]
    ws = ",".join(str(x) for x in info["w"])
    hs = ",".join(str(x) for x in info["h"])
    gs = ",".join(str(x) for x in info["gaps"])
    print(f"{seat:4s} {fid[:6]:6s} {value:>8d} {verdict:7s} "
          f"{info['n']}/{len(info['exp'])}  ({bb[0]},{bb[1]},{bb[2]},{bb[3]})"
          f"  w[{ws}] h[{hs}] gap[{gs}]")

print(f"\n=== 统计 ===")
print(f"组件高: n={len(stats['h'])} min={min(stats['h'])} "
      f"p50={np.median(stats['h'])} max={max(stats['h'])}")
print(f"组件宽: n={len(stats['w'])} min={min(stats['w'])} "
      f"p50={np.median(stats['w'])} max={max(stats['w'])}")
if stats["gap"]:
    print(f"间隙: n={len(stats['gap'])} min={min(stats['gap'])} "
          f"p50={np.median(stats['gap'])} max={max(stats['gap'])}")
# 期望字符数分布
import collections
print(f"期望字符数分布: {dict(collections.Counter(stats['expected']))}")
print(f"组件数分布: {dict(collections.Counter(stats['ncomp']))}")
print(f"粘连(MERGE) 出现: {sum(stats['merged'])}/{len(stats['merged'])}")
print(f"断裂(BROKEN) 出现: {sum(stats['broken'])}/{len(stats['broken'])}")
print(f"完美匹配(n==exp): {sum(stats['verdict_good'])}/{len(stats['verdict_good'])}")
