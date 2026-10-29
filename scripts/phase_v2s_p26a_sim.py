"""P26-A — 推荐策略模拟（纯内存，零生产/资产改动）：
S1) gold_rel 连通分割（34 GT + 2 未标注负例）
S2) 模板 = 从 GT 帧 gold 组件提取（f01 帧 6 座 → 覆盖 0,1,2,3,4,5,6,7,9,,,.,M）
S3) 逐组件 NCC（几何门 + margin）→ 组装 → parse
输出：34 GT 全表 raw vs exp + 负例 + 总结。
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    parse_stack_text)

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


def comps_of(band2d, min_area=4):
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


def glyph_bin(mask, comp):
    """组件二值形状（ROI 级 gold mask 的 tight crop——不重新求 gold_rel）。"""
    cx, cy, cw, chh, _ = comp
    return mask[cy:cy + chh, cx:cx + cw].astype(np.float64)


def render_expected(value):
    if value == 0:
        return "0"
    if value >= 1000000:
        return f"{value/1e6:.2f}M"
    return f"{value:,}"


# ---- 模板：从 f01（BUTTON_CHANGED_167）6 座 gold 组件按期望字符对齐 ----
def extract_templates():
    """f01 全部座 gold 组件 + 期望字符 → label->[bin shapes]。"""
    arr = load("BUTTON_CHANGED_167_1787992682749_full.png")
    tpls = defaultdict(list)
    f01 = {
        "S1": 101600, "S2": 169760, "S3": 0, "S4": 535072,
        "S5": 3240000, "S6": 123843,
    }
    for seat, value in f01.items():
        x0, y0, x1, y1 = STACK_ROI_1876[seat]
        roi = arr[y0:y1, x0:x1]
        m = gold_rel(roi)
        comps = comps_of(m)
        exp = render_expected(value)
        if len(comps) != len(exp):
            print(f"  [extract] {seat} n={len(comps)} != {len(exp)} skip")
            continue
        for comp, ch in zip(comps, exp):
            if ch == "," or ch == ".":
                continue
            tpls[ch].append(glyph_bin(m, comp))
    return tpls


TPL = extract_templates()
PREP = {}
for ch, insts in TPL.items():
    PREP[ch] = []
    for g in insts:
        t0 = g - g.mean()
        tn = float(np.sqrt((t0 * t0).sum()))
        if tn > 1e-9:
            PREP[ch].append((g, t0, tn))
print("模板统计:", {k: len(v) for k, v in PREP.items()})


def ncc_char(g):
    best, bsc, scores = None, -1.0, {}
    # pad=2（等尺寸模板会因 th>=g.shape 被跳过——与冻结链 _pad_tight 同因）
    gp = np.pad(g, 2)
    for ch, insts in PREP.items():
        sc = -1.0
        for (t, t0, tn) in insts:
            th, tw = t.shape
            if th >= gp.shape[0] or tw >= gp.shape[1] or tn < 1e-9:
                continue
            wins = np.lib.stride_tricks.sliding_window_view(gp, (th, tw))
            wm = wins.mean(axis=(2, 3), keepdims=True)
            wc = wins - wm
            wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
            s = np.abs((wc * t0).sum(axis=(2, 3))
                       / np.maximum(wn * tn, 1e-9))
            v = float(s.max())
            if v > sc:
                sc = v
        scores[ch] = sc
        if sc > bsc:
            best, bsc = ch, sc
    if best is None:
        return None, None, None
    ranked = sorted(scores.values(), reverse=True)
    margin = bsc - (ranked[1] if len(ranked) > 1 else 0.0)
    return best, bsc, margin


MIN_SCORE = 0.60
MIN_MARGIN = 0.05


def recognize(roi):
    m = gold_rel(roi)
    comps = comps_of(m)
    if not comps:
        return None
    chars = []
    for comp in comps:
        _, _, cw, chh, _ = comp
        if chh <= 9 and cw <= 10:
            chars.append(".")
            continue
        if chh <= 17 and cw <= 10:
            chars.append(",")
            continue
        if cw > 24:
            chars.append("M")
            continue
        g = glyph_bin(m, comp)
        ch, sc, mg = ncc_char(g)
        if ch is None or sc < MIN_SCORE or mg < MIN_MARGIN:
            chars.append("?")
        else:
            chars.append(ch)
    return "".join(chars)


cases = []
for ann in GT["annotations"]:
    fid = ann["frame_id"]
    st = (ann.get("fields") or {}).get("stack") or {}
    for seat, v in st.items():
        if v is not None:
            cases.append((fid, seat, v))

good = 0
for fid, seat, value in cases:
    arr = load(fid)
    x0, y0, x1, y1 = STACK_ROI_1876[seat]
    raw = recognize(arr[y0:y1, x0:x1])
    exp = render_expected(value)
    ok = raw == exp
    good += int(ok)
    parsed = parse_stack_text(raw) if raw else None
    print(f"{seat:4s} {value:>8d} raw={str(raw):>10s} exp={exp:>9s} "
          f"{'OK' if ok else 'XX':4s} parsed={parsed}")

print(f"\nraw==exp: {good}/{len(cases)}")

print("\n=== 未标注负例 ===")
for fid, seat in (("POT_CHANGED_102_1787992505794_full.png", "S6"),
                  ("POT_CHANGED_139_1787992592538_full.png", "S5")):
    arr = load(fid)
    x0, y0, x1, y1 = STACK_ROI_1876[seat]
    raw = recognize(arr[y0:y1, x0:x1])
    parsed = parse_stack_text(raw) if raw else None
    print(f"{fid[:6]} {seat}: raw={raw!r} parsed={parsed}")
