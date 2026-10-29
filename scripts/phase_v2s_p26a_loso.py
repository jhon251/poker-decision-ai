"""P26-A — 留一帧交叉验证（模板从未知帧的其余帧提取 → 测留出帧）。
评估：34 GT 中逐帧留出，用其余帧的 gold 组件建模板，测该帧全部 GT。
另：S3=0、S5 M 系列在留出模板缺失时的表现（无该字形 → 诚实 ?/UNKNOWN）。
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

FID = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
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

# 每帧 → (fid, seat, value) 列表
by_frame = defaultdict(list)
for fid, seat, value in cases:
    by_frame[fid].append((seat, value))


def extract_from(frames, exclude_fid=""):
    tpls = defaultdict(list)
    for fid in frames:
        if fid == exclude_fid:
            continue
        ann = next(a for a in GT["annotations"] if a["frame_id"] == fid)
        st = (ann.get("fields") or {}).get("stack") or {}
        arr = load(fid)
        for seat, value in st.items():
            if value is None:
                continue
            x0, y0, x1, y1 = STACK_ROI_1876[seat]
            roi = arr[y0:y1, x0:x1]
            m = gold_rel(roi)
            comps = comps_of(m)
            exp = render_expected(value)
            if len(comps) != len(exp):
                continue
            for comp, ch in zip(comps, exp):
                if ch in ",.":
                    continue
                tpls[ch].append(m[comp[1]:comp[1] + comp[3],
                                   comp[0]:comp[0] + comp[2]].astype(float))
    prep = {}
    for ch, insts in tpls.items():
        prep[ch] = []
        for g in insts:
            t0 = g - g.mean()
            tn = float(np.sqrt((t0 * t0).sum()))
            if tn > 1e-9:
                prep[ch].append((g, t0, tn))
    return prep


def ncc_char(g, PREP):
    gp = np.pad(g, 2)
    best, bsc, scores = None, -1.0, {}
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


def recognize(roi, PREP):
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
        g = m[comp[1]:comp[1] + comp[3], comp[0]:comp[0] + comp[2]] \
            .astype(float)
        ch, sc, mg = ncc_char(g, PREP)
        if ch is None or sc < MIN_SCORE or mg < MIN_MARGIN:
            chars.append("?")
        else:
            chars.append(ch)
    return "".join(chars)


fids = [ann["frame_id"] for ann in GT["annotations"]]
print(f"留一帧交叉验证（{len(fids)} 帧，每帧用其余帧模板）\n")
all_ok = 0
total = 0
for fid in fids:
    PREP = extract_from(fids, exclude_fid=fid)
    print(f"--- 留出 {fid[:7]}: 模板={{{', '.join(sorted(PREP))}}} ---")
    ann = next(a for a in GT["annotations"] if a["frame_id"] == fid)
    st = (ann.get("fields") or {}).get("stack") or {}
    arr = load(fid)
    for seat, value in st.items():
        if value is None:
            continue
        x0, y0, x1, y1 = STACK_ROI_1876[seat]
        raw = recognize(arr[y0:y1, x0:x1], PREP)
        exp = render_expected(value)
        ok = raw == exp
        all_ok += int(ok)
        total += 1
        print(f"  {seat} {value:>8d} raw={str(raw):>10s} exp={exp:>9s} "
              f"{'OK' if ok else 'XX'} parsed={parse_stack_text(raw) if raw else None}")

print(f"\n留一帧 raw==exp: {all_ok}/{total}")
