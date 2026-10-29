"""P26-A — 正/负分类 NCC 分数分布（LOSO 基础上）：
对每个 glyph 组件计算（正确标签分数 / 次佳标签分数 → margin）与
"对随机错误标签集合"的最高分数 → 分布。
决定诚实门限（不应为 PASS 而低于冻结链 0.7/0.08——除非数据说明分离更小）。
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
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


def build_templates(exclude_fid=""):
    tpls = defaultdict(list)
    for ann in GT["annotations"]:
        fid = ann["frame_id"]
        if fid == exclude_fid:
            continue
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


def ncc_all(g, PREP):
    """全标签分数。"""
    gp = np.pad(g, 2)
    scores = {}
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
    return scores


def score_analysis():
    """LOSO 分数分布：正确标签分数 vs 次佳（错误）。"""
    fids = [ann["frame_id"] for ann in GT["annotations"]]
    wrong_scores, right_scores, margins = [], [], []
    for fid in fids:
        PREP = build_templates(exclude_fid=fid)
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
                g = m[comp[1]:comp[1] + comp[3],
                      comp[0]:comp[0] + comp[2]].astype(float)
                scores = ncc_all(g, PREP)
                if ch not in scores:
                    continue
                ranked = sorted(scores.items(), key=lambda kv: kv[1],
                                reverse=True)
                right = scores[ch]
                # 次佳 = 首个标签 != ch
                second = next((v for lbl, v in ranked if lbl != ch), None)
                right_scores.append(right)
                if second is not None:
                    wrong_scores.append(second)
                    margins.append(right - second)
    rs = np.array(right_scores)
    ws = np.array(wrong_scores)
    ms = np.array(margins)
    print(f"正确标签分数: n={len(rs)} min={rs.min():.3f} p5={np.percentile(rs,5):.3f} "
          f"p10={np.percentile(rs,10):.3f} p50={np.median(rs):.3f}")
    print(f"次佳(错误)分数: n={len(ws)} p90={np.percentile(ws,90):.3f} "
          f"p95={np.percentile(ws,95):.3f} max={ws.max():.3f}")
    print(f"margin: min={ms.min():.3f} p5={np.percentile(ms,5):.3f} "
          f"p50={np.median(ms):.3f}")
    # 0.6/0.05 门 & 0.7/0.08 门
    n60 = int((rs >= 0.60).sum())
    print(f"正确≥0.60: {n60}/{len(rs)}; 错误≤0.60 但 margin 分析: "
          f"错误≥0.60: {int((ws >= 0.60).sum())}")
    print(f"margin≥0.05: {int((ms >= 0.05).sum())}/{len(ms)}; "
          f"margin≥0.08: {int((ms >= 0.08).sum())}/{len(ms)}")
    print(f"分位: right p10={np.percentile(rs,10):.3f} "
          f"wrong p95={np.percentile(ws,95):.3f}")
    # 混淆标签明细（top5 错误）
    from collections import Counter
    err = Counter()
    for fid in fids:
        PREP = build_templates(exclude_fid=fid)
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
                g = m[comp[1]:comp[1] + comp[3],
                      comp[0]:comp[0] + comp[2]].astype(float)
                scores = ncc_all(g, PREP)
                best = max(scores.items(), key=lambda kv: kv[1])
                if best[0] != ch:
                    err[(ch, best[0])] += 1
    print(f"混淆对: {dict(err)}")


score_analysis()
