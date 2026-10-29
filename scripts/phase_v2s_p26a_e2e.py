"""P26-A — 端到端模拟（零生产改动）：segmentation(connected) → NCC →
组装 → parse。用 coinpoker_assets_v1876pot 既有模板（Pot 已闭环字形库，
含 0-9,',','.','M'）；对比期望文本。

同时检查 2 个未标注座格（f03 S6 hero、f05 S5）不误报。
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
DIGIT_DIR = "data/vision2/coinpoker_assets_v1876pot"

#: 数据驱动 每座 stack 行 ROI（本诊断测得的 tight 范围 + 边距）
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


# ---- 模板装载（镜像 coinpoker_pot 的装载方式；仅供模拟）----
def load_templates():
    mp = Path(DIGIT_DIR) / "manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    tpls = {}
    for a in m.get("assets", []):
        lab = a.get("label")
        if lab not in "0123456789,.MK":
            continue
        p = Path(a["crop_ref"])
        if not p.exists():
            p = Path(DIGIT_DIR) / p.name
        if not p.exists():
            continue
        w2, h2, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h2, w2, 4)
        # alpha 掩码（glyph = 非透明）
        g = (arr[..., 3] > 128).astype(np.float64)
        ys, xs = np.where(g > 0)
        if len(xs) < 4:
            continue
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        t0 = g - g.mean()
        tn = float(np.sqrt((t0 * t0).sum()))
        tpls.setdefault(lab, []).append((g, t0, tn))
    return tpls


T = load_templates()
print("模板统计:", {k: len(v) for k, v in T.items()})


def ncc_char(g, tpls):
    best, bsc, scores = None, -1.0, {}
    for ch, insts in tpls.items():
        sc = -1.0
        for (t, t0, tn) in insts:
            th, tw = t.shape
            if th >= g.shape[0] or tw >= g.shape[1] or tn < 1e-9:
                continue
            wins = np.lib.stride_tricks.sliding_window_view(g, (th, tw))
            wm = wins.mean(axis=(2, 3), keepdims=True)
            wc = wins - wm
            wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
            s = np.abs((wc * t0).sum(axis=(2, 3)) / np.maximum(wn * tn, 1e-9))
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


def recognize(roi):
    m = gold_rel(roi)
    comps = comps_of(m)
    if not comps:
        return None, []
    chars, scores = [], []
    for (cx, cy, cw, chh, _) in comps:
        # 逗号/点：低+窄（几何 token）
        if chh <= 16 and cw <= 12:
            sub = m[cy:cy + chh, cx:cx + cw]
            # 逗号=下伸（底锚），点=中高
            rows = np.where(sub.sum(axis=1) > 0)[0]
            top = rows.min()
            chars.append("," if top >= 4 else ".")
            scores.append(0.10)
            continue
        sub = m[cy:cy + chh, cx:cx + cw]
        g = sub.astype(np.float64)
        ch, sc, mg = ncc_char(g, T)
        if ch is None:
            chars.append("?")
            scores.append(-1.0)
        else:
            chars.append(ch)
            scores.append(round(mg, 2))
    return "".join(chars), scores


cases = []
for ann in GT["annotations"]:
    fid = ann["frame_id"]
    st = (ann.get("fields") or {}).get("stack") or {}
    for seat, v in st.items():
        if v is not None:
            cases.append((fid, seat, v))

print(f"\n{'seat':4s} {'value':>8s} {'raw':>10s} {'exp':>9s} {'match':6s} "
      f"scores")
good = 0
for fid, seat, value in cases:
    arr = load(fid)
    x0, y0, x1, y1 = STACK_ROI_1876[seat]
    raw, scores = recognize(arr[y0:y1, x0:x1])
    exp = render_expected(value)
    ok = raw == exp
    good += int(ok)
    print(f"{seat:4s} {value:>8d} {str(raw):>10s} {exp:>9s} "
          f"{'OK' if ok else 'XX ':6s} {scores}")

print(f"\nraw==exp: {good}/{len(cases)}")

# 未标注座格（f03 S6 hero / f05 S5）不误报检查
print("\n=== 未标注座格（应无有效 gold → None）===")
for fid, seat in (("POT_CHANGED_102_1787992505794_full.png", "S6"),
                  ("POT_CHANGED_139_1787992592538_full.png", "S5")):
    arr = load(fid)
    x0, y0, x1, y1 = STACK_ROI_1876[seat]
    raw, _ = recognize(arr[y0:y1, x0:x1])
    print(f"{fid[:6]} {seat}: raw={raw!r}")
