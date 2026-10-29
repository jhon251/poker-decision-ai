"""P26-A — 端到端模拟 v2（精确镜像 PotRecognizer 预处理链）：
模板 = _bright_tight(RGB luma>120 收缩) → grayscale float；
候选 = gold_rel 组件 crop（RGB 原样）→ _bright_tight → pad → to_gray；
NCC + min_score 0.7 + margin 0.08；几何：h<10 且 w<=8 → '.'；w>24 → 'M'；
逗号/数字 NCC。<10px高窄组件→','；
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    parse_stack_text)
from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    PotRecognizer)

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

POT = PotRecognizer(window="1876")   # 复用冻结模板/匹配（只读）


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


def recognize(roi):
    m = gold_rel(roi)
    comps = comps_of(m)
    if not comps:
        return None, []
    chars = []
    for (cx, cy, cw, chh, _) in comps:
        y0, y1 = max(0, cy - 3), min(roi.shape[0], cy + chh + 3)
        x0, x1 = max(0, cx - 3), min(roi.shape[1], cx + cw + 3)
        crop = roi[y0:y1, x0:x1]
        lc = (crop[..., 0].astype(np.int16) * 0.299
              + crop[..., 1].astype(np.int16) * 0.587
              + crop[..., 2].astype(np.int16) * 0.114)
        rows = np.where((lc > 60).sum(axis=1) > 0)[0]
        hh = (rows.max() - rows.min() + 1) if len(rows) else 0
        if hh and hh < 10 and cw <= 8:
            chars.append(".")
            continue
        if cw > 24:
            chars.append("M")
            continue
        ch, sc, margin = POT._match_char(crop)
        if ch is None or sc is None or sc < POT.min_score \
                or margin < POT.min_margin:
            chars.append("?")
        else:
            chars.append(ch)
    return "".join(chars), None


cases = []
for ann in GT["annotations"]:
    fid = ann["frame_id"]
    st = (ann.get("fields") or {}).get("stack") or {}
    for seat, v in st.items():
        if v is not None:
            cases.append((fid, seat, v))

good = 0
format_ok = 0
for fid, seat, value in cases:
    arr = load(fid)
    x0, y0, x1, y1 = STACK_ROI_1876[seat]
    raw, _ = recognize(arr[y0:y1, x0:x1])
    exp = render_expected(value)
    ok = raw == exp
    good += int(ok)
    val = parse_stack_text(raw) if raw else None
    print(f"{seat:4s} {value:>8d} raw={str(raw):>10s} exp={exp:>9s} "
          f"{'OK' if ok else 'XX':4s} parse={val}")

print(f"\nraw==exp: {good}/{len(cases)}")

# 未标注座格
print("\n=== 未标注座格 ===")
for fid, seat in (("POT_CHANGED_102_1787992505794_full.png", "S6"),
                  ("POT_CHANGED_139_1787992592538_full.png", "S5")):
    arr = load(fid)
    x0, y0, x1, y1 = STACK_ROI_1876[seat]
    raw, _ = recognize(arr[y0:y1, x0:x1])
    print(f"{fid[:6]} {seat}: raw={raw!r}")
