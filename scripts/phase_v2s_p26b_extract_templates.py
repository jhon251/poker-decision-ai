"""P26-B — coinpoker_assets_v1876stack 模板提取（GT provenance）。

模板来源：6 个 GT 帧 stack 行 gold 组件（P26-A 实测 ROI + 背景相对判据），
按 GT 期望字符对齐；每字符多实例；逗号/点用几何 token（不建模板）。
manifest: {label, kind, crop_ref, source_frame_id, seat, gt_stack_value,
           bbox, expected_char, provenance}
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))
OUT = Path("data/vision2/coinpoker_assets_v1876stack")
OUT.mkdir(parents=True, exist_ok=True)

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
            & (l > bg * 1.6)).astype(np.uint8)


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


def glyph_png_data(mask):
    """二值掩码 → RGBA（白字 + alpha 掩码；同 pot 资产格式）。"""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (mask * 255).astype(np.uint8)
    return encode_rgba_png(w, h, rgba.tobytes())


assets = []
seen = 0
for ann in GT["annotations"]:
    fid = ann["frame_id"]
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
            print(f"  SKIP {fid[:6]} {seat} {value}: n={len(comps)} != "
                  f"{len(exp)}")
            continue
        for comp, ch in zip(comps, exp):
            if ch in ",.":
                continue
            if ch == "M":
                pass
            cx, cy, cw, chh, _ = comp
            sub = m[cy:cy + chh, cx:cx + cw]
            fname = f"stack_{ch}_{len(assets):03d}.png"
            (OUT / fname).write_bytes(glyph_png_data(sub))
            assets.append({
                "label": ch,
                "kind": "stack_glyph_1876",
                "crop_ref": str(OUT / fname).replace("\\", "/"),
                "bbox": [x0 + int(cx), y0 + int(cy), int(cw), int(chh)],
                "source_frame_id": fid,
                "seat": seat,
                "gt_stack_value": value,
                "expected_char": ch,
                "provenance": {
                    "source": "P26-A deep segmentation (GT frames)",
                    "method": "gold_rel mask + 8-conn component",
                    "gt_value": value,
                },
            })
            seen += 1

manifest = {
    "client": "coinpoker",
    "version": "v1876stack",
    "coordinate_system": "full_png_1876x1400",
    "ground_truth_status": "gt_derived_label",
    "note": ("P26-B: 从 6 个 GT 帧 stack 行 gold 组件提取；逗号/点用几何 "
             "token 不建模板；glyph 高≈34px（与 pot 30px 不同，不复用 pot）"),
    "assets": assets,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                              ensure_ascii=False),
                                  encoding="utf-8")
# 统计
from collections import Counter
cnt = Counter(a["label"] for a in assets)
print(f"提取 glyph 总数 = {len(assets)}; 分布 = {dict(cnt)}")
