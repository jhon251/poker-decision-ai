"""P29-B — 干净模板重建：用连通域 + 尺寸过滤提取 digit/suit tight glyph。
模板必须：真实 GT 帧、每实例 bbox 来自测量、不含卡缘/伪影。重建 manifest。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876herocards")
OUT.mkdir(parents=True, exist_ok=True)

# 每卡角标搜索窗（绝对坐标，连通域实测）：
WINDOWS = {
    "card1": {"rank": (822, 1022, 852, 1070), "suit": (822, 1076, 864, 1124)},
    "card2": {"rank": (938, 1014, 970, 1064), "suit": (932, 1070, 978, 1116)},
}
# 尺寸过滤（digit 与 suit 角标均 ~20-45 宽, 35-55 高；排除边缘线 >60 宽）
MIN_N = 120
MAX_W = 50
MIN_W = 12
MIN_H = 30
MAX_H = 56

GT = [
    ("POT_CHANGED_102_1787992505794_full.png", 1, "9", "c"),
    ("POT_CHANGED_102_1787992505794_full.png", 2, "3", "d"),
    ("8bf82f134651db0add626be0304a2ff1.png", 1, "5", "c"),
    ("8bf82f134651db0add626be0304a2ff1.png", 2, "3", "c"),
    ("POT_CHANGED_110_1787992526057_full.png", 1, "6", "c"),
    ("POT_CHANGED_110_1787992526057_full.png", 2, "3", "h"),
]


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def blocks(arr, x0, y0, x1, y1, min_n=MIN_N):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = (l < 120).astype(np.uint8)
    h, w = ink.shape
    mask = ink.astype(bool)
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]; seen[sy, sx] = True
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
            if len(cells) < min_n:
                continue
            xs = [c[0] for c in cells]; ys2 = [c[1] for c in cells]
            out.append((x0 + min(xs), y0 + min(ys2),
                        max(xs) - min(xs) + 1,
                        max(ys2) - min(ys2) + 1, len(cells)))
    return out


def glyph_png(mask):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (mask * 255).astype(np.uint8)
    return encode_rgba_png(w, h, rgba.tobytes())


assets = []
problems = []
for fp, ci, rank, suit in GT:
    arr = load(fp)
    key = f"card{ci}"
    for part, (r_, s_) in (("rank", (rank, suit)),):
        pass
    # rank
    for (rx, ry, rw, rh, rn) in blocks(arr, *WINDOWS[key]["rank"]):
        if MIN_W <= rw <= MAX_W and MIN_H <= rh <= MAX_H:
            # 提取 mask
            roi = arr[ry:ry + rh, rx:rx + rw]
            l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 \
                + roi[..., 2] * 0.114
            m = (l < 120)
            rf = f"hcr_{rank}_c{ci}_{rx}_{ry}.png"
            (OUT / rf).write_bytes(glyph_png(m))
            assets.append({
                "label": f"rank_{rank}", "kind": "herocards_rank_1876",
                "crop_ref": rf, "bbox": [rx, ry, rw, rh],
                "card": ci, "rank": rank, "glyph_size": [rw, rh],
                "source_frame_id": fp,
                "provenance": "VALID GT (user_annotated/image_observed)",
                "gt_status": "normal" if ("110_" in fp or "8bf8" in fp)
                else "dim+overlay",
            })
            break
    else:
        problems.append(f"{fp[:8]} c{ci} rank")
    # suit
    for (sx, sy, sw, sh, sn) in blocks(arr, *WINDOWS[key]["suit"]):
        if MIN_W <= sw <= MAX_W and MIN_H <= sh <= MAX_H:
            roi = arr[sy:sy + sh, sx:sx + sw]
            l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 \
                + roi[..., 2] * 0.114
            m = (l < 120)
            sf = f"hcs_{suit}_c{ci}_{sx}_{sy}.png"
            (OUT / sf).write_bytes(glyph_png(m))
            assets.append({
                "label": f"suit_{suit}", "kind": "herocards_suit_1876",
                "crop_ref": sf, "bbox": [sx, sy, sw, sh],
                "card": ci, "suit": suit, "glyph_size": [sw, sh],
                "source_frame_id": fp,
                "provenance": "VALID GT (user_annotated/image_observed)",
                "gt_status": "normal" if ("110_" in fp or "8bf8" in fp)
                else "dim+overlay",
            })
            break
    else:
        problems.append(f"{fp[:8]} c{ci} suit")

manifest = {
    "client": "coinpoker",
    "version": "v1876herocards",
    "coordinate_system": "full_png_1876x1400",
    "ground_truth_status": "gt_derived_label",
    "note": ("P29-B clean rebuild：连通域 + 尺寸过滤提取（排除卡缘/伪影）。"
             "rank={3,5,6,9} suit={c,d,h}（3 合法 GT）。未覆盖 → UNKNOWN。"),
    "assets": assets,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                              ensure_ascii=False),
                                  encoding="utf-8")
from collections import Counter
print("assets:", dict(Counter(a["label"] for a in assets)))
print("problems:", problems)
for a in assets:
    print(f"  {a['label']}: {a['glyph_size']} from {a['source_frame_id'][:6]} "
          f"c{a['card']}")
