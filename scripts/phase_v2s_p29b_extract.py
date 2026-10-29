"""P29-B — GT-provenance rank/suit 模板提取（3 合法 GT 帧）。

模板：rank 角标（卡左上，暗/红墨 tight）+ suit 角标（rank 下方小 suit）。
从 card1/card2 的已知 GT 身份提取；manifest 记录 frame_id/card/rank/suit/
bbox/provenance；不含任何自动推断——身份只来自 VALID GT。"""
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

# (frame, card_idx, rank, suit, card_bbox)
GT = [
    ("POT_CHANGED_102_1787992505794_full.png", 1, "9", "c", (812, 1011, 121, 178)),
    ("POT_CHANGED_102_1787992505794_full.png", 2, "3", "d", (836, 1011, 123, 178)),
    ("8bf82f134651db0add626be0304a2ff1.png", 1, "5", "c", (813, 1011, 121, 178)),
    ("8bf82f134651db0add626be0304a2ff1.png", 2, "3", "c", (837, 1011, 122, 178)),
    ("POT_CHANGED_110_1787992526057_full.png", 1, "6", "c", (813, 1011, 121, 178)),
    ("POT_CHANGED_110_1787992526057_full.png", 2, "3", "h", (837, 1011, 122, 178)),
]

# 角标布局实测（对 card bbox 的偏移）：
# rank 角标：card 左上角 (x0+8..x0+55, y0+5..y0+55)；suit 角标 rank 下方
RANK_OFF = (8, 5, 55, 58)      # 相对 card 左上 (dx, dy, w, h)
SUIT_OFF = (10, 60, 42, 60)


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def tighten_ink(sub):
    """暗/红墨 tight（卡面背景白）。"""
    r = sub[..., 0].astype(np.int16); g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = l < 120
    ys, xs = np.where(ink)
    if len(xs) < 8:
        return None
    return (int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def glyph_png(binmask):
    h, w = binmask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (binmask * 255).astype(np.uint8)
    return encode_rgba_png(w, h, rgba.tobytes())


assets = []
for fp, card_idx, rank, suit, cb in GT:
    arr = load(fp)
    cx, cy, cw, chh = cb
    # rank 角标
    r0 = arr[cy + RANK_OFF[1]:cy + RANK_OFF[1] + RANK_OFF[3],
             cx + RANK_OFF[0]:cx + RANK_OFF[0] + RANK_OFF[2]]
    rt = tighten_ink(r0)
    if rt is None:
        print(f"SKIP rank {fp[:6]} c{card_idx}")
        continue
    rx, ry, rw, rh = rt
    rmask = (r0[..., 0].astype(np.int16) * 0.299
             + r0[..., 1].astype(np.int16) * 0.587
             + r0[..., 2].astype(np.int16) * 0.114)[ry:ry + rh, rx:rx + rw] < 120
    rf = f"hcr_{rank}_{len(assets):03d}.png"
    (OUT / rf).write_bytes(glyph_png(rmask))
    assets.append({
        "label": f"rank_{rank}", "kind": "herocards_rank_1876",
        "crop_ref": str(OUT / rf).replace("\\", "/"),
        "bbox": list(cb),
        "card": card_idx, "rank": rank,
        "glyph_bbox_rel_card": [RANK_OFF[0] + rx, RANK_OFF[1] + ry, rw, rh],
        "source_frame_id": fp,
        "provenance": "VALID GT (user_annotated / image_observed V4 Flash)",
        "gt_status": "normal" if "110_" in fp or "8bf8" in fp else "dim+overlay",
    })
    # suit 角标
    s0 = arr[cy + SUIT_OFF[1]:cy + SUIT_OFF[1] + SUIT_OFF[3],
             cx + SUIT_OFF[0]:cx + SUIT_OFF[0] + SUIT_OFF[2]]
    st = tighten_ink(s0)
    if st is None:
        print(f"SKIP suit {fp[:6]} c{card_idx}")
        continue
    sx, sy, sw, sh = st
    smask = (s0[..., 0].astype(np.int16) * 0.299
             + s0[..., 1].astype(np.int16) * 0.587
             + s0[..., 2].astype(np.int16) * 0.114)[sy:sy + sh, sx:sx + sw] < 120
    sf = f"hcs_{suit}_{len(assets):03d}.png"
    (OUT / sf).write_bytes(glyph_png(smask))
    assets.append({
        "label": f"suit_{suit}", "kind": "herocards_suit_1876",
        "crop_ref": str(OUT / sf).replace("\\", "/"),
        "bbox": list(cb),
        "card": card_idx, "suit": suit,
        "glyph_bbox_rel_card": [SUIT_OFF[0] + sx, SUIT_OFF[1] + sy, sw, sh],
        "source_frame_id": fp,
        "provenance": "VALID GT (user_annotated / image_observed V4 Flash)",
        "gt_status": "normal" if ("110_" in fp or "8bf8" in fp) else "dim+overlay",
    })

manifest = {
    "client": "coinpoker",
    "version": "v1876herocards",
    "coordinate_system": "full_png_1876x1400",
    "ground_truth_status": "gt_derived_label",
    "note": ("P29-B: 从 3 个 VALID GT 帧提取（9c3d/5c3c/6c3h）。rank/suit 角标"
             "位于 card 左上角（rank 上、suit 下）。覆盖 rank={3,5,6,9},"
             " suit={c,d,h}——未覆盖完整 52 张；未覆盖 → UNKNOWN。"),
    "assets": assets,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                              ensure_ascii=False),
                                  encoding="utf-8")
from collections import Counter
print("资产统计:", dict(Counter(a["label"] for a in assets)))
