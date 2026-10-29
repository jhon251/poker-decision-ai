"""V3E-P1 round-2 — 确认包 v2（只读 + 仅写 gt_confirm_v3e_p1/v2/）。

横版逐帧合成图：pair ROI ×2 + 四角标 ×6（最近邻）+ 冻结模板对照面板。
机器只呈现像素与模板证据；标签仍以人工确认为准。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3e_p1_hero_audit import EXT_WIN, EV, glyph
from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
    _load_templates, HEROCARDS_PAIR_ROI)
from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png

OUT = Path("gt_confirm_v3e_p1/v2")
SCALE_ZOOM = 6
SCALE_PAIR = 2

# item_id → (frame, 附带模板对照面板 [(ch, label)])
ITEMS = [
    ("V2-01", "POT_CHANGED_250_1787992907809_full.png", []),
    ("V2-02", "POT_CHANGED_524_1787993575599_full.png", []),
    ("V2-03", "SCENE_CHANGED_429_1787993345715_full.png", []),
    ("V2-04", "POT_CHANGED_512_1787993546817_full.png", []),
    ("V2-05", "POT_CHANGED_606_1787993775320_full.png", []),
    ("V2-06", "TABLE_CHANGED_18_1787992293441_full.png", []),
    ("V2-07", "POT_CHANGED_404_1787993286628_full.png", []),
    ("V2-08", "POT_CHANGED_16_1787995303501_full.png", []),
    ("V2-09", "POT_CHANGED_208_1787992806955_full.png", []),
    ("V2-10", "SCENE_CHANGED_6_1787992258912_full.png", []),
    ("V2-11", "POT_CHANGED_708_1787994016446_full.png",
     [("4", "frozen rank_4 tmpl (suspect =6)")]),
    ("V2-12", "POT_CHANGED_293_1787993018309_full.png",
     [("J", "frozen rank_J tmpl (suspect =K)")]),
    ("V2-13", "POT_CHANGED_535_1787993603723_full.png", []),
    ("V2-14", "POT_CHANGED_454_1787993404728_full.png", []),
    ("V2-15", "POT_CHANGED_33_1787995356823_full.png", []),
    ("V2-16", "POT_CHANGED_420_1787993325856_full.png", []),
    ("V2-17", "POT_CHANGED_353_1787993164702_full.png", []),
    ("V2-18", "POT_CHANGED_47_1787995398974_full.png", []),
    ("V2-19", "POT_CHANGED_181_1787992731553_full.png", []),
    ("V2-20", "POT_CHANGED_290_1787993011134_full.png", []),
]
READINGS = {
    "POT_CHANGED_250_1787992907809_full.png": "As,3c",
    "POT_CHANGED_524_1787993575599_full.png": "Kd,Jc",
    "SCENE_CHANGED_429_1787993345715_full.png": "Qh,9s",
    "POT_CHANGED_512_1787993546817_full.png": "Td,6h",
    "POT_CHANGED_606_1787993775320_full.png": "7c,2c",
    "TABLE_CHANGED_18_1787992293441_full.png": "6c,3h",
    "POT_CHANGED_404_1787993286628_full.png": "As,5h",
    "POT_CHANGED_16_1787995303501_full.png": "Kc,3h",
    "POT_CHANGED_208_1787992806955_full.png": "Ah,5h",
    "SCENE_CHANGED_6_1787992258912_full.png": "8s,8h",
    "POT_CHANGED_708_1787994016446_full.png": "Ks,6c",
    "POT_CHANGED_293_1787993018309_full.png": "As,Kc",
    "POT_CHANGED_535_1787993603723_full.png": "Qd,Jc",
    "POT_CHANGED_454_1787993404728_full.png": "8c,3d",
    "POT_CHANGED_33_1787995356823_full.png": "4s,4h",
    "POT_CHANGED_420_1787993325856_full.png": "Qh,9s",
    "POT_CHANGED_353_1787993164702_full.png": "Th,5h",
    "POT_CHANGED_47_1787995398974_full.png": "4s,4h",
    "POT_CHANGED_181_1787992731553_full.png": "Jd,9h",
    "POT_CHANGED_290_1787993011134_full.png": "As,Kc",
}


def upscale(img, s):
    return np.repeat(np.repeat(img, s, axis=0), s, axis=1)


def tile(arr, win, scale):
    x0, y0, x1, y1 = win
    roi = arr[y0:y1, x0:x1].copy()
    return upscale(roi, scale)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    prep = _load_templates()
    PAD = 12
    manifest = {"items": [], "note": "v2 横版合成图；左=手牌区×2，右=角标×6 "
                "(c1_rank, c1_suit, c2_rank, c2_suit)；蓝框面板=冻结模板对照；"
                "476 已剔除（白簇 2335 < 8000，卡不在场）"}
    for item_id, frame, tmpls in ITEMS:
        w, h, rgba = decode_png((EV / frame).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        tiles = [tile(arr, HEROCARDS_PAIR_ROI, SCALE_PAIR)]
        for kind in ("rank", "suit"):
            for ci in (1, 2):
                tiles.append(tile(arr, EXT_WIN[ci][kind], SCALE_ZOOM))
        for ch, _label in tmpls:
            g = max(prep.get(ch, []), key=lambda p: p[0].shape[0] * p[0].shape[1])[0] \
                if prep.get(ch) else None
            if g is None:
                continue
            gh, gw = g.shape
            small = np.zeros((gh, gw, 4), dtype=np.uint8)
            small[..., 0] = 40
            small[..., 1] = 60
            small[..., 3] = 255
            small[g > 0] = [200, 220, 255, 255]
            big = upscale(small, SCALE_ZOOM)
            tiles.append(big)
        H = max(t.shape[0] for t in tiles) + 2 * PAD
        W = sum(t.shape[1] for t in tiles) + PAD * (len(tiles) + 1)
        sheet = np.zeros((H, W, 4), dtype=np.uint8)
        sheet[..., :3] = 28
        sheet[..., 3] = 255
        xx = PAD
        for t in tiles:
            yy = (H - t.shape[0]) // 2
            sheet[yy:yy + t.shape[0], xx:xx + t.shape[1]] = t
            xx += t.shape[1] + PAD
        f = OUT / f"{item_id}__{Path(frame).stem}.png"
        f.write_bytes(encode_rgba_png(sheet.shape[1], sheet.shape[0],
                                      sheet.tobytes()))
        # 回读验证
        w2, h2, rgba2 = decode_png(f.read_bytes())
        assert (w2, h2) == (sheet.shape[1], sheet.shape[0]), f
        manifest["items"].append({
            "id": item_id, "file": f.name, "frame": frame,
            "user_round2": READINGS[frame],
            "tmpl_panels": [c for c, _ in tmpls]})
        print(f.name, f"{w2}x{h2}")
    (OUT / "manifest_v2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"OK {len(manifest['items'])} composites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
