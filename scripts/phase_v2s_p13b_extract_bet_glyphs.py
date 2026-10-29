"""P13-B — Bet 字形模板提取（真实 1280×779 证据；provenance 入 manifest）。

对每个 (zone, value, frame) 证据实例：调用生产胶囊检测+字形分割（与识别器
同一算法）→ 按期望值 token 序列给字形标注 → 存 glyph PNG + manifest.json。
模板 = 真实证据（帧名/值/zone/组 hash 全部入 provenance）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    BET_ROIS_1280, detect_bet_capsule, gray_luma)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    segment_bet_glyphs)

OUT = Path("data/vision2/coinpoker_assets_v2bet1280")

# (zone, value, frame) —— 值/帧来自 P13-A 像素级分组（p13a_betlabels.json）
SAMPLES = [
    ("S_TL", 1000, "TABLE_CHANGED_100_1787928276694_full.png"),
    ("S_TL", 1000, "TABLE_CHANGED_101_1787928277940_full.png"),
    ("S_TL", 2000, "TABLE_CHANGED_213_1787928433101_full.png"),
    ("S_TL", 2000, "TABLE_CHANGED_215_1787928435301_full.png"),
    ("S_TR", 2000, "TABLE_CHANGED_100_1787928276694_full.png"),
    ("S_TL", 193400, "TABLE_CHANGED_117_1787928295861_full.png"),
    ("S_TL", 193400, "TABLE_CHANGED_119_1787928298017_full.png"),
    ("S_TR", 199680, "TABLE_CHANGED_117_1787928295861_full.png"),
    ("S_TL", 128190, "TABLE_CHANGED_413_1787928698111_full.png"),
    ("S_TL", 128190, "TABLE_CHANGED_420_1787928708495_full.png"),
    ("S_TL", 102000, "TABLE_CHANGED_185_1787928391857_full.png"),
    ("S_TR", 10450, "TABLE_CHANGED_3_1787928133694_full.png"),
    ("S_TR", 890401, "TABLE_CHANGED_20_1787928162646_full.png"),
    ("S_TL", 9280, "TABLE_CHANGED_60_1787928222972_full.png"),
    ("S_TL", 28000, "TABLE_CHANGED_398_1787928675733_full.png"),
    ("S_TL", 4000, "TABLE_CHANGED_145_1787928336066_full.png"),
]


def fmt(value: int) -> str:
    return f"{value:,}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    assets = []
    idx = 0
    skipped = []
    for zone, value, frame_name in SAMPLES:
        path = Path("data/vision2/coinpoker_evidence") / frame_name
        w, h, rgba = decode_png(path.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        lum = gray_luma(arr)
        roi = BET_ROIS_1280[zone]
        cap = detect_bet_capsule(lum, roi)
        if cap is None:
            skipped.append((zone, value, frame_name, "no_capsule"))
            continue
        glyphs = segment_bet_glyphs(cap["inner"])
        tokens = list(fmt(value))
        if glyphs is None or len(glyphs) != len(tokens):
            skipped.append((zone, value, frame_name,
                            f"glyph_count={None if glyphs is None else len(glyphs)}"
                            f" tokens={len(tokens)}"))
            continue
        x0, y0 = roi[0], roi[1]
        ix0, iy0 = cap["bbox"][0] - x0, cap["bbox"][1] - y0
        for gi, (gl, tok) in enumerate(zip(glyphs, tokens)):
            if tok == ",":
                continue              # 逗号 = 几何 token（无模板）
            gx = ix0 + gl["x"]
            gy = iy0 + gl["y"]
            crop = lum[y0 + gy:y0 + gy + gl["h"], x0 + gx:x0 + gx + gl["w"]]
            crop = np.clip(crop, 0, 255).astype(np.uint8)
            gray = np.repeat(crop[..., None], 3, axis=2)
            alpha = np.full(crop.shape + (1,), 255, dtype=np.uint8)
            rgba_out = np.concatenate([gray, alpha], axis=2)
            idx += 1
            fname = f"glyph_{idx:04d}_{tok}.png"
            (OUT / fname).write_bytes(encode_rgba_png(
                rgba_out.shape[1], rgba_out.shape[0], rgba_out.tobytes()))
            assets.append({
                "label": tok,
                "crop_ref": str(OUT / fname).replace("\\", "/"),
                "source_frame": frame_name,
                "zone": zone,
                "expected_value": value,
                "expected_tokens": fmt(value),
                "glyph_index": gi,
                "glyph_bbox_in_frame": [int(x0 + gx), int(y0 + gy),
                                        int(gl["w"]), int(gl["h"])],
                "class": "coinpoker_bet_1280",
            })
    manifest = {"class": "coinpoker_bet_1280", "window": "1280",
                "source": "real_capture_2026-08-28_p13a_session",
                "assets": assets}
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print("assets:", len(assets),
          Counter(a["label"] for a in assets))
    print("skipped:", skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
