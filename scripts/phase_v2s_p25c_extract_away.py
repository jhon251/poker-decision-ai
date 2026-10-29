"""P25-C — away 模板资产提取（f01 S3 白二值，GT provenance）+ manifest。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876status")
OUT.mkdir(parents=True, exist_ok=True)

FRAME = "BUTTON_CHANGED_167_1787992682749_full.png"
# away 白 bbox（f01 S3 实测）：(1585,232,63,106)；tight 化
w, h, rgba = decode_png((EV / FRAME).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
sub = arr[232:338, 1585:1648]
l = sub[..., 0] * 0.299 + sub[..., 1] * 0.587 + sub[..., 2] * 0.114
r = sub[..., 0].astype(np.int16); g = sub[..., 1].astype(np.int16)
b = sub[..., 2].astype(np.int16)
white = ((l > 140) & (np.abs(r - g) < 30) & (np.abs(g - b) < 30))
ys, xs = np.where(white)
print("away tight bbox:", xs.min(), ys.min(), xs.max() - xs.min() + 1,
      ys.max() - ys.min() + 1)
tx, ty, tw, th = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), \
    int(ys.max() - ys.min() + 1)
tight = white[ty:ty + th, tx:tx + tw]
# 二值 RGBA PNG（模板即为白掩码形态）
png = encode_rgba_png(tw, th, np.dstack([
    (tight * 255).astype(np.uint8)] * 3 + [
    np.full((th, tw), 255, dtype=np.uint8)]).astype(np.uint8).tobytes())
(OUT / "away_badge_white.png").write_bytes(png)
print("saved", OUT / "away_badge_white.png", tw, "x", th)

manifest = {
    "assets": [{
        "label": "away_badge_white",
        "kind": "status_away_white_binary",
        "crop_ref": "away_badge_white.png",
        "source_frame_id": FRAME,
        "source_seat": "S3",
        "gt_status": "away",
        "crop_box": [1585 + tx, 232 + ty, tw, th],
        "method": "white_luma_binary",
        "note": "P25-C: 白咖啡杯+暂离 二值模板（NCC 用）",
    }],
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                              ensure_ascii=False),
                                  encoding="utf-8")
print("manifest written")
