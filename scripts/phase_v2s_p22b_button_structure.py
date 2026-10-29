"""P22-B — 精测帧04 按钮结构（红环+白D）+ 提取白D模板。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform import encode_rgba_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876dealer")


def main() -> int:
    fid = "POT_CHANGED_12_1787995290794_full.png"
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    # 按钮区域（P21 标定 1469,745,30,30）——采样附近找最精确结构
    bx, by, bw, bh = 1469, 745, 30, 30
    roi = arr[by:by + bh, bx:bx + bw]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    red = (r > 120) & (g < 100) & (b < 100)
    white = l > 180
    print("帧04 按钮 30x30 红/白结构（R=红, W=白, .=其它）:")
    for i in range(bh):
        row = ""
        for j in range(bw):
            row += "R" if red[i, j] else ("W" if white[i, j] else ".")
        print(f"  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
