"""P22-B — 精确定位帧04按钮：紧凑红blob搜索（S5区域），输出精确结构。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.geometry import connected_components  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
    fid = "POT_CHANGED_12_1787995290794_full.png"
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    red = ((r > 120) & (g < 100) & (b < 100)).astype(np.uint8)
    # S5 区域 (1618,868) ±400px
    comps = [c for c in connected_components(red, min_area=50)
             if abs(int(c.bbox.x + c.bbox.width / 2) - 1618) < 400
             and abs(int(c.bbox.y + c.bbox.height / 2) - 868) < 400]
    for c in sorted(comps, key=lambda c: -(c.bbox.width * c.bbox.height))[:8]:
        x, y = int(c.bbox.x), int(c.bbox.y)
        ww, hh = int(c.bbox.width), int(c.bbox.height)
        print(f"红blob: ({x},{y},{ww},{hh}) area={c.area}")
        # 白D检查
        sub = l[y:y + hh, x:x + ww]
        print(f"   白像素={int((sub > 180).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
