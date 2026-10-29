"""P13-Step A — 只读客观像素重分析：V2H 18 帧（1848×1388）动作面板证据。

对 10 帧做：
- 颜色掩码（绿/红/蓝/黄/亮灰）→ 行带 × 列段候选矩形；
- 几何/面积/长宽比/位置跨帧稳定性 → 面板 vs 筹码/徽章/bet/pot/文字区分；
- 动作行（y-带）锚定 + 相邻 UI 扫描 → Check / Raise 候选搜索；
- 全部输出客观测量（非目视）；不修改生产代码；不创建识别器。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

FRAME_DIR = Path("picture data")
EVAL = json.loads(Path("data/vision2/pixel_v2e_eval.json")
                  .read_text(encoding="utf-8"))
FRAMES = [f["file"] for f in EVAL["frames"]]


def load(name: str) -> np.ndarray:
    w, h, rgba = decode_png((FRAME_DIR / name).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def masks_of(arr: np.ndarray) -> dict:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    return {
        "green": (g > r + 40) & (g > b + 40) & (g > 90),
        "red": (r > g + 50) & (r > b + 50) & (r > 110),
        "blue": (b > r + 40) & (b > g + 40) & (b > 100),
        "yellow": (r > 150) & (g > 110) & (b < 90) & (np.abs(r - g) < 90),
        # 亮色低饱和（灰/白按钮，含深底上的白色几何块）
        "bright_desat": (lum > 150) & (np.abs(r - g) < 40)
                       & (np.abs(g - b) < 40),
    }


def rects(mask: np.ndarray, min_h: int = 25, min_w: int = 40,
          min_area: int = 2500) -> list:
    """行带 > 220px → 列段 > 40px → 候选矩形。"""
    rows = mask.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 220 and run is None:
            run = i
        elif v <= 220 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        if be - bs < min_h:
            continue
        band = mask[bs:be]
        cols = band.sum(axis=0)
        runs, run = [], None
        for i, v in enumerate(cols):
            if v > 40 and run is None:
                run = i
            elif v <= 40 and run is not None:
                runs.append((run, i))
                run = None
        if run is not None:
            runs.append((run, len(cols)))
        for cs, ce in runs:
            if ce - cs < min_w:
                continue
            sub = mask[bs:be, cs:ce]
            ys, xs = np.where(sub)
            if len(ys) < min_area:
                continue
            out.append({"bbox": [int(xs.min() + cs), int(ys.min() + bs),
                                 int(xs.max() - xs.min() + 1),
                                 int(ys.max() - ys.min() + 1)],
                        "area": int(len(ys))})
    return out


def main() -> int:
    result = {}
    for name in FRAMES:
        arr = load(name)
        mk = masks_of(arr)
        fr = {}
        for color, m in mk.items():
            fr[color] = rects(m)
        result[name] = fr
    Path("data/vision2/p13_step_a_panels.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")

    # 汇总：跨帧候选
    from collections import defaultdict
    agg = defaultdict(list)           # (color, y//40, x//40) -> [ (frame, bbox, area) ]
    for name, fr in result.items():
        for color, rc in fr.items():
            for c in rc:
                k = (color, c["bbox"][1] // 40, c["bbox"][0] // 40)
                agg[k].append((name[:8], c["bbox"], c["area"]))
    print("=== 候选簇（色, y带, x带）≥2 帧 ===")
    rows = []
    for (color, yb, xb), hits in sorted(agg.items(),
                                        key=lambda kv: -len(kv[1])):
        if len(hits) >= 2:
            rows.append((len(hits), color, yb * 40, xb * 40,
                         hits[0][1], hits[0][2]))
    for n, color, y, x, bbox, area in rows[:40]:
        print(f"n={n:3d} color={color:10s} y~{y} x~{x} bbox={bbox} area={area}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
