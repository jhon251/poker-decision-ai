"""P13-Step A — 底部带全帧分类统计 + ASCII maps 落盘。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

ev = json.loads(Path("data/vision2/pixel_v2e_eval.json")
                .read_text(encoding="utf-8"))
FRAMES = [f["file"] for f in ev["frames"]]


def load(name: str) -> np.ndarray:
    w, h, rgba = decode_png((Path("picture data") / name).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def classify(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    cls = np.full(lum.shape, ord("."), dtype=np.uint8)
    cls[lum < 35] = ord("#")
    cls[(g > r + 40) & (g > b + 40) & (g > 90)] = ord("G")
    cls[(r > g + 50) & (r > b + 50) & (r > 110)] = ord("R")
    cls[(b > r + 40) & (b > g + 40) & (b > 100)] = ord("B")
    cls[(r > 150) & (g > 110) & (b < 90) & (np.abs(r - g) < 90)] = ord("Y")
    cls[(lum > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)] = ord("W")
    return cls


def ascii_map(cls: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              scale: int = 8) -> str:
    sub = cls[y0:y1, x0:x1]
    h2 = sub.shape[0] // scale
    w2 = sub.shape[1] // scale
    out = []
    for rr in range(h2):
        line = []
        for cc in range(w2):
            blk = sub[rr * scale:(rr + 1) * scale, cc * scale:(cc + 1) * scale]
            counts = {}
            for ch in np.unique(blk):
                counts[ch] = int((blk == ch).sum())
            picked = "."
            for prio in (ord("R"), ord("G"), ord("B"), ord("Y"), ord("W"),
                         ord("#"), ord(".")):
                if counts.get(prio, 0) >= max(4, scale * scale // 8):
                    picked = chr(prio)
                    break
            line.append(picked)
        out.append(f"{rr * scale + y0:4d} " + "".join(line))
    return "\n".join(out)


def main() -> int:
    lines = []
    stats = []
    for idx, name in enumerate(FRAMES, 1):
        arr = load(name)
        cls = classify(arr)
        bottom = cls[950:1388, 600:1848]
        counts = {ch: int((bottom == ord(ch)).sum())
                  for ch in "RGBYW#."}
        # 底部带红/绿/蓝/黄 bbox
        bboxes = {}
        for ch in "RGBY":
            m = (bottom == ord(ch)).astype(np.uint8)
            if m.sum() >= 200:
                ys, xs = np.where(m)
                bboxes[ch] = [int(xs.min() + 600), int(ys.min() + 950),
                              int(xs.max() - xs.min() + 1),
                              int(ys.max() - ys.min() + 1),
                              int(m.sum())]
        stats.append({"frame": idx, "name": name[:12],
                      "counts": counts, "bboxes": bboxes})
        lines.append(f"===== frame {idx} ({name[:12]}) =====")
        lines.append(ascii_map(cls, 600, 950, 1840, 1386))
        lines.append("")
    Path("data/vision2/p13_step_a_ascii.txt").write_text(
        "\n".join(lines), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    # V2G 绿 call 锚定采样
    for idx, name in enumerate(FRAMES, 1):
        arr = load(name)
        for zn, (x, y, w, h) in {"v2g_call_anch": (540, 1000, 140, 40),
                                 "v2h_call_anch": (800, 1140, 120, 65)}.items():
            sub = arr[y:y + h, x:x + w]
            r = sub[..., 0].astype(np.int16)
            g = sub[..., 1].astype(np.int16)
            b = sub[..., 2].astype(np.int16)
            print(f"f{idx} {zn:14s} mean_rgb=({int(r.mean())},{int(g.mean())},"
                  f"{int(b.mean())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
