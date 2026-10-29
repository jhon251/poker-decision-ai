"""P13-A — 精确区域测量 + 按钮搜索（只读分析工具）。

用法:
  python scripts/phase_v2s_p13a_measure.py --frame <png> --region x,y,w,h [--region ...]
      → 打印区域亮文本指标 + 保存 4x 放大 crop（p13a_crops/<name>.png）
  python scripts/phase_v2s_p13a_measure.py --hunt --since-filetime N
      → 全帧扫描下右区饱和块（按钮候选）+ 文本带；打印出现帧
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402


def load_frame(path: Path) -> np.ndarray:
    w, h, rgba = decode_png(path.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def luma_of(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    return (r * 0.299 + g * 0.587 + b * 0.114).astype(np.int16)


def measure(arr: np.ndarray, x0: int, y0: int, w: int, h: int,
            name: str, crop_dir: Path, scale: int = 4) -> dict:
    sub = arr[y0:y0 + h, x0:x0 + w]
    lum = luma_of(sub)
    mask = lum > 90
    ys, xs = np.where(mask)
    res = {"region": [x0, y0, w, h], "bright_px": int(mask.sum())}
    if len(ys) >= 20:
        iy0, iy1 = ys.min(), ys.max() + 1
        ix0, ix1 = xs.min(), xs.max() + 1
        inner = lum[iy0:iy1, ix0:ix1]
        bh, bw = inner.shape
        bright = inner[inner > 90]
        proj = {}
        for thr in (90, 110, 125, 140):
            cols = (inner > thr).sum(axis=0)
            runs, run = [], None
            for v in cols:
                if v > 0 and run is None:
                    run = 1
                elif v > 0:
                    run += 1
                elif run is not None:
                    runs.append(run)
                    run = None
            if run is not None:
                runs.append(run)
            proj[str(thr)] = {
                "zero_cols": int((cols == 0).sum()),
                "zero_ratio": round(float((cols == 0).mean()), 4),
                "ink_cores": len(runs),
                "max_ink_run": int(max(runs)) if runs else 0,
            }
        res.update({
            "text_bbox": [int(ix0 + x0), int(iy0 + y0), int(bw), int(bh)],
            "luma_min": int(bright.min()), "luma_max": int(bright.max()),
            "projection": proj,
        })
    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop = sub
        big = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)
        out = crop_dir / f"{name}_x{scale}.png"
        out.write_bytes(encode_rgba_png(big.shape[1], big.shape[0],
                                        big.tobytes()))
        res["crop"] = str(out)
    return res


def hunt(arr: np.ndarray) -> list:
    """下右象限饱和块（按钮候选）+ 文本带。"""
    h, w = arr.shape[:2]
    x0, y0 = int(w * 0.42), int(h * 0.60)
    sub = arr[y0:h, x0:w]
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    lum = luma_of(sub)
    colored = (sat > 60) & (lum > 60)
    rows = colored.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 60 and run is None:
            run = i
        elif v <= 60 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        block = colored[bs:be]
        if block.sum() < 3000:
            continue
        ys, xs = np.where(block)
        r2 = sub[bs:be][ys, xs][:, 0].mean()
        g2 = sub[bs:be][ys, xs][:, 1].mean()
        b2 = sub[bs:be][ys, xs][:, 2].mean()
        out.append({
            "bbox": [int(xs.min() + x0), int(ys.min() + y0 + bs),
                     int(xs.max() - xs.min() + 1),
                     int(ys.max() - ys.min() + 1)],
            "area": int(block.sum()),
            "mean_rgb": [int(r2), int(g2), int(b2)],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame")
    ap.add_argument("--region", action="append", nargs=4, type=int,
                    metavar=("X", "Y", "W", "H"))
    ap.add_argument("--hunt", action="store_true")
    ap.add_argument("--hunt-region", nargs=4, type=int,
                    default=None, metavar=("X", "Y", "W", "H"))
    ap.add_argument("--since-filetime", type=int, default=0)
    ap.add_argument("--crop-dir", default="data/vision2/p13a_crops")
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--out", default="data/vision2/p13a_digest.json")
    args = ap.parse_args()

    crop_dir = Path(args.crop_dir)
    if args.frame and args.region:
        arr = load_frame(Path(args.frame))
        name = Path(args.frame).stem
        out = []
        for i, (x0, y0, w, h) in enumerate(args.region):
            out.append(measure(arr, x0, y0, w, h, f"{name}_r{i}", crop_dir))
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return 0
    if args.hunt:
        since_s = args.since_filetime / 1e7 - 11644473600
        fulls = sorted(Path("data/vision2/coinpoker_evidence")
                       .glob("*_full.png"))
        new = [p for p in fulls if p.stat().st_mtime >= since_s]
        blocks_by_bbox: dict = {}
        frames_with_blocks = 0
        for path in new:
            arr = load_frame(path)
            blks = walk_hunt(args, arr)
            if blks:
                frames_with_blocks += 1
            for blk in blks:
                key = (blk["bbox"][0] // 20, blk["bbox"][1] // 20,
                       blk["bbox"][2] // 20, blk["bbox"][3] // 20)
                d = blocks_by_bbox.setdefault(key, {"n": 0, "frames": [],
                                                    "sample": blk})
                d["n"] += 1
                if len(d["frames"]) < 10:
                    d["frames"].append(path.name)
        print(json.dumps({
            "scanned": len(new),
            "frames_with_blocks": frames_with_blocks,
            "blocks": sorted(blocks_by_bbox.values(),
                             key=lambda d: -d["n"])[:30],
        }, indent=1, ensure_ascii=False))
        return 0
    if args.digest:
        since_s = args.since_filetime / 1e7 - 11644473600
        fulls = sorted(Path("data/vision2/coinpoker_evidence")
                       .glob("*_full.png"))
        new = [p for p in fulls if p.stat().st_mtime >= since_s]
        rows = []
        for path in new:
            arr = load_frame(path)
            h, w = arr.shape[:2]
            lum = luma_of(arr)
            y0b = int(h * 0.60)
            blocks = hunt_region(arr, (int(w * 0.30), y0b, int(w * 0.70),
                                       h - y0b))
            # 底池胶囊（中心上方）
            pot = segmented(np.asarray(arr[y0b - 140:y0b - 10,
                                           int(w * 0.62):int(w * 0.86)]))
            rows.append({
                "frame": path.stem,
                "buttons": blocks,
                "pot_row_n": pot,
            })
        out = Path(args.out)
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                       encoding="utf-8")
        nbtn = sum(1 for r in rows if r["buttons"])
        print(json.dumps({"frames": len(rows),
                          "frames_with_buttons": nbtn}, indent=1))
        return 0
    print("need --frame+--region or --hunt or --digest")
    return 2


def walk_hunt(args, arr):
    if args.hunt_region:
        return hunt_region(arr, args.hunt_region)
    return hunt(arr)


def segmented(arr: np.ndarray, thr: int = 90, min_ink: int = 20) -> int:
    """行带内候选数（简单计数）。"""
    lum = luma_of(arr)
    mask = lum > thr
    rows = mask.sum(axis=1)
    n = 0
    run = None
    for i, v in enumerate(rows):
        if v > 2 and run is None:
            run = i
        elif v <= 2 and run is not None:
            if i - run >= 6 and mask[run:i].sum() >= min_ink:
                n += 1
            run = None
    if run is not None and len(rows) - run >= 6 and \
            mask[run:].sum() >= min_ink:
        n += 1
    return n


def hunt_region(arr: np.ndarray, region: tuple) -> list:
    x0, y0, w, h = region
    sub = arr[y0:y0 + h, x0:x0 + w]
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    lum = luma_of(sub)
    colored = (sat > 60) & (lum > 60)
    rows = colored.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 60 and run is None:
            run = i
        elif v <= 60 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        block = colored[bs:be]
        if block.sum() < 3000:
            continue
        ys, xs = np.where(block)
        out.append({
            "bbox": [int(xs.min() + x0), int(ys.min() + y0 + bs),
                     int(xs.max() - xs.min() + 1),
                     int(ys.max() - ys.min() + 1)],
            "area": int(block.sum()),
            "mean_rgb": [int(r[bs:be][ys, xs].mean()),
                         int(g[bs:be][ys, xs].mean()),
                         int(b[bs:be][ys, xs].mean())],
        })
    return out


if __name__ == "__main__":
    raise SystemExit(main())
