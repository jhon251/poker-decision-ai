"""P13-A — Bet 标签精确测量 + 值分组 + 严格按钮搜索（只读）。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

# 1280×779 布局：Bet 标签数字行（whitetext 实测校准；加宽防 6-7 位截断）
LABEL_ROWS = {
    "A": (430, 314, 155, 36),   # 中左（pepe1213 下注区）
    "B": (664, 251, 148, 36),   # 上中右（CruelBlsck609 下注区）
    "C": (870, 650, 135, 38),   # 底部中央（第四玩家下注区）
}


def load_frame(path: Path) -> np.ndarray:
    w, h, rgba = decode_png(path.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def luma_of(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    return (r * 0.299 + g * 0.587 + b * 0.114).astype(np.int16)


def digit_metrics(lum: np.ndarray, x0: int, y0: int, w: int, h: int):
    sub = lum[y0:y0 + h, x0:x0 + w]
    out = {"bbox": [x0, y0, w, h]}
    for thr in (110, 140, 160, 190):
        cols = (sub > thr).sum(axis=0)
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
        out[str(thr)] = {
            "zero_cols": int((cols == 0).sum()),
            "ink_cores": len(runs),
            "max_ink_run": int(max(runs)) if runs else 0,
            "core_sizes": [int(r) for r in runs],
        }
    return out


def strip_hash(lum: np.ndarray, x0: int, y0: int, w: int, h: int,
               thr: int = 160) -> str:
    sub = (lum[y0:y0 + h, x0:x0 + w] > thr).astype(np.uint8)
    ys, xs = np.where(sub)
    if len(ys) < 10:
        return "EMPTY"
    crop = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return hashlib.sha1(crop.tobytes()).hexdigest()[:12]


def is_pill(lum: np.ndarray, x0: int, y0: int, w: int, h: int) -> bool:
    """bet pill 判定：数字行周边 8px 内暗背景占比高 + 行内白像素足。"""
    cx0, cy0 = max(0, x0 - 8), max(0, y0 - 6)
    cx1 = min(lum.shape[1], x0 + w + 8)
    cy1 = min(lum.shape[0], y0 + h + 6)
    ctx = lum[cy0:cy1, cx0:cx1]
    dark = float((ctx < 70).mean())
    white = int((lum[y0:y0 + h, x0:x0 + w] > 160).sum())
    return dark > 0.30 and white >= 30


def buttons(arr: np.ndarray) -> list:
    """饱和亮块（按钮候选）：sat>70 & lum>100，w>=50,h>=25,area>=4000。"""
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    lum = luma_of(arr)
    m = (sat > 70) & (lum > 100)
    rows = m.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 100 and run is None:
            run = i
        elif v <= 100 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        block = m[bs:be]
        ys, xs = np.where(block)
        if len(ys) < 4000:
            continue
        w = int(xs.max() - xs.min() + 1)
        h = int(ys.max() - ys.min() + 1)
        if w < 50 or h < 25:
            continue
        out.append({
            "bbox": [int(xs.min()), int(ys.min() + bs), w, h],
            "area": int(len(ys)),
            "mean_rgb": [int(r[bs:be][ys, xs].mean()),
                         int(g[bs:be][ys, xs].mean()),
                         int(b[bs:be][ys, xs].mean())],
        })
    return out


def main() -> int:
    since_s = 134324017101490527 / 1e7 - 11644473600
    fulls = sorted(Path("data/vision2/coinpoker_evidence").glob("*_full.png"))
    new = [p for p in fulls if p.stat().st_mtime >= since_s]

    results = {"frames": len(new),
               "labels": {k: {"seen": 0, "value_groups": {}} for k in
                          LABEL_ROWS},
               "button_frames": []}
    crop_dir = Path("data/vision2/p13a_crops3")
    crop_dir.mkdir(parents=True, exist_ok=True)
    for path in new:
        arr = load_frame(path)
        lum = luma_of(arr)
        for name, (x0, y0, w, h) in LABEL_ROWS.items():
            hsh = strip_hash(lum, x0, y0, w, h)
            if hsh == "EMPTY":
                continue
            if not is_pill(lum, x0, y0, w, h):
                continue
            rec = results["labels"][name]
            rec["seen"] += 1
            d = rec["value_groups"].setdefault(hsh, {"n": 0, "frames": []})
            d["n"] += 1
            d["frames"].append(path.stem)
            fid = path.stem.rsplit("_", 1)[0].rsplit("_", 1)[-1]
            d.setdefault("frame_ids", set()).add(fid)
            if not d.get("crop_saved"):
                d["crop_saved"] = True
                d["metrics"] = digit_metrics(lum, x0, y0, w, h)
                crop = arr[y0:y0 + h, x0:x0 + w]
                big = np.repeat(np.repeat(crop, 4, axis=0), 4, axis=1)
                Path(crop_dir / f"{name}_{hsh}.png").write_bytes(
                    encode_rgba_png(big.shape[1], big.shape[0],
                                    big.tobytes()))
        for blk in buttons(arr):
            results["button_frames"].append({"frame": path.stem,
                                             "block": blk})

    # JSON 序列化：set → list
    for rec in results["labels"].values():
        for g in rec["value_groups"].values():
            g["frame_ids"] = sorted(g.get("frame_ids", set()))
    out = Path("data/vision2/p13a_betlabels.json")
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps({
        "frames": results["frames"],
        "label_seen": {k: v["seen"] for k, v in results["labels"].items()},
        "label_groups": {k: len(v["value_groups"]) for k, v in
                         results["labels"].items()},
        "button_frames": len(results["button_frames"]),
    }, indent=1))
    for name, rec in results["labels"].items():
        print(f"--- {name} groups ---")
        for hsh, g in sorted(rec["value_groups"].items(),
                             key=lambda kv: -kv[1]["n"])[:12]:
            ids = sorted(g.get("frame_ids", set()))
            g["frame_ids_list"] = ids
            print(f"  hash={hsh} n={g['n']} distinct_ids={len(ids)} frames={g['frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
