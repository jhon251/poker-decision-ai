"""P13-Step B — 只读 Action 证据分析：按钮发现 + 跨帧状态变化。

- 仅分析 Step B 期间新采集的 *_full.png（since 时间戳之后）；
- 全帧饱和色按钮候选发现（颜色掩码 + 行带×列段 + 面积/长宽比门）→ 跨帧聚类；
- 复用冻结 PotObserver / StatusObserver 作为每帧客观状态信号（pot 值 / 徽章）——
  用于"按钮消失/变化 ↔ 状态变化"语义绑定观察（纯消费者，零识别改动）；
- 输出 data/vision2/p13_stepb_actions.json + stdout 摘要。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PotObserver, StatusObserver)

SINCE_FILETIME = 134324456119382219


def load(name: Path) -> np.ndarray:
    w, h, rgba = decode_png(name.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def color_masks(arr: np.ndarray) -> dict:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    return {
        "red": (r > g + 50) & (r > b + 50) & (r > 110),
        "green": (g > r + 40) & (g > b + 40) & (g > 90),
        "blue": (b > r + 40) & (b > g + 40) & (b > 100),
        "yellow": (r > 150) & (g > 110) & (b < 90) & (np.abs(r - g) < 90),
        "white": (lum > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40),
    }


def candidates(mask: np.ndarray, min_h: int = 40, min_w: int = 60,
               min_area: int = 3000, max_h: int = 160) -> list:
    rows = mask.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 150 and run is None:
            run = i
        elif v <= 150 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        if be - bs < min_h or be - bs > max_h:
            continue
        band = mask[bs:be]
        cols = band.sum(axis=0)
        runs, run = [], None
        for i, v in enumerate(cols):
            if v > 30 and run is None:
                run = i
            elif v <= 30 and run is not None:
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
    since_s = SINCE_FILETIME / 1e7 - 11644473600
    fulls = sorted(Path("data/vision2/coinpoker_evidence").glob("*_full.png"))
    new = [p for p in fulls if p.stat().st_mtime >= since_s]
    print("new_frames:", len(new))
    pot_obs = PotObserver()
    status_obs = StatusObserver()
    clusters: dict = {}
    per_frame = []
    for path in new:
        arr = load(path)
        w, h = arr.shape[:2]
        mk = color_masks(arr)
        fr = {"frame": path.stem, "size": [w, h], "cands": []}
        for color in ("red", "green", "blue", "yellow"):
            for c in candidates(mk[color]):
                fr["cands"].append({"color": color, **c})
        # 冻结观察者信号（当前帧）
        pot = pot_obs.observe(arr, w, h)
        status = status_obs.observe(arr, w, h)
        fr["pot"] = pot.get("pot")
        fr["status_badges"] = {s: v.get("status") for s, v in
                               (status.get("seats") or {}).items()}
        for c in fr["cands"]:
            key = (c["color"], c["bbox"][1] // 30, c["bbox"][0] // 30)
            d = clusters.setdefault(key, {"n": 0, "frames": [],
                                          "sample": None})
            d["n"] += 1
            if len(d["frames"]) < 8:
                d["frames"].append(path.stem)
            if d["sample"] is None:
                d["sample"] = c
        per_frame.append(fr)
    Path("data/vision2/p13_stepb_actions.json").write_text(
        json.dumps({"clusters": [
            {**{k: v for k, v in d.items() if k != "frames"},
             "distinct_frames": len(d["frames"]), "frames": d["frames"]}
            for d in sorted(clusters.values(), key=lambda d: -d["n"])],
                    "per_frame": per_frame},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    print("=== button-candidate clusters (top 30) ===")
    for d in sorted(clusters.values(), key=lambda d: -d["n"])[:30]:
        print(f"n={d['n']:4d} color={d['sample']['color']:7s} "
              f"bbox={d['sample']['bbox']} area={d['sample']['area']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
