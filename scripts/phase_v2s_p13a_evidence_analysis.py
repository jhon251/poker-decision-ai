"""P13-A — 只读证据分析 v2（不修改 capture 管线 / 不修改任何 Recognizer）。

输入：P13-A 期间新采集的 coinpoker_evidence/*_full.png（仅 since 时间戳之后）。
本版本窗口尺寸无关：对每帧做分段亮文本候选提取（行带 × 列段），
另对 1419×1066 帧额外做冻结 1419 Bet ROI 扫描（与封存剖面可比）。
Action：底部动作带（最后 ~11% 窗口高度）→ 饱和色 UI 块 + 文本带 + 块集状态变化。
输出：data/vision2/p13a_evidence_analysis.json + stdout 摘要。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    BET_ROIS_1419)


def load_frame(path: Path) -> np.ndarray:
    w, h, rgba = decode_png(path.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def luma_of(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    return (r * 0.299 + g * 0.587 + b * 0.114).astype(np.int16)


def _metrics(inner: np.ndarray, bx: int, by: int) -> dict | None:
    h, w = inner.shape
    if h < 6 or w < 6:
        return None
    bright = inner[inner > 90]
    if bright.size == 0:
        return None
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
            "col_max": int(cols.max()),
        }
    return {
        "bbox": [int(bx), int(by), int(w), int(h)],
        "luma_min": int(bright.min()),
        "luma_max": int(bright.max()),
        "bright_px": int(bright.size),
        "projection": proj,
    }


def segmented_candidates(lum: np.ndarray, x_base: int = 0,
                         y_base: int = 0, thr: int = 90,
                         min_h: int = 6, min_w: int = 6) -> list:
    """行带 × 列段 → 文本候选（窗口尺寸无关）。返回候选列表。"""
    mask = (lum > thr).astype(np.uint8)
    rows = mask.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 2 and run is None:
            run = i
        elif v <= 2 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        if be - bs < min_h:
            continue
        band = mask[bs:be]
        colink = band.sum(axis=0) > 2
        runs, run = [], None
        for i, v in enumerate(colink):
            if v and run is None:
                run = i
            elif not v and run is not None:
                if i - run >= min_w:
                    runs.append((run, i))
                run = None
        if run is not None and len(colink) - run >= min_w:
            runs.append((run, len(colink)))
        for cs, ce in runs:
            crop = lum[bs:be, cs:ce]
            m2 = crop > thr
            ys, xs = np.where(m2)
            if len(ys) < 20:
                continue
            inner = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            c = _metrics(inner, int(xs.min() + cs + x_base),
                         int(ys.min() + bs + y_base))
            if c is not None:
                out.append(c)
    return out


def scan_roi_text(arr: np.ndarray, roi: tuple) -> list:
    """冻结 1419 Bet ROI 内候选（仅 1419×1066 帧使用）。"""
    x0, y0, w, h = roi
    sub = arr[y0:y0 + h, x0:x0 + w]
    return segmented_candidates(luma_of(sub), x_base=x0, y_base=y0)


def find_colored_blocks(arr: np.ndarray, region: tuple,
                        min_area: int = 1500) -> list:
    x0, y0, x1, y1 = region
    sub = arr[y0:y1, x0:x1]
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    colored = sat > 45
    rows = colored.sum(axis=1)
    bands, run = [], None
    for i, v in enumerate(rows):
        if v > 40 and run is None:
            run = i
        elif v <= 40 and run is not None:
            bands.append((run, i))
            run = None
    if run is not None:
        bands.append((run, len(rows)))
    out = []
    for bs, be in bands:
        block = colored[bs:be]
        ys, xs = np.where(block)
        if len(ys) < min_area:
            continue
        r_in = sub[bs:be][ys, xs][:, 0]
        g_in = sub[bs:be][ys, xs][:, 1]
        b_in = sub[bs:be][ys, xs][:, 2]
        out.append({
            "bbox": [int(xs.min() + x0), int(ys.min() + y0 + bs),
                     int(xs.max() - xs.min() + 1),
                     int(ys.max() - ys.min() + 1)],
            "area": int(len(ys)),
            "mean_rgb": [int(r_in.mean()), int(g_in.mean()),
                         int(b_in.mean())],
        })
    return out


def summarize(clusters: dict) -> list:
    out = []
    for key, d in sorted(clusters.items(), key=lambda kv: -kv[1]["n"]):
        ex = d["samples"][0]
        out.append({
            "cluster_key": list(key),
            "instances": d["n"],
            "distinct_frames": len(d["frames"]),
            "sample": ex,
            "n_samples_kept": len(d["samples"]),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir",
                    default="data/vision2/coinpoker_evidence")
    ap.add_argument("--since-filetime", type=int, required=True)
    ap.add_argument("--out", default="data/vision2/p13a_evidence_analysis.json")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    # FileTime（1601 起 100ns）→ Unix 秒
    since_s = args.since_filetime / 1e7 - 11644473600
    fulls = sorted(Path(args.evidence_dir).glob("*_full.png"))
    new = [p for p in fulls if p.stat().st_mtime >= since_s]
    if args.max_frames:
        new = new[:args.max_frames]
    print(f"new_full_frames={len(new)}")

    bet_clusters: dict = {}
    roi_clusters: dict = {}
    bet_roi_presence = {seat: {"frames_seen": 0, "frames_empty": 0}
                        for seat in BET_ROIS_1419}
    action_blocks: dict = {}
    action_text: dict = {}
    sizes = set()
    frames_read = 0
    frame_block_sets: list = []
    total_cands = 0
    for path in new:
        try:
            arr = load_frame(path)
        except Exception as e:                       # noqa: BLE001
            print(f"decode_skip {path.name}: {e}")
            continue
        frames_read += 1
        h, w = arr.shape[:2]
        sizes.add((w, h))
        lum = luma_of(arr)
        # 全窗候选（用 0.85~1.0 高度带外全部，但排除底部按钮条避免污染：
        # 底部条另测）——全窗扫描，含整窗所有亮文本
        cands = segmented_candidates(lum)
        # 去除完全落入底部条(y>=0.89h)的候选（按钮文本单独统计）
        bottom_y0 = int(h * 0.89)
        cands_table = [c for c in cands if c["bbox"][1] < bottom_y0 - 2]
        cands_bottom = [c for c in cands if c["bbox"][1] >= bottom_y0 - 2]
        total_cands += len(cands_table)
        for c in cands_table:
            key = (c["bbox"][0] // 12, c["bbox"][1] // 12, c["bbox"][3] // 6)
            d = bet_clusters.setdefault(
                key, {"n": 0, "frames": set(), "samples": []})
            d["n"] += 1
            d["frames"].add(path.name)
            if len(d["samples"]) < 8:
                d["samples"].append(c)
        for c in cands_bottom:
            key = (c["bbox"][0] // 12, c["bbox"][1] // 12, c["bbox"][3] // 6)
            d = action_text.setdefault(
                key, {"n": 0, "frames": set(), "samples": []})
            d["n"] += 1
            d["frames"].add(path.name)
            if len(d["samples"]) < 8:
                d["samples"].append(c)
        # 冻结 1419 ROI（仅同尺寸帧）
        if (w, h) == (1419, 1066):
            for seat, roi in BET_ROIS_1419.items():
                cc = scan_roi_text(arr, roi)
                st = bet_roi_presence[seat]
                st["frames_seen"] += 1
                if not cc:
                    st["frames_empty"] += 1
                for c in cc:
                    key = (seat, c["bbox"][0] // 8, c["bbox"][1] // 8,
                           c["bbox"][3] // 6)
                    d = roi_clusters.setdefault(
                        key, {"n": 0, "frames": set(), "samples": []})
                    d["n"] += 1
                    d["frames"].add(path.name)
                    if len(d["samples"]) < 8:
                        d["samples"].append(c)
        # 底部动作块
        blocks = find_colored_blocks(arr, (0, bottom_y0, w, h))
        frame_block_sets.append((path.name,
                                 tuple(sorted(b["bbox"][0] // 16
                                              for b in blocks))))
        for blk in blocks:
            key = (blk["bbox"][0] // 16, blk["bbox"][1] // 16)
            d = action_blocks.setdefault(
                key, {"n": 0, "frames": set(), "samples": []})
            d["n"] += 1
            d["frames"].add(path.name)
            if len(d["samples"]) < 5:
                d["samples"].append(blk)

    transitions = 0
    for i in range(1, len(frame_block_sets)):
        if frame_block_sets[i][1] != frame_block_sets[i - 1][1]:
            transitions += 1

    result = {
        "frames": {
            "new_full_frames": len(new),
            "frames_read": frames_read,
            "frame_sizes": sorted(sizes),
            "block_set_transitions": transitions,
            "total_table_candidates": total_cands,
        },
        "bet_roi_presence_1419": bet_roi_presence,
        "bet_roi_clusters_1419": summarize(roi_clusters),
        "table_text_candidates": summarize(bet_clusters),
        "bottom_text_candidates": summarize(action_text),
        "action_ui_blocks": summarize(action_blocks),
    }
    Path(args.out).write_text(json.dumps(result, indent=1,
                                         ensure_ascii=False),
                              encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False)[:9000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
