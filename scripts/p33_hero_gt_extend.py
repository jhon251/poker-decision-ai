"""V3D — GT 扩充提取（rank-only）+ 逐卡 round-trip + 零通过清理。

输入：logs/hero_gt_ext_draft.json（分诊草案，rank 为新增 label，suit 为
已有 label 且像素验证通过）。
机制：完全复刻 phase_v2s_p29b_rebuild.py（绝对窗 + 8 邻接连通域 + 尺寸
过滤 12≤w≤50 / 30≤h≤56 / MIN_N=120 + tight mask）。不改 detector/阈值。
每 label 最多 3 实例；round-trip 逐卡要求识别器输出与人工标注完全一致；
某 label 若零帧通过 → 删除其资产与 manifest 条目（不留死模板）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876herocards")
DRAFT = Path("logs/hero_gt_ext_draft.json")

# 与 phase_v2s_p29b_rebuild.py 完全一致的提取窗与尺寸过滤
WINDOWS = {1: {"rank": (822, 1022, 852, 1070), "suit": (822, 1076, 864, 1124)},
           2: {"rank": (938, 1014, 970, 1064), "suit": (932, 1070, 978, 1116)}}
MIN_N, MAX_W, MIN_W, MIN_H, MAX_H = 120, 50, 12, 30, 56
CAP_PER_LABEL = 3


def blocks(arr, x0, y0, x1, y1):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = (l < 120).astype(bool)
    h, w = ink.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not ink[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and ink[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(cells) < MIN_N:
                continue
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            out.append((x0 + min(xs), y0 + min(ys),
                        max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
    return out


def glyph_png(mask):
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = (mask * 255).astype(np.uint8)
    return encode_rgba_png(w, h, rgba.tobytes())


def main() -> int:
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))["draft"]
    if not draft:
        print("draft 为空——零改动退出")
        return 1

    # 每 (rank, card) 选 ≤CAP 个实例：优先 suit 匹配分高（渲染更干净）
    by_label = {}
    for row in draft:
        by_label.setdefault((row["rank"], row["card"]), []).append(row)
    chosen = []
    for (rk, ci), rows in sorted(by_label.items()):
        rows2 = sorted(rows, key=lambda r: -r["suit_best"][1])[:CAP_PER_LABEL]
        chosen.extend(rows2)
    print(f"chosen rows: {len(chosen)} "
          f"({sorted({(r['rank'], r['card']) for r in chosen})})", flush=True)

    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    # 幂等：清除上次 V3D 追加的条目与其资产，再重新提取
    prev = [a for a in manifest["assets"]
            if a.get("provenance", "").startswith("V3D GT EXT (user_annotated")]
    if prev:
        for a in prev:
            p = OUT / a["crop_ref"]
            if p.exists():
                p.unlink()
        manifest["assets"] = [a for a in manifest["assets"]
                              if a not in prev]
        print(f"purged {len(prev)} previous V3D entries", flush=True)
    n_before = len(manifest["assets"])
    added, seen_files, results = [], set(), []
    for row in chosen:
        fp, ci, rk, sk = row["frame"], row["card"], row["rank"], row["suit"]
        w, h, rgba = decode_png((EV / fp).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        gx = None
        for (rx, ry, rw, rh) in blocks(arr, *WINDOWS[ci]["rank"]):
            roi = arr[ry:ry + rh, rx:rx + rw]
            l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 \
                + roi[..., 2] * 0.114
            gx = (l < 120)
            bbox = [rx, ry, rw, rh]
            break
        if gx is None:
            results.append({**row, "extract": "NO_COMPONENT"})
            continue
        fn = f"hcr_{rk}_c{ci}_{bbox[0]}_{bbox[1]}.png"
        if fn not in seen_files:      # 同手牌 burst 帧 glyph 相同 → 去重
            (OUT / fn).write_bytes(glyph_png(gx))
            seen_files.add(fn)
            added.append({"label": f"rank_{rk}",
                          "kind": "herocards_rank_1876",
                          "crop_ref": fn, "bbox": bbox, "card": ci,
                          "rank": rk,
                          "glyph_size": [bbox[2], bbox[3]],
                          "source_frame_id": fp,
                          "provenance": "V3D GT EXT (user_annotated "
                                        "2026-09-04, calibration-confirmed; "
                                        "pixel-vetted)",
                          "gt_status": "normal"})
            print(f"extracted {fn} {bbox}", flush=True)
        else:
            print(f"dedup {fn} ({fp[:24]})", flush=True)

    # 写 manifest（追加；note 幂等）
    base_note = manifest.get("note", "")
    cut = base_note.find(" | V3D 扩充")
    if cut >= 0:
        base_note = base_note[:cut]
    manifest["note"] = (base_note
                        + " | V3D 扩充：rank {"
                        + ",".join(sorted({a['rank'] for a in added}))
                        + "} 来自 user_annotated 校准确认标注。")
    manifest["assets"].extend(added)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # round-trip：逐卡要求识别器输出与人工标注完全一致
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 \
        import HeroCardsRecognizer1876
    rec = HeroCardsRecognizer1876()
    passed_labels = set()
    rr = []
    for row in chosen:
        fp, ci, rk, sk = row["frame"], row["card"], row["rank"], row["suit"]
        w, h, rgba = decode_png((EV / fp).read_bytes())
        r = rec.recognize(np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4))
        card = (r.get("cards") or [{}])[ci - 1] if r.get("cards") else {}
        got = (card.get("status"), card.get("rank"), card.get("suit"))
        ok = got == ("OK", rk, sk)
        rr.append({**row, "roundtrip": "PASS" if ok else "FAIL",
                   "got": [str(x) for x in got], "frame_reason": r.get("reason")})
        if ok:
            passed_labels.add(f"rank_{rk}")
        print(f"round-trip {fp[:34]} c{ci} {rk}{sk} -> {got} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    # 清理零通过 label（删资产 + 删 manifest 条目）
    dead = sorted({a["label"] for a in added} - passed_labels)
    if dead:
        keep = [a for a in manifest["assets"] if a["label"] not in dead]
        dropped_files = {a["crop_ref"] for a in manifest["assets"]
                         if a["label"] in dead}
        for fn in dropped_files:
            p = OUT / fn
            if p.exists():
                p.unlink()
        manifest["assets"] = keep
        (OUT / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"removed zero-pass labels: {dead}", flush=True)

    (Path("logs") / "hero_gt_extend_result.json").write_text(
        json.dumps({"added": len(added), "labels": sorted(passed_labels),
                    "removed": dead, "roundtrip": rr}, ensure_ascii=False,
                   indent=1), encoding="utf-8")
    print(f"assets {n_before} -> {len(manifest['assets'])}; "
          f"labels passing round-trip: {sorted(passed_labels)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
