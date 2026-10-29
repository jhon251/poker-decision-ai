# -*- coding: utf-8 -*-
"""R13-EXE-2 — v3b header 字重模板集裁制（coinpoker_v3 → v3b_header）。

取证结论（2026-10-07，与审计对照）：
- coinpoker_v3 全 1,889 帧 header 渲染与 1876 GT 模板源（f01/frame167）
  不同字重与间距（v3 笔画更粗、间距更宽；整词 NCC 0.24）；
- v3 帧 stake 文本 `1K/2K (320)` 位于 ABS x∈[139,216)——现 STAKE_ROI
  x1=200 会截掉 `0)`（旧布局止于 x≈189，不截）；
- header 布局跨语料首/中/尾帧 IoU=1.0000（共享模板前提成立）。

本脚本从 v3 帧裁制整套 header glyph 模板（digit 0/1/2/3/4/7/9 + K +
slash + 括号 + dash），存 coinpoker_assets_v3b_header/；
digit{5,6,8} 语料未出现 → NOT_COVERED（遇即 UNKNOWN，诚实降级不变）。

红线：只裁模板不动识别器逻辑；粗体 v1876 资产零改动。
输出：
  data/vision2/coinpoker_assets_v3b_header/hdr_*.png (RGBA, alpha=ink)
  data/vision2/coinpoker_assets_v3b_header/manifest.json（v1876 格式 +
  font_variant 字段）
  data/vision2/coinpoker_assets_v3b_header/stake_span_mask_v3b.npy
  data/vision2/coinpoker_assets_v3b_header/proof_sheet.png（目视验证图）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_betting_header import (  # noqa: E402
    _text_mask)

OUT_DIR = Path("data/vision2/coinpoker_assets_v3b_header")
SRC_FRAME = Path(
    "data/vision2/coinpoker_v3/POT_CHANGED_114_1791199840586_full.png")
STABILITY_FRAMES = [
    Path("data/vision2/coinpoker_v3/BUTTON_CHANGED_178_1791197763010_full.png"),
    Path("data/vision2/coinpoker_v3/SETTLED_BOARD_CHECK_115_1791201027720_full.png"),
    Path("data/vision2/coinpoker_v3/TABLE_CHANGED_9_1791201710751_full.png"),
]

#: frame 114 实测（列墨量剖面取证，phase_r13 冻结）：
#: NLH(33-59) hand_id 1491749(67-120) dash(127-131) 1K/2K (320)(139-215)
#: 合并 run 的切分点取列墨量谷底（74→105 | K/→153 | (3→188 | 0)→210）
GLYPHS = [
    # (文件名, label, x0, x1, 备注) —— manifest 顺序 = loader last-wins 顺序
    ("hdr_digit_1_hand.png", "digit_1", 67, 70, "hand_id 首位 1"),
    ("hdr_digit_4b.png", "digit_4", 105, 112, "hand_id 74 合并切分右半"),
    ("hdr_digit_7.png", "digit_7", 98, 104, "hand_id 74 合并切分左半"),
    ("hdr_digit_9b.png", "digit_9", 114, 120, "hand_id 末位 9"),
    ("hdr_digit_1.png", "digit_1", 139, 142, "stake 1K 首位（有效模板）"),
    ("hdr_K1.png", "K", 146, 152, "1K 的 K（与 slash 谷底切分）"),
    ("hdr_slash.png", "slash", 153, 159, "SB/BB 分隔斜杠"),
    ("hdr_digit_2.png", "digit_2", 161, 167, "2K 首位"),
    ("hdr_K2.png", "K", 169, 176, "2K 的 K（独立 run，有效模板）"),
    ("hdr_paren_open.png", "paren_open", 183, 187, "ante 左括号（(3 切分）"),
    ("hdr_digit_3.png", "digit_3", 188, 194, "ante 320 首位"),
    ("hdr_digit_2b.png", "digit_2", 196, 202, "ante 320 第 2 位（有效）"),
    ("hdr_digit_0.png", "digit_0", 204, 209, "ante 320 末位"),
    ("hdr_paren_close.png", "paren_close", 210, 215, "ante 右括号"),
    ("hdr_digit_4.png", "digit_4", 73, 80, "hand_id 独立 4（有效模板）"),
    ("hdr_digit_9.png", "digit_9", 82, 88, "hand_id 独立 9（有效模板）"),
]
#: stake 文本 span（`1K/2K (320)`，不含 hand-id/dash）——整词滑动参考
STAKE_SPAN_ABS = (139, 216)


def tight_crop(mask: np.ndarray, x0: int, x1: int) -> np.ndarray | None:
    sub = (mask[:, x0:x1 + 1] > 0)
    ys, xs = np.where(sub)
    if len(xs) < 3:
        return None
    return sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def save_tpl(mask: np.ndarray, name: str) -> tuple[int, int]:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = (mask * 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(OUT_DIR / name)
    return h, w


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(Image.open(SRC_FRAME).convert("RGBA"))
    m = _text_mask(frame, (0, 0, 240, 26))

    # ── 1. 跨帧稳定性断言（同渲染才允许共享模板）─────────────────────
    for sf in STABILITY_FRAMES:
        sm = _text_mask(np.asarray(Image.open(sf).convert("RGBA")),
                        (0, 0, 240, 26))
        iou = float(((m > 0) & (sm > 0)).sum() / max(
            ((m > 0) | (sm > 0)).sum(), 1))
        assert iou > 0.97, f"header 布局不稳定 {sf.name}: IoU={iou:.3f}"
        print(f"stability {sf.name}: IoU={iou:.4f}")

    # ── 2. glyph 裁制（目视验证点：切分块标注 split）─────────────────
    manifest_entries, tiles = [], []
    for fname, label, a, b, note in GLYPHS:
        g = tight_crop(m, a, b)
        assert g is not None, f"{fname} 空"
        h, w = save_tpl(g, fname)
        manifest_entries.append({
            "label": label,
            "crop_ref": fname,
            "bbox": [int(a), 9, int(b - a + 1), int(h)],
            "shape": [int(h), int(w)],
            "source_frame_id": SRC_FRAME.name,
            "gt_text": "1K/2K (320)",
            "note": note + (" [split]" if "切分" in note else ""),
            "kind": "betting_header_v3b",
        })
        tiles.append((fname, g))
        print(f"saved {fname:24s} label={label:11s} shape=({h},{w})")

    # ── 3. stake span 整词参考掩码（`1K/2K (320)`，x∈[139,216)）─────
    span = (m[:, STAKE_SPAN_ABS[0]:STAKE_SPAN_ABS[1]] > 0).astype(np.float64)
    np.save(OUT_DIR / "stake_span_mask_v3b.npy", span)
    print("stake span mask:", span.shape, "ink:", int(span.sum()))

    # ── 4. 目视验证 proof sheet ──────────────────────────────────────
    S, pad = 10, 6
    W = sum((g.shape[1] + 1) * S + pad for _, g in tiles) + pad
    H = max(g.shape[0] for _, g in tiles) * S + 40
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(sheet)
    x = pad
    for fname, g in tiles:
        img = Image.fromarray(((1 - g) * 255).astype(np.uint8))
        img = img.resize((g.shape[1] * S, g.shape[0] * S), Image.NEAREST)
        sheet.paste(img.convert("RGB"), (x, 26))
        d.text((x, 8), fname.replace("hdr_", "").replace(".png", ""), fill=0)
        d.text((x, 18), f"{g.shape[1]}x{g.shape[0]}", fill=120)
        x += (g.shape[1] + 1) * S + pad
    sheet.save(OUT_DIR / "proof_sheet.png")
    print("proof sheet saved")

    # ── 5. manifest（v1876 格式 + font_variant）─────────────────────
    manifest = {
        "client": "coinpoker",
        "version": "v3bheader",
        "font_variant": "v3b",
        "gt_text": "1K/2K (320)",
        "gt_values": {"sb": 1000, "bb": 2000, "ante": 320},
        "source_frame": str(SRC_FRAME).replace("\\", "/"),
        "stake_span_abs": list(STAKE_SPAN_ABS),
        "note": ("R13-EXE-2 header glyph 模板（coinpoker_v3 渲染，跨帧 "
                 "IoU=1.0；逐字形目视验证 proof_sheet.png）。NOT_COVERED: "
                 "digit{5,6,8}、小数、M 后缀（v3 语料未出现）→ 遇即 "
                 "UNKNOWN。v1876 粗体资产零改动，两套并存由识别器按分"
                 "自动选择。"),
        "assets": manifest_entries,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("manifest saved:", OUT_DIR / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
