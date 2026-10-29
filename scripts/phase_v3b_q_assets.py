"""V-2b (Round 12) — v3b 裁剪模板（Q 黑/红 + 10 的 "0" 组件）——审计工件，未注册。

任务书假设「Q 模板从未存在」——实测不成立：v2zv/rank_Qc.png（V2ZT 轮）
一直在 DEFAULT_ASSET_DIRS 里加载。真正根因（本轮实测，
docs/V2B_Q_TEMPLATE_REPORT.md §2）：1876 render class 下 rank 字形浓墨列
（white 0.19-0.25）被 `_ink_width` 白卡约束（white≥0.25）排除出墨宽跨度
→ "10" 双字形测成 21-22px < 32 → 墨宽门把全部双字形 T 模板（v2zo 'T'、
v2zv 'T@black'）排除出单字形模式；Q 在全族复核以 0.900-0.910 胜出
（窄条 J 防护剔除 J 后余量 ≥0.09）→ 27 帧 T→Q。

本脚本裁出的 4 个模板曾按任务书注册进 DEFAULT_ASSET_DIRS 尾部做 A/B：
400 帧 v3 语料实测**有害**——"0" 字形与 Q 在 NCC 空间仅差 0.05-0.10，
T0 模板把真 Q 槽位的 primary/rescue 余量饿到 0.03-0.08（门 0.08/0.09）
→ 43 个真 Q 槽位 rank 掉成 UNCERTAIN（q 模板互为近亲同理）。
注册已撤销，本目录仅作审计保留（裁剪方法/目视验证记录见下与
docs/vision_v2_board/q_template_preview/）。

经模拟验证的正确修复（一行，越出本任务边界，未实施——Round 13 提案）：
`_ink_width` 卡列判定追加析取支 `(white≥0.15 & ink≥0.30)`（浓墨字形列
仍属卡面；桌面列 white≈0 不受影响）。400 帧模拟：T→Q 27/27 全修、
0 回退；三层口径 rank 层 72.15%→84.93%、FULL suit 层 89.05%→99.50%。

裁剪源帧为 v3 语料 400 帧人工标注帧（1876x1400，slot 几何
(590,551,140,180) 与 BoardObserver 同源），人工读牌已目视核验
（docs/vision_v2_board/q_template_preview/crop_source_cards.png）：
- SCENE_CHANGED_146_1791197609014 slot1 = Qs（黑 Q）
- POT_CHANGED_41_1791197068161      slot2 = Qd（红 Q）
- POT_CHANGED_148_1791197620554     slot2 = Th（红 10，取 "0" 组件）
- POT_CHANGED_17_1791201749537      slot0 = Ts（黑 10，取 "0" 组件）
方法：V2ZV 先例 rank_zone_content_bbox / 最大连通组件（禁 ML）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.geometry import connected_components  # noqa: E402
from poker_decision_ai.vision2_platform import (  # noqa: E402
    decode_png, encode_rgba_png)

# V3F-P15 render class 1876 冻结几何（与 SLOT_GEOM_BY_WINDOW 同源）
SLOT_X0, SLOT_Y0, SLOT_W, SLOT_H = 590, 551, 140, 180
CORPUS = Path("data/vision2/coinpoker_v3")
OUT = Path("data/vision2/coinpoker_assets_v3b")

SOURCES = [
    # (输出名, 源帧, slot, 人工牌面, 模式)
    ("board_Q_black", "SCENE_CHANGED_146_1791197609014_full.png", 1, "Qs", "glyph"),
    ("board_Q_red", "POT_CHANGED_41_1791197068161_full.png", 2, "Qd", "glyph"),
    ("board_T0_red", "POT_CHANGED_148_1791197620554_full.png", 2, "Th", "zero"),
    ("board_T0_black", "POT_CHANGED_17_1791201749537_full.png", 0, "Ts", "zero"),
]
# label / color_variant → 模板 key（尾目录覆盖语义下必须全部为新 key）
LABELS = {
    "board_Q_black": ("Q", "black"),
    "board_Q_red": ("Q", "red"),
    "board_T0_red": ("T", "red"),
    "board_T0_black": ("T", "black0"),
}


def ink_mask(arr):
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    dark = luma < 115
    red = (r > 100) & (r > g + 40) & (r > b + 40)
    return (dark | red).astype("uint8")


def rank_glyph(slot):
    """V2ZV 先例：rank 区内容 bbox（含 1px 白边）。"""
    zone = slot[6:6 + 57, 6:6 + 49]
    m = ink_mask(zone)[3:-3, 3:-3]
    ys, xs = np.where(m)
    if len(xs) < 40:
        return None, None
    x0, x1, y0, y1 = xs.min() + 3, xs.max() + 3, ys.min() + 3, ys.max() + 3
    crop = zone[y0 - 1:y1 + 3, x0 - 1:x1 + 3]
    return crop, (int(x0) - 1, int(y0) - 1, int(crop.shape[1]),
                  int(crop.shape[0]))


def zero_glyph(slot):
    """10 的 "0" 字形组件：rank 区最大连通组件（"0" 椭圆面积 > "1" 竖条）。"""
    zone = slot[6:6 + 57, 6:6 + 49]
    m = ink_mask(zone)
    comps = connected_components(m.tolist(), min_area=60)
    if not comps:
        return None, None
    big = comps[0]                      # 面积最大 = "0"（"1" 为窄竖条）
    b = big.bbox
    bx, by = int(b.x), int(b.y)
    bw, bh = int(b.width), int(b.height)
    crop = zone[max(by - 1, 0):by + bh + 3, max(bx - 1, 0):bx + bw + 3]
    return crop, (max(bx - 1, 0), max(by - 1, 0), int(crop.shape[1]),
                  int(crop.shape[0]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "client": "coinpoker", "version": "v3b",
        "coordinate_system": "full_png_1876x1400",
        "coordinate_fix": "V3F-P15 render class 1876 冻结槽位几何 "
                          "(590,551,140,180)，与 BoardObserver 同源",
        "ground_truth_status": "visually_labeled",
        "registration": "NOT_REGISTERED——A/B 实测有害（真 Q 槽位余量饥饿，"
                        "43 槽回退），DEFAULT_ASSET_DIRS 未追加本目录；"
                        "本目录仅作审计保留。正确修复为 _ink_width 一行"
                        "判据扩展（Round 13 提案，见报告）。",
        "note": "V-2b (Round 12)：1876 尺度 Q（黑/红）与 10 的 '0' 字形组件"
                "（T@red / T@black0）。任务书假设「Q 模板从未存在」不成立"
                "（v2zv Q 自 V2ZT 轮就在加载）；根因=1876 类墨宽门退化使双"
                "字形 T 在单字形模式缺席。全部为新 key（'Q@black'/'Q@red'/"
                "'T@red'/'T@black0'），若注册也不覆盖 v2zo 'T'/v2zv "
                "'T@black'/v2zv 'Q'。裁剪源为 v3 语料人工标注分歧帧（人工"
                "读牌目视核验 docs/vision_v2_board/q_template_preview/），"
                "方法=rank_zone_content_bbox / 最大连通组件（V2ZV 先例，"
                "禁 ML）。",
        "quality_gate": {"method": "rank_zone_content_bbox+tight_glyph"},
        "assets": []}
    for name, fname, slot_idx, truth, mode in SOURCES:
        w, h, rgba = decode_png((CORPUS / fname).read_bytes())
        assert (w, h) == (1876, 1400), (fname, w, h)
        full = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        x0 = SLOT_X0 + slot_idx * SLOT_W
        slot = full[SLOT_Y0:SLOT_Y0 + SLOT_H, x0:x0 + SLOT_W]
        crop, b = (rank_glyph(slot) if mode == "glyph" else zero_glyph(slot))
        if crop is None or crop.size < 150:
            print(f"FAIL extract {name}")
            return 1
        r = crop[..., 0].astype(int)
        g = crop[..., 1].astype(int)
        b_ = crop[..., 2].astype(int)
        luma = r * 0.299 + g * 0.587 + b_ * 0.114
        ink = float(((luma < 115) | ((r > 100) & (r > g + 40)
                                     & (r > b_ + 40))).mean())
        red_r = float(((r > 100) & (r > g + 40) & (r > b_ + 40)).mean())
        if ink < 0.10:
            print(f"FAIL ink low {name}")
            return 1
        label, variant = LABELS[name]
        p = OUT / f"{name}.png"
        p.write_bytes(encode_rgba_png(crop.shape[1], crop.shape[0],
                                      crop.tobytes()))
        manifest["assets"].append({
            "label": label, "color_variant": variant,
            "kind": "card_glyph",
            "bbox": [x0 + b[0], SLOT_Y0 + b[1], b[2], b[3]],
            "crop_ref": str(p), "coordinate_system": "full_png",
            "coordinate_fix": "V3F-P15 render class 1876",
            "source_frame_id": fname,
            "human_truth": f"slot{slot_idx}={truth}",
            "ink_ratio": round(ink, 3),
            "color_family": "red" if red_r > 0.10 else "black",
            "ground_truth_status": "visually_labeled",
            "confidence": 0.95,
            "provenance": {
                "annotation": "V-3a 人工标注（项目所有者盲标）+ 裁剪前目视"
                              "复核源卡面",
                "label_source": "agent_vision_annotation",
                "method": "rank_zone_content_bbox" if mode == "glyph"
                          else "rank_zone_largest_component (10 的 0 字形)",
                "template_key": f"{label}@{variant}",
                "purpose": "1876 类单字形模式的 Q/T 证据（T→Q 根因修复）"}})
        print(f"{name}: {crop.shape[1]}x{crop.shape[0]} ink={ink:.3f} "
              f"fam={'red' if red_r > 0.10 else 'black'} key={label}@{variant}"
              f" src={fname} slot{slot_idx}={truth}")
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"manifest -> {OUT / 'manifest.json'} ({len(manifest['assets'])} assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
