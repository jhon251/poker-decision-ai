"""Phase V2E — Pot/Hero 增量重提取（真实截图像素证据驱动）。

- pot：金色掩码 + 间隙游程 → 每帧字符候选（potchar_*），替换旧的多字符长串候选；
- hero：实测修正 ROI（realclient.layout.real_layout_specs）→ 卡槽 → 卡上
  字形候选（herochar_*）；
- 合并 v3 资产（board 已有真实标签**原样保留**）→ 新候选聚类（簇 id 偏移
  不与既有簇冲突）→ 输出 manifest v4 + worklist 增量；
- 全流程零猜测：新候选 ground_truth_status=unlabeled_candidate，
  semantic_label=null，待视觉确认后标注。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientTemplate, RealClientTemplateLibrary, assign_clusters,
    assign_role_labels)
from poker_decision_ai.vision2.pixel.realclient.extract import (  # noqa: E402
    _canonicalize, _card_candidates)
from poker_decision_ai.vision2.pixel.realclient.layout import (  # noqa: E402
    real_layout_specs)
from poker_decision_ai.vision2.pixel.realclient.pot_seg import (  # noqa: E402
    segments_gold_chars)
from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile  # noqa: E402
from poker_decision_ai.vision2.realtime.coords import Rect  # noqa: E402
from poker_decision_ai.vision2.realtime.roi import ROIManager  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2_platform import (  # noqa: E402
    decode_png, encode_rgba_png)

CLIENT = "unknown_client_1848x1388"
OUT_ASSET_DIR = "data/vision2/realclient_assets_v4"


def main() -> int:
    ap = argparse.ArgumentParser(description="V2E pot/hero re-collect")
    ap.add_argument("--dir", type=str, default="picture data")
    ap.add_argument("--in-manifest", dest="in_manifest", type=str,
                    default="data/vision2/realclient_assets_v3")
    ap.add_argument("--out", type=str, default="data/vision2/realclient_assets_v4")
    args = ap.parse_args()
    src_dir = Path(args.in_manifest).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = Path(args.dir).resolve()
    files = sorted(png_dir.glob("*.png"))

    # v3（board 标签保留）
    m3 = json.loads((src_dir / "manifest.json").read_text(encoding="utf-8"))
    lib = RealClientTemplateLibrary.from_json(str(src_dir / "manifest.json"))
    max_cluster = max((a.get("cluster_id") or -1) for a in m3.get("assets", []))
    next_cluster = max_cluster + 1
    new_assets = []

    for i, p in enumerate(files):
        w, h, rgba = decode_png(p.read_bytes())
        img = ImageData.from_rgba_bytes(w, h, rgba)
        geo = __import__(
            "poker_decision_ai.vision2.realtime.coords",
            fromlist=["GeometryContext"]).GeometryContext(
            screen_size=(float(w), float(h)), window_rect=None,
            client_offset=(0.0, 0.0),
            client_size_physical=(float(w), float(h)), dpi_scale=1.0)
        prof = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
        mgr = ROIManager(prof, real_layout_specs())
        rois, _ = mgr.resolve()
        crops, _ = mgr.extract_all(img, rois)

        # --- pot 金色字符（带 30% 上下文 padding，防止紧裁极性误判） ---
        pot = crops.get("pot")
        if pot is not None:
            rects, _ = segments_gold_chars(pot)
            for j, r in enumerate(rects):
                px = max(1.0, r.width * 0.3)
                py = max(1.0, r.height * 0.3)
                padded = Rect(max(0.0, r.x - px), max(0.0, r.y - py),
                              min(pot.width, r.x + r.width + px) -
                              max(0.0, r.x - px),
                              min(pot.height, r.y + r.height + py) -
                              max(0.0, r.y - py))
                sub = pot.crop(padded)
                if sub is None:
                    continue
                sub = _canonicalize(sub)
                tid = f"potchar_f{i+1}_{j:02d}"
                a = RealClientTemplate(
                    template_id=tid, kind="digit_glyph",
                    source_frame=p.name, source_roi="pot", client=CLIENT,
                    crop_bbox=r.to_dict(), ground_truth_status="unlabeled_candidate",
                    cluster_id=next_cluster)
                png_path = out_dir / f"{tid}.png"
                png_path.write_bytes(encode_rgba_png(
                    sub.width, sub.height, sub.rgba_bytes()))
                a.png_ref = f"data/vision2/realclient_assets_v4/{tid}.png"
                lib.add(a, sub)
                new_assets.append(a)
                next_cluster += 1

        # --- hero 修正 ROI 白卡专用字形提取 ---
        hero = crops.get("hero_cards")
        if hero is not None:
            from poker_decision_ai.vision2.pixel.realclient.extract import (
                extract_hero_card_glyphs)
            for a in extract_hero_card_glyphs(i + 1, hero, CLIENT,
                                              source_frame=p.name):
                tid = a.template_id
                roi_c = hero.crop(Rect(a.crop_bbox["x"], a.crop_bbox["y"],
                                       a.crop_bbox["width"],
                                       a.crop_bbox["height"]))
                if roi_c is None or roi_c.width < 3 or roi_c.height < 3:
                    continue
                roi_c = _canonicalize(roi_c)
                png_path = out_dir / f"{tid}.png"
                png_path.write_bytes(encode_rgba_png(
                    roi_c.width, roi_c.height, roi_c.rgba_bytes()))
                a.png_ref = f"data/vision2/realclient_assets_v4/{tid}.png"
                lib.add(a, roi_c)
                new_assets.append(a)
                next_cluster += 1

    # 新候选聚簇（同 kind 内）；偏移避免与既有簇 id 冲突
    assignment = assign_clusters(new_assets, lib.images)
    offset = max((a.get("cluster_id") or -1) for a in m3.get("assets", [])) + 1
    for a in new_assets:
        if a.template_id in assignment:
            a.cluster_id = assignment[a.template_id] + offset
    # 赋角色
    assign_role_labels(lib)
    lib.version = 4
    for a in lib.assets.values():
        a.asset_version = 4
    lib.to_json(str(out_dir / "manifest.json"))

    kinds = Counter(a.kind for a in lib.assets.values())
    print(f"=== V2E pot/hero 增量重提取 ===")
    print(f"  资产总数: {len(lib.assets)}  (v3: {len(m3['assets'])}, 新增: {len(new_assets)})")
    print(f"  新增 pot 字符: {sum(1 for a in new_assets if a.template_id.startswith('potchar'))}")
    print(f"  新增 hero 字形: {sum(1 for a in new_assets if a.template_id.startswith('herochar'))}")
    print(f"  board 标签保留: {sum(1 for a in lib.assets.values() if a.semantic_label)}")
    print(f"  manifest v4: {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
