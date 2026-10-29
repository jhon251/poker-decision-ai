"""Phase V2D — 真实资产角色标注 + 标注工作清单生成。

- 加载 v1 manifest（V2C 冻结基线不动）→ 按 ROI provenance 确定性赋角色
  （role_identified，label_source=roi_provenance）→ 输出 manifest v2；
- 生成 labeling_worklist_v2.json：每簇角色/候选数/来源帧/建议标签词表/
  PNG 引用——供人/视觉模型按簇批量标注（标注一个簇 = 全部同类资产）；
- 明确声明：角色级 ≠ 字符级；字符级标注完成前，识别后端恒 UNAVAILABLE。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientTemplateLibrary, assign_role_labels, role_for)

#: 角色 → 建议标签词表（标注者按此填写 semantic_label）
SUGGESTED_LABELS = {
    "pot_digit_role": "0123456789.,:",
    "bet_digit_role": "0123456789.",
    "stack_digit_role": "0123456789.",
    "table_meta_digit_role": "0123456789.:",
    "hero_card_glyph_role": "AKQJT98765432shdc",
    "board_card_glyph_role": "AKQJT98765432shdc",
    "button_disc_role": "button|not_button",
    "player_area_disc_role": "player_icon|chip_stack|other",
    "action_area_disc_role": "action_button|other",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="V2D role labels + worklist")
    ap.add_argument("--assets", type=str,
                    default="data/vision2/realclient_assets_v1")
    ap.add_argument("--out-dir", type=str,
                    default="data/vision2/realclient_assets_v2")
    args = ap.parse_args()
    src_dir = Path(args.assets).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_v1 = src_dir / "manifest.json"
    if not manifest_v1.exists():
        print(f"未找到 manifest: {manifest_v1}")
        return 2

    library = RealClientTemplateLibrary.from_json(str(manifest_v1))
    n = assign_role_labels(library)
    library.version = 2
    for asset in library.assets.values():
        asset.asset_version = 2
    library.to_json(str(out_dir / "manifest.json"))

    # 工作清单（按簇）
    clusters = defaultdict(list)
    for a in library.assets.values():
        if a.cluster_id is not None:
            clusters[a.cluster_id].append(a)
    worklist = []
    for cid in sorted(clusters):
        items = clusters[cid]
        kinds = {a.kind for a in items}
        roles = sorted({a.role for a in items if a.role})
        frames = sorted({a.source_frame for a in items if a.source_frame})
        pngs = sorted({a.png_ref for a in items if a.png_ref})
        suggest = []
        for r in roles:
            if r in SUGGESTED_LABELS:
                suggest.append(SUGGESTED_LABELS[r])
        worklist.append({
            "cluster_id": cid, "candidate_count": len(items),
            "kinds": sorted(kinds), "roles": roles,
            "source_frames": frames,
            "suggested_label_vocab": suggest,
            "png_refs": pngs[:12],
            "status": "awaiting_visual_label",
        })
    work_path = out_dir / "labeling_worklist_v2.json"
    work_path.write_text(json.dumps({
        "schema_version": 2,
        "note": ("角色级语义由 ROI provenance 推导（role_identified）；"
                 "字符级 semantic_label 需视觉标注者（人或 vision-capable "
                 "模型）完成；标注一个簇 = 标注全部同类资产。"),
        "clusters": worklist,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    role_counts = Counter(a.role for a in library.assets.values() if a.role)
    print(f"=== V2D 角色标注完成 ===")
    print(f"  资产: {len(library.assets)}  已赋角色: {n}")
    for role, cnt in sorted(role_counts.items()):
        print(f"  {role:<26} {cnt}")
    print(f"  字符级已标注: 0（保持 null + UNAVAILABLE，不猜）")
    print(f"  manifest v2: {out_dir / 'manifest.json'}")
    print(f"  工作清单: {work_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
