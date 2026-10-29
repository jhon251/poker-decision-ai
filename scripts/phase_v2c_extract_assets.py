"""Phase V2C — 真实客户端资产提取（10 张截图，只读，不修改截图）。

- 每帧：解码 → bootstrap ROI → 候选提取（digit/card_slot/card_glyph/
  disc/region）→ 无监督聚类（同形分簇）；
- 产出：data/vision2/realclient_assets_v1/ 下每候选 PNG 裁剪 +
  manifest JSON；全部 ground_truth_status="unlabeled_candidate"、
  semantic_label=null——**绝不把候选标记为已识别**；
- 标注入口：后续由具备视觉能力的标注者（人或视觉模型）按 cluster_id
  批量填写 semantic_label 后，识别后端才会启用（见报告）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientTemplateLibrary, assign_clusters, extract_candidates)
from poker_decision_ai.vision2.pixel.realclient.extract import (  # noqa: E402
    _canonicalize)
from poker_decision_ai.vision2.realtime.anchors import bootstrap_profile  # noqa: E402
from poker_decision_ai.vision2.realtime.capture import CapturedFrame  # noqa: E402
from poker_decision_ai.vision2.realtime.coords import (  # noqa: E402
    GeometryContext, Rect)
from poker_decision_ai.vision2.realtime.frame_manager import Frame  # noqa: E402
from poker_decision_ai.vision2.realtime.image import ImageData  # noqa: E402
from poker_decision_ai.vision2.realtime.roi import (  # noqa: E402
    ROIManager, default_roi_specs)
from poker_decision_ai.vision2_platform import (  # noqa: E402
    decode_png, encode_rgba_png)

CLIENT = "unknown_client_1848x1388"


def main() -> int:
    ap = argparse.ArgumentParser(description="V2C real-client asset extraction")
    ap.add_argument("--dir", type=str, default="picture data")
    ap.add_argument("--out", type=str,
                    default="data/vision2/realclient_assets_v1")
    args = ap.parse_args()
    png_dir = Path(args.dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(png_dir.glob("*.png"))
    if not files:
        print(f"未找到 PNG: {png_dir}")
        return 2

    library = RealClientTemplateLibrary(client=CLIENT, version=1)
    all_assets = []
    for i, p in enumerate(files):
        w, h, rgba = decode_png(p.read_bytes())
        geo = GeometryContext(screen_size=(float(w), float(h)),
                              window_rect=None, client_offset=(0.0, 0.0),
                              client_size_physical=(float(w), float(h)),
                              dpi_scale=1.0)
        cap = CapturedFrame(image=ImageData.from_rgba_bytes(w, h, rgba),
                            captured_at=float(i), geometry=geo,
                            source=f"real_screenshot:{p.name}",
                            synthetic_test_only=False)
        frame = Frame.from_capture(cap, i + 1)
        prof = bootstrap_profile("real", geo, {"table": (0.0, 0.0, 1.0, 1.0)})
        mgr = ROIManager(prof, default_roi_specs())
        rois, missing = mgr.resolve()
        crops, cm = mgr.extract_all(frame.image, rois)
        missing.update(cm)
        assets = extract_candidates(frame.frame_id, crops, CLIENT,
                                    source_frame=p.name)
        for a in assets:
            roi_crop = crops.get(a.source_roi)
            if roi_crop is None or a.crop_bbox is None:
                continue
            b = a.crop_bbox
            sub = roi_crop.crop(Rect(float(b["x"]), float(b["y"]),
                                     float(b["width"]), float(b["height"])))
            if sub is None or sub.width < 2 or sub.height < 2:
                continue
            # 字形类资产极规范化（深字浅底 → 反色）；结构资产保持原样
            if a.kind in ("digit_glyph", "card_glyph"):
                sub = _canonicalize(sub)
            png_name = f"{a.template_id}.png"
            png_path = out_dir / png_name
            png_path.write_bytes(encode_rgba_png(sub.width, sub.height,
                                                 sub.rgba_bytes()))
            a.png_ref = f"data/vision2/realclient_assets_v1/{png_name}"
            library.add(a, sub)
            all_assets.append(a)

    # 无监督聚类（同形分簇；无语义）
    assignment = assign_clusters(all_assets, library.images)
    for a in all_assets:
        if a.template_id in assignment:
            a.cluster_id = assignment[a.template_id]

    manifest = out_dir / "manifest.json"
    library.to_json(str(manifest))

    from collections import Counter
    kind_counts = Counter(a.kind for a in all_assets)
    print(f"=== V2C 真实客户端资产提取（{len(files)} 张截图） ===")
    print(f"  候选总数: {len(all_assets)}  聚类数: {len(set(assignment.values()))}")
    for kind, n in sorted(kind_counts.items()):
        print(f"  {kind:<18} {n}")
    print(f"  标注状态: 全部 unlabeled_candidate / semantic_label=null")
    print(f"  manifest: {manifest}")
    print("  标注入口: 由具备视觉能力的标注者对 cluster_id 批量标注后，")
    print("  识别后端（RealClient*Backend）才会启用；当前后端恒 UNAVAILABLE。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
