"""Phase V2E — 字符级标注应用入口（文件式，可审计）。

用法：
  1) 查看/编辑标注文件（首次运行自动生成全 null 模板，绝不猜测）：
       data/vision2/realclient_assets_v2/labels.json
    格式（两种均可）：
       {"7": "A"}                                  # cluster_id -> label
       {"7": {"label": "A", "status": "visually_labeled",
              "label_source": "human_review", "confidence": 0.95}}
  2) 运行本脚本 → 应用合法标注 → 输出：
       data/vision2/realclient_assets_v3/manifest.json   （library v3）
       data/vision2/realclient_assets_v3/labeling_audit_v3.json
  3) 标注后的 library 自动流入 labeled_templates → Card/Digit 后端。

校验：标签必须 ∈ 簇角色词表；已标注簇拒绝静默覆盖；无角色簇拒绝。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientTemplateLibrary, apply_cluster_labels, label_vocab_for)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="V2E apply cluster labels")
    ap.add_argument("--assets", type=str,
                    default="data/vision2/realclient_assets_v2")
    ap.add_argument("--out-dir", type=str,
                    default="data/vision2/realclient_assets_v3")
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument("--status", type=str, default="visually_labeled")
    ap.add_argument("--label-source", type=str, default="human_review")
    args = ap.parse_args()

    src_dir = Path(args.assets).resolve()
    out_dir = Path(args.out_dir).resolve()
    labels_path = Path(args.labels).resolve() if args.labels \
        else src_dir / "labels.json"

    manifest_v2 = src_dir / "manifest.json"
    if not manifest_v2.exists():
        print(f"未找到 manifest: {manifest_v2}")
        return 2
    library = RealClientTemplateLibrary.from_json(str(manifest_v2))

    if not labels_path.exists():
        # 生成全 null 模板（绝不猜测）+ 指引
        template = {}
        clusters = {}
        for a in library.assets.values():
            if a.cluster_id is not None:
                clusters.setdefault(a.cluster_id, a)
        for cid in sorted(clusters):
            a = clusters[cid]
            template[str(cid)] = {
                "label": None,
                "status": "unlabeled_candidate",
                "label_source": None,
                "confidence": None,
                "role": a.role,
                "suggested_vocab": label_vocab_for(a.role),
            }
        _write_json(labels_path, template)
        print(f"已生成标注模板（全部 label=null，绝不猜测）: {labels_path}")
        print("请人工/视觉模型查看 contact_sheet.html 后填写 labels.json")
        return 0

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    cluster_labels = {}
    meta = {}
    for key, val in raw.items():
        if not isinstance(val, dict):
            cluster_labels[int(key)] = val
            meta[int(key)] = {}
        else:
            cluster_labels[int(key)] = val.get("label")
            meta[int(key)] = val
    report = apply_cluster_labels(
        library, cluster_labels,
        label_source=args.label_source,
        status=args.status,
        confidence=None, force=False)

    library.version = 3
    for a in library.assets.values():
        a.asset_version = 3
    library.to_json(str(out_dir / "manifest.json"))

    # 审计 manifest（每簇全字段；可审计）
    audit = []
    by_cluster = {}
    for a in library.assets.values():
        if a.cluster_id is not None:
            by_cluster.setdefault(a.cluster_id, []).append(a)
    for cid in sorted(by_cluster):
        items = by_cluster[cid]
        first = items[0]
        audit.append({
            "cluster_id": cid,
            "role": first.role,
            "candidate_count": len(items),
            "source_frames": sorted({a.source_frame for a in items
                                     if a.source_frame}),
            "png_refs": sorted({a.png_ref for a in items if a.png_ref}),
            "semantic_label": first.semantic_label,   # 无证据 → null
            "ground_truth_status": first.ground_truth_status,
            "label_source": first.label_source,
            "confidence": first.confidence,
            "asset_version": first.asset_version,
        })
    audit_path = out_dir / "labeling_audit_v3.json"
    _write_json(audit_path, {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_file": str(labels_path),
        "applied": report.to_dict(),
        "clusters": audit,
    })

    print(f"=== V2E 标注应用完成 ===")
    print(f"  applied_clusters: {len(report.applied_clusters)} "
          f"applied_assets: {report.applied_assets}")
    for r in report.rejected:
        print(f"  rejected: cluster={r['cluster_id']} label={r['label']} "
              f"reason={r['reason']}")
    print(f"  字符级已标注资产: "
          f"{sum(1 for a in library.assets.values() if a.semantic_label)}")
    print(f"  manifest v3: {out_dir / 'manifest.json'}")
    print(f"  audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
