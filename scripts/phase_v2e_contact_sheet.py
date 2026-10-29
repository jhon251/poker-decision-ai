"""Phase V2E — 人工审阅联系表生成（HTML，仅引用现有 PNG，不修改资产）。

输出 data/vision2/realclient_assets_v2/contact_sheet.html：
按簇列出 cluster_id / role / 候选数 / 建议词表 / 全部候选 PNG 缩略图，
供人工或视觉模型对照审阅并填写 labels.json。
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient import (  # noqa: E402
    RealClientTemplateLibrary, label_vocab_for)


def main() -> int:
    ap = argparse.ArgumentParser(description="V2E contact sheet")
    ap.add_argument("--assets", type=str,
                    default="data/vision2/realclient_assets_v2")
    ap.add_argument("--out", type=str,
                    default="data/vision2/realclient_assets_v2/"
                            "contact_sheet.html")
    args = ap.parse_args()
    src_dir = Path(args.assets).resolve()
    manifest = src_dir / "manifest.json"
    if not manifest.exists():
        print(f"未找到 manifest: {manifest}")
        return 2
    library = RealClientTemplateLibrary.from_json(str(manifest))

    by_cluster = defaultdict(list)
    for a in library.assets.values():
        if a.cluster_id is not None:
            by_cluster[a.cluster_id].append(a)

    rows = []
    for cid in sorted(by_cluster):
        items = by_cluster[cid]
        first = items[0]
        imgs = "".join(
            f'<img src="{html.escape(p)}" alt="{html.escape(p)}" '
            f'style="max-height:56px;margin:2px;border:1px solid #ccc;">'
            for p in sorted({a.png_ref for a in items if a.png_ref}))
        vocab = label_vocab_for(first.role) or ""
        rows.append(
            f'<tr><td>{cid}</td><td>{html.escape(first.role or "")}</td>'
            f'<td>{len(items)}</td><td>{html.escape(vocab)}</td>'
            f'<td>{imgs}</td></tr>')
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>V2E contact sheet</title></head>
<body><h1>Real-client asset contact sheet（人工审阅用）</h1>
<p>共 {len(by_cluster)} 簇。审阅后填写 labels.json（label=null 表示未确认，
绝不猜测）。PNG 为真实截图裁剪（只读引用）。</p>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>cluster</th><th>role</th><th>count</th><th>vocab</th><th>crops</th></tr>
{''.join(rows)}
</table></body></html>"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"联系表已生成: {out}（{len(by_cluster)} 簇）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
