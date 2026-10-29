"""V2C — 候选资产无监督聚类（同形资产聚簇；无语义）。

用途：把 10 张截图中外观相同的字形/组件聚成簇——未来标注一个簇即可
标注全部同类资产（大幅降低标注成本）。聚类不赋予任何语义标签；
cluster_id 只是"外观相同"的确定性分组。
"""
from __future__ import annotations

from ...realtime.image import ImageData, ncc_template_match


def _normalize(img: ImageData, size: int = 32) -> ImageData:
    """最近邻归一化到 size×size（确定性）。"""
    out = bytearray(size * size * 4)
    for y in range(size):
        sy = min(img.height - 1, int(y * img.height / size))
        for x in range(size):
            sx = min(img.width - 1, int(x * img.width / size))
            i = (sy * img.width + sx) * 4
            o = (y * size + x) * 4
            out[o:o + 4] = img.data[i:i + 4]
    return ImageData.from_rgba_bytes(size, size, bytes(out))


def assign_clusters(assets: list, images: dict, *, threshold: float = 0.75
                    ) -> dict:
    """贪心聚簇（同 kind 内两两 NCC ≥ 阈值合并）。返回 {template_id: cluster_id}。

    不修改语义标签；cluster_id 仅为外观分组。
    """
    assignment: dict = {}
    next_cluster = 0
    by_kind: dict = {}
    for a in assets:
        by_kind.setdefault(a.kind, []).append(a)
    for kind in sorted(by_kind):
        group = by_kind[kind]
        norms = {}
        for a in group:
            img = images.get(a.template_id)
            if img is not None:
                norms[a.template_id] = _normalize(img)
        for a in group:
            if a.template_id in assignment:
                continue
            tid = a.template_id
            if tid not in norms:
                continue
            cluster = next_cluster
            next_cluster += 1
            assignment[tid] = cluster
            for b in group:
                bid = b.template_id
                if bid in assignment or bid not in norms:
                    continue
                found = ncc_template_match(norms[bid], norms[tid],
                                           score_threshold=threshold)
                if found is not None and found[2] >= threshold:
                    assignment[bid] = cluster
    return assignment
