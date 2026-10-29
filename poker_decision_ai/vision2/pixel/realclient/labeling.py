"""V2E — 字符级标注应用 API（最小标注工具链核心）。

- 输入：cluster_id → semantic_label 的标注字典（由人工或未来视觉模型
  提供）；**本模块绝不猜测标签**；
- 校验：标签必须属于该簇角色词表；整簇一致标注；已标注簇拒绝静默覆盖
  （force=True 显式覆盖，审计可追溯）；
- 输出：LabelingReport（应用/拒绝明细）+ 更新后的 library（所有候选
  同步 semantic_label/label_source/status/confidence/asset_version）；
- 可审计：每个标签可追溯到 cluster + label_source；semantic_label 在无
  真实证据时保持 null。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .assets import RealClientTemplateLibrary

#: 角色 → 允许的字符级标签词表（标注入口唯一校验源）
ROLE_VOCAB = {
    "pot_digit_role": "0123456789.,:",
    "bet_digit_role": "0123456789.,",
    "stack_digit_role": "0123456789.,",
    "table_meta_digit_role": "0123456789.:",
    "hero_card_glyph_role": "AKQJT98765432shdc",
    "board_card_glyph_role": "AKQJT98765432shdc",
    "button_disc_role": "button,not_button",
    "player_area_disc_role": "player_icon,chip_stack,other",
    "action_area_disc_role": "action_button,other",
}


def _cluster_majority_role(library: RealClientTemplateLibrary,
                           cluster_id: int) -> str | None:
    roles = [a.role for a in library.assets.values()
             if a.cluster_id == cluster_id and a.role]
    if not roles:
        return None
    return max(set(roles), key=roles.count)


def label_vocab_for(role: str | None) -> str | None:
    return ROLE_VOCAB.get(role or "")


@dataclass
class LabelingReport:
    applied_clusters: list = field(default_factory=list)
    applied_assets: int = 0
    rejected: list = field(default_factory=list)   # [{cluster_id, label, reason}]

    def to_dict(self) -> dict:
        return {"applied_clusters": self.applied_clusters,
                "applied_assets": self.applied_assets,
                "rejected": self.rejected}


def apply_cluster_labels(library: RealClientTemplateLibrary,
                         cluster_labels: dict, *,
                         label_source: str,
                         status: str = "visually_labeled",
                         confidence: float | None = None,
                         force: bool = False) -> LabelingReport:
    """簇级标注应用。cluster_labels: {cluster_id: "label"}。

    校验规则：
      - 簇必须存在且有已知角色（无角色簇 → 拒绝，需人工指定角色）；
      - 标签必须 ∈ 角色词表（未知标签 → 拒绝，绝不猜测）；
      - 簇内已有语义标签且 force=False → 拒绝（防止静默覆盖）。
    """
    report = LabelingReport()
    for cluster_id in sorted(cluster_labels):
        label = cluster_labels[cluster_id]
        if not isinstance(label, str) or label == "":
            report.rejected.append(
                {"cluster_id": cluster_id, "label": label,
                 "reason": "empty_or_non_string"})
            continue
        role = _cluster_majority_role(library, cluster_id)
        if role is None:
            report.rejected.append(
                {"cluster_id": cluster_id, "label": label,
                 "reason": "cluster_role_unknown"})
            continue
        vocab = label_vocab_for(role)
        if vocab is None:
            report.rejected.append(
                {"cluster_id": cluster_id, "label": label,
                 "reason": f"role_not_labelable({role})"})
            continue
        if label not in vocab:
            report.rejected.append(
                {"cluster_id": cluster_id, "label": label,
                 "reason": f"label_not_in_vocab({vocab})"})
            continue
        members = [a for a in library.assets.values()
                   if a.cluster_id == cluster_id]
        if not members:
            report.rejected.append(
                {"cluster_id": cluster_id, "label": label,
                 "reason": "cluster_empty"})
            continue
        already = any(a.semantic_label is not None for a in members)
        if already and not force:
            report.rejected.append(
                {"cluster_id": cluster_id, "label": label,
                 "reason": "already_labeled_without_force"})
            continue
        for a in members:
            a.semantic_label = label
            a.ground_truth_status = status
            a.label_source = label_source
            a.confidence = confidence
        report.applied_clusters.append(cluster_id)
        report.applied_assets += len(members)
    return report
