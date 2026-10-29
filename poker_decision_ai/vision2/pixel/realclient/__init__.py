"""V2C — Real Client Visual Domain Adaptation（真实客户端视觉域适配）。

职责：
  - 真实客户端视觉资产体系（完整 provenance；未标注候选绝不标记为已识别）；
  - 从真实截图提取未标注候选（复用 V2B 分割/几何，不新建核心）；
  - 无监督聚类（同形资产分簇，降低标注成本）；
  - 识别后端（严格门控：无已标注真实资产 → UNAVAILABLE → null）。

红线：
  - synthetic 资产仅测试（source="test"），绝不冒充真实识别效果；
  - 不重新设计 Pixel Core / Vision 架构；
  - 无任何策略/决策/自动操作内容。
"""
from __future__ import annotations

from .assets import (DIGITS, GROUND_TRUTH_STATUSES, KINDS, RANKS, SUITS,
                     RealClientTemplate, RealClientTemplateLibrary)
from .extract import extract_candidates
from .cluster import assign_clusters
from .roles import ROLE_TABLE, assign_role_labels, role_for
from .labeling import (LabelingReport, ROLE_VOCAB, apply_cluster_labels,
                       label_vocab_for)
from .backends import (RealClientCardLibrary, RealClientCardRecognitionBackend,
                       RealClientDigitOCRBackend)

__all__ = [
    "DIGITS", "GROUND_TRUTH_STATUSES", "KINDS", "RANKS", "SUITS",
    "RealClientTemplate", "RealClientTemplateLibrary",
    "extract_candidates", "assign_clusters",
    "ROLE_TABLE", "assign_role_labels", "role_for",
    "LabelingReport", "ROLE_VOCAB", "apply_cluster_labels", "label_vocab_for",
    "RealClientCardLibrary", "RealClientCardRecognitionBackend",
    "RealClientDigitOCRBackend",
]
