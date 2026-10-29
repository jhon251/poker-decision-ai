"""V2C — 真实客户端视觉资产体系（Real Client Template Library）。

硬约束：
  - 资产必须携带完整 provenance（来源帧/ROI/语义标签/客户端布局/裁剪框/
    版本/置信度/标注状态）；
  - 未标注候选资产 ground_truth_status="unlabeled_candidate"、semantic_label
    =None——**绝不标记为已识别**；
  - 只有被标注（visually_labeled/verified/test）的资产才可参与识别；
  - synthetic 资产（source="test"）仅用于测试机制，绝不冒充真实识别效果。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# 资产类型（冻结白名单）
KINDS = ("card_slot", "card_glyph", "digit_glyph", "disc_candidate",
         "region_snapshot", "unknown")
# 标注状态（schema v2 扩展：+ role_identified）。
# unlabeled_candidate=无任何语义；role_identified=仅角色级语义（由 ROI
# provenance 确定性推导）；visually_labeled/verified=字符级已标注；
# test=仅机制测试。
GROUND_TRUTH_STATUSES = ("unlabeled_candidate", "role_identified",
                         "visually_labeled", "verified", "test")
# 语义标签词表（按 kind）
RANKS = "AKQJT98765432"
SUITS = "shdc"
DIGITS = "0123456789.,:"
# 数字单位后缀（gold 文本末尾的 K/M/B）；真实模板待视觉标注（机制已就绪）
UNIT_CHARS = "MKB"


@dataclass
class RealClientTemplate:
    """单条真实客户端视觉资产（全 provenance；未知 = null）。

    schema v2 新字段：role（角色级语义，由 ROI provenance 确定性推导，
    如 pot_digit_role）——与 semantic_label（字符级）分离；role 存在
    不代表字符身份已知。
    """
    template_id: str
    kind: str = "unknown"
    semantic_label: str | None = None      # 字符级标签；未标注 → null（禁止填 ""）
    role: str | None = None                # 角色级标签（可为 null）
    source_frame: str | None = None        # 来源截图文件名
    source_roi: str | None = None          # 语义 ROI 名
    client: str = "unknown_client_1848x1388"
    crop_bbox: dict | None = None          # 源帧 client 坐标 {x,y,width,height}
    asset_version: int = 1
    confidence: float | None = None        # 未校准 → null
    ground_truth_status: str = "unlabeled_candidate"
    label_source: str | None = None        # 标注来源（视觉标注/人工/测试/ROI 推导）
    cluster_id: int | None = None          # 无监督聚类 id（未标注语义）
    png_ref: str | None = None             # 裁剪 PNG 引用（数据目录内）

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"非法 kind {self.kind}")
        if self.ground_truth_status not in GROUND_TRUTH_STATUSES:
            raise ValueError(f"非法 ground_truth_status {self.ground_truth_status}")
        if self.semantic_label is not None:
            self._validate_label(self.kind, self.semantic_label)
        if self.role is not None and not isinstance(self.role, str) \
                or (self.role == ""):
            raise ValueError("role 必须为 null 或非空字符串")
        if self.asset_version < 1:
            raise ValueError("asset_version 必须 ≥ 1")
        if self.template_id in (None, ""):
            raise ValueError("template_id 必填")

    @staticmethod
    def _validate_label(kind: str, label: str) -> None:
        if kind == "card_glyph":
            if label not in RANKS + SUITS:
                raise ValueError(f"card_glyph 标签必须属于 rank/suit：{label}")
        elif kind == "digit_glyph":
            if label not in DIGITS:
                raise ValueError(f"digit_glyph 标签必须属于数字集：{label}")
        elif kind in ("card_slot", "disc_candidate", "region_snapshot"):
            pass                                # 结构资产标签自由文本（可为 null）
        else:
            raise ValueError(f"kind {kind} 不支持语义标签")

    @property
    def is_labeled(self) -> bool:
        """字符级已标注（角色级 role_identified 不算）。

        值识别后端只认字符级标签——role 存在绝不开启值识别。
        """
        return (self.semantic_label is not None
                and self.ground_truth_status in
                ("visually_labeled", "verified", "test"))

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id, "kind": self.kind,
            "semantic_label": self.semantic_label, "role": self.role,
            "source_frame": self.source_frame, "source_roi": self.source_roi,
            "client": self.client, "crop_bbox": self.crop_bbox,
            "asset_version": self.asset_version,
            "confidence": self.confidence,
            "ground_truth_status": self.ground_truth_status,
            "label_source": self.label_source,
            "cluster_id": self.cluster_id, "png_ref": self.png_ref,
        }


class RealClientTemplateLibrary:
    """真实客户端资产库（版本化；识别只能使用已标注资产）。"""

    def __init__(self, client: str = "unknown_client_1848x1388",
                 version: int = 1):
        self.client = client
        self.version = version
        self.assets: dict = {}                # template_id -> RealClientTemplate
        self.images: dict = {}                # template_id -> ImageData

    def add(self, asset: RealClientTemplate, image=None) -> None:
        asset.client = self.client
        asset.asset_version = self.version
        self.assets[asset.template_id] = asset
        if image is not None:
            self.images[asset.template_id] = image

    def get(self, template_id: str) -> RealClientTemplate | None:
        return self.assets.get(template_id)

    def by_kind(self, kind: str) -> list:
        return [a for a in self.assets.values() if a.kind == kind]

    def labeled_assets(self) -> list:
        return [a for a in self.assets.values() if a.is_labeled]

    def unlabeled_assets(self) -> list:
        return [a for a in self.assets.values() if not a.is_labeled]

    def labeled_templates(self, kind: str, vocab: str,
                          source_roi: str | None = None) -> dict:
        """已标注资产 → {semantic_label: ImageData}（仅 vocab 内标签）。

        识别后端唯一入口：未标注资产一律不参与识别；
        source_roi 非空时仅限该 ROI 来源的资产（hero/board 域隔离）。
        """
        out = {}
        for a in self.assets.values():
            if not a.is_labeled or a.kind != kind:
                continue
            if a.semantic_label not in vocab:
                continue
            if source_roi is not None and a.source_roi != source_roi:
                continue
            img = self.images.get(a.template_id)
            if img is not None:
                out[a.semantic_label] = img
        return out

    def cluster_counts(self) -> dict:
        counts: dict = {}
        for a in self.assets.values():
            if a.cluster_id is not None:
                counts[a.cluster_id] = counts.get(a.cluster_id, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "client": self.client, "version": self.version,
            "assets": [a.to_dict() for a in self.assets.values()],
        }

    def to_json(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                     encoding="utf-8")

    @classmethod
    def from_json(cls, path: str, images_loader=None) -> "RealClientTemplateLibrary":
        """manifest 恢复（images_loader: template_id -> ImageData|None）。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        lib = cls(client=data.get("client", "unknown_client_1848x1388"),
                  version=int(data.get("version", 1)))
        for d in data.get("assets", []):
            asset = RealClientTemplate(
                template_id=d["template_id"], kind=d.get("kind", "unknown"),
                semantic_label=d.get("semantic_label"),
                role=d.get("role"),
                source_frame=d.get("source_frame"),
                source_roi=d.get("source_roi"),
                client=d.get("client", lib.client),
                crop_bbox=d.get("crop_bbox"),
                asset_version=int(d.get("asset_version", 1)),
                confidence=d.get("confidence"),
                ground_truth_status=d.get("ground_truth_status",
                                          "unlabeled_candidate"),
                label_source=d.get("label_source"),
                cluster_id=d.get("cluster_id"), png_ref=d.get("png_ref"))
            img = images_loader(asset.template_id) if images_loader else None
            lib.add(asset, img)
        return lib
