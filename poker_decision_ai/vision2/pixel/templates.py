"""模板库与批量模板匹配（V2B 启用 TEMPLATE_MATCHED 证据）。

- 复用 V1 `ncc_template_match`（FFT/纯 Python 双路径）；
- TemplateLibrary：template_id + image + roi_hint + source + min_score；
- 匹配只回答"哪个模板在哪个位置、分数多少"，不解释牌局语义；
- builtin 模板 source 恒为 builtin_synthetic（真实客户端模板需另行标定）。
"""
from __future__ import annotations

from dataclasses import dataclass

from ..realtime.image import ImageData, ncc_template_match


@dataclass
class TemplateEntry:
    template_id: str
    image: ImageData
    roi_hint: str | None = None          # 限定 ROI；None=通用
    source: str = "builtin_synthetic"
    min_score: float = 0.7

    def to_dict(self) -> dict:
        return {"template_id": self.template_id,
                "width": self.image.width, "height": self.image.height,
                "roi_hint": self.roi_hint, "source": self.source,
                "min_score": self.min_score}


class TemplateLibrary:
    def __init__(self, entries=()):
        self.entries = list(entries)

    def add(self, entry: TemplateEntry) -> None:
        self.entries.append(entry)

    def by_hint(self, roi_hint: str | None) -> list:
        return [e for e in self.entries
                if e.roi_hint is None or e.roi_hint == roi_hint]

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries],
                "count": len(self.entries)}


@dataclass(frozen=True)
class TemplateMatch:
    template_id: str
    x: float
    y: float
    score: float
    roi: str | None = None
    bbox: dict | None = None
    source: str = "builtin_synthetic"

    def to_dict(self) -> dict:
        return {"template_id": self.template_id, "x": self.x, "y": self.y,
                "score": round(self.score, 4), "roi": self.roi,
                "bbox": self.bbox, "source": self.source}


def polarity_normalized(image: ImageData) -> ImageData:
    """极性归一化（确定性）：整体亮度均值 ≥128（深字形浅底）→ 反色。

    模板恒为亮字形黑底；匹配前把深字浅底区域反色后直接 NCC。
    """
    if image.channels != 4 or image.data is None:
        return image
    total = sum(image.data[i] for i in range(0, len(image.data), 4))
    n = image.width * image.height
    if n == 0 or total / n < 128.0:
        return image
    return invert_image(image)


def invert_image(image: ImageData) -> ImageData:
    """RGB 反色（确定性）。"""
    if image.channels != 4 or image.data is None:
        return image
    out = bytearray(image.data)
    for i in range(0, len(out), 4):
        out[i] = 255 - out[i]
        out[i + 1] = 255 - out[i + 1]
        out[i + 2] = 255 - out[i + 2]
    return ImageData.from_rgba_bytes(image.width, image.height, bytes(out))


def foreground_is_dark(mask: list, gray) -> bool:
    """mask 前景亮度 < 背景亮度 → 前景为深色（需反色匹配亮模板）。"""
    fg_s = bg_s = 0.0
    fg_n = bg_n = 0
    for y in range(gray.height):
        for x in range(gray.width):
            v = int(gray.data[y][x])
            if mask[y][x]:
                fg_s += v
                fg_n += 1
            else:
                bg_s += v
                bg_n += 1
    if fg_n == 0 or bg_n == 0:
        return False
    return (fg_s / fg_n) < (bg_s / bg_n)


def match_library(image: ImageData, library: TemplateLibrary,
                  roi_hint: str | None = None,
                  score_threshold: float | None = None) -> tuple:
    """对库中（按 roi_hint 过滤的）每个模板做 NCC → 达标匹配降序返回。"""
    matches = []
    for entry in library.by_hint(roi_hint):
        threshold = score_threshold if score_threshold is not None \
            else entry.min_score
        found = ncc_template_match(image, entry.image,
                                   score_threshold=threshold)
        if found is not None:
            x, y, score = found
            matches.append(TemplateMatch(
                template_id=entry.template_id, x=float(x), y=float(y),
                score=float(score), roi=roi_hint, source=entry.source,
                bbox={"x": float(x), "y": float(y),
                      "width": float(entry.image.width),
                      "height": float(entry.image.height)}))
    matches.sort(key=lambda m: -m.score)
    return tuple(matches)


def best_match(image: ImageData, library: TemplateLibrary,
               roi_hint: str | None = None) -> TemplateMatch | None:
    matches = match_library(image, library, roi_hint)
    return matches[0] if matches else None
