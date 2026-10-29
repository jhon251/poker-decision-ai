"""V2ZE — CoinPoker Hero Card 识别（Runtime 集成，确定性）。

- 模板：coinpoker_assets_v2zd_hero manifest（Q/s/9/d，full_png 坐标，
  visually_labeled）；
- 布局：hero_card_layout_v1.json（card0/card1 的 rank/suit bbox）；
- 识别：rank 族 {Q,9}、suit 族 {s,d} —— 双极性/二值灰度匹配（复用既有
  _match_family），置信 ≥0.7 + 判别余量 ≥0.08 才输出；
- 失败 → value=None + UNCERTAIN；不猜测不存在的牌。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .backends import _match_family

RANKS = "AKQJT98765432"
SUITS = "shdc"
MIN_SCORE = 0.7
MIN_MARGIN = 0.08


class CoinPokerHeroRecognizer:
    name = "coinpoker_hero_template_recognition"

    def __init__(self, assets_dir: str, layout_path: str,
                 min_score: float = MIN_SCORE, min_margin: float = MIN_MARGIN):
        self.min_score = min_score
        self.min_margin = min_margin
        from ...realtime.image import ImageData
        manifest = json.loads(
            (Path(assets_dir) / "manifest.json").read_text(encoding="utf-8"))
        self.templates = {}
        for a in manifest["assets"]:
            w, h, rgba = _decode_png(Path(a["crop_ref"]).read_bytes())
            self.templates[a["label"]] = ImageData.from_rgba_bytes(w, h, rgba)
        self.templates = {k: _tight_glyph(v) for k, v in
                          self.templates.items()}
        self.layout = json.loads(Path(layout_path).read_text(
            encoding="utf-8"))
        self.ranks = {k: v for k, v in self.templates.items() if k in RANKS}
        self.suits = {k: v for k, v in self.templates.items() if k in SUITS}

    def available(self) -> bool:
        return bool(self.ranks) and bool(self.suits)

    @staticmethod
    def _color_family(img) -> str:
        """字形颜色族：red / black（确定性像素统计，红通道主导）。"""
        import numpy as _np
        arr = _np.frombuffer(img.data, dtype=_np.uint8).reshape(
            img.height, img.width, img.channels or 4)
        r = arr[..., 0].astype(_np.int16)
        g = arr[..., 1].astype(_np.int16)
        b = arr[..., 2].astype(_np.int16)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        red = ((r > 100) & (r > g + 40) & (r > b + 40)).mean()
        dark = (luma < 115).mean()
        return "red" if red > max(0.05, dark * 0.5) else "black"

    def _family_templates(self, templates: dict, family: str) -> dict:
        return {k: v for k, v in templates.items()
                if self._color_family(v) == family}

    def recognize_card(self, frame: np.ndarray, card: dict) -> dict:
        """单卡（layout 项）→ {rank, suit, confidence, raw_text}。

        颜色族约束：先分类字形颜色 → 仅与该族模板匹配 → margin gate。
        """
        from ...realtime.image import ImageData
        rank_b, suit_b = card.get("rank_bbox"), card.get("suit_bbox")
        result = {"rank": None, "suit": None, "confidence": None,
                  "raw_text": None, "status": "UNCERTAIN"}
        if rank_b is None or suit_b is None:
            return result
        fh, fw = frame.shape[0], frame.shape[1]
        if (rank_b[0] + rank_b[2] > fw or suit_b[0] + suit_b[2] > fw
                or rank_b[1] + rank_b[3] > fh
                or suit_b[1] + suit_b[3] > fh):
            return result
        rank_crop = frame[rank_b[1]:rank_b[1] + rank_b[3],
                          rank_b[0]:rank_b[0] + rank_b[2]]
        suit_crop = frame[suit_b[1]:suit_b[1] + suit_b[3],
                          suit_b[0]:suit_b[0] + suit_b[2]]
        rank_img = _tight_glyph(ImageData.from_rgba_bytes(
            rank_crop.shape[1], rank_crop.shape[0], rank_crop.tobytes()))
        suit_img = _tight_glyph(ImageData.from_rgba_bytes(
            suit_crop.shape[1], suit_crop.shape[0], suit_crop.tobytes()))
        rf = self._color_family(rank_img)
        sf = self._color_family(suit_img)
        rank_fam = self._family_templates(self.ranks, rf)
        suit_fam = self._family_templates(self.suits, sf)
        r, rs, _ = _match_family(rank_img, rank_fam, binary=False)
        s, ss, _ = _match_family(suit_img, suit_fam, binary=False)
        if (r is None or s is None or rs is None or ss is None
                or rs < self.min_score or ss < self.min_score):
            return result
        rm = _margin(rs, rank_fam, rank_img)
        sm = _margin(ss, suit_fam, suit_img)
        if rm < self.min_margin or sm < self.min_margin:
            return result
        return {"rank": r, "suit": s,
                "confidence": round(min(rs, ss), 4),
                "raw_text": f"{r}{s}", "status": "OK",
                "rank_family": rf, "suit_family": sf}

    def recognize(self, rgba_frame: bytes, w: int, h: int,
                  frame_id: int) -> dict:
        """整帧 → hero 双卡结果（value/raw_text/confidence/source/
        evidence_frame_id）。"""
        frame = np.frombuffer(rgba_frame, dtype=np.uint8).reshape(h, w, 4)
        cards = []
        for card in self.layout["cards"]:
            res = self.recognize_card(frame, card)
            res["card_bbox"] = card.get("card_bbox")
            res["source"] = self.name
            res["evidence_frame_id"] = frame_id
            cards.append(res)
        ok = any(c["status"] == "OK" for c in cards)
        return {"cards": cards, "status": "OK" if ok else "UNCERTAIN",
                "value": None if not ok else [c["raw_text"] for c in cards
                                              if c["status"] == "OK"]}


def _margin(score, templates, img):
    if len(templates) < 2:
        return 1.0
    scores = []
    for ch, tpl in templates.items():
        _, s, _ = _match_family(img, {ch: tpl}, binary=False)
        scores.append(s if s is not None else -1.0)
    scores.sort(reverse=True)
    return scores[0] - (scores[1] if len(scores) > 1 else 0.0)


def _tight_glyph(img) -> "ImageData":
    """模板内容收紧：非白像素 bbox → 紧字形（去除冗余白边）。"""
    import numpy as _np
    from ...realtime.image import ImageData as _ID
    arr = _np.frombuffer(img.data, dtype=_np.uint8).reshape(
        img.height, img.width, img.channels or 4)
    r = arr[..., 0].astype(_np.int16)
    g = arr[..., 1].astype(_np.int16)
    b = arr[..., 2].astype(_np.int16)
    content = (r < 190) | (g < 190) | (b < 190)
    ys, xs = _np.where(content)
    if len(ys) < 4:
        return img
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    tight = arr[y0:y1, x0:x1]
    return _ID.from_rgba_bytes(tight.shape[1], tight.shape[0],
                               tight.tobytes())


def _decode_png(data):
    """解码 asset PNG → (w, h, rgba)；平台层（stdlib）。"""
    from ....vision2_platform import decode_png  # noqa: PLC0415
    return decode_png(data)
