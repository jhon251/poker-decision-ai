"""V2ZX — Cash Game Perception Layer（统一感知状态 + pot/dealer 识别）。

- CashGameState：hand/hero/board/pot/table/players/evidence 统一结构，
  每字段可追溯 frame→ROI→recognition→state；
- uncertainty 语义：OBSERVED/RECOGNIZED/UNCERTAIN/UNKNOWN/NOT_OBSERVED；
- PotRecognizer：底池数字带内滑动搜索（不固定 ROI——处理位置漂移），
  严格门控；无 → null+UNCERTAIN；
- DealerRecognizer：全桌红 D 六边形 blob 检测 → button_detected/bbox；
- 纪律：RECOGNIZED≠GT；stack→bet / bet→action / pot→stack 禁止；
  hero_seat/position 无证据=UNKNOWN。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .coinpoker_board import CoinPokerBoardRecognizer
from .coinpoker_hero import CoinPokerHeroRecognizer, _tight_glyph
from .coinpoker_board import _pad_tight
from ...realtime.image import ImageData, to_gray

UNKNOWN = "UNKNOWN"
NOT_OBSERVED = "NOT_OBSERVED"


@dataclass
class CashGameState:
    session_id: str = "session"
    frame_id: int = 0
    timestamp: float = 0.0
    hand_id: str | None = None
    street: str = UNKNOWN
    hero_cards: list = field(default_factory=list)
    hero_seat: str = UNKNOWN
    board_cards: list = field(default_factory=list)
    board_cardinality: int = 0
    pot: dict | None = None
    button: dict | None = None
    players: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "frame_id": self.frame_id,
                "timestamp": self.timestamp, "hand_id": self.hand_id,
                "street": self.street, "hero_cards": self.hero_cards,
                "hero_seat": self.hero_seat,
                "board_cards": self.board_cards,
                "board_cardinality": self.board_cardinality,
                "pot": self.pot, "button": self.button,
                "players": self.players, "evidence": self.evidence,
                "provenance": self.provenance,
                "ground_truth_status": "runtime_state_non_gt"}


def _block_downscale(g, factor=4):
    """4× block-average 降采样（保留结构——非最近邻采样）。"""
    h, w = g.shape
    h2, w2 = h // factor, w // factor
    return g[:h2 * factor, :w2 * factor].reshape(
        h2, factor, w2, factor).mean(axis=(1, 3))


def _bright_tight(img):
    """深色底亮字形：亮内容（luma>150）bbox 修剪。"""
    from ...realtime.image import ImageData as _ID, to_gray
    g = np.asarray(to_gray(img).data, dtype=float)
    ys, xs = np.where(g > 150)
    if len(ys) < 4:
        return img
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    arr = np.frombuffer(img.data, dtype=np.uint8).reshape(
        img.height, img.width, img.channels or 4)
    tight = arr[y0:y1, x0:x1]
    return _ID.from_rgba_bytes(tight.shape[1], tight.shape[0],
                               tight.tobytes())


class PotRecognizer:
    name = "coinpoker_pot_digit_band_search"

    #: V3F-P15 —— 已废止（WRONG GEOMETRY）：旧假设 pot 在窗口顶部。
    #: P21 视觉确认 + P22-A 像素实测：pot 数字区在桌面上部 x≈908–1060, y≈488–543
    #: （"底池"标签在 x<908，自然分离；数字白色 R≈231，非金色）。
    #: 1419 保持既有比例带（0.71-0.78 h × 0.62-0.78 w）。
    POT_ROI_1876 = (908, 488, 1060, 543)
    #: 1876 pot 数字模板目录（P22-A 从 6 个真实 GT 帧提取，GT provenance 入 manifest）
    POT_ASSET_DIR_1876 = "data/vision2/coinpoker_assets_v1876pot"
    #: V-5（Round16）微级别 pot 字形目录（目录列表**尾部追加**；kind=
    #: pot_digit_micro 隔离——v3c_micro 混载目录，仅该 kind 入 pot 模板集）。
    #: 微级别 pot 数字 ≈20-24px（1876 基础模板 ≈30px 对滑窗超候选被跳过 →
    #: digit_gate 失败）；裸小数格式（'0.25'）接受见 recognize() 格式门。
    POT_ASSET_DIR_1876_MICRO = "data/vision2/coinpoker_assets_v3c_micro"
    POT_MICRO_KIND = "pot_digit_micro"
    #: 尺度一致性门（V-5）：模板高 < 0.75×候选高 → 跳过（防微模板在大字形
    #: 内部伪匹配吞 margin——与 coinpoker_stack._ncc_1876 同因同门）
    POT_NCC_H_RATIO_MIN = 0.75

    def __init__(self, asset_dirs=None, min_score=0.7, min_margin=0.08,
                 window: str = "1419"):
        from .coinpoker_board import _prep_templates
        asset_dirs = asset_dirs or (
            ("data/vision2/coinpoker_assets_v1876pot",
             self.POT_ASSET_DIR_1876_MICRO)          # V-5 尾追加（1876）
            if window == "1876" else
            ("data/vision2/coinpoker_assets_v2zb",
             "data/vision2/coinpoker_assets_v2zy"))
        self.min_score = min_score
        self.min_margin = min_margin
        self.window = window
        self.digits = {}            # label -> list[ImageData]（多实例）
        # V3F-P22-A：1876 允许 'M'（K/M 格式如 1.61M）；'.' 不加载（几何检测，
        # h<10→点——避免与逗号 NCC 互混）；1419 仅 digits+comma。
        labels_allowed = "0123456789,M" if window == "1876" else "0123456789,"
        for d in asset_dirs:
            m = json.loads((Path(d) / "manifest.json").read_text(
                encoding="utf-8"))
            for a in m["assets"]:
                if a["label"] not in labels_allowed:
                    continue
                # V-5 kind 隔离：微级别混载目录仅入 pot_digit_micro 资产；
                # 基础目录（无 kind 过滤需求）维持原状零变化
                if d == self.POT_ASSET_DIR_1876_MICRO \
                        and a.get("kind") != self.POT_MICRO_KIND:
                    continue
                w, h, rgba = _decode_png(Path(a["crop_ref"]).read_bytes())
                img = _bright_tight(ImageData.from_rgba_bytes(w, h, rgba))
                # 碎片过滤（V2ZZ-C）：宽 <12 且非逗号/点 → 提取伪影，禁用
                if a["label"] not in (",", ".") and img.width < 12:
                    continue
                self.digits.setdefault(a["label"], []).append(img)
        self._prep = {}
        for label, imgs in self.digits.items():
            self._prep[label] = []
            for im in imgs:
                g = np.asarray(to_gray(im).data, dtype=float)
                t0 = g - g.mean()
                tn = float(np.sqrt((t0 * t0).sum()))
                self._prep[label].append((g, t0, tn))

    def _seg(self, sub: np.ndarray) -> list:
        """V2ZZ-C 列投影分割（与提取器一致：luma>150、min 3px）。"""
        r = sub[..., 0].astype(int)
        g = sub[..., 1].astype(int)
        b = sub[..., 2].astype(int)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        col = (luma > 150).sum(axis=0) > 1
        runs = []
        x = 0
        while x < len(col):
            if not col[x]:
                x += 1
                continue
            s = x
            while x < len(col) and col[x]:
                x += 1
            if x - s >= 3:
                runs.append((s, x))
        return runs

    def _match_char(self, crop: np.ndarray) -> tuple:
        from .coinpoker_board import _slide_prepped, _pad_tight
        bright = _bright_tight(ImageData.from_rgba_bytes(
            crop.shape[1], crop.shape[0], crop.tobytes()))
        glyph_h = bright.height            # V-5：加垫前字高（尺度门基准）
        tight = _pad_tight(bright)
        from ...realtime.image import to_gray
        g = np.asarray(to_gray(tight).data, dtype=float)
        best, bsc, bmargin = None, -1.0, 0.0
        best_scores = {}                      # label -> 最佳分（多实例取 max）
        h_min = self.POT_NCC_H_RATIO_MIN * glyph_h      # V-5 尺度一致性门
        for ch, insts in self._prep.items():
            best_sc_ch = -1.0
            for (t, t0, tn) in insts:
                th, tw = t.shape
                if th >= g.shape[0] or tw >= g.shape[1] or tn < 1e-9:
                    continue
                if th < h_min:
                    continue
                wins = np.lib.stride_tricks.sliding_window_view(g, (th, tw))
                wm = wins.mean(axis=(2, 3), keepdims=True)
                wc = wins - wm
                wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
                num = (wc * t0).sum(axis=(2, 3))
                scores = np.abs(num / np.maximum(wn * tn, 1e-9))
                sc = float(scores.max())
                if sc > best_sc_ch:
                    best_sc_ch = sc
            best_scores[ch] = best_sc_ch
            if best_sc_ch > bsc:
                best, bsc = ch, best_sc_ch
        if best is None:
            return None, None, None
        scores2 = sorted(best_scores.values(), reverse=True)
        margin = bsc - (scores2[1] if len(scores2) > 1 else 0.0)
        return best, bsc, margin

    def recognize(self, frame: np.ndarray) -> dict:
        """pot 数字串识别（V2ZZ-B pill 区）+ 分割 + 逐字符匹配 + 有序组装。"""
        # V3F-P22-A：1876 pot = 桌面上部白色数字区（P21 视觉确认 + 像素实测），
        # 用 1876 模板（GT 提取）+ 固定 ROI；1419 比例带不变。
        if self.window == "1876":
            x0, y0, x1, y1 = self.POT_ROI_1876
            band = frame[y0:y1, x0:x1]
        else:
            fh, fw = frame.shape[0], frame.shape[1]
            band = frame[int(fh * 0.71):int(fh * 0.78), int(fw * 0.62):
                         int(fw * 0.78)]
        if band.size < 100:
            return {"value": None, "raw_text": None, "status": UNKNOWN,
                    "reason": "empty_band"}
        luma = (band[..., 0].astype(np.int16) * 0.299
                + band[..., 1].astype(np.int16) * 0.587
                + band[..., 2].astype(np.int16) * 0.114)
        if (luma > 140).sum() < 60:
            return {"value": None, "raw_text": None, "status": UNKNOWN,
                    "reason": "no_bright_digits"}
        runs = self._seg(band)
        if not runs:
            return {"value": None, "raw_text": None, "status": UNKNOWN,
                    "reason": "no_char_runs"}
        chars = []
        for (s, e) in runs:
            crop = band[:, s:e]
            # V3F-P22-A（仅 1876）：几何判别 点(h<10) vs 逗号/数字/M（NCC）。
            # 防止 '.' 模板与 ',' 模板 NCC 互混（点=小圆点 h≈6；逗号 h≈15）。
            if self.window == "1876":
                lc = (crop[..., 0].astype(np.int16) * 0.299
                      + crop[..., 1].astype(np.int16) * 0.587
                      + crop[..., 2].astype(np.int16) * 0.114)
                rows = np.where((lc > 150).sum(axis=1) > 0)[0]
                hh = (rows.max() - rows.min() + 1) if len(rows) else 0
                if hh and hh < 10 and (e - s) <= 8:
                    chars.append(".")
                    continue
                if (e - s) > 24:
                    # 几何"M"：唯一宽字母（数字 17-20px，M 26px）——避免 NCC
                    # 与数字子匹配 margin 不足
                    chars.append("M")
                    continue
            ch, sc, margin = self._match_char(crop)
            if (ch is None or sc is None or sc < self.min_score
                    or margin < self.min_margin):
                return {"value": None, "raw_text": None,
                        "status": "UNCERTAIN", "reason": "digit_gate",
                        "partial": "".join(chars)}
            chars.append(ch)
        text = "".join(chars)
        import re
        # V3F-P22-A：1876 支持 K/M 格式（1.61M→1610000，用 parse_stack_text 解析
        # 保持一致）；1419 保持纯逗号格式。
        km_ok = re.match(r"^(\d+)(?:\.(\d+))?[KM]$", text) is not None
        # V-5（Round16）微级别裸小数：'0.25' → value 原样文本（货币单位；
        # production_bridge._strict_num float 化 → 0.25，与 V-4 header
        # `_to_num` 小数分支同单位语义——bet/pot/stack 直比较）。单点格式、
        # 整数部分可空（'.25' 不允许——语料未见且几何点判定会先行吞掉）。
        dec_ok = re.match(r"^\d+\.\d+$", text) is not None
        if not (re.match(r"^\d{1,3}(,\d{3})*$", text) or km_ok or dec_ok):
            return {"value": None, "raw_text": None,
                    "status": "UNCERTAIN", "reason": "format_invalid",
                    "partial": text}
        return {"value": text, "raw_text": text, "status": "RECOGNIZED",
                "reason": "ordered_assembly", "digits": chars}


class PlayerStatusRecognizer:
    """V2ZZ-B — 徽章模板匹配（视觉确认的 full_png bbox + NCC + 严格门控）。

    支持：folded（『弃牌』）；away/all-in/disconnected 模板待视觉确认
    → 未注册 → NOT_OBSERVED。徽章位置随座位变化——按座搜索（seat ROI 未
    建立 → 当前提供通用检测：在全帧搜索徽章模板）。
    """

    def __init__(self, asset_dir="data/vision2/coinpoker_assets_v2zy",
                 min_score=0.7, min_margin=0.08):
        self.min_score = min_score
        self.min_margin = min_margin
        m = json.loads((Path(asset_dir) / "manifest.json").read_text(
            encoding="utf-8"))
        self.badges = {}
        for a in m["assets"]:
            if a["kind"] != "status_badge":
                continue
            w, h, rgba = _decode_png(Path(a["crop_ref"]).read_bytes())
            self.badges[a["label"]] = _bright_tight(
                ImageData.from_rgba_bytes(w, h, rgba))

    def recognize(self, frame: np.ndarray) -> dict:
        if not self.badges:
            return {"status": "UNKNOWN", "badges": {},
                    "reason": "no_badge_templates"}
        fh, fw = frame.shape[0], frame.shape[1]
        g_full = np.asarray(to_gray(ImageData.from_rgba_bytes(
            fw, fh, frame.tobytes())).data, dtype=np.float32)
        # V2ZZ-D：4× block-average 金字塔（保留结构，非采样——不猜坐标）
        g = _block_downscale(g_full, 4)
        found = {}
        for label, tpl in self.badges.items():
            t_full = np.asarray(to_gray(tpl).data, dtype=np.float32)
            t = _block_downscale(t_full, 4)
            th, tw = t.shape
            if th < 2 or tw < 2 or th >= g.shape[0] or tw >= g.shape[1]:
                continue
            wins = np.lib.stride_tricks.sliding_window_view(g, (th, tw))
            wm = wins.mean(axis=(2, 3), keepdims=True)
            wc = wins - wm
            wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
            t0 = t - t.mean()
            tn = float(np.sqrt((t0 * t0).sum()))
            scores = np.abs((wc * t0).sum(axis=(2, 3))
                            / np.maximum(wn * tn, 1e-9))
            score = float(scores.max())
            idx = np.unravel_index(np.argmax(scores), scores.shape)
            cy, cx = int(idx[0]) * 4, int(idx[1]) * 4
            if score >= 0.5:
                # 精化：原分辨率 ±(4*模板尺寸) 窗口 stride1
                th_f, tw_f = t_full.shape
                y0 = max(0, cy - th_f)
                x0 = max(0, cx - tw_f)
                y1 = min(g_full.shape[0], cy + th_f * 2)
                x1 = min(g_full.shape[1], cx + tw_f * 2)
                fine = g_full[y0:y1, x0:x1]
                if fine.shape[0] >= th_f and fine.shape[1] >= tw_f:
                    fw2 = np.lib.stride_tricks.sliding_window_view(
                        fine, (th_f, tw_f))
                    fwm = fw2.mean(axis=(2, 3), keepdims=True)
                    fwc = fw2 - fwm
                    fwn = np.sqrt((fwc * fwc).sum(axis=(2, 3)))
                    tf0 = t_full - t_full.mean()
                    tfn = float(np.sqrt((tf0 * tf0).sum()))
                    fsc = np.abs((fwc * tf0).sum(axis=(2, 3))
                                 / np.maximum(fwn * tfn, 1e-9))
                    score = float(fsc.max())
                    fidx = np.unravel_index(np.argmax(fsc), fsc.shape)
                    cx, cy = x0 + int(fidx[1]), y0 + int(fidx[0])
            if score >= self.min_score:
                found[label] = {"score": round(score, 3),
                                "bbox": [cx, cy, tw_f, th_f]}
        if not found:
            return {"status": "NOT_OBSERVED", "badges": {},
                    "reason": "no_badge_match"}
        return {"status": "RECOGNIZED", "badges": found,
                "reason": "badge_match"}


class DealerRecognizer:
    name = "coinpoker_dealer_red_d_blob"

    def recognize(self, frame: np.ndarray) -> dict:
        r = frame[..., 0].astype(np.int16)
        g = frame[..., 1].astype(np.int16)
        b = frame[..., 2].astype(np.int16)
        red = (r > 150) & (g < 100) & (b < 100)
        ys, xs = np.where(red)
        if len(xs) < 40:
            return {"button_detected": False, "bbox": None,
                    "status": NOT_OBSERVED}
        return {"button_detected": True,
                "bbox": [int(xs.min()), int(ys.min()),
                         int(xs.max() - xs.min() + 1),
                         int(ys.max() - ys.min() + 1)],
                "status": "OBSERVED", "note": "red blob (D candidate)"}


def assemble_cash_state(board_rec, hero_rec, pot_rec, dealer_rec,
                        rgba_frame, w, h, frame_id, session_id,
                        street_from_board=None) -> CashGameState:
    import time
    frame = np.frombuffer(rgba_frame, dtype=np.uint8).reshape(h, w, 4)
    board = board_rec.recognize(rgba_frame, w, h, frame_id)
    cards = [(c["slot_index"], c["raw_text"]) for c in board["cards"]
             if c["status"] == "OK"]
    hero = hero_rec.recognize(rgba_frame, w, h, frame_id)
    pot = pot_rec.recognize(frame)
    dealer = dealer_rec.recognize(frame)
    street = street_from_board(board["cards"]) if street_from_board else \
        UNKNOWN
    st = CashGameState(session_id=session_id, frame_id=frame_id,
                       timestamp=time.time(),
                       street=street,
                       hero_cards=[c["raw_text"] for c in hero["cards"]
                                   if c["status"] == "OK"],
                       board_cards=[c for _, c in sorted(cards)],
                       board_cardinality=len(cards), pot=pot,
                       button=dealer, players=[],
                       evidence={"frame_ids": [frame_id],
                                 "board_source": board_rec.name,
                                 "hero_source": hero_rec.name},
                       provenance={"recognition": "deterministic",
                                   "ground_truth_status":
                                       "runtime_state_non_gt"})
    return st


def _decode_png(data):
    from ....vision2_platform import decode_png  # noqa: PLC0415
    return decode_png(data)
