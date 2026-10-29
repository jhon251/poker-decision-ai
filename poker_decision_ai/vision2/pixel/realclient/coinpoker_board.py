"""V2ZG — CoinPoker Board Card Runtime 识别（确定性，模板族匹配）。

- 卡槽：V2ZA 实测 (669+i*143, 807, 143, 178) full_png；
- 模板：v2zh+v2zo+v2zv+v2zk manifest（rank 含 J/4 补全、suit {s,d}）；
- 颜色族：S2 复用（角区红色主导 → 仅红族模板）；
- 双门控：conf≥0.7 + margin≥0.08；失败 → 严格全族复核（≥0.90/≥0.10）；
- V-2（Round 9）半置信分层输出：
  * FULL（status=OK）：rank+suit 双证据 → 最终牌面 + 计数；
  * RANK_ONLY：rank 过门 + suit 未定 → 仅计数（street 只需要张数）；
  * UNCERTAIN：rank 门失败/空槽/越界（不猜测）；
- "10" 双字形墨宽几何门 + 中心单证据 0.80 严门（s→c 误认防护）；
- 不猜测缺失牌（模板未覆盖的 rank/suit → RANK_ONLY/UNCERTAIN）。
"""
from __future__ import annotations

import os

import json
from pathlib import Path

import numpy as np

from .backends import _slide_family_best
from .coinpoker_hero import _tight_glyph, _decode_png
from ...realtime.image import ImageData, to_gray

RANKS = "AKQJT98765432"
SUITS = "shdc"
MIN_SCORE = 0.7
MIN_MARGIN = 0.08
SLOT_X0, SLOT_Y0, SLOT_W, SLOT_H = 664, 551, 140, 177
#: V3F-P15 — per render class 槽位几何（1419 冻结；1876 由 1220 帧实测：
#: 卡槽 x=590/730/870/1010/1150，pitch 140，卡 ~120×180，顶 y≈551）。
#: 派发依据 = render class（几何），非 frame id / scene / 值。
SLOT_GEOM_BY_WINDOW = {
    "1419": (664, 551, 140, 177),
    "1876": (590, 551, 140, 180),
}
#: 模板优先级：v2zh（完整字形，实测槽位）优先；v2zo（完整 T）末位覆盖。
#: v2x/v2y 碎片目录自 V2ZP 移除——碎片（8×5 级）为提取伪影，v2zh 已有同
#: label 完整版本；保留它们曾占 78% 的 NCC 窗口数（性能诊断证实）。
DEFAULT_ASSET_DIRS = ("data/vision2/coinpoker_assets_v2zh",
                      "data/vision2/coinpoker_assets_v2zo",
                      "data/vision2/coinpoker_assets_v2zv",
                      "data/vision2/coinpoker_assets_v2zk",
                      # V-4（Round15）尾追加：微级别 ₮ header 模板目录。
                      # 本识别器只加载 manifest kind=="card_glyph" 的资产，
                      # v3c_micro 全部为 betting_header_v3c_micro kind →
                      # 加载即过滤，rank/suit 模板集零变化（见 V-4 报告）。
                      "data/vision2/coinpoker_assets_v3c_micro")
#: 已废弃碎片目录（审计保留，不加载）
DEPRECATED_ASSET_DIRS = ("data/vision2/coinpoker_assets_v2x",
                         "data/vision2/coinpoker_assets_v2y")
CENTER_SUIT_DIR = "data/vision2/coinpoker_assets_v2zj"
#: corner-only suit 独立可靠性开关（V2ZL）：至今无任何验证正确的 corner-only
#: 输出（2d 两度错误）→ 默认禁止单 corner 输出；待独立验证后由配置启用。
CORNER_ONLY_SUIT_ALLOWED = False
#: V2ZQ 搜索分区（V2ZH bbox 证据：rank y 9-62（2/A/8/9/K/5/T/7 实测）、
#: 角标花色 y 61-89（c/d/s/h 实测）→ rank 区 y 6-63（含 5 的 y62 行）、
#: suit 区 y 62-90）
RANK_ZONE = (6, 6, 49, 57)      # card-relative: y 6..62 含边界行（高 57）
SUIT_ZONE = (6, 62, 55, 28)     # card-relative: y 62..89（高 28）
EMPTY_SLOT_MIN_WHITE = 0.12     # 白卡存在门（空槽直接跳过识别）
# ── V-2（Round 9）半置信分层 + 严格复核参数 ─────────────────────────────
#: 严格二阶全族 rank 复核门（V-2 RC-1 修复：主搜索被颜色族硬过滤时，真
#: rank 的唯一模板变体可能在另一族——灰度 NCC 本无色别。全族强证据
#: （0.90/0.09）优先于主族弱证据；语料实测：真 rank 全族证据 score≥0.94、
#: margin≥0.09（8♣/6♣ 最强混淆对 0.94 vs 0.845），错救需错字形 ≥0.90 且
#: 胜次 0.09+——全语料混淆矩阵中错字形胜出最大观测 0.893（Q-on-Ts，1331），
#: 无 ≥0.90 错救样本；SETTLED 全集前后对照：V-1 漏读 333 帧 → 终态 1 帧、
#: GT 帧 0 误报（docs/V2_BOARD_RECALL_REPORT.md §3/§4）。
RANK_RESCUE_MIN_SCORE = 0.90
RANK_RESCUE_MIN_MARGIN = 0.09
#: 中心大花色单证据可独立支撑 FULL 的最低分（V-2 RC-2 校准：中心 NCC 把
#: 黑桃系统性认成梅花 @0.755-0.762（错花色总体），正确 d 证据 @0.82-0.84
#: （对总体）——0.80 分界。低于此值的中心证据不再单独支撑花色 → 该卡
#: 降级 RANK_ONLY（计数有效，suit 不入最终牌面）。比旧 0.70 更严。
CENTER_ONLY_MIN_SCORE = 0.80
#: "10" 双字形墨宽几何门（V-2 RC-3：双字形 "10" 墨宽 ≥32px，单字形
#: rank ≤28px——墨宽 ≥32 时仅双字形模板可竞争，消除 T margin<0.08 的
#: 单字形互抢；墨宽 <32 时排除双字形模板，消除单卡被 T 吸引）。
TWO_GLYPH_INK_WIDTH = 32
#: 窄条字形防护（V-2 RC-4，0 误报纪律）：J（14px 竖条钩）类窄条模板与
#: 任意字形竖笔画的基线 NCC 相似度高（实测 A♥ 角标 J@red 吸引 0.74-0.85、
#: Q♦ 0.845——均过旧 0.7/0.08 主门）。规则：窄条候选（模板墨宽 <18px）
#: 得分 <0.90 先剔除再 argmax；窄条夺冠还需判别余量 ≥0.20，不足 → 整槽
#: 拒绝（不猜、不降级次名）。真 J 实测 0.96-1.0 / margin 0.40+ 不受影响。
NARROW_BAR_INK_WIDTH = 18
NARROW_BAR_MIN_SCORE = 0.90
NARROW_BAR_MIN_MARGIN = 0.20
#: 分层输出 tier 值
TIER_FULL = "FULL"              # rank+suit 双证据（进最终牌面 + 计数）
TIER_RANK_ONLY = "RANK_ONLY"    # rank 过门 + suit 未定（仅计数，不入牌面）


def _prep_templates(templates: dict) -> dict:
    """模板预处理缓存（V2ZP）：灰度 + 均值 + 模长——与 _slide_family_best
    内部数学完全一致，仅从每调用迁移到 init 一次。"""
    out = {}
    for ch, tpl in templates.items():
        g = np.asarray(to_gray(tpl).data, dtype="float")
        th, tw = g.shape
        t0 = g - g.mean()
        tn = float(np.sqrt((t0 * t0).sum()))
        out[ch] = (g, t0, tn)
    return out


def _pad_tight(img, pad: int = 6):
    """紧字形加白边（V2ZQ）：使搜索区严格大于模板——滑动匹配的
    `>=` 跳过规则在"紧字形==模板尺寸"时会把完美匹配跳过（根因）。
    几何填充，不改滑动/阈值语义。"""
    w, h = img.width, img.height
    out = np.full((h + 2 * pad, w + 2 * pad, 4), 255, dtype=np.uint8)
    arr = np.frombuffer(img.data, dtype=np.uint8).reshape(
        h, w, img.channels or 4)
    out[pad:pad + h, pad:pad + w] = arr
    from ...realtime.image import ImageData as _ID
    return _ID.from_rgba_bytes(out.shape[1], out.shape[0], out.tobytes())


def _slide_prepped(slot_g, prepped: dict, stride: int = 4) -> tuple:
    """V2ZP 本地滑动 NCC（数学 = _slide_family_best；模板预处理已缓存）。

    输出签名与 _slide_family_best 完全一致：
    (最佳字符, 分, 位置, 跨字符判别余量)。

    P30-9-IL 复盘 ✗：曾在此接入"有界搜索"（缩小滑窗范围 ✓）提速 ⇒ 实测**收益 0%** ✗
    （此处输入本就是紧裁小卡块 ⇒ 缩不动 ✓）⇒ **已回退** ✓（不留没有实测收益的改动 ✓）。
    """
    char_scores = {}
    for ch, (g, t0, tn) in prepped.items():
        th, tw = g.shape
        if th >= slot_g.shape[0] or tw >= slot_g.shape[1]:
            continue
        if tn < 1e-9:
            continue
        us = 2 if th >= 55 else 1
        wins = np.lib.stride_tricks.sliding_window_view(slot_g, (th, tw))
        sub = wins[::us, ::us]
        wm = sub.mean(axis=(2, 3), keepdims=True)
        wc = sub - wm
        wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
        num = (wc * t0).sum(axis=(2, 3))
        scores = num / np.maximum(wn * tn, 1e-9)
        abs_scores = np.abs(scores)
        idx = np.unravel_index(np.argmax(abs_scores), abs_scores.shape)
        char_scores[ch] = (float(abs_scores[idx]),
                           (int(idx[1] * us), int(idx[0] * us)))
    if not char_scores:
        return (None, -1.0, None, -1.0)
    ordered = sorted(char_scores.items(), key=lambda kv: -kv[1][0])
    best_ch, (best_sc, best_pos) = ordered[0]
    second = ordered[1][1][0] if len(ordered) > 1 else 0.0
    return best_ch, best_sc, best_pos, best_sc - second


class CoinPokerBoardRecognizer:
    name = "coinpoker_board_template_recognition"

    def __init__(self, asset_dirs=DEFAULT_ASSET_DIRS, min_score=MIN_SCORE,
                 min_margin=MIN_MARGIN, center_suit_dir=CENTER_SUIT_DIR,
                 corner_only_allowed=CORNER_ONLY_SUIT_ALLOWED,
                 window: str = "1419"):
        self.min_score = min_score
        self.min_margin = min_margin
        self.corner_only_allowed = corner_only_allowed
        # V3F-P15：槽位几何按 render class 派发（1419 冻结默认；1876 实测）
        self.window = window
        self.slot_x0, self.slot_y0, self.slot_w, self.slot_h = \
            SLOT_GEOM_BY_WINDOW.get(window, SLOT_GEOM_BY_WINDOW["1419"])
        self.templates = {}
        for d in asset_dirs:
            m = json.loads((Path(d) / "manifest.json").read_text(
                encoding="utf-8"))
            for a in m["assets"]:
                if a["kind"] != "card_glyph":
                    continue
                w, h, rgba = _decode_png(Path(a["crop_ref"]).read_bytes())
                key = a["label"] if not a.get("color_variant") \
                    else f"{a['label']}@{a['color_variant']}"
                self.templates[key] = _tight_glyph(
                    ImageData.from_rgba_bytes(w, h, rgba))
        self.ranks = {k: v for k, v in self.templates.items()
                      if k.split("@")[0] in RANKS}
        self.suits = {k: v for k, v in self.templates.items()
                      if k.split("@")[0] in SUITS}
        # 中心大花色模板（V2ZJ）：每 suit 取最大实例（最完整字形）
        self.center_suits = {}
        if Path(center_suit_dir).exists():
            m = json.loads((Path(center_suit_dir) / "manifest.json")
                           .read_text(encoding="utf-8"))
            by_label = {}
            for a in m["assets"]:
                if a["label"] not in SUITS:
                    continue
                w, h, rgba = _decode_png(Path(a["crop_ref"]).read_bytes())
                img = _tight_glyph(ImageData.from_rgba_bytes(w, h, rgba))
                cur = by_label.get(a["label"])
                if cur is None or img.width * img.height > \
                        cur.width * cur.height:
                    by_label[a["label"]] = img
            self.center_suits = by_label
        # V2ZP：模板预处理缓存（灰度/均值/模长 init 一次）+ 家族登记
        self._prep_ranks = _prep_templates(self.ranks)
        self._prep_suits = _prep_templates(self.suits)
        self._prep_center = _prep_templates(self.center_suits)
        self._fam = {ch: ("red" if self._red_dominant(tpl) else "black")
                     for ch, tpl in
                     {**self.ranks, **self.suits,
                      **self.center_suits}.items()}
        # V-2 RC-3：模板墨宽（紧字形宽）——"10" 双字形 ≥32px vs 单字形 ≤28px
        self._tpl_w = {ch: tpl.width for ch, tpl in
                       {**self.ranks, **self.suits,
                        **self.center_suits}.items()}

    def _fam_prepped(self, prepped: dict, red_dom: bool) -> dict:
        """按颜色族过滤预处理模板（必要条件双向，与 _family_for 等价）。"""
        return {ch: v for ch, v in prepped.items()
                if (self._fam.get(ch) == "red") == red_dom}

    def _glyph_width_prepped(self, prepped: dict, ink_w: int) -> dict:
        """V-2 RC-3 墨宽几何门：字形宽与搜索墨宽同类才可比。

        ink_w ≥ TWO_GLYPH_INK_WIDTH → 仅双字形模板（"10"）；
        ink_w <  TWO_GLYPH_INK_WIDTH → 仅单字形模板。
        过滤后为空 → 退回全量（新皮肤墨宽漂移时不放大召回损失）。
        """
        want_two = ink_w >= TWO_GLYPH_INK_WIDTH
        sel = {ch: v for ch, v in prepped.items()
               if (self._tpl_w.get(ch, 0) >= TWO_GLYPH_INK_WIDTH) == want_two}
        return sel if sel else prepped

    def _ink_width(self, corner_arr: np.ndarray) -> int:
        """V-2 RC-3 墨宽测量（卡面白底约束 + 墨密度阈值）。

        列分类：卡面列 = 白像素占比 ≥0.25（桌面列 ≈0——墨迹定义 luma<190
        本身含暗桌面，须以白底存在区分卡面/桌面）；墨迹列 = 卡面列且墨
        密度 ≥0.15（卡边阴影列 0.05-0.12 剔除，字形笔画列 ≥0.2 保留）。
        墨宽 = 首末墨迹列跨度（"10" 双字形整体 38-39px vs 单字形 ≤28px；
        字形内部镂空/笔画间隙不影响首末跨度）。无有效列 → 退回搜索区宽。

        V-2b（2026-10-07，所有者批准解冻）：追加析取支 (white ≥0.15 &
        ink ≥0.30)——1876 渲染下浓墨字形列（"10" 与 T 角面）white 仅
        0.19–0.25，被 0.25 硬门排除后墨宽误测 21px：双字形 "10" 模板被
        几何门整类排除（10→Q 27 帧误读），单字形 T 模板（39px 双色版）
        同样被排除（Th→Qh，人工标注确证）。析取支要求更高墨密度（≥0.30），
        桌面/阴影列（≤0.12）不可能通过——只扩展浓墨列可见性。
        **两支取并集的最大跨度**（V-2b 审计第二轮修正）：base 优先时
        T 槽实测仍 21px（base 支在浓墨 T 角面上非空但收窄）——并集才覆盖
        "白底弱但墨浓"的字形；高墨密度门槛保证桌面列（≤0.12）不可通过。
        """
        luma = (corner_arr[..., 0].astype(np.int16) * 0.299
                + corner_arr[..., 1].astype(np.int16) * 0.587
                + corner_arr[..., 2].astype(np.int16) * 0.114)
        white = (luma > 200).mean(axis=0)
        ink = (luma < 190).mean(axis=0)
        base = np.where((white >= 0.25) & (ink >= 0.15))[0]
        alt = np.where((white >= 0.15) & (ink >= 0.30))[0]
        use = np.union1d(base, alt)
        if len(use) == 0:
            return int(corner_arr.shape[1])
        return int(use[-1] - use[0] + 1)

    def _slide_rank(self, prepped: dict, g: np.ndarray) -> tuple:
        """V-2 rank 滑动匹配（含窄条字形防护）。

        数学与 _slide_prepped 一致；差异仅在候选治理：
        1) 窄条候选（_tpl_w < NARROW_BAR_INK_WIDTH）得分
           < NARROW_BAR_MIN_SCORE → 先剔除再 argmax（防 J 吸引）；
        2) 窄条夺冠需判别余量 ≥ NARROW_BAR_MIN_MARGIN，不足 → 全拒
           （不猜、不降级次名——0 误报纪律）。
        返回 (char, score, margin) 或 (None, None, None)。

        P30-9-IL 复盘 ✗：曾在此接入"有界搜索"（缩小滑窗范围 ✓）⇒ 实测**收益 0%** ✗
        （此处输入本就是紧裁小卡块 ⇒ 缩不动 ✓）⇒ **已回退** ✓（不留没有实测收益的改动 ✓）。
        """
        char_scores = {}
        for ch, (gg, t0, tn) in prepped.items():
            th, tw = gg.shape
            if th >= g.shape[0] or tw >= g.shape[1]:
                continue
            if tn < 1e-9:
                continue
            us = 2 if th >= 55 else 1
            wins = np.lib.stride_tricks.sliding_window_view(g, (th, tw))
            sub = wins[::us, ::us]
            wm = sub.mean(axis=(2, 3), keepdims=True)
            wc = sub - wm
            wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
            num = (wc * t0).sum(axis=(2, 3))
            scores = num / np.maximum(wn * tn, 1e-9)
            abs_scores = np.abs(scores)
            idx = np.unravel_index(np.argmax(abs_scores), abs_scores.shape)
            char_scores[ch] = (float(abs_scores[idx]),
                               (int(idx[1] * us), int(idx[0] * us)))
        if not char_scores:
            return (None, None, None)

        def _narrow(ch: str) -> bool:
            return self._tpl_w.get(ch, 99) < NARROW_BAR_INK_WIDTH

        kept = {ch: v for ch, v in char_scores.items()
                if not _narrow(ch) or v[0] >= NARROW_BAR_MIN_SCORE}
        if not kept:
            return (None, None, None)
        ordered = sorted(kept.items(), key=lambda kv: -kv[1][0])
        best_ch, (best_sc, _) = ordered[0]
        second = ordered[1][1][0] if len(ordered) > 1 else 0.0
        margin = best_sc - second
        if _narrow(best_ch) and margin < NARROW_BAR_MIN_MARGIN:
            return (None, None, None)
        return best_ch, best_sc, margin

    def available(self) -> bool:
        return bool(self.ranks) and bool(self.suits)

    def coverage(self) -> dict:
        return {"ranks": sorted(self.ranks), "suits": sorted(self.suits)}

    @staticmethod
    def _red_dominant(img) -> bool:
        arr = np.frombuffer(img.data, dtype=np.uint8).reshape(
            img.height, img.width, img.channels or 4)
        r = arr[..., 0].astype(np.int16)
        g = arr[..., 1].astype(np.int16)
        b = arr[..., 2].astype(np.int16)
        red = ((r > 100) & (r > g + 40) & (r > b + 40)).mean()
        return red > 0.04

    def _family_for(self, img, templates) -> dict:
        """颜色族必要条件（双向）：红主导 → 仅红族；黑主导 → 仅黑族。"""
        red_dom = self._red_dominant(img)
        if red_dom:
            return {k: v for k, v in templates.items()
                    if self._red_dominant(v)}
        return {k: v for k, v in templates.items()
                if not self._red_dominant(v)}

    def _family(self, templates, red_only: bool):
        if not red_only:
            return templates
        return {k: v for k, v in templates.items()
                if self._red_dominant(v)}

    def _center_evidence(self, frame: np.ndarray, slot_idx: int,
                         red_dom: bool) -> dict:
        """中心大花色证据（卡片中心区域 → 族过滤滑动匹配）。

        V-2：返回体增加 gate_ok（原始门判定）与 margin——上层据
        CENTER_ONLY_MIN_SCORE 决定中心单证据能否独立支撑 FULL（0.80 严
        于原始门 0.70：0.75-0.76 的 s→c 误认总体不得再单独出 FULL 花色）。
        """
        x0 = self.slot_x0 + slot_idx * self.slot_w
        # V3F-P15：中心花色偏移按 render class 尺度比例缩放（1419 基准
        # 卡 140×177 → 中心区 x0+18/y0+92/94×78；1876 卡 120×180）
        sx = self.slot_w / 140.0
        sy = self.slot_h / 177.0
        cx = int(round(x0 + 18 * sx))
        cy = int(round(self.slot_y0 + 92 * sy))
        cw, chh = int(round(94 * sx)), int(round(78 * sy))
        fh, fw = frame.shape[0], frame.shape[1]
        if cx + cw > fw or cy + chh > fh:
            return {"suit": None, "score": None, "gate_ok": False,
                    "margin": None}
        center = frame[cy:cy + chh, cx:cx + cw]
        if center.size < 100:
            return {"suit": None, "score": None, "gate_ok": False,
                    "margin": None}
        center_img = _pad_tight(_tight_glyph(ImageData.from_rgba_bytes(
            center.shape[1], center.shape[0], center.tobytes())))
        suits = self._fam_prepped(self._prep_center or self._prep_suits,
                                  red_dom)
        g = np.asarray(to_gray(center_img).data, dtype=float)
        sn, ss, _, sm = _slide_prepped(g, suits)
        if sn is None or ss is None or ss < self.min_score \
                or sm < self.min_margin:
            return {"suit": None,
                    "score": round(ss, 4) if ss is not None else None,
                    "gate_ok": False,
                    "margin": round(sm, 4) if sm is not None else None}
        return {"suit": sn, "score": round(ss, 4), "gate_ok": True,
                "margin": round(sm, 4)}

    def recognize_slot(self, frame: np.ndarray, slot_idx: int) -> dict:
        """单卡槽（V-2 半置信分层）：rank（族过滤主搜 + 严格全族复核 +
        墨宽几何门）→ tier FULL / RANK_ONLY / UNCERTAIN。

        - FULL（status=OK）：rank 过门 + suit 双证据一致（或中心单证据
          ≥CENTER_ONLY_MIN_SCORE）——进最终牌面 + 计数；
        - RANK_ONLY：rank 过门 + suit 未定（冲突/双败/角标被禁/中心低于
          严门/复核家族矛盾）——**仅计数**（street 只需要张数），suit 不
          入最终牌面；rank 证据标准与 FULL 完全相同（0 误报纪律不降）；
        - UNCERTAIN：rank 门失败/空槽/越界（同 V-1，不猜测）。
        """
        fh, fw = frame.shape[0], frame.shape[1]
        x0 = self.slot_x0 + slot_idx * self.slot_w
        if x0 + self.slot_w > fw:
            return {"rank": None, "suit": None, "confidence": None,
                    "raw_text": None, "status": "UNCERTAIN",
                    "tier": None, "reason": "slot_oob"}
        # V2ZP 空槽白卡门：无白卡区域 → 跳过识别（确定性内容检查，
        # 结果与原 rank_gate 路径一致 = UNCERTAIN）
        slot_card = frame[self.slot_y0:self.slot_y0 + self.slot_h,
                          x0:x0 + self.slot_w]
        if slot_card.size < 100:
            return {"rank": None, "suit": None, "confidence": None,
                    "raw_text": None, "status": "UNCERTAIN",
                    "tier": None, "reason": "empty_slot"}
        luma_card = (slot_card[..., 0].astype(np.int16) * 0.299
                     + slot_card[..., 1].astype(np.int16) * 0.587
                     + slot_card[..., 2].astype(np.int16) * 0.114)
        if float((luma_card > 200).mean()) < EMPTY_SLOT_MIN_WHITE:
            return {"rank": None, "suit": None, "confidence": None,
                    "raw_text": None, "status": "UNCERTAIN",
                    "tier": None, "reason": "empty_slot"}
        corner = frame[self.slot_y0 + RANK_ZONE[1]:self.slot_y0 + RANK_ZONE[1]
                       + RANK_ZONE[3],
                       x0 + RANK_ZONE[0]:x0 + RANK_ZONE[0] + RANK_ZONE[2]]
        suit_zone = frame[self.slot_y0 + SUIT_ZONE[1]:self.slot_y0 + SUIT_ZONE[1]
                          + SUIT_ZONE[3],
                          x0 + SUIT_ZONE[0]:x0 + SUIT_ZONE[0]
                          + SUIT_ZONE[2]]
        if corner.size < 100:
            return {"rank": None, "suit": None, "confidence": None,
                    "raw_text": None, "status": "UNCERTAIN",
                    "tier": None, "reason": "corner_empty"}
        corner_img = _pad_tight(_tight_glyph(ImageData.from_rgba_bytes(
            corner.shape[1], corner.shape[0], corner.tobytes())))
        suit_img = _pad_tight(_tight_glyph(ImageData.from_rgba_bytes(
            suit_zone.shape[1], suit_zone.shape[0], suit_zone.tobytes())))
        red_dom = self._red_dominant(corner_img)
        # V-2 RC-3：墨宽几何门（"10" 双字形 vs 单字形不可互抢）——
        # 墨宽按卡面白底约束测量（桌面侵入不计）
        ink_w = self._ink_width(corner)
        ranks = self._glyph_width_prepped(
            self._fam_prepped(self._prep_ranks, red_dom), ink_w)
        suits = self._fam_prepped(self._prep_suits, red_dom)
        g = np.asarray(to_gray(corner_img).data, dtype=float)
        gs = np.asarray(to_gray(suit_img).data, dtype=float)
        # 主族搜索 + 全族复核（V-2 RC-1）：全族强证据（≥0.90/≥0.09）优先
        # 于主族弱证据——真 rank 的唯一模板变体可能在另一族（灰度 NCC 本
        # 无色别），主族缺席时弱赢家（如 5♣ 上 6@0.77）不得凭主门通过。
        rn, rs, rm = self._slide_rank(ranks, g)
        rn = rn.split("@")[0] if rn is not None else None
        primary_pass = (rs is not None and rs >= self.min_score
                        and rm >= self.min_margin)
        ranks_all = self._glyph_width_prepped(self._prep_ranks, ink_w)
        rn2, rs2, rm2 = self._slide_rank(ranks_all, g)
        rescue_pass = (rs2 is not None and rs2 >= RANK_RESCUE_MIN_SCORE
                       and rm2 >= RANK_RESCUE_MIN_MARGIN)
        rank_ok = False
        rank_rescued = False
        if rescue_pass and (not primary_pass
                            or rs2 >= rs or rn2.split("@")[0] != rn):
            rn = rn2.split("@")[0]
            rs, rm = rs2, rm2
            rank_ok = True
            rank_rescued = True
        elif primary_pass:
            rn, rs, rm = rn, rs, rm
            rank_ok = True
        cn, cs, _, cm = _slide_prepped(gs, suits)
        cn = cn.split("@")[0] if cn is not None else None
        if not rank_ok:
            return {"rank": None, "suit": None, "confidence": None,
                    "raw_text": None, "status": "UNCERTAIN",
                    "tier": None, "reason": "rank_gate",
                    "rank_score": round(rs, 4) if rs is not None else None}
        corner_ev = {"suit": cn, "score": cs} if (cn is not None
                                                  and cs is not None
                                                  and cs >= self.min_score
                                                  and cm >= self.min_margin) \
            else {"suit": None, "score": cs if cs is not None else None}
        center_ev = self._center_evidence(frame, slot_idx,
                                          self._red_dominant(corner_img))
        # ── V-2 suit 融合（分层）─────────────────────────────────────
        suit = None
        if corner_ev["suit"] is not None and center_ev.get("suit") \
                is not None:
            if corner_ev["suit"] != center_ev["suit"]:
                # P30-9-CJ：角标支配（证据强度裁决）——阈值从严，否则行为不变
                _dom_on = str(os.environ.get("BOARD_SUIT_CORNER_DOM", "1")).lower() \
                    not in ("0", "off", "false", "no", "")
                _c_sc = float(corner_ev.get("score") or 0.0)
                _k_sc = float(center_ev.get("score") or 0.0)
                if (_dom_on and _c_sc >= 0.95 and (_c_sc - _k_sc) >= 0.15
                        and _k_sc < CENTER_ONLY_MIN_SCORE):
                    return {"rank": rn, "suit": corner_ev["suit"],
                            "confidence": round(min(rs, _c_sc), 4),
                            "raw_text": f"{rn}{corner_ev['suit']}", "status": "OK",
                            "tier": TIER_FULL, "rank_rescued": rank_rescued,
                            "suit_resolved_by": "corner_dominance",
                            "evidence": {"corner": corner_ev, "center": center_ev}}
                # 冲突 → RANK_ONLY（rank 证据完好，suit 不入牌面）
                return self._rank_only(rn, rs, "suit_conflict", rank_rescued,
                                       corner_ev, center_ev)
            suit = corner_ev["suit"]
            score = round(min(corner_ev["score"], center_ev["score"]), 4)
        elif corner_ev["suit"] is not None:
            # 角标单证据：策略默认禁用（V2ZL 无独立验证正确先例）→ RANK_ONLY
            if not self.corner_only_allowed:
                # ── P30-9-IF：**人头牌的中心不是花色** ✗✗ ──────────────────
                #   实测（用户帧，真值 4♣3♥**K♥**）：4♣/3♥ 的角标与中心都匹配（0.97 ✓）；
                #   而 **K♥** 的角标 **0.991** ✓ 极高、中心却只有 **0.272** ✗ ——
                #   因为 J/Q/K 的卡面中心是**王冠/人头图案**而非花色 ✓ ⇒ 中心路径对
                #   **所有**人头牌系统性失效 ✗（此前只能降级为 RANK_ONLY ✗）。
                #   ⇒ 允许**角标高分 + 红黑颜色一致**双证据支撑 FULL ✓：
                #     * 角标 ≥0.95（远高于历史误认区间 0.75–0.76 ✗）；
                #     * 颜色校验：红花色（h/d）要求红墨主导 ✓、黑花色（s/c）要求非红 ✓。
                #   两条都满足才接受 ✓（否则维持 RANK_ONLY ✓，不降门槛 ✗）。
                _c_sc = float(corner_ev.get("score") or 0.0)
                _cs = str(corner_ev.get("suit") or "")
                _red_ok = bool(self._red_dominant(corner_img))
                _color_ok = (_red_ok if _cs in ("h", "d") else not _red_ok)
                if _c_sc >= 0.95 and _color_ok:
                    return {"rank": rn, "suit": _cs,
                            "confidence": round(min(rs, _c_sc), 4),
                            "raw_text": f"{rn}{_cs}", "status": "OK",
                            "tier": TIER_FULL, "rank_rescued": rank_rescued,
                            "suit_resolved_by": "corner_high_score_with_color",
                            "evidence": {"corner": corner_ev, "center": center_ev,
                                          "red_dominant": _red_ok}}
                return self._rank_only(rn, rs,
                                       "corner_only_suit_not_reliable",
                                       rank_rescued, corner_ev, center_ev)
            suit, score = corner_ev["suit"], corner_ev["score"]
        elif center_ev.get("suit") is not None:
            # 中心单证据：仅 ≥CENTER_ONLY_MIN_SCORE 可独立支撑 FULL
            #（0.80 严门——0.75-0.76 的 s→c 误认总体不得再单独出 FULL）
            if (center_ev.get("score") or 0) >= CENTER_ONLY_MIN_SCORE:
                suit, score = center_ev["suit"], center_ev["score"]
            else:
                return self._rank_only(rn, rs, "center_suit_below_strict_bar",
                                       rank_rescued, corner_ev, center_ev)
        else:
            # 双证据皆缺 → RANK_ONLY（rank 计数有效，suit 未定）
            return self._rank_only(rn, rs, "suit_gate", rank_rescued,
                                   corner_ev, center_ev)
        return {"rank": rn, "suit": suit, "confidence": round(min(rs, score),
                                                              4),
                "raw_text": f"{rn}{suit}", "status": "OK",
                "tier": TIER_FULL,
                "rank_rescued": rank_rescued,
                "evidence": {"corner": corner_ev, "center": center_ev}}

    @staticmethod
    def _rank_only(rank, score, reason, rescued, corner_ev, center_ev) -> dict:
        """RANK_ONLY 半置信输出：rank 可信 + suit 未定（仅计数）。"""
        return {"rank": rank, "suit": None, "confidence": None,
                "raw_text": None, "status": "RANK_ONLY",
                "tier": TIER_RANK_ONLY, "reason": reason,
                "rank_score": round(score, 4) if score is not None else None,
                "rank_rescued": rescued,
                "evidence": {"corner": corner_ev, "center": center_ev}}

    def _auto_geometry(self, frame: np.ndarray) -> dict:
        """P30-9-IF：**按帧自动选/校正槽位几何** ✓。

        实测根因：构造函数默认 `window="1419"`（槽位 X0=**664** ✗），
        而 1876 渲染的真实牌位是 **590 / 732 / 875** ✓（几何表里本来就有 `'1876': (590, …)` ✓）
        ⇒ 只要调用方没显式传 `window="1876"`，三个槽就整体错位 ~74px ✗ ⇒ 全部 `rank_gate` ✗
        （用户帧实测：牌面 4♣3♥K♥ 三张完整可见，却三张全读不出 ✗）。
        修法两步：① 按帧宽选几何 ✓；② 用**白卡斑块**实测 x0 并校正 ✓（容忍将来布局再变 ✓）。
        返回校正信息（写入 evidence ✓，可审计 ✓）。
        """
        info: dict = {"window": self.window, "slot_x0": self.slot_x0,
                       "detected_x0": None, "corrected": False}
        try:
            key = str(int(frame.shape[1]))
            if key in SLOT_GEOM_BY_WINDOW and self.window != key:
                self.window = key
                (self.slot_x0, self.slot_y0,
                 self.slot_w, self.slot_h) = SLOT_GEOM_BY_WINDOW[key]
                info.update({"window": key, "slot_x0": self.slot_x0})
        except Exception:                                      # noqa: BLE001
            pass
        # ② 白卡斑块实测（列方向白像素占比 ✓）
        try:
            band = frame[max(0, self.slot_y0 - 40): self.slot_y0 + self.slot_h + 40,
                          :, :3].astype(np.int16)
            r, g, b = band[..., 0], band[..., 1], band[..., 2]
            l = 0.299 * r + 0.587 * g + 0.114 * b
            white = (l > 150) & (r > 140) & (g > 140) & (b > 140)
            colfrac = white.mean(axis=0)
            on = colfrac > 0.35
            runs, run = [], None
            for i, v in enumerate(on):
                if v and run is None:
                    run = [i, i]
                elif v and run is not None:
                    run[1] = i
                elif run is not None:
                    if run[1] - run[0] >= max(60, self.slot_w // 2):
                        runs.append(tuple(run))
                    run = None
            if run is not None and run[1] - run[0] >= max(60, self.slot_w // 2):
                runs.append(tuple(run))
            if runs:
                x0 = int(runs[0][0])
                info["detected_x0"] = x0
                if abs(x0 - int(self.slot_x0)) > 20:
                    self.slot_x0 = x0
                    info.update({"slot_x0": x0, "corrected": True})
        except Exception:                                      # noqa: BLE001
            pass
        return info

    def recognize(self, rgba_frame: bytes, w: int, h: int,
                  frame_id: int) -> dict:
        """整板识别（V-2 分层聚合）。

        - value：FULL 卡（rank+suit 双证据）——最终牌面唯一来源；
        - countable_cards：FULL+RANK_ONLY 的 rank token（按 slot 序）——
          street 张数判定的输入口径（street 只需要几张）；
        - rank_only_cards：suit 未定的 rank（provenance 分级标注）。
        """
        frame = np.frombuffer(rgba_frame, dtype=np.uint8).reshape(h, w, 4)
        geo = self._auto_geometry(frame)                       # P30-9-IF ✓
        cards = []
        for i in range(5):
            res = self.recognize_slot(frame, i)
            res["slot_index"] = i
            res["source"] = self.name
            res["evidence_frame_id"] = frame_id
            cards.append(res)
        ok = any(c["status"] == "OK" for c in cards)
        full = [(c["slot_index"], c["raw_text"]) for c in cards
                if c["status"] == "OK"]
        countable = [(c["slot_index"], c["rank"]) for c in cards
                     if c["status"] in ("OK", "RANK_ONLY")]
        return {"cards": cards, "status": "OK" if ok else "UNCERTAIN",
                "value": [t for _, t in sorted(full)],
                "countable_cards": [t for _, t in sorted(countable)],
                "geometry": geo,                               # P30-9-IF：几何来源可审计 ✓
                "rank_only_cards": [c["rank"] for c in cards
                                    if c["status"] == "RANK_ONLY"]}
