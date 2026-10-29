"""V3F-P5R — Stack Recognition Foundation（当前帧纯识别器）。

- 逐座：金文本定位（座锚点/裁剪窗自适应）→ 字符分割 → 数字模板 NCC
  （模板来自 pot 家族 {,0,2,3,6,7,8} + v2p5r 真实提取字形（如有））→
  有序组装 → 文本规范化（千位/K/M/小数）→ int 值；
- 语义：UNKNOWN 原样（未识别 ≠ 0/前值/默认）；1419 裁剪座 → UNKNOWN；
- 独立：不依赖 Position/Board/Street/Hero Runtime；不改 pot 识别器语义。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ...realtime.image import ImageData, to_gray
from ..geometry import connected_components

OCCUPIED, EMPTY, UNKNOWN = "OCCUPIED", "EMPTY", "UNKNOWN"

DIGIT_DIRS = ("data/vision2/coinpoker_assets_v2zb",
              "data/vision2/coinpoker_assets_v2zy",
              "data/vision2/coinpoker_assets_v2p5r")

SEAT_ANCHORS_1419 = {
    "S1": (255, 320), "S3": (1000, 200), "S4": (300, 880), "S5": (1000, 1000),
}
SEAT_ANCHORS_1848 = {
    "S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
    "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160),
}
MAX_ASSOC_DIST = 320.0
MIN_SCORE = 0.7
MIN_MARGIN = 0.08

PLAIN_RE = re.compile(r"^\d{1,3}(,\d{3})*$")
KM_RE = re.compile(r"^(\d+)(?:\.(\d+))?([KM])$")
#: V-5 微级别裸小数（stack pill『9.64』；单位=货币单位 float——与 V-4
#: header `_to_num` 同语义选择：bet/pot/header 观测同为货币单位，
#: 美分换算会使 stack=964 与 bet=0.05 差 100 倍）。既有 PLAIN/K/M
#: 返回值零变化（int）。
DEC_RE = re.compile(r"^(\d+)\.(\d+)$")


def parse_stack_text(raw: str | None) -> int | float | None:
    """文本 → 数值（仅真实识别成功后调用）。
    '201,000'→201000 · '2.28M'→2280000 · '1.3M'→1300000 · '999.5K'→999500 ·
    '9.64'→9.64（V-5 裸小数 float，货币单位）；
    非法/未知 → None（不猜测）。"""
    if not raw:
        return None
    s = raw.strip().replace("，", ",")
    if PLAIN_RE.match(s):
        return int(s.replace(",", ""))
    m = KM_RE.match(s)
    if m:
        base = int(m.group(1))
        frac_str = m.group(2) or ""
        mult = 1000 if m.group(3) == "K" else 1000000
        frac = int(frac_str) / (10 ** len(frac_str)) if frac_str else 0.0
        return int(round((base + frac) * mult))
    m = DEC_RE.match(s)
    if m:
        return float(s)
    return None


# ── V3F-P26-B：1876 Stack 专用路径 ────────────────────────────────────
#: 每座 Stack 行 ROI（P26-A 实测 tight bbox + 边距；布局常量——与
#: SEAT_ROIS_1419 / BET_ROIS_1876 同类；非 frame-id 特判）
SEAT_STACK_ROIS_1876 = {
    "S1": (165, 413, 360, 485),
    "S2": (845, 298, 1035, 370),
    "S3": (1520, 413, 1715, 485),
    "S4": (125, 978, 320, 1050),
    "S5": (1560, 978, 1745, 1050),
    "S6": (838, 1236, 1040, 1308),
}

#: 1876 stack glyph 模板目录（P26-B 从 6 个 GT 帧 stack 行提取，
#: GT provenance 入 manifest；glyph 高≈34px——与 pot 30px 不同，单独库）
STACK_ASSET_DIR_1876 = "data/vision2/coinpoker_assets_v1876stack"
#: V-5 微级别 stack 字形目录（kind=stack_glyph_micro 隔离装载入口保留）。
#: **默认不并入** STACK_ASSET_DIRS_1876——实测微桌 stack pill 字形与 K 桌
#: 同尺度（微 34-36px vs 基础 34px，渲染差异在 pill 底亮度非字号），双实
#: 例并置使跨标签 NCC margin 坍缩（K 帧 POT_CHANGED_108 实测 4/6 座
#: glyph_gate_failed：margin 0.01-0.066 < 0.08 冻结门）——零回退红线优先，
#: 微桌 stack 识别由判据自适应 + 小数解析闭合（≥6 座全 RECOGNIZED）。
#: 裁制资产（36 实例，proof_sheet 目视核）留库供独立微几何域启用。
STACK_ASSET_DIR_1876_MICRO = "data/vision2/coinpoker_assets_v3c_micro"
STACK_ASSET_DIRS_1876 = (STACK_ASSET_DIR_1876,)
STACK_MICRO_KIND = "stack_glyph_micro"
#: 尺度一致性门（V-5）：模板高 ≥ 0.75×候选高才参与 NCC（防微模板在
#: 大字形内部伪匹配——'0' 模板配 '8' 内环拉低 margin）
STACK_NCC_H_RATIO_MIN = 0.75

#: background-relative gold 判据（P26-A 冻结）：金文本金-橙饱和度
#: (r>b+30, g>=b+18, r>=g*0.9) + 相对亮度（NORMAL/DIM 同判据）
#: V-5（Round16）微级别自适应：RGB 相对差门按实测放宽到 B 档
#: (r>b+15, g>=b+8)——2,338 帧全量对照：负样本区（felt/bet/board/dealer）
#: B 档 gold 像素 =0（0 误报论证），暗态金字捕获 +15~24%（见
#: docs/V5_MICRO_ADAPT_REPORT.md §2）。基础档常量保留（历史语义/测试引用）。
STACK_GOLD_R_DELTA = 30
STACK_GOLD_G_DELTA = 18
STACK_GOLD_LUMA_RATIO = 1.6
#: V-5 微级别自适应档（实测论证参数化，替代任务书建议公式
#: max(1.6, 72/max(bg,20))——该式在语料微帧（bg≈10-25、文字 l p50≈41-99）
#: 上把阈值抬到 72，暗态金字过半被拒（gold 反降），与修复方向相反；见报告 §2）：
#: - bg ≤ STACK_GOLD_DIM_BG_MAX(60)：pill 暗底域，倍率降至 1.3 +
#:   绝对下限 28（防近黑底噪声：bg≤17 时下限严于旧 1.6×bg，其余域宽于旧门
#:   ——旧判据在该域从不满捕获，语料零回退由全量 A/B 证明）；
#: - bg > 60：亮底/遮挡域（浏览器覆盖帧 bg≈213-248），冻结 1.6×bg 不变
#:   （该域旧门即 0 捕获，放宽会引入覆盖 UI 伪金）。
STACK_GOLD_LUMA_RATIO_DIM = 1.3
STACK_GOLD_DIM_BG_MAX = 60
STACK_GOLD_LUMA_FLOOR = 28
#: V-5.1（Round17）暗域"上限带 + 严格暖门"：`thr = max(bg*1.3, 28)` 保留，
#: 但 (STACK_GOLD_LUMA_CEIL, thr] 亮度带内的像素改用**文本级饱和度门**准入。
#:
#: 机制（用户实况失效链）：微桌 hero pill bg≈45、暗金字 l≈41-99——V-5 式
#: 阈在 bg=45 给 58.5，高于暗金字主体（p50≈41）→ 只捕最亮尾（用户实测
#: 441px，5 字形完整需 ≈1,200-1,400px）→ 组件碎裂 → glyph 门拒 →
#: stack/pot UNKNOWN。上限 36 使暗金字主体（p10≈38-41）进入捕获域。
#:
#: 为什么上限带不能沿用 B 档 RGB：语料暗金字帧（POT_CHANGED_192/234/
#: SCENE_15）实测 pill 底为**暖色渐变**（r-b p50=14-22、g-b p50=13-16，
#: B 档 r>b+15 边缘通过），底亮随场景抬到 40+ 时整片暖底会以 B 档过门
#: → 掩码连成巨块 → 组件断裂。文本级严格暖门（r>b+26 & g≥b+18 &
#: r≥0.95g）与底色实测分离：文本 r-b p10=30-34 / g-b p10=21-24 全过，
#: 暖底 r-b p50≤22 / g-b p50≤16 全拒（.tmp/v51_probe_warmsep 输出）。
#: thr 以上亮度域仍用 B 档（亮金字核，V-5 语义不变）。
#:
#: 语料影响面：bg≤27.7 域 thr≤36 → 上限带为空 → 逐像素零变化；
#: (27.7,60] 域 = 空桌绿毡（RGB 门 0 像素，g>r 恒拒）+ 3 个高亮 pill 帧
#: （金核 l≈197 在 thr 域，上限带增量仅 AA 边缘）——全语料 A/B 识别
#: 逐字段零变化（.tmp/v51_regress_c*.jsonl）。
STACK_GOLD_LUMA_CEIL = 36
STACK_GOLD_R_DELTA_STRICT = 26
STACK_GOLD_G_DELTA_STRICT = 18
STACK_GOLD_RG_RATIO_STRICT = 0.95
STACK_GOLD_R_DELTA_DIM = 15
STACK_GOLD_G_DELTA_DIM = 8

#: 几何 token（P26-A 实测）：点 h≤9/w≤10；逗号 h≤17/w≤10；M w>24
STACK_DOT_H_MAX = 9
STACK_COMMA_H_MAX = 17
STACK_TOKEN_W_MAX = 10
STACK_M_W_MIN = 25


def stack_gold_1876(roi: np.ndarray) -> tuple:
    """背景相对 gold 掩码（P26-A 冻结判据 + V-5 微级别自适应）+ 背景中位 luma。

    亮度/色度门（V-5 分档 + V-5.1 上限带严格暖门，论证见模块头）：
      bg > 60（亮底/遮挡域，冻结）→ B 档(30/18) & l > bg*1.6；
      bg ≤ 60（暗 pill 域）→ thr = max(bg*1.3, 28)：
        l > thr 域用 B 档(15/8)——V-5 语义不变；
        (36, thr] 上限带用严格暖门(26/18/0.95)——暗金字主体准入
        （bg≤27.7 时该带为空 → 逐像素零变化）。
    实测对照：暗金字帧文本 r-b p10=30-34/g-b p10=21-24 全过严格门，
    暖 pill 底 r-b p50≤22/g-b p50≤16 全拒；绿毡域 RGB 门 0 像素
    （docs/V51_STACK_POT_REPORT.md §判据）。用户实况形态（pill bg≈45、
    暗金字 l≈41-99）：V-5 门 58.5 只捕最亮尾（用户实测 441px）→
    V-5.1 上限带全捕获。

    空/截断 ROI（出界帧、不完整裁剪）→ 全零掩码（UNKNOWN 语义，不猜）。
    """
    if roi.size == 0 or roi.shape[0] < 4 or roi.shape[1] < 4:
        zeros = np.zeros((max(0, roi.shape[0]), max(0, roi.shape[1])),
                         dtype=np.uint8)
        return zeros, 0.0
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    bg = float(np.median(l))
    if bg <= 0.0:
        return np.zeros_like(l, dtype=np.uint8), bg
    if bg > STACK_GOLD_DIM_BG_MAX:
        # 亮底/遮挡域：V-5 冻结语义（浏览器覆盖帧 bg≈213-248 旧门即 0
        # 捕获，放宽会引入覆盖 UI 伪金）。
        mask = ((r > b + STACK_GOLD_R_DELTA)
                & (g >= b + STACK_GOLD_G_DELTA)
                & (r >= g * 0.9)
                & (l > bg * STACK_GOLD_LUMA_RATIO)).astype(np.uint8)
        return mask, bg
    # 暗 pill 域：thr 以上 = B 档（V-5 不变）；(CEIL, thr] 上限带 =
    # 严格暖门（暗金字准入；thr≤CEIL 时带为空 → 与 V-5 逐像素同）。
    thr = max(bg * STACK_GOLD_LUMA_RATIO_DIM, STACK_GOLD_LUMA_FLOOR)
    color_dim = (r > b + STACK_GOLD_R_DELTA_DIM) \
        & (g >= b + STACK_GOLD_G_DELTA_DIM) & (r >= g * 0.9)
    color_strict = (r > b + STACK_GOLD_R_DELTA_STRICT) \
        & (g >= b + STACK_GOLD_G_DELTA_STRICT) \
        & (r >= g * STACK_GOLD_RG_RATIO_STRICT)
    band_hi = l > thr
    band_mid = (l > STACK_GOLD_LUMA_CEIL) & ~band_hi
    mask = ((color_dim & band_hi) | (color_strict & band_mid)).astype(
        np.uint8)
    return mask, bg


def stack_comps_8conn(mask: np.ndarray, min_area: int = 4) -> list:
    """8-连通组件（P26-A 已验证：34/34 组件数==期望字符数，无粘连/断裂）。"""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] \
                                and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(cells) < min_area:
                continue
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            out.append((min(xs), min(ys), max(xs) - min(xs) + 1,
                        max(ys) - min(ys) + 1, len(cells)))
    return sorted(out)


def _load_1876_templates(asset_dir: str = STACK_ASSET_DIR_1876,
                         kind: str | None = None) -> dict:
    """1876 stack glyph 模板装载（alpha 掩码二值 + 预处理）。

    V-5：kind 非 None 时仅装载该 kind 资产（微级别混载目录隔离——
    v3c_micro 同时存 header/pot/bet/stack 字形，互不混入）。
    """
    mp = Path(asset_dir) / "manifest.json"
    if not mp.exists():
        return {}
    m = json.loads(mp.read_text(encoding="utf-8"))
    prep: dict = {}
    for a in m.get("assets", []):
        lab = a.get("label")
        if lab not in "0123456789M":
            continue
        if kind is not None and a.get("kind") != kind:
            continue
        p = Path(a["crop_ref"])
        if not p.exists():
            p = Path(asset_dir) / Path(a["crop_ref"]).name
        if not p.exists():
            continue
        from poker_decision_ai.vision2_platform.png_decode import decode_png
        w, h, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        g = (arr[..., 3] > 128).astype(np.float64)
        ys, xs = np.where(g > 0)
        if len(xs) < 4:
            continue
        g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        t0 = g - g.mean()
        tn = float(np.sqrt((t0 * t0).sum()))
        if tn > 1e-9:
            prep.setdefault(lab, []).append((g, t0, tn))
    return prep


def _load_1876_template_dirs(asset_dirs=STACK_ASSET_DIRS_1876) -> dict:
    """多目录 stack 模板装载（V-5：基础库 + 微级别库尾追加；同 label
    多实例并存——NCC 逐实例取 max，标签集不变）。"""
    prep: dict = {}
    for d in asset_dirs:
        for lab, insts in _load_1876_templates(
                d, kind=STACK_MICRO_KIND if d == STACK_ASSET_DIR_1876_MICRO
                else None).items():
            prep.setdefault(lab, []).extend(insts)
    return prep


def _ncc_1876(g: np.ndarray, prep: dict) -> tuple:
    """候选二值（pad 2——与冻结链 _pad_tight 同因：等尺寸模板不被跳过）
    vs 全部模板 → (best_label, score, margin)。

    V-5 尺度一致性门：模板高 < 0.75×候选高 → 跳过（微模板在大字形内部
    的伪匹配会吞掉 margin——'0' 模板配 '8' 内环；同 1419 路径几何一致性
    门同因。既有单尺度库全通过，零行为变化）。

    ⚠️ P30-9-IL 复盘 ✗：曾在此接入"有界搜索"（缩小滑窗范围 ✓）提速，
    实测**收益 0%** ✗（此处输入本就是紧裁小块 ⇒ 缩不动 ✓）⇒ **已回退** ✓
    （纪律：没有实测收益的改动不留 ✓；观测 A/B 20/20 帧一致 ✓ 证明回退无副作用 ✓）。
    """
    gp = np.pad(g, 2)
    cand_h = g.shape[0]
    h_min = STACK_NCC_H_RATIO_MIN * cand_h
    best, bsc, scores = None, -1.0, {}
    for ch, insts in prep.items():
        sc = -1.0
        for (t, t0, tn) in insts:
            th, tw = t.shape
            if th >= gp.shape[0] or tw >= gp.shape[1] or tn < 1e-9:
                continue
            if th < h_min:
                continue
            wins = np.lib.stride_tricks.sliding_window_view(gp, (th, tw))
            wm = wins.mean(axis=(2, 3), keepdims=True)
            wc = wins - wm
            wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
            s = np.abs((wc * t0).sum(axis=(2, 3))
                       / np.maximum(wn * tn, 1e-9))
            v = float(s.max())
            if v > sc:
                sc = v
        scores[ch] = sc
        if sc > bsc:
            best, bsc = ch, sc
    if best is None:
        return None, None, None
    ranked = sorted(scores.values(), reverse=True)
    margin = bsc - (ranked[1] if len(ranked) > 1 else 0.0)
    return best, bsc, margin


class StackRecognizer1876:
    """1876 Stack 专用识别器（V3F-P26-B）。

    - per-seat 固定 ROI → 背景相对 gold 掩码 → 8-conn 组件 → 几何 token
      （点/逗号/M）+ 数字 NCC（GT-provenance 模板，0.7/0.08 冻结门）→
      组装 → parse_stack_text（复用冻结解析）；
    - UNKNOWN-first：无组件/任何组件不可靠/格式非法 → {stack: None,
      status: UNKNOWN}（不猜测被遮挡/未知值）；
    - 独立：不引用 Pot/Bet/Status/Position；无 previous/cache；
    - 1419 路径零改动（本类仅 1876 用）。
    """

    name = "coinpoker_stack_v3f_p26b_1876"

    def __init__(self, asset_dirs=STACK_ASSET_DIRS_1876):
        self.rois = SEAT_STACK_ROIS_1876
        self._prep = _load_1876_template_dirs(asset_dirs)

    def _recognize_seat(self, frame: np.ndarray, seat: str) -> dict:
        roi_bb = self.rois[seat]
        x0, y0, x1, y1 = roi_bb
        roi = frame[y0:y1, x0:x1]
        mask, bg = stack_gold_1876(roi)
        comps = stack_comps_8conn(mask)
        # ── V-5 精度恢复三件套（RGB 门 B 档放宽的伴随修正；全语料 K 簇
        #    1,889 帧对照零回退后冻结）─────────────────────────────────
        # a) 底缘辉光线剥离：B 档在个别 K 帧 pill 底缘引入全宽 AA 辉光线
        #    （实测 83×2-3px），把逗号与相邻数字桥接成宽组件（'492,332'
        #    → 逗号丢失）。中窄高组件（10≤h≤24 且 w≥20——数字/M 高 34-36
        #    不在内）底部逐行剥离"行宽 ≥90% 组件宽"的行（≤4 行）后重割。
        changed = False
        for (cx, cy, cw, chh, _a) in comps:
            if cw >= 20 and 10 <= chh <= 24:
                trim = 0
                for t in range(min(5, chh - 8)):
                    if mask[cy + chh - 1 - t, cx:cx + cw].sum() >= 0.4 * cw:
                        trim += 1
                    else:
                        break
                if trim:
                    mask[cy + chh - trim:cy + chh, cx:cx + cw] = 0
                    changed = True
        if changed:
            comps = stack_comps_8conn(mask)
        # b) 数字带包含：带外 AA 碎片（'217,168' 门失败例）剔除——带 = 高≥20
        #    组件包络 [top-6, bot+9]（下松量容纳逗号下伸实测 7px）。
        big = [c for c in comps if c[3] >= 20]
        if big:
            top = min(c[1] for c in big)
            bot = max(c[1] + c[3] for c in big)
            comps = [c for c in comps if c[3] >= 20
                     or (c[1] >= top - 6 and c[1] + c[3] <= bot + 9)]
        # c) 扁平碎片剔除：h≤6 且 w>12 非任何合法 token（点≤10w/逗号≤10w/
        #    数字≥19w 且 h≥34）——'.2.33M' 1px 伪点与 'M214,848' 底缘细条；
        #    1px 高碎片（h<2）同样剔除。
        comps = [c for c in comps
                 if c[3] >= 2 and not (c[3] <= 6 and c[2] > 12)]
        if not comps:
            return {"stack": None, "raw_text": None, "status": "UNKNOWN",
                    "reason": "no_gold_components"}
        if not self._prep:
            return {"stack": None, "raw_text": None, "status": "UNKNOWN",
                    "reason": "no_templates"}
        chars = []
        scores = []
        for (cx, cy, cw, chh, _area) in comps:
            sub = mask[cy:cy + chh, cx:cx + cw]
            if chh <= STACK_DOT_H_MAX and cw <= STACK_TOKEN_W_MAX:
                chars.append(".")
                scores.append(1.0)
                continue
            if chh <= STACK_COMMA_H_MAX and cw <= STACK_TOKEN_W_MAX:
                chars.append(",")
                scores.append(1.0)
                continue
            if cw >= STACK_M_W_MIN:
                chars.append("M")
                scores.append(1.0)
                continue
            g = sub.astype(np.float64)
            ch, sc, margin = _ncc_1876(g, self._prep)
            if ch is None or sc is None or sc < 0.7 or margin < 0.08:
                return {"stack": None, "raw_text": None, "status": "UNKNOWN",
                        "reason": "glyph_gate_failed",
                        "evidence": {"score": sc, "margin": margin}}
            chars.append(ch)
            scores.append(round(margin, 3))
        raw = "".join(chars)
        value = parse_stack_text(raw)
        if value is None:
            return {"stack": None, "raw_text": raw, "status": "UNKNOWN",
                    "reason": "stack_format_rejected"}
        return {"stack": value, "raw_text": raw, "status": "RECOGNIZED",
                "reason": "stack_glyph_assembly",
                "confidence": round(min(scores), 3)}

    def recognize(self, frame: np.ndarray) -> dict:
        return {seat: self._recognize_seat(frame, seat)
                for seat in sorted(self.rois)}


def _cut_low_ink(cols, min_w: int = 3) -> list:
    """递归低墨切割（泛化）：列墨 <30% 峰值为切点，递归处理所得的块。
    用于相邻字符粘连（含内部逗号）的通用断开。"""
    if cols is None or len(cols) < 2 * min_w + 1:
        return []
    peak = max(1.0, float(cols.max()))
    cuts = [i for i in range(min_w, len(cols) - min_w)
            if cols[i] < 0.3 * peak]
    if not cuts:
        return []
    pieces = []
    prev = 0
    for c in cuts + [len(cols)]:
        if c - prev >= min_w:
            sub = cols[prev:c]
            if sub.max() >= 0.45 * peak:
                inner_p = _cut_low_ink(sub, min_w)
                if inner_p:
                    pieces.extend((prev + a, prev + b) for a, b in inner_p)
                else:
                    pieces.append((prev, c))
        prev = c
    return pieces


def _bright_tight(img: ImageData) -> ImageData:
    g = np.asarray(to_gray(img).data, dtype=float)
    ys, xs = np.where(g > 120)
    if len(ys) < 4:
        return img
    arr = np.frombuffer(img.data, dtype=np.uint8).reshape(
        img.height, img.width, img.channels or 4)
    return ImageData.from_rgba_bytes(
        xs.max() - xs.min() + 1, ys.max() - ys.min() + 1,
        arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].tobytes())


def _pad_tight(img: ImageData, pad: int = 8) -> ImageData:
    arr = np.pad(np.frombuffer(img.data, dtype=np.uint8).reshape(
        img.height, img.width, img.channels or 4),
        ((pad, pad), (pad, pad), (0, 0)))
    return ImageData.from_rgba_bytes(arr.shape[1], arr.shape[0],
                                     arr.tobytes())


class StackRecognizer:
    """当前帧 → per-seat {raw_text, value, status, confidence}。"""

    name = "coinpoker_stack_v2p5r"

    def __init__(self, asset_dirs=DIGIT_DIRS, window: str = "1419",
                 text_layout: str | None =
                 "data/vision2/stack_text_layout_v1.json"):
        self.digits = {}                  # label -> [prepped (g, t0, tn)]
        for d in asset_dirs:
            mp = Path(d) / "manifest.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text(encoding="utf-8"))
            for a in m.get("assets", []):
                lab = a.get("label")
                if lab not in "0123456789,.KM":
                    continue
                p = Path(a["crop_ref"])
                if not p.exists():
                    continue
                try:
                    from poker_decision_ai.vision2_platform import decode_png
                    w2, h2, rgba = decode_png(p.read_bytes())
                except Exception:
                    continue
                img = _bright_tight(ImageData.from_rgba_bytes(w2, h2, rgba))
                if lab not in ",.KM" and img.width < 8:
                    continue
                # 装载尺寸带（数据质量清洗——P5T-5）：堆栈尺度证据带 [5,45]px
                # （校准源字高 28-34；剔除 >45 跨尺度伪影（58/66 类）与 M
                # 残片窄带；不改变任何识别门限）
                if not (5 <= img.height <= 45):
                    continue
                if lab in "MK" and img.width < 8:
                    continue
                g = np.asarray(to_gray(img).data, dtype=float)
                t0 = g - g.mean()
                tn = float(np.sqrt((t0 * t0).sum()))
                self.digits.setdefault(lab, []).append((g, t0, tn))
        self.anchors = (SEAT_ANCHORS_1848 if window == "1848"
                        else SEAT_ANCHORS_1419)
        self.cropped = () if window == "1848" else ("S2", "S6")
        self.window = window
        self.text_windows = {}
        if text_layout and Path(text_layout).exists():
            tl = json.loads(Path(text_layout).read_text(encoding="utf-8"))
            for k, v in tl.get("text_windows", {}).items():
                if isinstance(v, (list, tuple)) and len(v) >= 4:
                    self.text_windows[k] = {"bbox": list(v[:4]),
                                            "split_thr": int(v[4])
                                            if len(v) > 4 else 2}

    def _gold_mask(self, frame: np.ndarray) -> np.ndarray:
        # 与 P5S 校准提取的权威掩码一致（一致性，非阈值变更）
        r = frame[..., 0].astype(np.int16)
        g = frame[..., 1].astype(np.int16)
        b = frame[..., 2].astype(np.int16)
        l = r * 0.299 + g * 0.587 + b * 0.114
        return ((r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (l > 90)
                ).astype(np.uint8)

    def _split(self, mask: np.ndarray) -> list:
        """列投影分割：抗锯齿桥断阈值（≥2 有效列）+ 1px 间隙合并
        （逗号/数字粘连稳定化——P5T 修复；非阈值/门限降低）。"""
        cols = mask.sum(axis=0)
        active = cols >= 2
        comps, run = [], None
        for i, a in enumerate(active):
            if a and run is None:
                run = i
            elif not a and run is not None:
                comps.append([run, i])
                run = None
        if run is not None:
            comps.append([run, len(active)])
        i = 0
        while i < len(comps) - 1:
            if comps[i + 1][0] - comps[i][1] <= 1:
                comps[i][1] = comps[i + 1][1]
                comps.pop(i + 1)
            else:
                i += 1
        return [tuple(c) for c in comps]

    def _match_char(self, g: np.ndarray,
                    top_ratio: float = 0.0) -> tuple:
        best, bsc, best_scores = None, -1.0, {}
        for ch, insts in self.digits.items():
            sc_ch = -1.0
            for (t, t0, tn) in insts:
                th, tw = t.shape
                if th >= g.shape[0] or tw >= g.shape[1] or tn < 1e-9:
                    continue
                # 几何一致性门：模板高度必须 ≥ 候选高度 55%（点/逗号区分；
                # 静态几何约束，非阈值降低——防第 5 个字符 '.'/'，'混淆）
                if th < 0.55 * g.shape[0]:
                    continue
                # 宽度一致性门（精度）：模板宽 ≥ 候选宽 55%——防 '4'/'1'
                # 竖直笔画混淆（'1'(11px) 不能匹配 '4'(20px+)）
                if tw < 0.55 * g.shape[1]:
                    continue
                # 点/逗号格式互斥：'.' 仅匹配小字形（<12px 高）；',' 不匹配
                # 极小字形（<10px 高）——独立于数字化，K/M 与千位格式天然区分
                if ch == "." and g.shape[0] >= 12:
                    continue
                if ch == "," and g.shape[0] < 10:
                    continue
                # 底部锚定门（实验 P5T-4 发现真逗号顶部含墨点——未正确定义；
                # 禁用保留位（top_ratio > 5.0 永不触发）——不引入伪规则）
                if ch == "," and top_ratio > 5.0:
                    continue
                wins = np.lib.stride_tricks.sliding_window_view(g,
                                                               (th, tw))
                wm = wins.mean(axis=(2, 3), keepdims=True)
                wc = wins - wm
                wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
                scores = np.abs((wc * t0).sum(axis=(2, 3))
                                / np.maximum(wn * tn, 1e-9))
                sc = float(scores.max())
                if sc > sc_ch:
                    sc_ch = sc
            best_scores[ch] = sc_ch
            if sc_ch > bsc:
                best, bsc = ch, sc_ch
        if best is None:
            return None, None
        ranked = sorted(best_scores.values(), reverse=True)
        margin = bsc - (ranked[1] if len(ranked) > 1 else 0.0)
        return best, margin

    def _read_stack_text(self, frame: np.ndarray, cx: int, cy: int) -> dict:
        """金文本定位 → 字符序列 → 组装文本。返回 {raw, status, confidence}。"""
        x0 = max(0, cx - 170)
        y0 = max(0, cy - 40)
        roi = frame[y0:cy + 50, x0:cx + 170]
        if roi.size == 0:
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0}
        gold = self._gold_mask(roi)
        ys, xs = np.where(gold > 0)
        if len(ys) < 30:
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0}
        inner = gold[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        comps = self._split(inner)
        if not comps:
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0}
        chars, scores = [], []
        for (cs, ce) in comps:
            seg = inner[:, cs:ce]
            rws = np.where(seg.sum(axis=1) > 0)[0]
            if len(rws) < 6:
                continue
            sub = seg[rws[0]:rws[-1] + 1]
            img = _pad_tight(ImageData.from_rgba_bytes(
                sub.shape[1], sub.shape[0],
                np.stack([sub * 255] * 4, axis=-1).astype(np.uint8).tobytes()),
                pad=2)
            g = np.asarray(to_gray(img).data, dtype=float) / 255.0
            top_h = max(1, int(sub.shape[0] * 0.4))
            top_ink = float(sub[:top_h].sum()) / max(1.0,
                                                     float(sub.sum()))
            ch, margin = self._match_char(g, top_ratio=top_ink)
            if ch is None or margin is None:
                continue
            chars.append(ch)
            scores.append(round(margin, 3))
        raw = "".join(chars)
        conf = round(min(scores) if scores else 0.0, 3)
        # 粘连拒绝（精度）：掩码宽≥45 却只有 <4 字符 → 分割明显失败 → UNKNOWN
        if len(chars) < 4 and inner.shape[1] >= 45:
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0}
        if not fresh_ok(raw, scores):
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0}
        return {"raw": raw, "status": "RECOGNIZED", "confidence": conf}

    def recognize_in_region(self, frame: np.ndarray,
                            region: tuple) -> dict:
        """raw-pixel-first 区域识别（P5T-8 工作流）：座级大区域 →
        行带投影发现文本行 → 最优带识别。

        - 区域 = 座位几何先验（table-stable）；文本窗由 raw 像素推导
          （无人工坐标/视图链）；
        - 最优带 = 字符分割后经门控产生 RECOGNIZED 的带（含格式门）；
        - 无带可靠 → UNKNOWN。
        """
        x0, y0, x1, y1 = region
        roi = frame[y0:y1, x0:x1]
        gold = self._gold_mask(roi)
        rows = gold.sum(axis=1)
        bands, run = [], None
        for i, v in enumerate(rows):
            if v > 2 and run is None:
                run = i
            elif v <= 2 and run is not None:
                bands.append((run, i))
                run = None
        if run is not None:
            bands.append((run, len(rows)))
        bands = [b for b in bands if b[1] - b[0] >= 10]
        best = None
        for (bs, be) in bands:
            seg = gold[bs:be]
            ys, xs = np.where(seg > 0)
            if len(ys) < 30:
                continue
            inner = seg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            read = self._chars_from_mask(roi, inner, ys + bs, xs)
            if read["status"] == "RECOGNIZED":
                # 最长串优先（最佳证据带）：完整数值带（7 字符）胜过碎片带
                if best is None or len(read["raw"]) > len(best["raw"]):
                    best = read
        if best is None:
            return {"raw_text": None, "value": None, "status": "UNKNOWN",
                    "reason": "no_reliable_band"}
        # 合理性门（泛化）：真实 stack 值 ≥4 字符（'1.3M' 最短）——
        # 1-3 字符=片段/标签伪值 → 拒（非值级硬编码）
        if len(best["raw"]) < 4:
            return {"raw_text": None, "value": None, "status": "UNKNOWN",
                    "reason": "too_short_fragment"}
        return {"raw_text": best["raw"], "value": parse_stack_text(best["raw"]),
                "status": "RECOGNIZED", "confidence": best["confidence"],
                "reason": "raw_band_discovered"}

    def recognize_tight(self, frame: np.ndarray, bbox: tuple) -> dict:
        """紧致 bbox 识别（公开现有窗口路径的薄入口——V3F-P8R）。

        - 单行 stack 文本的紧致 region（无行带切分——window 路径逻辑）；
        - 零识别核心算法改动（仅委托 _read_window）。
        """
        read = self._read_window(frame, {"bbox": list(bbox[:4]),
                                         "split_thr": 2})
        value = parse_stack_text(read["raw"]) if read["status"] == \
            "RECOGNIZED" else None
        return {"raw_text": read["raw"], "value": value,
                "status": read["status"], "confidence": read["confidence"],
                "reason": read.get("reason", "stack_text_recognized")
                if read["status"] == "RECOGNIZED" else "insufficient_evidence"}

    def recognize_in_window(self, frame: np.ndarray,
                            window_key: str) -> dict:
        """显式（源前缀）校准窗识别——P5T 逐源闭环验证入口；
        帧→窗派发属 consumer 侧（StackObserver 范围，本阶段不实现）。"""
        win = self.text_windows.get(window_key)
        if win is None:
            return {"raw_text": None, "value": None, "status": "UNKNOWN",
                    "reason": "no_window"}
        read = self._read_window(frame, win)
        value = parse_stack_text(read["raw"]) if read["status"] == \
            "RECOGNIZED" else None
        return {"raw_text": read["raw"], "value": value,
                "status": read["status"], "confidence": read["confidence"],
                "reason": "stack_text_recognized"
                if read["status"] == "RECOGNIZED"
                else "insufficient_evidence"}

    def recognize(self, frame: np.ndarray) -> dict:
        result = {}
        golds = SeatLikeGoldBlobs(frame)
        for seat in list(self.anchors) + list(self.cropped):
            if seat in self.cropped:
                result[seat] = {"raw_text": None, "value": None,
                                "status": "UNKNOWN",
                                "reason": "cropped_window"}
                continue
            ax, ay = self.anchors[seat]
            best = None
            for (bx, by) in golds:
                d = float(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
                if d <= MAX_ASSOC_DIST and (best is None or d < best[1]):
                    best = ((bx, by), d)
            if best is None:
                result[seat] = {"raw_text": None, "value": None,
                                "status": "UNKNOWN",
                                "reason": "no_stack_text_at_seat"}
                continue
            read = self._read_stack_text(frame, best[0][0], best[0][1])
            value = parse_stack_text(read["raw"]) if read["status"] == \
                "RECOGNIZED" else None
            result[seat] = {"raw_text": read["raw"], "value": value,
                            "status": read["status"],
                            "confidence": read["confidence"],
                            "reason": "stack_text_recognized"
                            if read["status"] == "RECOGNIZED"
                            else "insufficient_evidence"}
        return result

    def _find_window_for(self, seat: str, frame: np.ndarray):
        """校准窗：座级窗键匹配（'S4_BL'/'1328_S3' 等——由布局键）。"""
        for k, win in self.text_windows.items():
            if k.startswith(seat + "_"):
                return win
        return None

    def _read_window(self, frame: np.ndarray, win) -> dict:
        bbox = win["bbox"] if isinstance(win, dict) else tuple(win)
        thr = win.get("split_thr", 2) if isinstance(win, dict) else 2
        x0, y0, x1, y1 = bbox
        roi = frame[y0:y1, x0:x1]
        gold = self._gold_mask(roi)
        ys, xs = np.where(gold > 0)
        if len(ys) < 30:
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0}
        inner = gold[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        return self._chars_from_mask(roi, inner, ys, xs, split_thr=thr)

    def _chars_from_mask(self, roi: np.ndarray, inner: np.ndarray,
                         ys: np.ndarray, xs: np.ndarray,
                         split_thr: int = 2) -> dict:
        cols = inner.sum(axis=0)
        active = cols >= split_thr
        comps, run = [], None
        for i, a in enumerate(active):
            if a and run is None:
                run = i
            elif not a and run is not None:
                comps.append([run, i])
                run = None
        if run is not None:
            comps.append([run, len(active)])
        i = 0
        while i < len(comps) - 1:
            if comps[i + 1][0] - comps[i][1] <= 1:
                comps[i][1] = comps[i + 1][1]
                comps.pop(i + 1)
            else:
                i += 1
        if not comps:
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0}
        # 超宽块重切（泛化——P5T-5续）：宽 ≥1.7×中位宽的块在最小墨列切分
        # （相邻重复字符粘连如 "000"/"88" 的通用断开；非值级特判）
        widths = [c[1] - c[0] for c in comps]
        med = float(np.median(widths)) if widths else 0.0
        if med > 0:
            comps2 = []
            for (cs, ce) in comps:
                w = ce - cs
                if w >= 1.7 * med and w >= 30:
                    cols_sub = inner[:, cs:ce].sum(axis=0) \
                        if hasattr(inner, "shape") else None
                    from . import coinpoker_stack as _m
                    pieces = _cut_low_ink(cols_sub)
                    if pieces:
                        for (a, b) in pieces:
                            comps2.append((cs + a, cs + b))
                        continue
                comps2.append((cs, ce))
            comps = comps2
        chars, scores = [], []
        for (cs, ce) in comps:
            seg = inner[:, cs:ce]
            rws = np.where(seg.sum(axis=1) > 0)[0]
            if len(rws) < 6:
                continue
            sub = seg[rws[0]:rws[-1] + 1]
            img = _pad_tight(ImageData.from_rgba_bytes(
                sub.shape[1], sub.shape[0],
                np.stack([sub * 255] * 4, axis=-1).astype(np.uint8).tobytes()),
                pad=2)
            g = np.asarray(to_gray(img).data, dtype=float) / 255.0
            top_h = max(1, int(sub.shape[0] * 0.4))
            top_ink = float(sub[:top_h].sum()) / max(1.0,
                                                     float(sub.sum()))
            ch, margin = self._match_char(g, top_ratio=top_ink)
            if ch is None or margin is None:
                continue
            chars.append(ch)
            scores.append(round(margin, 3))
        raw = "".join(chars)
        conf = round(min(scores) if scores else 0.0, 3)
        # 粘连拒绝（精度）：掩码宽≥45 却只有 <4 字符 → 分割明显失败 → UNKNOWN
        if len(chars) < 4 and inner.shape[1] >= 45:
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0}
        if not fresh_ok(raw, scores):
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0}
        return {"raw": raw, "status": "RECOGNIZED", "confidence": conf}


def SeatLikeGoldBlobs(frame: np.ndarray) -> list:
    """金色光标（金文本/筹码金）中心列表（浅封装，测试可复用）。"""
    try:
        from .seat_position import SeatOccupancyDetector
        return SeatOccupancyDetector._gold_blobs(frame)
    except Exception:
        return []


def fresh_ok(raw: str | None, scores: list) -> bool:
    """全字符门：任何字符 margin < MIN_MARGIN 或组装文本非法 → 拒。"""
    if not raw or not scores or min(scores) < MIN_MARGIN:
        return False
    if not (PLAIN_RE.match(raw) or KM_RE.match(raw) or DEC_RE.match(raw)):
        return False
    return True
