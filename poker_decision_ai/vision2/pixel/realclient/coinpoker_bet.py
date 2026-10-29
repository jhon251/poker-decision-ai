"""V3F-P6-P1 — Bet Recognition Foundation（1419 类）+ V3F-P13-B（1280 类）。

BetRecognizer：座间下注区 ROI → 复用 Stack 已验证原语（行带/分割/数字
模板/逗号几何/金额解析——组合，非复制）→ per-seat bet 值。

- Bet ROI ≠ Stack ROI（座间筹码区；严禁读座牌区/Stack 金额）；
- UNKNOWN-first（None ≠ 0 ≠ no-bet——语义分离）；
- 无状态/无缓存；street 仅作上下文 metadata（不推导数值）。

P13-B（window="1280"）：1280×779 新证据类 Bet 胶囊识别——
胶囊检测（亮墨行带 + 暗底校验）→ 字形分割（列投影 + 低墨重切）→
数字 NCC 模板（真实证据提取，provenance 入 manifest）/ 逗号几何 token →
金额解析 → UNKNOWN-first（任何歧义/截断/越门 → None）。
"""
from __future__ import annotations

import json
import re

import numpy as np

#: 座间下注区 ROI（full_png 1419×1066——1328 帧实测/目视校准；P6-P1）
BET_ROIS_1419 = {
    "S1": (500, 420, 700, 500),      # "1,000"（S1 下注，1328）
    "S3": (500, 300, 700, 380),      # "2,000"（S3 下注，1328）
    "S4": (960, 770, 1160, 860),     # "1,600"（S4 下注，1328）
}
BET_ROIS_F1 = {                      # V2R 帧 1
    "S1": (500, 420, 700, 500),      # "1,000"
    "S4": (960, 770, 1160, 860),     # "960"
}

#: 1280×779 窗口 Bet 胶囊 ROI（P13-A 新证据类；P13-B 校准）。
#: 胶囊几何（P13-B 实测，P13-A 帧 100/117 全帧目视验证）：
#:   S_TL 胶囊中心 x=504.5 · y 311-357 (h=46)；S_TR 中心 x=740.5 · y 248-294。
#: 宽度随数值变化 117-160px（居中锚定）→ ROI 中心锚定 + 半宽 95。
#: ownership：S_TL=顶左座下注区（帧 117 全帧验证=pepe1213 的 193,400/跟注）；
#:           S_TR=顶右座下注区（帧 117 全帧验证=CruelBlsck609 的 199,680/全押）。
#: 独立于 Stack 区 / Pot 胶囊 / Status 徽章 / 玩家名牌（几何不相交——防混淆测试）。
BET_ROIS_1280 = {
    "S_TL": (409, 303, 191, 62),
    "S_TR": (645, 240, 191, 62),
}

#: V3F-P24-C — 1876 per-seat ROI（P24-A 诊断实测）：
#: 暗胶囊 DarkPath：S1/S2/S6（暗底白字，P16-E bbox + P24-A 修正）；
#: 亮底 BrightPath：S5（P24-B 视觉确认 bright-pill variant，luma≈70>55）。
BET_ROIS_1876_DARK = {
    "S1": (440, 455, 120, 40),
    "S2": (680, 390, 120, 40),
    "S6": (875, 790, 120, 44),
}
#: S5 bright-pill ROI（P24-B：白字 (1360-1468, 895-925) 于亮灰/青 pill）
BET_ROI_1876_BRIGHT = (1350, 886, 130, 48)

#: P30-9-X — 1876 **补齐覆盖**：此前只有 S1/S2/S6（暗）+S5（亮）四个 ROI，
#: 另两个投注位没有 ROI ⇒ 那两个座位的下注**恒不可见**（用户实测：面对加注时
#: 看不到"有人下注"，导致输出不可能的 bet/check）。
#: 新增 ROI 由**真实帧网格扫描实测**得到（同一 1876 布局、两个不同牌桌帧一致）：
#:   · 右上投注位：胶囊实测 (1346,462,62,29)='0,12'（user_2250）、
#:     (1330,462,93,29)='2,000'（canonical）⇒ ROI 取其外扩包围盒
#:   · 中下右投注位：胶囊实测 (1065,947,70,29)='0,05'（user_2250）、
#:     (1066,947,68,29)='0,25'（user_2300）、(1053,947,93,29)='2,000'（canonical）
#: 纯**加法**：不改既有 ROI、不改任何阈值/门（CAPSULE/NCC/margin 全部原样）。
BET_ROIS_1876_EXTRA = {
    "S3": (1316, 450, 124, 44),
    "S4": (1036, 933, 124, 46),
}

#: 1280 Bet 字形资源目录（P13-B：模板提取自真实 Bet 证据，provenance 入 manifest）
BET_GLYPH_DIR = "data/vision2/coinpoker_assets_v2bet1280"
#: V-5（Round16）微级别 bet 字形目录（目录列表**尾部追加**；kind=
#: bet_digit_micro 隔离——v3c_micro 混载目录仅该 kind 入 bet 模板集）。
#: 实测 1876 bet capsule 数字 ≈13-16px（微级别 0.05/0.1 与 K 桌 1,000
#: 同尺度）——v2bet1280 模板 26-31px 对其全部超尺寸被跳过（此前恒
#: 模板缺失 → UNKNOWN）；微模板补齐该尺度带。0.55 几何门 + 0.08
#: margin 门不动（冻结精度语义）。
BET_GLYPH_DIR_MICRO = "data/vision2/coinpoker_assets_v3c_micro"
BET_MICRO_KIND = "bet_digit_micro"
#: V-5 微级别小数胶囊（几何逗号 token=小数点）：尾组 1-2 位。K 世界千位
#: 分组尾组恒 3 位（'1,000'）——两者互斥，纯格式区分（见 parse_bet_amount）。
BET_MICRO_DEC_RE = re.compile(r"^(\d{1,3}),(\d{1,2})$")

#: 胶囊视觉常量（P13-A 实测：胶囊内底 luma≈40-48；felt≈60-73；数字 luma→255）
CAPSULE_DARK_MAX = 55.0
DIGIT_WHITE_MIN = 150.0
CAPSULE_DARK_RATIO = 0.30
#: 字形几何带（P13-A/B 实测：数字 h 26-31；数字 w 15-20；'1' 8-11；
#: 逗号二进制掩码 h≈6（AA 头部 <150 亮度；下伸基线）w 5-7 底部锚定——
#: 逗号以几何 token 识别（低+窄+底部锚定），数字以 NCC 模板识别）
GLYPH_H_BAND = (4, 34)
GLYPH_W_BAND = (4, 24)
#: 全字符 margin 门（与 Stack MIN_MARGIN 一致——不降门槛）
BET_MIN_MARGIN = 0.08
#: 金额 token 数带（V-5 下限 4→3：微级别 '0.5'/'0.1' = 3 token（含小数点
#: 几何 token）；K 世界 3 token = 3 位裸整数（'999'，格式门后为合法 int）。
#: 截断风险论证：胶囊检测有 ROI 边缘截断防护（贴边即拒），中段丢字形仅
#: 可能来自 <6px 碎行（数字 28-31px 不满足）——'1,000'→'1,00' 类截断无
#: 产生路径；上限 8 不变）
BET_TOKENS_BAND = (3, 8)


def gray_luma(frame: np.ndarray) -> np.ndarray:
    """RGBA ndarray → 亮度（int16，0.299R+0.587G+0.114B）。"""
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    return (r * 0.299 + g * 0.587 + b * 0.114).astype(np.int16)


def detect_bet_capsule(lum: np.ndarray, roi: tuple,
                       *, check_dark: bool = True) -> dict | None:
    """胶囊检测（1280 类）：亮墨行带（≥12px/行，≤2 行间隙合并）→ 最大墨量带
    （数字行；排除上方筹码白边等稀疏白噪）→ 列紧致 → 暗底胶囊校验。

    V3F-P24-C：check_dark=False 用于 1876 S5 bright-pill variant（亮底白字，
    背景 luma≈70 > CAPSULE_DARK_MAX=55，旧 dark_ratio 校验会误拒）。

    返回 {"inner": 紧致二值掩码(np.uint8), "bbox": [x,y,w,h], "dark_ratio"} 或 None。
    None = 无胶囊（无下注 / 非胶囊 UI）——UNKNOWN 语义，非 0。
    """
    x0, y0, w, h = roi
    sub = lum[y0:y0 + h, x0:x0 + w]
    white = (sub > DIGIT_WHITE_MIN).astype(np.uint8)
    rows = white.sum(axis=1)
    runs, run = [], None
    for i, v in enumerate(rows):
        if v >= 12 and run is None:
            run = i
        elif v < 12 and run is not None:
            runs.append([run, i])
            run = None
    if run is not None:
        runs.append([run, len(rows)])
    i = 0
    while i < len(runs) - 1:
        if runs[i + 1][0] - runs[i][1] <= 2:
            runs[i][1] = runs[i + 1][1]
            runs.pop(i + 1)
        else:
            i += 1
    if not runs:
        return None
    best = max(runs, key=lambda r: int(white[r[0]:r[1]].sum()))
    band_ink = int(white[best[0]:best[1]].sum())
    if band_ink < 20:
        return None
    # 边缘截断防护（partial crop）：带贴 ROI 上下边 → 拒
    if best[0] <= 1 or best[1] >= h - 1:
        return None
    band = white[best[0]:best[1]]
    ys, xs = np.where(band > 0)
    ix0, iy0 = int(xs.min()), best[0]
    ix1, iy1 = int(xs.max()) + 1, best[1]
    # 横向截断防护
    if ix0 <= 0 or ix1 >= w:
        return None
    # 胶囊校验：周围暗底占比（P13-A 实测胶囊<55 / felt≥60）——仅 check_dark 时
    if check_dark:
        cx0, cy0 = max(0, ix0 - 6), max(0, iy0 - 6)
        cx1, cy1 = min(w, ix1 + 6), min(h, iy1 + 6)
        ctx = sub[cy0:cy1, cx0:cx1]
        dark_ratio = float((ctx < CAPSULE_DARK_MAX).mean())
        if dark_ratio < CAPSULE_DARK_RATIO:
            return None
    else:
        dark_ratio = None            # bright-pill variant：不校验暗底
    inner = band[:, ix0:ix1]
    return {"inner": inner, "bbox": [int(ix0 + x0), int(iy0 + y0),
                                     int(ix1 - ix0), int(iy1 - iy0)],
            "dark_ratio": dark_ratio}


def segment_bet_glyphs(inner: np.ndarray) -> list | None:
    """字形分割（1280 类）：列投影组件 + 1px 间隙合并 + 超宽低墨重切
    （复用 Stack 已验证的 `_cut_low_ink`——只读调用）→ 几何带内字形列表。

    任何字形越带（>GLYPH_H_BAND/GLYPH_W_BAND）→ None（整串不可靠 → UNKNOWN）。
    """
    from .coinpoker_stack import _cut_low_ink
    cols = inner.sum(axis=0)
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
    if not comps:
        return None
    widths = [c[1] - c[0] for c in comps]
    med = float(np.median(widths)) if widths else 0.0
    if med > 0:
        comps2 = []
        for (cs, ce) in comps:
            wd = ce - cs
            if wd >= 1.7 * med and wd >= 30:
                pieces = _cut_low_ink(inner[:, cs:ce].sum(axis=0))
                if pieces:
                    for (a, b) in pieces:
                        comps2.append((cs + a, cs + b))
                    continue
            comps2.append((cs, ce))
        comps = comps2
    glyphs = []
    for (cs, ce) in comps:
        seg = inner[:, cs:ce]
        rws = np.where(seg.sum(axis=1) > 0)[0]
        if len(rws) < 6:
            continue
        sub = seg[rws[0]:rws[-1] + 1]
        gh, gw = sub.shape
        if not (GLYPH_H_BAND[0] <= gh <= GLYPH_H_BAND[1]
                and GLYPH_W_BAND[0] <= gw <= GLYPH_W_BAND[1]):
            return None
        glyphs.append({"mask": sub, "x": int(cs), "y": int(rws[0]),
                       "w": int(gw), "h": int(gh)})
    if not glyphs:
        return None
    return glyphs


def _ncc_best(g: np.ndarray, tpls: dict) -> tuple:
    """通用 NCC 匹配（与 Stack 同一数学：均值去除 + 归一化滑动窗点积）。

    返回 (best_label, best_score, margin)。几何门：模板高/宽 ≥ 候选 55%。
    """
    best, bsc = None, -1.0
    scores = {}
    for label, insts in tpls.items():
        sc_l = -1.0
        for (t, t0, tn) in insts:
            th, tw = t.shape
            if th >= g.shape[0] or tw >= g.shape[1] or tn < 1e-9:
                continue
            if th < 0.55 * g.shape[0] or tw < 0.55 * g.shape[1]:
                continue
            wins = np.lib.stride_tricks.sliding_window_view(g, (th, tw))
            wm = wins.mean(axis=(2, 3), keepdims=True)
            wc = wins - wm
            wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
            s = np.abs((wc * t0).sum(axis=(2, 3))
                       / np.maximum(wn * tn, 1e-9))
            sc = float(s.max())
            if sc > sc_l:
                sc_l = sc
        scores[label] = sc_l
        if sc_l > bsc:
            best, bsc = label, sc_l
    if best is None:
        return None, None, None
    ranked = sorted(scores.values(), reverse=True)
    margin = bsc - (ranked[1] if len(ranked) > 1 else 0.0)
    return best, bsc, margin


def classify_bet_glyphs(glyphs: list, templates: dict) -> tuple | None:
    """字形 → (标签序列, margin 序列)。任何字形歧义 → None。

    逗号 = 几何 token（证据实测：二进制掩码窄且低、底部锚定——与数字 26-31px
    高度差 >4×，几何判别决定性）：h < 0.5×带高 且 w ≤ 8 且 底部锚定。
    数字 = NCC 模板匹配（margin 门 BET_MIN_MARGIN——与 Stack 一致，不降）。
    其它低/宽/越界形状 → None（歧义 → UNKNOWN）。
    """
    band_h = float(max(gl["h"] for gl in glyphs))
    low_h = 0.5 * band_h
    digits = {k: v for k, v in templates.items() if k in "0123456789"}
    labels = []
    margins = []
    for gl in glyphs:
        if gl["h"] < low_h:
            # 逗号几何：窄 + 底部锚定（掩码底 ≥ 带底-6）
            if gl["w"] <= 8 and gl["y"] + gl["h"] >= band_h - 6:
                labels.append(",")
                margins.append(BET_MIN_MARGIN)
                continue
            return None
        if not digits:
            return None
        # 候选 padding（同 Stack `_pad_tight` pad=2——模板必须严格小于候选，
        # 否则等尺寸模板被几何门略过；padding 后搜索窗 ≥1）
        g = np.pad(gl["mask"].astype(np.float64), 2)
        ch, _, mg = _ncc_best(g, digits)
        if ch is None or mg is None or mg < BET_MIN_MARGIN:
            return None
        labels.append(ch)
        margins.append(round(mg, 3))
    return labels, margins


def recognize_bet_capsule(lum: np.ndarray, roi: tuple,
                          templates: dict, *, check_dark: bool = True) -> dict:
    """单胶囊识别（1280 类）——UNKNOWN-first：无胶囊/分割失败/歧义/格式非法
    → {bet: None, status: UNKNOWN}（None ≠ 0 ≠ 前值 ≠ 推测）。"""
    cap = detect_bet_capsule(lum, roi, check_dark=check_dark)
    if cap is None:
        return {"bet": None, "raw_text": None, "status": "UNKNOWN",
                "confidence": 0.0, "reason": "no_bet_capsule"}
    band_h = cap["bbox"][3]
    if not (GLYPH_H_BAND[0] <= band_h <= GLYPH_H_BAND[1]):
        return {"bet": None, "raw_text": None, "status": "UNKNOWN",
                "confidence": 0.0, "reason": "capsule_scale_out_of_band"}
    glyphs = segment_bet_glyphs(cap["inner"])
    if glyphs is None:
        return {"bet": None, "raw_text": None, "status": "UNKNOWN",
                "confidence": 0.0, "reason": "glyph_segmentation_failed"}
    if not (BET_TOKENS_BAND[0] <= len(glyphs) <= BET_TOKENS_BAND[1]):
        return {"bet": None, "raw_text": None, "status": "UNKNOWN",
                "confidence": 0.0, "reason": "token_count_out_of_band"}
    labels = classify_bet_glyphs(glyphs, templates)
    if labels is None:
        return {"bet": None, "raw_text": None, "status": "UNKNOWN",
                "confidence": 0.0, "reason": "ambiguous_glyph"}
    raw = "".join(labels[0])
    from .coinpoker_stack import fresh_ok
    # V-5：微级别小数（'0,28'——几何逗号 token=小数点）与既有 K 格式
    # （fresh_ok：PLAIN/K/M）并立；margin 门两者同严（BET_MIN_MARGIN，
    # 不降门槛）。
    micro_dec = bool(raw) and BET_MICRO_DEC_RE.match(raw) is not None \
        and labels[1] and min(labels[1]) >= BET_MIN_MARGIN
    if not (fresh_ok(raw, labels[1]) or micro_dec):
        return {"bet": None, "raw_text": None, "status": "UNKNOWN",
                "confidence": 0.0, "reason": "bet_format_rejected"}
    value = parse_bet_amount(raw)
    if value is None:
        return {"bet": None, "raw_text": None, "status": "UNKNOWN",
                "confidence": 0.0, "reason": "bet_format_rejected"}
    return {"bet": value, "raw_text": raw, "status": "RECOGNIZED",
            "confidence": float(min(labels[1])),
            "reason": "bet_capsule_recognized"}


def load_bet_templates(glyph_dir: str = BET_GLYPH_DIR,
                       kind: str | None = None) -> dict:
    """Bet 字形模板装载（仅真实证据提取资产；无资产 → 空 → 恒 UNKNOWN）。

    V-5：kind 非 None 时仅装载该 kind 资产（微级别混载目录隔离）。
    """
    from pathlib import Path as _P
    mp = _P(glyph_dir) / "manifest.json"
    if not mp.exists():
        return {}
    m = json.loads(mp.read_text(encoding="utf-8"))
    tpls: dict = {}
    for a in m.get("assets", []):
        lab = a.get("label")
        if lab not in "0123456789":
            continue
        if kind is not None and a.get("kind") != kind:
            continue
        p = _P(a["crop_ref"])
        if not p.exists():
            p = _P(glyph_dir) / _P(a["crop_ref"]).name
        if not p.exists():
            continue
        from poker_decision_ai.vision2_platform import decode_png
        w2, h2, rgba = decode_png(p.read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h2, w2, 4)
        g = gray_luma(arr).astype(np.float64) / 255.0
        t0 = g - g.mean()
        tn = float(np.sqrt((t0 * t0).sum()))
        tpls.setdefault(lab, []).append((g, t0, tn))
    return tpls


class BetRecognizer:
    """当前帧 → per-seat {bet, raw_text, status, confidence, reason}。"""

    name = "coinpoker_bet_v3f_p6p1"

    def __init__(self, rois: dict | None = None, window: str = "1419",
                 street: str | None = None):
        from .coinpoker_stack import StackRecognizer
        self.stack = StackRecognizer(window=window)
        self.window = window
        self.street = street          # 仅上下文 metadata（不用于推导数值）
        if window in ("1280", "1876"):
            if rois is not None:
                self.rois = rois
            elif window == "1280":
                self.rois = dict(BET_ROIS_1280)
            else:
                # P30-9-X：1876 暗胶囊 ROI + 实测补齐的两个投注位（纯加法）
                self.rois = {**BET_ROIS_1876_DARK, **BET_ROIS_1876_EXTRA}
            # V-5：微级别模板尾追加（kind 隔离；同 label 多实例并存——
            # _ncc_best 逐实例取 max）
            self._templates = load_bet_templates()
            for lab, insts in load_bet_templates(
                    BET_GLYPH_DIR_MICRO, kind=BET_MICRO_KIND).items():
                self._templates.setdefault(lab, []).extend(insts)
            self._has_bet_glyphs = bool(self._templates)
            return
        self.rois = rois or BET_ROIS_1419
        self._templates = {}
        # bet 尺度字形族（v2p6b1——真实提取；当前资产未落成 → 恒 UNKNOWN，
        # 精度优先：无 bet 尺度证据不冒险识别——防跨尺度错值）
        from pathlib import Path as _P
        self._has_bet_glyphs = False
        mp = _P("data/vision2/coinpoker_assets_v2p6b1/manifest.json")
        if mp.exists():
            try:
                m = json.loads(mp.read_text(encoding="utf-8"))
                self._has_bet_glyphs = bool(m.get("assets"))
            except Exception:
                self._has_bet_glyphs = False

    def _read_region(self, frame: np.ndarray, region: tuple) -> dict:
        """区域→行带发现→最优带识别（复用 Stack 原语——只读组合）。
        Bet 文本=白/亮色（非金色）→ 亮度掩码（l>90）。"""
        x0, y0, x1, y1 = region
        roi = frame[y0:y1, x0:x1]
        l = roi[..., 0].astype(np.int16) * 0.299 + \
            roi[..., 1].astype(np.int16) * 0.587 + \
            roi[..., 2].astype(np.int16) * 0.114
        gold = (l > 90).astype(np.uint8)
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
            if len(ys) < 20:
                continue
            inner = seg[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            read = self.stack._chars_from_mask(roi, inner, ys + bs, xs,
                                               split_thr=2)
            if read["status"] == "RECOGNIZED":
                read["glyph_height"] = int(inner.shape[0])
                if best is None or len(read["raw"]) > len(best["raw"]):
                    best = read
        if best is None:
            return {"raw": None, "status": "UNKNOWN", "confidence": 0.0,
                    "glyph_height": 0}
        return best

    def recognize(self, frame: np.ndarray) -> dict:
        if self.window == "1280":
            return self._recognize_1280(frame)
        if self.window == "1876":
            return self._recognize_1876(frame)
        result = {}
        if not self._has_bet_glyphs:
            return {seat: {"bet": None, "raw_text": None,
                           "status": "UNKNOWN",
                           "reason": "no_bet_scale_glyphs"}
                    for seat in self.rois}
        for seat, region in self.rois.items():
            if region[0] + region[2] > frame.shape[1] or \
                    region[1] + region[3] > frame.shape[0]:
                result[seat] = {"bet": None, "raw_text": None,
                                "status": "UNKNOWN",
                                "reason": "roi_out_of_frame"}
                continue
            read = self._read_region(frame, region)
            if read["status"] != "RECOGNIZED":
                result[seat] = {"bet": None, "raw_text": None,
                                "status": "UNKNOWN",
                                "reason": "no_reliable_bet_text"}
                continue
            # 尺度一致性保护（精度）：bet 文本≈13px——堆栈模板(28-34px)不可
            # 用且无 bet 尺度字形时 → 拒识（防跨尺度错值 "0,000"）
            if read.get("glyph_height", 0) < 20:
                result[seat] = {"bet": None, "raw_text": None,
                                "status": "UNKNOWN",
                                "reason": "scale_mismatch_no_bet_glyphs"}
                continue
            value = parse_bet_amount(read["raw"])
            if value is None:
                result[seat] = {"bet": None, "raw_text": None,
                                "status": "UNKNOWN",
                                "reason": "bet_format_rejected"}
                continue
            result[seat] = {"bet": value, "raw_text": read["raw"],
                            "status": "RECOGNIZED",
                            "confidence": read["confidence"],
                            "reason": "bet_text_recognized"}
        return result

    def _recognize_1280(self, frame: np.ndarray) -> dict:
        """1280×779 新证据类（P13-B）：胶囊 → 字形 → 识别 → 解析，UNKNOWN-first。"""
        lum = gray_luma(frame)
        result = {}
        for seat, roi in self.rois.items():
            if roi[0] + roi[2] > frame.shape[1] or \
                    roi[1] + roi[3] > frame.shape[0]:
                result[seat] = {"bet": None, "raw_text": None,
                                "status": "UNKNOWN",
                                "reason": "roi_out_of_frame"}
                continue
            result[seat] = recognize_bet_capsule(lum, roi, self._templates)
        return result

    def _recognize_1876(self, frame: np.ndarray) -> dict:
        """V3F-P24-C — 1876：DarkPath（S1/S2/S6 暗胶囊）+ BrightPath（S5 亮底
        pill variant，check_dark=False）。UNKNOWN-first；无 previous/cache。"""
        lum = gray_luma(frame)
        result = {}
        for seat, roi in self.rois.items():
            if roi[0] + roi[2] > frame.shape[1] or \
                    roi[1] + roi[3] > frame.shape[0]:
                result[seat] = {"bet": None, "raw_text": None,
                                "status": "UNKNOWN",
                                "reason": "roi_out_of_frame"}
                continue
            result[seat] = recognize_bet_capsule(lum, roi, self._templates,
                                                 check_dark=True)
        # BrightPath：S5 亮底 pill（P24-B 视觉确认；不执行暗底校验）
        if BET_ROI_1876_BRIGHT[0] + BET_ROI_1876_BRIGHT[2] <= frame.shape[1] \
                and BET_ROI_1876_BRIGHT[1] + BET_ROI_1876_BRIGHT[3] \
                <= frame.shape[0]:
            result["S5"] = recognize_bet_capsule(
                lum, BET_ROI_1876_BRIGHT, self._templates, check_dark=False)
        return result


def parse_bet_amount(raw: str | None) -> int | float | None:
    """下注金额解析（复用 Stack 解析契约——独立入口，值域同构）。

    V-5 微级别小数分支：胶囊几何 token（低+窄+底部锚定）在 K 世界是千位
    逗号（'1,000'，尾组恰 3 位）、在微级别世界是小数点（'0,28'，尾组
    1-2 位）——按**尾组位数**区分（纯格式规则，非值域猜测）：尾组 1-2 位
    → 货币单位 float（'0,28'→0.28，与 V-4 header `_to_num` 同单位语义）；
    尾组 3 位 → 原千位解析（int）不变。'12,345' 尾组 3 位 → 12345 不变。
    """
    from .coinpoker_stack import parse_stack_text
    if raw:
        t = raw.strip()
        m = BET_MICRO_DEC_RE.match(t)
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")
        # V-5 前导零守卫（精度）：'005'（'0.05' 的小数点 token 被段间丢弃的
        # 段品）按 PLAIN 会解析成 5——百倍伪值。真实金额渲染无前导零；
        # '0'（单字符，过牌/无注语义位）不受影响。
        if len(t) > 1 and t.isdigit() and t[0] == "0":
            return None
    return parse_stack_text(raw)


def no_bet_proof(frame: np.ndarray, region: tuple) -> bool:
    """无下注态判定（仅视觉证据充分时）：区域内无金/亮文本信号。"""
    from .coinpoker_stack import StackRecognizer
    r = StackRecognizer()
    roi = frame[region[1]:region[3], region[0]:region[2]]
    gold = r._gold_mask(roi)
    l = roi[..., 0].astype(np.int16) * 0.299 + \
        roi[..., 1].astype(np.int16) * 0.587 + \
        roi[..., 2].astype(np.int16) * 0.114
    return int((l > 100).sum()) < 30
