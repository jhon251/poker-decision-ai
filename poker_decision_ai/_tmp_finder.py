"""P30-9-AD：自校准 bet 胶囊发现器（不依赖硬编码 ROI）。

流程：
  ① 绿色台面掩码 → 桌面 bbox / 中心 / 半径（自校准）
  ② 台面内白字掩码的行带切分 → 候选文本块（廉价、向量化）
  ③ 候选块交给**既有** `recognize_bet_capsule`（同模板/同判据，不降门）
  ④ 按相对桌面中心的角度归到最近座位（锚点角度，像素无关）
  ⑤ 排除名牌环带与中心底池盘（径向带过滤）

只做"发现 + 归属"，不发明数值：模板/格式/margin 判定沿用既有实现。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

#: 白字亮度门（与 coinpoker_bet.DIGIT_WHITE_MIN 同源语义，集中在此便于审计）
WHITE_MIN = 170
#: 台面（绿色）判定：G 明显高于 R/B
FELT_G_MIN_DIFF = 18
#: 径向带：排除名牌环（外圈）与中心底池盘
R_INNER, R_OUTER = 0.20, 0.80
#: 候选块尺寸带
H_BAND = (14, 56)
W_BAND = (24, 220)
#: 归属最大锚点距离（px；1876 实测 bet 区距锚点 206~375）⇒ 超出不归属（不猜）
MAX_ANCHOR_DIST = 430.0
#: 与锚点质心的最小距离（px）：更近者判为底池标签（实测底池 ~130px，投注位 ≥300px）
MIN_CENTER_DIST = 300.0


def felt_geometry(frame: np.ndarray) -> dict:
    """绿色台面 → {cx, cy, rx, ry}（自校准；失败 ⇒ None）。"""
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    felt = (g - r >= FELT_G_MIN_DIFF) & (g - b >= FELT_G_MIN_DIFF) & (g > 40)
    ys, xs = np.where(felt)
    if len(xs) < 5000:
        return {"ok": False, "reason": "felt_not_found", "n_px": int(len(xs))}
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    return {"ok": True, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "cx": cx, "cy": cy, "rx": rx, "ry": ry, "n_px": int(len(xs))}


def _white_mask(frame: np.ndarray) -> np.ndarray:
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    return (lum > WHITE_MIN) & (sat < 70)


def candidate_boxes(frame: np.ndarray, geo: dict) -> list[tuple]:
    """台面内白字行带 → 候选文本块（径向带过滤，排除名牌/底池）。"""
    if not geo.get("ok"):
        return []
    h, w = frame.shape[:2]
    white = _white_mask(frame)
    # 只在台面 bbox（略放宽）内搜索；不用径向带（径向带依赖中心估计，实测不稳）
    bx0 = max(0, int(geo["x0"]) - 40)
    bx1 = min(w, int(geo["x1"]) + 40)
    by0 = max(0, int(geo["y0"]) - 40)
    by1 = min(h, int(geo["y1"]) + 40)
    m = np.zeros_like(white)
    m[by0:by1, bx0:bx1] = white[by0:by1, bx0:bx1]
    rows = m.sum(axis=1)
    boxes: list[tuple] = []
    run = None
    for y, v in enumerate(rows):
        if v >= 6 and run is None:
            run = y
        elif v < 6 and run is not None:
            boxes.append((run, y))
            run = None
    if run is not None:
        boxes.append((run, len(rows)))
    out: list[tuple] = []
    for y0, y1 in boxes:
        if not (H_BAND[0] <= (y1 - y0) <= H_BAND[1]):
            continue
        band = m[y0:y1]
        cols = band.sum(axis=0)
        c0 = None
        gap = 0
        for x, v in enumerate(cols):
            if v > 0:
                if c0 is None:
                    c0 = x
                gap = 0
            elif c0 is not None:
                gap += 1
                if gap > 26:                 # 允许字间空隙
                    if W_BAND[0] <= (x - gap - c0) <= W_BAND[1]:
                        out.append((c0 - 6, y0 - 6, (x - gap) - c0 + 12, (y1 - y0) + 12))
                    c0 = None
                    gap = 0
        if c0 is not None:
            x = len(cols)
            if W_BAND[0] <= (x - c0) <= W_BAND[1]:
                out.append((c0 - 6, y0 - 6, (x - c0) + 12, (y1 - y0) + 12))
    return out


def _anchor_center(anchors: dict) -> tuple:
    """锚点质心（近似桌面中心；只用锚点表）。"""
    xs = [a[0] for a in anchors.values()]
    ys = [a[1] for a in anchors.values()]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def background_kind(frame: np.ndarray, box: tuple) -> dict:
    """胶囊**背景**判据：felt（裸台面=下注）/ plate（名牌=筹码堆）/ pill（底池胶囊）。"""
    x0, y0, bw, bh = box
    h, w = frame.shape[:2]
    pad = 8
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(w, x0 + bw + pad), min(h, y0 + bh + pad)
    sub = frame[Y0:Y1, X0:X1]
    if sub.size == 0:
        return {"kind": "unknown"}
    # **只取外圈环形**（排除胶囊自身的暗底/亮底，否则恒被判成名牌）
    ring = np.ones(sub.shape[:2], dtype=bool)
    ring[pad:sub.shape[0] - pad, pad:sub.shape[1] - pad] = False
    if ring.sum() < 40:
        return {"kind": "unknown"}
    sub = sub[ring]
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    green = float(((g - r >= FELT_G_MIN_DIFF) & (g - b >= FELT_G_MIN_DIFF)).mean())
    dark = float((lum < 90).mean())
    blue = float(((b - g >= 6) & (lum < 120)).mean())
    if green >= 0.45:
        kind = "felt"
    elif blue >= 0.35:
        kind = "pill"
    elif dark >= 0.45:
        kind = "plate"
    else:
        kind = "unknown"
    return {"kind": kind, "green": round(green, 3), "dark": round(dark, 3),
            "blue": round(blue, 3)}


def load_templates() -> dict:
    """模板集（与生产同源：v2bet1280 + v3c_micro 的 bet_digit_micro）。"""
    from poker_decision_ai.vision2.pixel.realclient import coinpoker_bet as cb
    tpl = cb.load_bet_templates()
    for lab, insts in cb.load_bet_templates(
            cb.BET_GLYPH_DIR_MICRO, kind=cb.BET_MICRO_KIND).items():
        tpl.setdefault(lab, []).extend(insts)
    return tpl


def find_bet_capsules(frame: np.ndarray, anchors: dict, *,
                      templates: dict | None = None,
                      log: list | None = None) -> dict:
    """发现并归属 bet 胶囊 → {"seats": {seat: value}, "capsules": [...], "geo": {...}}。"""
    from poker_decision_ai.vision2.pixel.realclient import coinpoker_bet as cb
    geo = felt_geometry(frame)
    res: dict = {"seats": {}, "capsules": [], "geo": geo}
    if not geo.get("ok"):
        return res
    tpl = templates if templates is not None else load_templates()
    lum = cb.gray_luma(frame)
    ccx, ccy = _anchor_center(anchors)
    for box in candidate_boxes(frame, geo):
        x0, y0, bw, bh = box
        r = None
        for check_dark in (True, False):        # 暗底优先，亮底兜底（判据不降）
            try:
                r = cb.recognize_bet_capsule(lum, box, tpl,
                                              check_dark=check_dark)
            except Exception:                                      # noqa: BLE001
                r = None
            if r and r.get("status") == "RECOGNIZED":
                break
        if not r or r.get("status") != "RECOGNIZED":
            if log is not None and r is not None:
                log.append({"bbox": box, "reason": r.get("reason")})
            continue
        bg = background_kind(frame, box)
        cx, cy = x0 + bw / 2.0, y0 + bh / 2.0
        _dc = math.hypot(cx - ccx, cy - ccy)
        if _dc < MIN_CENTER_DIST:
            # 离桌面中心太近 ⇒ 底池标签（实测底池位于中心附近 ~130px）
            if log is not None:
                log.append({"bbox": box, "value": r.get("bet"),
                            "skipped": "too_close_to_center",
                            "dist": round(_dc, 1)})
            continue
        if bg.get("kind") != "felt":
            if log is not None:
                log.append({"bbox": box, "value": r.get("bet"),
                            "skipped": "background=" + str(bg.get("kind")), **bg})
            continue
        best, bd = None, 1e9
        for s, (ax, ay) in anchors.items():
            d = math.hypot(cx - ax, cy - ay)
            if d < bd:
                best, bd = s, d
        if best is not None:
            ax, ay = anchors[best]
            if bd > MAX_ANCHOR_DIST:
                best = None                      # 太远 ⇒ 不归属
            elif math.hypot(cx - ccx, cy - ccy) >= math.hypot(ax - ccx, ay - ccy):
                best = None                      # 不在"锚点与中心之间" ⇒ 不归属
        rec = {"bbox": box, "value": r.get("bet"), "raw": r.get("raw_text"),
               "confidence": r.get("confidence"), "background": bg.get("kind"),
               "seat": best, "anchor_dist": round(bd, 1)}
        res["capsules"].append(rec)
        if rec["seat"] is not None:
            cur = res["seats"].get(rec["seat"])
            if cur is None or (rec["value"] or 0) > (cur or 0):
                res["seats"][rec["seat"]] = rec["value"]
    return res
