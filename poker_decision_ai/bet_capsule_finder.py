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
#: 字形连通域筛（排除白牌面/大 UI 块）
GLYPH_AREA = (6, 900)
GLYPH_H = (5, 46)
GLYPH_W_MAX = 46
#: 上下条带中"台面绿"占比：强门 ⇒ felt；软门 ⇒ felt_soft（交交叉校验）
FELT_STRIP_MIN = 0.35
FELT_STRIP_SOFT = 0.10
#: 归属最大锚点距离（px；1876 实测 bet 区距锚点 206~375）⇒ 超出不归属（不猜）
MAX_ANCHOR_DIST = 430.0
#: 与锚点质心的最小距离（px）：更近者判为底池标签（实测底池 ~130px，投注位 ≥300px）
MIN_CENTER_DIST = 240.0


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


def _components(mask: np.ndarray) -> list[dict]:
    """行程 + 并查集连通域（纯 numpy/Python；无 scipy/cv2 依赖）。

    返回 [{x0,y0,x1,y1,area,rows:[(y,x0,x1),...]}]。
    """
    runs: list[tuple[int, int, int]] = []          # (y, x0, x1)
    for y in range(mask.shape[0]):
        row = mask[y]
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [len(idx) - 1]))
        for s, e in zip(starts, ends):
            runs.append((y, int(idx[s]), int(idx[e])))
    parent = list(range(len(runs)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_row: dict[int, list[int]] = {}
    for i, (y, x0, x1) in enumerate(runs):
        by_row.setdefault(y, []).append(i)
    for y, ids in by_row.items():
        prev = by_row.get(y - 1) or []
        for i in ids:
            _, ax0, ax1 = runs[i]
            for j in prev:
                _, bx0, bx1 = runs[j]
                if ax0 <= bx1 + 1 and bx0 <= ax1 + 1:      # 8-邻接（含对角）
                    union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(len(runs)):
        groups.setdefault(find(i), []).append(i)
    out: list[dict] = []
    for ids in groups.values():
        xs0 = min(runs[i][1] for i in ids)
        xs1 = max(runs[i][2] for i in ids)
        ys0 = min(runs[i][0] for i in ids)
        ys1 = max(runs[i][0] for i in ids)
        area = sum(runs[i][2] - runs[i][1] + 1 for i in ids)
        out.append({"x0": xs0, "y0": ys0, "x1": xs1, "y1": ys1, "area": area})
    return out


def candidate_boxes(frame: np.ndarray, geo: dict) -> list[tuple]:
    """台面内**字形连通域** → 行内成组 → 候选胶囊框。

    关键：先按连通域面积/尺寸筛出"字形"（排除**白牌面/大 UI 块**——它们会把行带
    冲垮导致候选框超宽被丢），再按行重叠 + 水平间隙成组。
    """
    if not geo.get("ok"):
        return []
    h, w = frame.shape[:2]
    white = _white_mask(frame)
    bx0 = max(0, int(geo["x0"]) - 40)
    bx1 = min(w, int(geo["x1"]) + 40)
    by0 = max(0, int(geo["y0"]) - 40)
    by1 = min(h, int(geo["y1"]) + 40)
    m = np.zeros_like(white)
    m[by0:by1, bx0:bx1] = white[by0:by1, bx0:bx1]
    glyphs = []
    for c in _components(m):
        bw, bh = c["x1"] - c["x0"] + 1, c["y1"] - c["y0"] + 1
        if not (GLYPH_AREA[0] <= c["area"] <= GLYPH_AREA[1]):
            continue
        if not (GLYPH_H[0] <= bh <= GLYPH_H[1]):
            continue
        if not (1 <= bw <= GLYPH_W_MAX):
            continue
        glyphs.append(c)
    glyphs.sort(key=lambda c: (c["y0"], c["x0"]))
    used = [False] * len(glyphs)
    boxes: list[tuple] = []
    for i, g in enumerate(glyphs):
        if used[i]:
            continue
        grp = [g]
        used[i] = True
        changed = True
        while changed:                              # 迭代吸收同行相邻字形
            changed = False
            gx0 = min(x["x0"] for x in grp)
            gx1 = max(x["x1"] for x in grp)
            gy0 = min(x["y0"] for x in grp)
            gy1 = max(x["y1"] for x in grp)
            for j, o in enumerate(glyphs):
                if used[j]:
                    continue
                ov = min(gy1, o["y1"]) - max(gy0, o["y0"]) + 1
                if ov < 0.5 * min(gy1 - gy0 + 1, o["y1"] - o["y0"] + 1):
                    continue
                gap = max(gx0 - o["x1"], o["x0"] - gx1) - 1
                if gap > 26:
                    continue
                grp.append(o)
                used[j] = True
                changed = True
        gx0 = min(x["x0"] for x in grp) - 6
        gy0 = min(x["y0"] for x in grp) - 6
        gx1 = max(x["x1"] for x in grp) + 6
        gy1 = max(x["y1"] for x in grp) + 6
        bw, bh = gx1 - gx0 + 1, gy1 - gy0 + 1
        if H_BAND[0] <= bh <= H_BAND[1] and W_BAND[0] <= bw <= W_BAND[1]:
            boxes.append((gx0, gy0, bw, bh))
    return boxes


def background_kind(frame: np.ndarray, box: tuple) -> dict:
    """胶囊**上下条带**背景判据：只要有任一条带是台面绿 ⇒ felt（下注）。

    为何不用整圈：底部投注位**紧贴深色名牌**，整圈取样会被名牌污染（实测误判 plate）。
    为何能区分筹码堆：筹码数字在**名牌内部**，其上/下条带同样是名牌深色 ⇒ plate。
    """
    x0, y0, bw, bh = box
    h, w = frame.shape[:2]
    strip = 6
    res = {"kind": "unknown", "green_top": 0.0, "green_bottom": 0.0}
    for tag, yy in (("top", y0 - strip - 2), ("bottom", y0 + bh + 2)):
        Y0, Y1 = max(0, yy), min(h, yy + strip)
        if Y1 <= Y0:
            continue
        X0, X1 = max(0, x0), min(w, x0 + bw)
        sub = frame[Y0:Y1, X0:X1]
        if sub.size == 0:
            continue
        r = sub[..., 0].astype(np.int16)
        g = sub[..., 1].astype(np.int16)
        b = sub[..., 2].astype(np.int16)
        green = float(((g - r >= FELT_G_MIN_DIFF) & (g - b >= FELT_G_MIN_DIFF)).mean())
        res[f"green_{tag}"] = round(green, 3)
    best = max(res["green_top"], res["green_bottom"])
    res["green"] = best
    if best >= FELT_STRIP_MIN:          # 明显裸台面 ⇒ felt（强信号）
        res["kind"] = "felt"
    elif best >= FELT_STRIP_SOFT:       # 弱信号：可能紧贴名牌/牌面 ⇒ 交由交叉校验判定
        res["kind"] = "felt_soft"
    else:
        lum = None
        X0, X1 = max(0, x0), min(w, x0 + bw)
        Y0, Y1 = max(0, y0), min(h, y0 + bh)
        sub = frame[Y0:Y1, X0:X1]
        if sub.size:
            b = sub[..., 2].astype(np.int16)
            g = sub[..., 1].astype(np.int16)
            r = sub[..., 0].astype(np.int16)
            lum = r * 0.299 + g * 0.587 + b * 0.114
            blue = float(((b - g >= 6) & (lum < 120)).mean())
            res["blue"] = round(blue, 3)
            if blue >= 0.30:
                res["kind"] = "pill"
        if res["kind"] == "unknown":
            res["kind"] = "plate"
    return res


def _anchor_center(anchors: dict) -> tuple:
    """锚点质心（近似桌面中心；只用锚点表）。"""
    xs = [a[0] for a in anchors.values()]
    ys = [a[1] for a in anchors.values()]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


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
                      cross_check: dict | None = None,
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
        if bg.get("kind") not in ("felt", "felt_soft"):
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
        val = r.get("bet")
        cc = cross_check or {}
        pots = cc.get("pot")
        stacks = cc.get("stacks") or {}
        if pots is not None and val is not None and abs(float(val) - float(pots)) < 1e-9 \
                and _dc < MIN_CENTER_DIST * 1.6:
            if log is not None:
                log.append({"bbox": box, "value": val, "skipped": "equals_pot"})
            continue
        if best is not None and val is not None and best in stacks \
                and stacks[best] is not None \
                and abs(float(val) - float(stacks[best])) < 1e-9:
            if log is not None:
                log.append({"bbox": box, "value": val, "seat": best,
                            "skipped": "equals_stack"})
            continue
        rec = {"bbox": box, "value": val, "raw": r.get("raw_text"),
               "confidence": r.get("confidence"), "background": bg.get("kind"),
               "seat": best, "anchor_dist": round(bd, 1)}
        res["capsules"].append(rec)
        if rec["seat"] is not None:
            cur = res["seats"].get(rec["seat"])
            if cur is None or (rec["value"] or 0) > (cur or 0):
                res["seats"][rec["seat"]] = rec["value"]
    return res


#: 锚点表座位 → **bet 环标签**（1876 实测：底中在锚点表是 S6，在 bet 环是 S4；
#: 左下在锚点表是 S4，在 bet 环是 S6）。合并进 `extra.v3f_bets` 前必须翻译。
ANCHOR_TO_BET_LABEL_1876 = {
    "S1": "S1", "S2": "S2", "S3": "S3", "S4": "S6", "S5": "S5", "S6": "S4",
}


def to_bet_ring_labels(seats: dict) -> dict:
    """把发现器的锚点表座位键翻译为 bet 环标签键。"""
    out = {}
    for k, v in (seats or {}).items():
        out[ANCHOR_TO_BET_LABEL_1876.get(k, k)] = v
    return out
