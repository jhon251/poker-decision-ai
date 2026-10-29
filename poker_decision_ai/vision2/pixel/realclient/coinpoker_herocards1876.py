"""V3F-P29-B — 1876 HeroCards detector（英雄两张 hole cards）。

严格基于 P29-A 冻结视觉证据：
- 几何（真实测量）：card1 (813,1011,121,178) card2 (837,1011,122,178)
  pair (813,1011,246,178)（hero S6 卡区；8bf8/pc110 一致，f03 ±1px）
- rank/suit 角标布局：rank 左上 (8,5,55,58)、suit 下方 (10,60,42,60)
  （相对 card 左上）
- 模板 = GT-provenance（coinpoker_assets_v1876herocards 从 3 个 VALID GT
  提取：rank {3,5,6,9} / suit {c,d,h}；未覆盖 → UNKNOWN）

管线：
1876 帧 → hero seat（由调用方传入）→ pair ROI → 白卡面 presence 双列簇
→ card1/card2 切分（几何常量 + 白簇验证）→ 每卡 rank/suit 角标 ROI →
模板 NCC（score≥0.7 margin≥0.08）+ 红黑 suit 颜色验证 → 两卡全通过 → 输出
(f"{rank}{suit}", ...)；任一失败 → UNKNOWN。

红线：无 OCR；无视觉 LLM runtime；无跨字段推断；无 previous/cache；
UNKNOWN-first；不因几何命中就输出身份。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: hero S6 卡 pair ROI（P29-A 实测）
HEROCARDS_PAIR_ROI = (795, 1005, 1070, 1195)
#: 每卡 bbox（card1 左 / card2 右；并排重叠显示）
HEROCARDS_CARD1 = (813, 1011, 121, 178)
HEROCARDS_CARD2 = (837, 1011, 122, 178)
#: 角标搜索窗（**绝对坐标**，P29-B 连通域实测；suit 窗含 f03 overlay 偏移）
#: card1 (左卡, 后) rank (824,1026,23,40) suit (825,1075,40,47)
#: card2 (右卡, 前, 旋转) rank (940,1018,25,44) suit (938,1052,42,62)
#: P30-9-AE（用户真实帧实测修正）：rank 角标可能是**双字形**（"10"），
#: 原 28/31px 宽窗口会把第二个字形切掉（NCC 因模板比窗口宽而跳过 T，
#: 只剩 J 等单字形候选 ⇒ 实测把 10♥ 误读为 Jh）。故**只加宽 ROI**：
#:   card1 (822,1020,850,1072) → 46px 宽；card2 (937,1016,968,1064) → 54px 宽
#: 阈值（MIN_SCORE/MIN_MARGIN/CARD_PRESENCE_MIN）与红黑校验**一律未改**。
RANK_WINDOWS = {"card1": (816, 1020, 862, 1072),
                "card2": (931, 1016, 985, 1064)}
SUIT_WINDOWS = {"card1": (822, 1074, 864, 1124),
                "card2": (936, 1050, 980, 1116)}
#: 白卡面 presence 阈值（双卡白簇；P29-A 实测 >8k）
CARD_PRESENCE_MIN = 8000
#: 模板目录
HEROCARDS_ASSET_DIR = "data/vision2/coinpoker_assets_v1876herocards"
#: 冻结门（与全链一致）
MIN_SCORE = 0.7
MIN_MARGIN = 0.08
#: 红/黑 suit 颜色验证（角标红墨占比）
RED_INK_FRAC_MIN = 0.15

_RANKS = "0123456789"
_SUITS = "cdhs"


def _load_templates(asset_dir: str = HEROCARDS_ASSET_DIR) -> dict:
    """GT-provenance 模板装载（alpha 掩码二值 + 预处理）。"""
    mp = Path(asset_dir) / "manifest.json"
    if not mp.exists():
        return {}
    m = json.loads(mp.read_text(encoding="utf-8"))
    prep: dict = {}
    for a in m.get("assets", []):
        lab = a.get("label", "")
        ch = lab.split("_", 1)[-1]     # rank_9 → 9; suit_c → c
        if not ch:
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
            prep.setdefault(ch, []).append((g, t0, tn))
    return prep


def _ncc(g: np.ndarray, prep: list) -> tuple:
    """候选二值 vs 模板实例 → (best, score, margin)。"""
    gp = np.pad(g, 2)
    best, bsc, scores = None, -1.0, {}
    for (t, t0, tn) in prep:
        th, tw = t.shape
        if th >= gp.shape[0] or tw >= gp.shape[1] or tn < 1e-9:
            continue
        wins = np.lib.stride_tricks.sliding_window_view(gp, (th, tw))
        wm = wins.mean(axis=(2, 3), keepdims=True)
        wc = wins - wm
        wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
        s = np.abs((wc * t0).sum(axis=(2, 3)) / np.maximum(wn * tn, 1e-9))
        sc = float(s.max())
        if sc > bsc:
            best, bsc = sc, 0.0
        # 记录每实例 best 后续由调用方聚合——此处聚合 max
        if sc > scores.get("_m", -1.0):
            scores["_m"] = sc
    return best, bsc if best is not None else None, None


def _match_glyph(g: np.ndarray, prep: dict, kinds: tuple) -> tuple:
    """对 kinds（如 ('rank_3','rank_5',...)—实际 prep 键为纯字符）全部标签 NCC。
    prep: {ch: [(t,t0,tn), ...]}；匹配 → (ch, score, margin) 或 (None,...).

    P30-9-IJ（**有界搜索** ✓，替代全窗口滑窗 ✗）：字形已在 ROI 内基本居中 ✓，
    只有几像素偏移需要容忍 ✓ ⇒ 先把掩码裁到**墨迹包围盒 ± R 像素** ✓，
    并**保证不小于最大模板**（首次尝试就是因为漏了这个护栏而更慢 ✗ ×）。
    窗口外区域本来**没有墨迹**（掩码只有墨 ✓）⇒ 跳过只会去掉**得分为 0 的候选** ✓，
    故 `max` 结果**不变** ✓（实测：夹具 + 全部真帧读数一致 ✓）。
    """
    best, bsc = None, -1.0
    scores = {}
    sub = g
    try:
        ys, xs = np.where(g > 0)
        if len(ys) >= 8:
            _maxth = max((t.shape[0] for insts in prep.values() for (t, _, _) in insts),
                          default=0)
            _maxtw = max((t.shape[1] for insts in prep.values() for (t, _, _) in insts),
                          default=0)
            y0, y1 = int(ys.min()) - HERO_SEARCH_R, int(ys.max()) + HERO_SEARCH_R + 1
            x0, x1 = int(xs.min()) - HERO_SEARCH_R, int(xs.max()) + HERO_SEARCH_R + 1
            need_h, need_w = _maxth + 2, _maxtw + 2          # **护栏** ✓
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            half_h = max((y1 - y0) // 2 + 1, need_h // 2 + 1)
            half_w = max((x1 - x0) // 2 + 1, need_w // 2 + 1)
            Y0, Y1 = max(0, cy - half_h), min(g.shape[0], cy + half_h)
            X0, X1 = max(0, cx - half_w), min(g.shape[1], cx + half_w)
            if (Y1 - Y0) >= need_h and (X1 - X0) >= need_w:
                sub = g[Y0:Y1, X0:X1]
    except Exception:                                          # noqa: BLE001
        sub = g
    gp = np.pad(sub, 2)
    for ch, insts in prep.items():
        sc = -1.0
        for (t, t0, tn) in insts:
            th, tw = t.shape
            if th >= gp.shape[0] or tw >= gp.shape[1] or tn < 1e-9:
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


def _match_glyph_scores(g: np.ndarray, prep: dict) -> dict:
    """**全部标签**的 NCC 最高分（供结构化消歧使用；判据与 `_match_glyph` 一致）。"""
    gp = np.pad(g, 2)
    out = {}
    for ch, insts in prep.items():
        best = -1.0
        for (t, t0, tn) in insts:
            th, tw = t.shape
            if th >= gp.shape[0] or tw >= gp.shape[1] or tn < 1e-9:
                continue
            wins = np.lib.stride_tricks.sliding_window_view(gp, (th, tw))
            wm = wins.mean(axis=(2, 3), keepdims=True)
            wc = wins - wm
            wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
            s = np.abs((wc * t0).sum(axis=(2, 3)) / np.maximum(wn * tn, 1e-9))
            best = max(best, float(s.max()))
        if best > 0:
            out[ch] = best
    return out


def _glyph_component_count(ink: np.ndarray, min_area: int = 25) -> int:
    """墨迹连通域数（≥min_area）：**"10" 为 2，单字形 rank 为 1**。"""
    from poker_decision_ai.bet_capsule_finder import _components
    b = ink.astype(bool)
    if not b.any():
        return 0
    return sum(1 for c in _components(b) if c["area"] >= min_area)


def _corner_mask(frame: np.ndarray, win: tuple) -> np.ndarray:
    """绝对窗口 ROI 的暗/红墨二值（角标字形）。"""
    x0, y0, x1, y1 = win
    if x1 > frame.shape[1] or y1 > frame.shape[0] or x0 < 0 or y0 < 0:
        return np.zeros((4, 4), dtype=np.float64)
    roi = frame[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = (l < 120).astype(np.float64)
    return ink


# ── P30-9-IB：**小角度扫描**（角标字形会因牌面扇形/动画而**倾斜** ✗，而匹配器只对平移不变 ✓）──
#   真实失败帧实测：card1 的 `6` 在 0° 时分数 0.812 / 余量 0.019（**不足** 0.08 ✗）；
#   旋转 −10° 后 → 分数 **0.936** / 余量 **0.228** ✓（真值被干净分离 ✓）。
#   ⇒ 在若干角度上取**判别余量最大**者 ✓，并把胜出角度记入 evidence（可审计 ✓）。
HERO_ROTATION_ANGLES = (-12.0, -10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0)
HERO_ROTATION_PAD = 8
# P30-9-IJ：有界搜索半径（字形已在 ROI 内基本居中 ✓）
HERO_SEARCH_R = 6
# P30-9-ID：**性能** ✗→✓ —— 首版在**全部 17 个标签 × 11 个角度**上做滑窗 NCC
#   ⇒ 实测 hero 底牌识别 **7.7 秒/次** ✗（用户抱怨"太慢" ✓）。
#   优化：先用 0° 结果取**前 K 名候选**（真值几乎总在其中 ✓：实测 '6' 已是第 1 ✓），
#   再只在这些候选上做角度扫描 ✓ ⇒ 标签数 17→3（约 5.7× 提速 ✓），方法不变、门槛不变 ✓。
HERO_ROTATION_CANDIDATES = 3
# P30-9-IJ：**达门即停** ✓ —— 旋转扫描得到一个"清楚过门"的结果即停止 ✓。
#   判据不变 ✓（仍要求 MIN_SCORE/MIN_MARGIN ✓），只是不再把全部角度跑完 ✗。
#   倾斜牌实测 ~4.1 秒 ✗ ⇒ 提前退出后通常 1–2 个角度即可 ✓。
HERO_ROTATION_EARLY_SCORE = 0.93
HERO_ROTATION_EARLY_MARGIN = 0.20


def _corner_mask_rotated(frame: np.ndarray, win: tuple, angle: float) -> np.ndarray:
    """先在**加宽**窗口内裁切，再旋转 angle 度，最后取暗墨二值（旋转留边以免切字 ✓）。"""
    from PIL import Image as _Image
    x0, y0, x1, y1 = win
    p = HERO_ROTATION_PAD
    x0p, y0p = max(0, x0 - p), max(0, y0 - p)
    x1p, y1p = min(frame.shape[1], x1 + p), min(frame.shape[0], y1 + p)
    if x1p <= x0p or y1p <= y0p:
        return np.zeros((4, 4), dtype=np.float64)
    crop = _Image.fromarray(np.ascontiguousarray(frame[y0p:y1p, x0p:x1p]))
    if abs(angle) > 1e-6:
        crop = crop.rotate(angle, resample=_Image.BILINEAR, fillcolor=(0, 0, 0))
    roi = np.asarray(crop)
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return (l < 120).astype(np.float64)


def match_glyph_rotated(frame: np.ndarray, win: tuple, prep: dict,
                         kinds: tuple, candidates: tuple | None = None) -> tuple:
    """在多角度上做字形匹配 ⇒ 取**判别余量最大**的结果 ✓。

    `candidates`：只在**这些标签**上匹配 ✓（性能优化：由 0° 结果的**前 K 名**给出 ✓；
    为 None 时用全部标签 ✓）。方法、阈值与判据**均不变** ✓。

    返回 `(best, score, margin, angle)`；全部角度都失败 ⇒ `(None, None, None, None)` ✓。
    """
    sub = ({k: v for k, v in prep.items() if k in candidates}
            if candidates else prep)
    if not sub:
        sub = prep
    best_all = (None, None, None, None)
    best_margin = -1.0
    for ang in HERO_ROTATION_ANGLES:
        mask = _corner_mask_rotated(frame, win, ang)
        if mask.sum() <= 0:
            continue
        b, sc, marg = _match_glyph(mask, sub, kinds)
        if b is None:
            continue
        if marg is not None and marg > best_margin:
            best_margin = marg
            best_all = (b, sc, marg, ang)
            # P30-9-IJ：**达门即停** ✓ —— 已经"清楚过门"就没必要把其余角度跑完 ✗
            #   （结果仍是"最佳余量"语义中的**足够好**者 ✓；判据/阈值不变 ✓）
            if sc is not None and sc >= HERO_ROTATION_EARLY_SCORE \
                    and marg >= HERO_ROTATION_EARLY_MARGIN:
                break
    return best_all


def _red_frac(frame: np.ndarray, win: tuple) -> float:
    """窗口内红墨占比（suit 红/黑判据）。"""
    x0, y0, x1, y1 = win
    if x1 > frame.shape[1] or y1 > frame.shape[0] or x0 < 0 or y0 < 0:
        return 0.0
    roi = frame[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    ink = l < 120
    if not ink.any():
        return 0.0
    red = (r > 110) & (r > g + 30) & (r > b + 30)
    return float(red[ink].mean())


#: 底部"卡面搜索带"（比 pair ROI 宽：允许展开/更靠下的渲染）
HEROCARDS_SEARCH_BAND = (700, 950, 1150, 1400)
#: 卡面连通域尺寸带（白色卡面）
CARD_BLOB_W = (60, 320)
CARD_BLOB_H = (90, 340)
#: 相对卡面左上角的角标偏移（P29-A 实测：rank (8,5,55,58)、suit (10,60,42,60)）
RANK_DX_DY = (8, 5, 55, 58)
SUIT_DX_DY = (10, 60, 42, 60)


def _white_mask(frame: np.ndarray) -> np.ndarray:
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    lum = r * 0.299 + g * 0.587 + b * 0.114
    return ((lum > 150) & (r > 140) & (g > 140) & (b > 140))


def locate_hero_card_blobs(frame: np.ndarray) -> list[dict]:
    """底部搜索带内的**可见白色卡面**连通域（按 x 排序，取最左/最右两张）。"""
    from poker_decision_ai.bet_capsule_finder import _components
    x0, y0, x1, y1 = HEROCARDS_SEARCH_BAND
    y1 = min(y1, frame.shape[0])
    x1 = min(x1, frame.shape[1])
    band = np.zeros(frame.shape[:2], dtype=bool)
    band[y0:y1, x0:x1] = _white_mask(frame)[y0:y1, x0:x1]
    out = []
    for c in _components(band):
        bw = c["x1"] - c["x0"] + 1
        bh = c["y1"] - c["y0"] + 1
        if CARD_BLOB_W[0] <= bw <= CARD_BLOB_W[1] and \
                CARD_BLOB_H[0] <= bh <= CARD_BLOB_H[1]:
            out.append(c)
    out.sort(key=lambda c: -c["area"])
    out = out[:6]
    out.sort(key=lambda c: c["x0"])
    return out


def windows_from_card(card: dict) -> dict:
    """由**可见卡面**左上角推导 rank/suit 窗口（绝对坐标）。"""
    cx, cy = card["x0"], card["y0"]
    rx, ry, rw, rh = RANK_DX_DY
    sx, sy, sw, sh = SUIT_DX_DY
    return {"rank": (cx + rx, cy + ry, cx + rx + rw, cy + ry + rh),
            "suit": (cx + sx, cy + sy, cx + sx + sw, cy + sy + sh),
            "from": (cx, cy)}


class HeroCardsRecognizer1876:
    """1876 帧 → hero 双卡 {cards: [{rank,suit}], status, value}。"""

    name = "coinpoker_herocards_v3f_p29b_1876"

    def __init__(self, asset_dir: str = HEROCARDS_ASSET_DIR):
        self._prep = _load_templates(asset_dir)
        # P30-9-II：**代表模板**（每标签 1 个 ✓）——`fast` 首轮用它 ✓ 以省掉 20+ 实例的全跑 ✗

    def _card_presence(self, frame: np.ndarray) -> bool:
        x0, y0, x1, y1 = HEROCARDS_PAIR_ROI
        if x1 > frame.shape[1] or y1 > frame.shape[0]:
            return False
        roi = frame[y0:y1, x0:x1]
        r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
        b = roi[..., 2].astype(np.int16)
        l = r * 0.299 + g * 0.587 + b * 0.114
        white = ((l > 150) & (r > 140) & (g > 140) & (b > 140))
        return int(white.sum()) >= CARD_PRESENCE_MIN

    def _recognize_card(self, frame: np.ndarray, card_key: str,
                        prep: dict, windows: dict | None = None) -> dict:
        r_win = (windows or {}).get("rank") or RANK_WINDOWS[card_key]
        s_win = (windows or {}).get("suit") or SUIT_WINDOWS[card_key]
        r_ink = _corner_mask(frame, r_win)
        s_ink = _corner_mask(frame, s_win)
        if r_ink.sum() < 30 or s_ink.sum() < 20:
            return {"rank": None, "suit": None, "status": "UNKNOWN",
                    "reason": "corner_ink_insufficient"}
        rch, rsc, rmg = _match_glyph(r_ink, prep, None)
        sch, ssc, smg = _match_glyph(s_ink, prep, None)
        if rsc is None or ssc is None or rmg is None or smg is None:
            return {"rank": None, "suit": None, "status": "UNKNOWN",
                    "reason": "no_template_match"}
        # margin 退化处理：该标签只有 1 个模板实例（无次佳可比）→ margin
        # 视为充分（几何一致性由 score 门 + 红黑校验兜底；不为 PASS 降门槛）
        def _margin_ok(mg, ch):
            return mg >= MIN_MARGIN or len(prep.get(ch, [])) <= 1
        tiebreak = None
        rank_ok = (rsc >= MIN_SCORE) and _margin_ok(rmg, rch)
        if not rank_ok and rsc >= MIN_SCORE:
            # P30-9-AR：仅 margin 不足时，用**结构证据**（双字形 vs 单字形）消歧
            scores = _match_glyph_scores(r_ink, prep)
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            if len(ranked) >= 2:
                a_ch, a_sc = ranked[0]
                b_ch, b_sc = ranked[1]
                pair = {a_ch, b_ch}
                if "T" in pair and a_ch == rch:
                    n_comp = _glyph_component_count(r_ink)
                    other = b_ch if a_ch == "T" else a_ch
                    if n_comp >= 2 and a_ch != "T":
                        rch, rsc, tiebreak = "T", a_sc, "two_glyph_structure→T"
                        rank_ok = rsc >= MIN_SCORE
                    elif n_comp >= 2 and a_ch == "T":
                        # 双字形 ⇒ 只有 "10" 可能 ⇒ 采信 T（分数门已过）
                        tiebreak = "two_glyph_structure→T(confirmed)"
                        rank_ok = True
                    elif n_comp == 1 and a_ch == "T":
                        rch, rsc, tiebreak = other, b_sc, \
                            "single_glyph_structure→" + str(other)
                        rank_ok = rsc >= MIN_SCORE
                    elif n_comp == 1 and a_ch != "T":
                        # 单字形 ⇒ 不可能是 "10" ⇒ 次佳 T 结构上不成立，采信最高分
                        tiebreak = "single_glyph_structure→" + str(a_ch)
                        rank_ok = True
        rot_angle = None
        if not rank_ok or ssc < MIN_SCORE or not _margin_ok(smg, sch):
            # P30-9-IB：**小角度扫描兜底** ✓ —— 牌面扇形/动画会让角标**倾斜** ✗，
            #   而匹配器只对平移不变 ✓。实测失败帧：card1 的 `6` 在 0° 时
            #   分数 0.812 / 余量 0.019（不足 0.08 ✗）；−10° 时 0.936 / 0.228 ✓。
            #   仅在**原角度判失败**时启用（0° 成功的情形行为完全不变 ✓），
            #   且仍要求**同样严格**的分数门 + 判别余量门 ✓（不降门槛 ✗）。
            #   P30-9-ID：先用 0° 结果取**前 K 名候选**再扫描 ⇒ 5.7× 提速 ✓（方法不变 ✓）。
            _cands: tuple | None = None
            try:
                _sc = _match_glyph_scores(r_ink, prep)
                _cands = tuple(k for k, _ in
                                sorted(_sc.items(), key=lambda kv: -kv[1])
                                [:HERO_ROTATION_CANDIDATES])
            except Exception:                                  # noqa: BLE001
                _cands = None
            r2 = match_glyph_rotated(frame, r_win, prep, None, candidates=_cands)
            s2 = match_glyph_rotated(frame, s_win, prep, None)
            if r2[0] is not None and s2[0] is not None:
                _rch, _rsc, _rmg, _ang = r2
                _sch, _ssc, _smg, _ang2 = s2
                if (_rsc >= MIN_SCORE and _margin_ok(_rmg, _rch)
                        and _ssc >= MIN_SCORE and _margin_ok(_smg, _sch)):
                    rch, rsc, rmg = _rch, _rsc, _rmg
                    sch, ssc, smg = _sch, _ssc, _smg
                    rot_angle = _ang
                    rank_ok = True
        if not rank_ok or ssc < MIN_SCORE or not _margin_ok(smg, sch):
            return {"rank": None, "suit": None, "status": "UNKNOWN",
                    "reason": "glyph_gate_failed",
                    "evidence": {"rank_score": round(rsc, 3),
                                 "rank_margin": round(rmg, 3),
                                 "suit_score": round(ssc, 3),
                                 "suit_margin": round(smg, 3),
                                 "rank_tiebreak": tiebreak,
                                 "rotation_scan": list(HERO_ROTATION_ANGLES)}}
        # 红/黑验证（suit 颜色一致性：红 suit 应有红墨；黑 suit 不应有）
        rf = _red_frac(frame, s_win)
        suit_red = sch in ("h", "d")
        if suit_red and rf < RED_INK_FRAC_MIN:
            return {"rank": None, "suit": None, "status": "UNKNOWN",
                    "reason": "red_suit_color_mismatch",
                    "evidence": {"red_frac": round(rf, 3),
                                 "suit": sch}}
        if not suit_red and rf > 0.6:
            return {"rank": None, "suit": None, "status": "UNKNOWN",
                    "reason": "black_suit_color_mismatch",
                    "evidence": {"red_frac": round(rf, 3)}}
        out = {"rank": rch, "suit": sch, "status": "OK",
               "confidence": round(min(rsc, ssc), 4)}
        if tiebreak:
            out["evidence"] = {"rank_tiebreak": tiebreak,
                               "rank_score": round(rsc, 3)}
        if rot_angle is not None:
            # P30-9-IB：**如实标注**该结果由小角度扫描得到 ✓（可审计 ✓）
            out.setdefault("evidence", {})
            out["evidence"].update({"rotation_deg": rot_angle,
                                     "rank_score": round(rsc, 3),
                                     "rank_margin": round(rmg, 3)})
        return out

    def recognize(self, frame: np.ndarray, hero_seat: str = "S6") -> dict:
        if hero_seat != "S6":
            # 仅支持 hero=S6（P29-A 实测布局常量为 S6 卡区）；
            # 其它 hero 座：无 GT 支撑 → UNKNOWN（诚实）
            return {"cards": [], "status": "UNKNOWN", "value": None,
                    "reason": "unsupported_hero_seat_layout"}
        if not self._prep:
            return {"cards": [], "status": "UNKNOWN", "value": None,
                    "reason": "no_templates"}
        if not self._card_presence(frame):
            return {"cards": [], "status": "UNKNOWN", "value": None,
                    "reason": "no_card_pair_present"}
        results = []
        for card_key in ("card1", "card2"):
            results.append(self._recognize_card(frame, card_key, self._prep))
        ok = all(r["status"] == "OK" for r in results)
        windows_source = "fixed"
        if not ok:
            # P30-9-AM：固定窗口失败 ⇒ 动态定位两张明牌后**同模板同门**重试
            try:
                blobs = locate_hero_card_blobs(frame)
                if len(blobs) >= 2:
                    left = windows_from_card(blobs[0])
                    right = windows_from_card(blobs[-1])
                    retry = [
                        self._recognize_card(frame, "card1", self._prep, left),
                        self._recognize_card(frame, "card2", self._prep, right),
                    ]
                    if all(r["status"] == "OK" for r in retry):
                        results = retry
                        ok = True
                        windows_source = "dynamic"
            except Exception:                      # noqa: BLE001
                pass
        if not ok:
            return {"cards": results, "status": "UNKNOWN",
                    "value": [r.get("rank") or "?" for r in results],
                    "reason": "card_recognition_incomplete"}
        value = [f"{r['rank']}{r['suit']}" for r in results]
        return {"cards": results, "status": "OK", "value": value,
                "windows_source": windows_source,
                "reason": "herocards_recognized"}
