"""V3F-P25-C — 1876 Status Recognizer（folded / disconnected / away / allin）。

识别对象（P25-A/B.1/B.2/B.3 已冻结的视觉事实，本实现禁止重新审计）：
- folded：座位框左上角**暗青色圆角标签**（~103×40，每座布局常量）+ 深色
  「盖牌」字形。判据 = teal 颜色比例（g>1.5r, |g−b|≤14, luma 6-110），
  NORMAL/DIM 双场景同判据（P25-B.3：16/16 ≥0.84，20/20 ≤0.05）。
- disconnected：座位框左上角**暗玻璃棒 + 灰白「断开」字**（luma 90-140 中性）。
  判据 = 暗底覆盖 + 灰字量 + 文本列结构（≥12px 列 run）——与狼毛/头像灰
  噪（run≤5）分离。
- away：**白色咖啡杯 + 「暂离」白字**（头像圆上）。判据 = 白二值 + 模板
  NCC（f01 S3 GT provenance；bounded ±170，P25-C 实测 POS 0.81-0.82 /
  NEG ≤0.25）。
- allin：**独立亮牌路径**（非徽章）：白卡面 + 胜率/OUTS 胶囊（饱和红/
  teal/蓝带 + 白字被彩色包夹行）。白卡与 hero 自己亮牌（f03 S6 白卡
  15.5k 无胶囊）由胶囊门区分；「过牌」绿标签（f02 S1，白字但无白卡面）
  由白卡门区分。

红线：禁止用 1419 badge 外观；禁止牌面/胜率数值反推；禁止 Action 词；
UNKNOWN-first（无证据/冲突/无法归属 → UNKNOWN，无 fallback/cache/previous）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: 每座 folded/disc 徽章 ROI（P25-B.3 实测布局常量——与 SEAT_ROIS_1419 /
#: BET_ROIS_1876 同类；非 frame-id 特判）
STATUS_TAG_ROIS_1876 = {
    "S1": (138, 325, 104, 41),
    "S2": (816, 211, 103, 40),
    "S3": (1494, 325, 103, 41),
    "S4": (101, 891, 103, 40),
    "S5": (1531, 891, 103, 40),
    "S6": (816, 1149, 103, 40),
}

#: folded teal 覆盖阈值（P25-B.3：正例 16/16 ≥0.84，负例 20/20 ≤0.05；
#: 阈值取分布中点 0.5，不在正例底线处勉强）
FOLDED_TEAL_MIN = 0.5
#: folded teal 覆盖上限（P25-C 补测：真实 tag 为"徽章"——字形与圆角恒非
#: teal，实测正例峰值 0.87；纯 teal 填充帧（空白/均匀底色）覆盖=1.0，
#: 用 0.95 上限排除——上限高于全部 16 个真实正例，非调低）
FOLDED_TEAL_MAX = 0.95

#: disconnected 判据（P25-C 实测）：暗玻璃底 ≥0.4 ＋ 灰字量 ≥250 ＋
#: 最大列 run ≥12px（正例 404px/run18；全部负例 ≤199px/run≤5）
DISC_GLASS_MIN = 0.40
DISC_GRAY_MIN = 250
DISC_GRAY_RUN_MIN = 12

#: away 模板资产（f01 S3 白咖啡杯+暂离；GT provenance）
AWAY_ASSET_DIR = "data/vision2/coinpoker_assets_v1876status"
AWAY_TEMPLATE_FILE = "away_badge_white.png"
AWAY_NCC_MIN = 0.6
AWAY_SEARCH_RADIUS = 170

#: allin 判据（P25-C 实测）：白卡面 ≥20000px ＋ 彩带白字包夹行 ≥20
#: （正例 30070+/27 行；hero 亮牌 15585/0 行；过牌绿标签 1609/18）
ALLIN_WHITE_MIN = 20000
ALLIN_CAPSULE_ROWS_MIN = 20

#: teal 颜色比例常量（NORMAL/DIM 同判据，P25-B.3 冻结）
_TEAL_G_MIN = 1.5
_TEAL_ABS_MAX = 14

#: 亮二值阈值（away 模板 / 白卡面）
_WHITE_LUMA_MIN = 140.0
_CARD_WHITE_MIN = 200.0
_CARD_WHITE_LUMA = 190.0


def _luma(arr: np.ndarray) -> np.ndarray:
    return (arr[..., 0].astype(np.int16) * 0.299
            + arr[..., 1].astype(np.int16) * 0.587
            + arr[..., 2].astype(np.int16) * 0.114)


def teal_coverage(frame: np.ndarray, roi: tuple) -> float:
    """TAG ROI 内 teal 比例（folded 主判据，P25-B.3 冻结）。"""
    x0, y0, w, h = roi
    sub = frame[y0:y0 + h, x0:x0 + w]
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    mask = ((g > r * _TEAL_G_MIN) & (np.abs(g - b) <= _TEAL_ABS_MAX)
            & (l > 6) & (l < 110))
    return float(mask.mean())


def disconnected_signature(frame: np.ndarray, roi: tuple) -> dict:
    """TAG ROI 内 disc 签名：glass 底 / 灰字量 / 最大灰列 run。

    灰字 = luma 90-140 近中性；文本结构 = 列投影 run（≥6px 高聚集）合并
    ≤3px 间隙；「断开」字形呈 ≥12px 列块，头像灰噪呈 ≤5px 散斑。
    """
    x0, y0, w, h = roi
    sub = frame[y0:y0 + h, x0:x0 + w]
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    glass = float((l < 45).mean())
    gray = ((l >= 90) & (l <= 140) & (np.abs(r - g) < 18)
            & (np.abs(g - b) < 18))
    cold = gray.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v >= 6 and run is None:
            run = [i, i]
        elif v >= 6 and run is not None:
            run[1] = i
        else:
            if run is not None:
                runs.append(run)
                run = None
    if run is not None:
        runs.append(run)
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= 3:
            merged[-1][1] = rr[1]
        else:
            merged.append(list(rr))
    max_run = max((bb - aa + 1 for aa, bb in merged), default=0)
    return {"glass": round(glass, 3), "gray_n": int(gray.sum()),
            "max_run": int(max_run)}


def _load_away_template() -> np.ndarray | None:
    """装载 away 白字模板（二值 float，f01 S3 GT provenance）。"""
    from poker_decision_ai.vision2_platform.png_decode import decode_png
    p = Path(AWAY_ASSET_DIR) / AWAY_TEMPLATE_FILE
    if not p.exists():
        return None
    w, h, rgba = decode_png(p.read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    return (arr[..., 0] > 128).astype(np.float64)


_AWAY_TMPL = None
_AWAY_T0 = None
_AWAY_TN = None


def _away_template_ready() -> bool:
    global _AWAY_TMPL, _AWAY_T0, _AWAY_TN
    if _AWAY_TMPL is None:
        _AWAY_TMPL = _load_away_template()
        if _AWAY_TMPL is not None:
            _AWAY_T0 = _AWAY_TMPL - _AWAY_TMPL.mean()
            _AWAY_TN = float(np.sqrt((_AWAY_T0 * _AWAY_T0).sum()))
    return _AWAY_TMPL is not None and _AWAY_TN is not None and _AWAY_TN > 1e-9


def _white_bin(arr: np.ndarray, x0: int, y0: int, w: int, h: int,
               lo: float = _WHITE_LUMA_MIN) -> np.ndarray:
    x0, y0 = max(0, x0), max(0, y0)
    x1 = min(arr.shape[1], x0 + w)
    y1 = min(arr.shape[0], y0 + h)
    sub = arr[y0:y1, x0:x1]
    l = _luma(sub)
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    return ((l > lo) & (np.abs(r - g) < 30) & (np.abs(g - b) < 30)
            ).astype(np.float64)


#: 粗搜步长与"精修邻域半径"（P30-9-AT）
_AWAY_COARSE_STRIDE = 6
_AWAY_REFINE_RADIUS = 8


def _away_score(win: np.ndarray, th: int, tw: int, dy: int, dx: int) -> float:
    sub = win[dy:dy + th, dx:dx + tw]
    s0 = sub - sub.mean()
    dn = float(np.sqrt((s0 * s0).sum()))
    if dn < 1e-9:
        return -1.0
    return float((s0 * _AWAY_T0).sum() / (dn * _AWAY_TN))


def _away_ncc_full(frame: np.ndarray, ax: int, ay: int,
                   radius: int, stride: int) -> float:
    """原实现（stride 网格全搜）——对拍用。"""
    if not _away_template_ready():
        return -1.0
    th, tw = _AWAY_TMPL.shape
    win = _white_bin(frame, ax - radius, ay - radius, 2 * radius, 2 * radius)
    if win.shape[0] < th or win.shape[1] < tw:
        return -1.0
    best = -1.0
    for dy in range(0, win.shape[0] - th + 1, stride):
        for dx in range(0, win.shape[1] - tw + 1, stride):
            v = _away_score(win, th, tw, dy, dx)
            if v > best:
                best = v
    return best


def away_ncc(frame: np.ndarray, ax: int, ay: int,
             radius: int = AWAY_SEARCH_RADIUS, stride: int = 2) -> float:
    """锚点 ±radius 白二值 → away 模板滑窗 max|NCC|（**由粗到精**，P30-9-AT）。

    先以 `_AWAY_COARSE_STRIDE` 粗搜取候选，再在其 ±`_AWAY_REFINE_RADIUS`
    邻域内按 `stride` 精修 ⇒ 保留 stride 网格上的峰值（近似，但分类一致）。
    粗搜最高分 < 0.30 时视为明显无徽章，直接返回（不再精修）。
    """
    if not _away_template_ready():
        return -1.0
    th, tw = _AWAY_TMPL.shape
    win = _white_bin(frame, ax - radius, ay - radius, 2 * radius, 2 * radius)
    if win.shape[0] < th or win.shape[1] < tw:
        return -1.0
    gy = win.shape[0] - th + 1
    gx = win.shape[1] - tw + 1
    best, bdy, bdx = -1.0, 0, 0
    cs = _AWAY_COARSE_STRIDE
    for dy in range(0, gy, cs):
        for dx in range(0, gx, cs):
            v = _away_score(win, th, tw, dy, dx)
            if v > best:
                best, bdy, bdx = v, dy, dx
    if best < 0.30:                      # 明显无徽章：直接返回（不精修）
        return best
    r = _AWAY_REFINE_RADIUS
    y0, y1 = max(0, bdy - r), min(gy - 1, bdy + r)
    x0, x1 = max(0, bdx - r), min(gx - 1, bdx + r)
    for dy in range(y0, y1 + 1, stride):
        for dx in range(x0, x1 + 1, stride):
            v = _away_score(win, th, tw, dy, dx)
            if v > best:
                best = v
    return best


def allin_signature(frame: np.ndarray, ax: int, ay: int) -> dict:
    """亮牌路径：白卡面量 ＋ 彩带（红/teal/蓝）包夹白字行数。

    - 白卡面 = 锚点 ±150/±140 内 luma>190 中性大区（双卡同时亮出）；
    - 胶囊行 = 彩色带（行色彩 ≥60px）内、白像素左右 25px 均有彩的行；
    - 「过牌」绿标签（无白卡面）由白卡门拒绝；hero 自身亮牌（无胶囊）
      由胶囊门拒绝。
    """
    x0, y0 = max(0, ax - 150), max(0, ay - 140)
    x1, y1 = min(frame.shape[1], ax + 150), min(frame.shape[0], ay + 140)
    sub = frame[y0:y1, x0:x1]
    r = sub[..., 0].astype(np.int16)
    g = sub[..., 1].astype(np.int16)
    b = sub[..., 2].astype(np.int16)
    l = _luma(sub)
    white = ((l > _CARD_WHITE_LUMA) & (r > _CARD_WHITE_MIN)
             & (g > _CARD_WHITE_MIN) & (b > _CARD_WHITE_MIN))
    white_n = int(white.sum())
    red = ((r > 150) & (g < 95) & (b < 95)).astype(np.int16)
    teal = ((g > 120) & (g > r + 90) & (g > b + 20)).astype(np.int16)
    blue = ((b > 100) & (b > r + 40) & (b > g + 30)).astype(np.int16)
    wbin = white.astype(np.int16)
    col = red | teal | blue
    run = best_run = 0
    for y in range(col.shape[0]):
        row_c, row_w = col[y], wbin[y]
        cnt = 0
        for i in np.flatnonzero(row_w):
            a = max(0, i - 25)
            bb = min(row_c.shape[0], i + 26)
            if row_c[a:i].sum() >= 3 and row_c[i + 1:bb].sum() >= 3:
                cnt += 1
        if cnt >= 8 and int(row_c.sum()) >= 60:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    return {"white_n": white_n, "pill_rows": int(best_run)}


class StatusRecognizer1876:
    """1876 帧 → per-seat 状态（UNKNOWN-first，无 fallback/cache）。"""

    name = "coinpoker_status_v1876"

    def __init__(self):
        self.rois = STATUS_TAG_ROIS_1876
        self.seat_anchors = {
            "S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
            "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160),
        }

    def _seat_status(self, frame: np.ndarray, seat: str) -> dict:
        roi = self.rois[seat]
        ax, ay = self.seat_anchors[seat]
        teal = teal_coverage(frame, roi)
        disc = disconnected_signature(frame, roi)
        away = away_ncc(frame, ax, ay)
        allin = allin_signature(frame, ax, ay)
        hits = []
        if FOLDED_TEAL_MAX >= teal >= FOLDED_TEAL_MIN:
            hits.append("folded")
        if (disc["glass"] >= DISC_GLASS_MIN and disc["gray_n"] >= DISC_GRAY_MIN
                and disc["max_run"] >= DISC_GRAY_RUN_MIN):
            hits.append("disconnected")
        if away >= AWAY_NCC_MIN:
            hits.append("away")
        if (allin["white_n"] >= ALLIN_WHITE_MIN
                and allin["pill_rows"] >= ALLIN_CAPSULE_ROWS_MIN):
            hits.append("allin")
        if len(hits) > 1:
            return {"status": "UNKNOWN", "reason": "conflicting_evidence",
                    "evidence": {"teal": round(teal, 3), "disc": disc,
                                 "away_ncc": round(away, 3), "allin": allin,
                                 "hits": hits}}
        if not hits:
            return {"status": "UNKNOWN", "reason": "no_status_evidence",
                    "evidence": {"teal": round(teal, 3), "disc": disc,
                                 "away_ncc": round(away, 3), "allin": allin}}
        return {"status": hits[0], "reason": "visual_evidence",
                "evidence": {"teal": round(teal, 3), "disc": disc,
                             "away_ncc": round(away, 3), "allin": allin}}

    def recognize(self, frame: np.ndarray) -> dict:
        seats = {}
        for seat in sorted(self.rois):
            seats[seat] = self._seat_status(frame, seat)
        recognized = [s for s in seats if seats[s]["status"] != "UNKNOWN"]
        status = ("RECOGNIZED" if recognized
                  else "NOT_OBSERVED")
        return {"status": status, "seats": seats,
                "reason": ("status_recognized" if recognized
                           else "no_status_evidence")}
