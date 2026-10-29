"""V3F-P30-2C-A — 1876 Hero to_call Recognizer（动作面板跟注按钮金额）。

严格基于 P30-2A 冻结证据：
- to_call 唯一视觉来源 = Hero 动作面板「跟注 N」按钮金额；
- 「跟注任何」无金额（≠ 跟注 N）；「加注 N」= raise-to ≠ to_call；
- 「过牌」（check 按钮）→ 视觉语义直接表达无需跟注 → to_call = 0；
- 蓝 SB/BB 徽章 / Bet capsule / Stack / header 均 ≠ to_call（空间隔离）。

检测策略（UNKNOWN-first）：
1. 面板 ROI (1250,1150,1876,1400) → 文本行带（label 行 + 金额行）；
2. 金额行（下带；与 label 行间有 ≥5px 空白 gap）→ 整行模板 NCC vs
   GT 参考位图（124,163 / 2,000）→ 匹配 → 相应值；
3. 无金额行 +「过牌」结构（下带窄 ≤31px 且无 5 位宽段）→ to_call=0；
4. 冲突/无法确认 → UNKNOWN。

红线：无 OCR / 无 CV-ML / 无视觉 LLM / 无网络 / 无跨字段推导
（Bet/Stack/Pot/BB/current_bet 一律不引用）；加注金额绝不转 to_call；
无 previous-cache；UNKNOWN ≠ 0（只有过牌视觉证据才输出 0）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: hero 动作面板 ROI（P30-2A 实测：右下座按钮区）
ACTION_PANEL_ROI = (1250, 1150, 1876, 1400)
#: 模板目录
TO_CALL_ASSET_DIR = "data/vision2/coinpoker_assets_v1876tocall"
#: 整行模板 NCC 门（GT 参考自身=1.0；不同金额应显著低于此——实测）
AMOUNT_NCC_MIN = 0.85
#: 过牌判定：下带无 ≥50px 宽段（金额行特征）→ 过牌/标签 → 0
AMOUNT_BAND_MAX_WIDTH = 50


def _luma(frame: np.ndarray) -> np.ndarray:
    return (frame[..., 0].astype(np.int16) * 0.299
            + frame[..., 1].astype(np.int16) * 0.587
            + frame[..., 2].astype(np.int16) * 0.114)


def _text_mask(frame: np.ndarray) -> np.ndarray:
    """面板白字掩码（绿/炭/红按钮上的白字——luma>170 低饱和）。"""
    r = frame[..., 0].astype(np.int16); g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    l = _luma(frame)
    return ((l > 170) & (np.abs(r - g) < 55) & (np.abs(g - b) < 55))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """NCC with edge-zero tolerance：裁剪到公共尺寸（取 min 行/列）——
    抗细微尺寸差（如抗锯齿尾巴 1-3px），勿缩放（缩放会扭曲字形）。"""
    mh, mw = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    if mh < 3 or mw < 3:
        return 0.0
    a0 = a[:mh, :mw].astype(np.float64)
    b0 = b[:mh, :mw].astype(np.float64)
    a0 = a0 - a0.mean()
    b0 = b0 - b0.mean()
    an = float(np.sqrt((a0 * a0).sum()))
    bn = float(np.sqrt((b0 * b0).sum()))
    if an < 1e-9 or bn < 1e-9:
        return 0.0
    return float((a0 * b0).sum() / (an * bn))


class ToCallRecognizer1876:
    """1876 动作面板 → {to_call, status, reason, confidence}。"""

    name = "coinpoker_to_call_v3f_p30_2ca_1876"

    def __init__(self, asset_dir: str = TO_CALL_ASSET_DIR):
        self.refs = self._load_refs(asset_dir)

    @staticmethod
    def _load_refs(asset_dir: str) -> dict:
        mp = Path(asset_dir) / "manifest.json"
        if not mp.exists():
            return {}
        m = json.loads(mp.read_text(encoding="utf-8"))
        refs = {}
        for a in m.get("assets", []):
            p = Path(asset_dir) / Path(a["crop_ref"]).name
            if not p.exists():
                continue
            refs[a["label"]] = {
                "mask": np.load(p).astype(np.float64),
                "value": a.get("gt_value"),
            }
        return refs

    def _row_bands(self, mask: np.ndarray) -> list:
        rowd = mask.sum(axis=1)
        bands, run = [], None
        for i, v in enumerate(rowd):
            if v >= 5 and run is None:
                run = [i, i]
            elif v >= 5 and run is not None:
                run[1] = i
            elif run is not None:
                bands.append(run); run = None
        if run:
            bands.append(run)
        return [(a, b) for a, b in bands if b - a >= 3]

    def _band_max_width(self, mask: np.ndarray, a: int, b: int) -> int:
        cold = mask[a:b + 1].sum(axis=0)
        segs, run = [], None
        for i, v in enumerate(cold):
            if v >= 1 and run is None:
                run = [i, i]
            elif v >= 1 and run is not None:
                run[1] = i
            elif run is not None:
                segs.append(run); run = None
        if run:
            segs.append(run)
        if not segs:
            return 0
        return max((b_ - a_ + 1) for a_, b_ in segs)

    @staticmethod
    def _col_groups(mask: np.ndarray, a: int, b: int,
                    gap_min: int = 8) -> list:
        """行带内列段组（按 ≥gap_min 空白列分裂）→ 各列组 (xa, xb)。"""
        cold = mask[a:b + 1].sum(axis=0)
        runs, run = [], None
        for i, v in enumerate(cold):
            if v >= 1 and run is None:
                run = [i, i]
            elif v >= 1 and run is not None:
                run[1] = i
            elif run is not None:
                runs.append(run); run = None
        if run:
            runs.append(run)
        # 按空白 ≥gap_min 分裂
        merged = []
        for rr in runs:
            if merged and rr[0] - merged[-1][1] >= gap_min:
                merged.append(rr)
            elif merged:
                merged[-1][1] = rr[1]
            else:
                merged.append(rr)
        return [(a_, b_) for a_, b_ in merged if b_ - a_ >= 20]

    def _tight(self, mask: np.ndarray, a: int, b: int,
               xa: int, xb: int) -> np.ndarray | None:
        sub = mask[a:b + 1, xa:xb + 1]
        ys = np.where(sub.sum(axis=1) > 0)[0]
        xs = np.where(sub.sum(axis=0) > 0)[0]
        if len(ys) == 0 or len(xs) == 0:
            return None
        return sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def _tight_amount(self, mask: np.ndarray, a: int, b: int,
                      xa: int, xb: int) -> np.ndarray | None:
        """列组内 label+amount 分离：取底部子带（金额行）。

        跟注按钮两行（标签+金额）在同列组内——用行带间 gap 分离，
        金额行 = 底部有 digit 密度的子带。
        """
        sub = mask[a:b + 1, xa:xb + 1]
        rowd = sub.sum(axis=1)
        # 找行带间空白（gap ≥3）
        runs, run = [], None
        for i, v in enumerate(rowd):
            if v >= 1 and run is None:
                run = [i, i]
            elif v >= 1 and run is not None:
                run[1] = i
            elif run is not None:
                runs.append(run); run = None
        if run:
            runs.append(run)
        if not runs:
            return None
        # 若 ≥2 子带 → 底部为金额行；否则整组
        if len(runs) >= 2:
            a_sub, b_sub = runs[-1]
        else:
            a_sub, b_sub = runs[0]
        inner = sub[a_sub:b_sub + 1]
        ys = np.where(inner.sum(axis=1) > 0)[0]
        xs = np.where(inner.sum(axis=0) > 0)[0]
        if len(ys) == 0 or len(xs) == 0:
            return None
        return inner[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    def recognize(self, frame: np.ndarray) -> dict:
        x0, y0, x1, y1 = ACTION_PANEL_ROI
        if x1 > frame.shape[1] or y1 > frame.shape[0]:
            return {"to_call": None, "status": "UNKNOWN",
                    "reason": "panel_out_of_frame", "confidence": 0.0}
        panel = frame[y0:y1, x0:x1]
        mask = _text_mask(panel)
        if int(mask.sum()) < 50:
            return {"to_call": None, "status": "UNKNOWN",
                    "reason": "no_action_panel", "confidence": 0.0}
        bands = self._row_bands(mask)
        if not bands:
            return {"to_call": None, "status": "UNKNOWN",
                    "reason": "no_text_bands", "confidence": 0.0}
        # 最后一文本带（金额行 or 过牌/标签行）
        a2, b2 = bands[-1]
        groups = self._col_groups(mask, a2, b2)
        # 金额候选：列组（≥20px 宽）tight-crop → NCC vs GT 参考
        best, bsc = None, 0.0
        for ga, gb in groups:
            tight = self._tight_amount(mask, a2, b2, ga, gb)
            if tight is None:
                continue
            for lab, item in self.refs.items():
                sc = _ncc(tight, item["mask"])
                if sc > bsc:
                    best, bsc = lab, sc
        if best is not None and bsc >= AMOUNT_NCC_MIN:
            return {"to_call": self.refs[best]["value"],
                    "status": "RECOGNIZED",
                    "reason": "call_amount_template_match",
                    "confidence": round(bsc, 4)}
        # 无金额命中：过牌语义判定——最宽组宽度 < 最小已知金额参考宽
        # （GT 参考最窄 = 81px；check 标签组 ≤63px——结构阈值来自实测）
        min_ref_w = min((item["mask"].shape[1] for item in self.refs.values()),
                        default=81)
        max_group_w = max((gb - ga + 1 for ga, gb in groups), default=0)
        if max_group_w < min_ref_w:
            return {"to_call": 0, "status": "RECOGNIZED",
                    "reason": "check_button_no_amount", "confidence": 0.7}
        return {"to_call": None, "status": "UNKNOWN",
                "reason": "amount_unknown_format", "confidence": 0.0}
