"""V3F-P13-ACTION FOUNDATION — 离线回归测试。

- 真实证据帧（第5轮+补采 1876×1400）→ 按钮可见帧聚类 → 用户标注对齐；
- 断言：fold 按钮必然在（行存在判据）；raise 按钮与用户标注一致；
  callcheck 输出统一类目 call_or_check（Call/Check 视觉不可分——诚实合并，
  绝不强行拆分）；
- render-class 隔离：1419/1280 → EVIDENCE_LIMITED；
- 无按钮行帧 → NOT_OBSERVED；UNKNOWN-first；无 OCR/文字语义/跨字段。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_action import (  # noqa: E402
    ACTION_WINDOW, ActionRecognizer)

EV = Path("data/vision2/coinpoker_evidence")
SINCE5 = 134324658401327108
SINCE6 = 134324688462724951
ANNOT5 = {2: "fold", 3: "check", 4: "call", 6: "fold", 8: "fold", 9: "fold",
          10: "raise_pre", 11: "call", 12: "raise", 13: "allin", 14: "fold",
          15: "fold", 16: "fold", 17: "fold", 19: "call", 20: "fold",
          21: "fold", 22: "call", 23: "fold", 25: "fold"}
ANNOT6 = {1: "check", 2: "raise", 3: "raise", 4: "check"}

REC = ActionRecognizer()


def _load(p: Path) -> np.ndarray:
    w, h, rgba = decode_png(p.read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _ts(p: Path) -> float:
    return int(re.search(r"_(\d+)_full$", p.stem).group(1)) / 1000.0


def _frames(since_ft: int) -> list:
    since_s = since_ft / 1e7 - 11644473600
    frames = [p for p in sorted(EV.glob("*_full.png"))
              if p.stat().st_mtime >= since_s]
    frames.sort(key=_ts)
    return frames


def _red_row(arr: np.ndarray) -> int:
    """与识别器同判据：fold 区域 (1116-1354, 1263-1392) 红掩码。"""
    r = arr[..., 0].astype(np.int16)
    g = arr[..., 1].astype(np.int16)
    b = arr[..., 2].astype(np.int16)
    m = ((r[1263:1392, 1116:1354] > g[1263:1392, 1116:1354] + 50)
         & (r[1263:1392, 1116:1354] > b[1263:1392, 1116:1354] + 50)
         & (r[1263:1392, 1116:1354] > 110))
    return int(m.sum())


# 模块级缓存：frames + occurrence 聚类（解码一次复用）
_OCC = {}


def _occ(since_ft: int):
    if since_ft in _OCC:
        return _OCC[since_ft]
    frames = _frames(since_ft)
    marks = [(i, p, _red_row(_load(p))) for i, p in enumerate(frames)]
    occ, cur = [], None
    for i, p, m in marks:
        if m > 12000:
            if cur is None:
                cur = [i, i]
            elif _ts(frames[i]) - _ts(frames[cur[1]]) < 20:
                cur[1] = i
            else:
                occ.append(cur)
                cur = [i, i]
    if cur:
        occ.append(cur)
    _OCC[since_ft] = (frames, occ)
    return frames, occ


# ---------------------------------------------------------------- 真实证据闭环

def test_all_button_frames_fold_present():
    """全部按钮可见帧（fold 区域红掩码>12000）：fold 按钮必然存在。"""
    n_checked = 0
    for since in (SINCE5, SINCE6):
        frames, occ = _occ(since)
        for s, e in occ:
            for i in range(s, e + 1):
                arr = _load(frames[i])
                if _red_row(arr) <= 12000:
                    continue              # 簇边界无按钮帧跳过
                res = REC.recognize(arr)
                assert res["status"] == "RECOGNIZED", frames[i].stem
                assert res["actions"]["fold"] is True
                n_checked += 1
    assert n_checked >= 100


def test_raise_button_matches_user_annotations():
    """raise/allin 标注帧（按钮可见帧内）：raise 按钮必在。"""
    frames5, occ5 = _occ(SINCE5)
    frames6, occ6 = _occ(SINCE6)
    checked = 0
    for n, (s, e) in enumerate(occ5, 1):
        if ANNOT5.get(n) in ("raise", "allin", "raise_pre"):
            for i in range(s, e + 1):
                arr = _load(frames5[i])
                if _red_row(arr) <= 12000:
                    continue
                assert REC.recognize(arr)["actions"]["raise"] is True, \
                    frames5[i].stem
                checked += 1
    for n, (s, e) in enumerate(occ6, 1):
        if ANNOT6.get(n) == "raise":
            for i in range(s, e + 1):
                arr = _load(frames6[i])
                if _red_row(arr) <= 12000:
                    continue
                assert REC.recognize(arr)["actions"]["raise"] is True, \
                    frames6[i].stem
                checked += 1
    assert checked >= 3


def test_callcheck_unified_category():
    """Call/Check 视觉不可分 → 统一类目 call_or_check（绝不输出 call/check）。"""
    frames5, occ5 = _occ(SINCE5)
    seen = set()
    for n, (s, e) in enumerate(occ5, 1):
        for i in range(s, e + 1):
            arr = _load(frames5[i])
            if _red_row(arr) <= 12000:
                continue
            res = REC.recognize(arr)
            cc = res["actions"]["callcheck"]
            assert cc in ("call_or_check", "UNKNOWN"), cc
            seen.add(cc)
    assert "call_or_check" in seen


def test_check_and_call_occurrences_same_output():
    """check 与 call 标注的 occurrence 输出一致（合并类目——诚实不做区分）。"""
    frames5, occ5 = _occ(SINCE5)
    for n in (3, 4):          # occ3=check, occ4=call
        s, e = occ5[n - 1]
        outs = set()
        for i in range(s, e + 1):
            arr = _load(frames5[i])
            if _red_row(arr) <= 12000:
                continue
            outs.add(REC.recognize(arr)["actions"]["callcheck"])
        assert "call_or_check" in outs
        assert "call" not in outs and "check" not in outs


# ---------------------------------------------------------------- UNKNOWN 纪律

def test_no_row_not_observed():
    """无按钮行帧 → NOT_OBSERVED（不猜 0/空集）。"""
    frames = _frames(SINCE5)
    for p in frames:
        if _red_row(_load(p)) <= 12000:
            res = REC.recognize(_load(p))
            assert res["status"] in ("NOT_OBSERVED", "EVIDENCE_LIMITED")
            return
    pytest.skip("无空闲帧")


def test_render_class_isolation():
    """1419×1066 / 1280×779 / 1966×1106 → EVIDENCE_LIMITED（按钮不可见类）。"""
    for shape in ((1066, 1419, 4), (779, 1280, 4), (1106, 1966, 4)):
        res = REC.recognize(np.zeros(shape, dtype=np.uint8))
        assert res["status"] == "EVIDENCE_LIMITED"
        assert res["actions"]["fold"] is None


def test_unknown_first_semantics():
    """UNKNOWN-first：无行 → fold=False（明确），raise/callcheck=None（未定义）。"""
    frames = _frames(SINCE5)
    for p in frames[:40]:
        res = REC.recognize(_load(p))
        if res["status"] == "NOT_OBSERVED":
            assert res["actions"]["fold"] is False
            assert res["actions"]["raise"] is None
            assert res["actions"]["callcheck"] is None


# ---------------------------------------------------------------- 模块边界

def test_module_no_text_semantics_no_cross_field():
    """识别器主体零 OCR/文字语义/跨字段：不引用 stack/bet/status/pot 模块。"""
    import poker_decision_ai.vision2.pixel.realclient.coinpoker_action as A
    src = Path(A.__file__).read_text(encoding="utf-8")
    body = src[src.index("class ActionRecognizer"):]
    for bad in ("coinpoker_stack", "coinpoker_bet", "cashgame",
                "seat_position", "recognize_text", "frame_id",
                "previous", "cache", "timestamp"):
        assert bad not in body, bad


def test_window_geometry_constant():
    assert ACTION_WINDOW == (1876, 1400)
    rec = ActionRecognizer(window="1876")
    assert rec.name == "coinpoker_action_foundation_v1"
