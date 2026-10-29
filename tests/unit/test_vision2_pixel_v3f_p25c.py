"""V3F-P25-C — Status 1876 Production Implementation 单元测试。

覆盖（用户 spec 13 项）：
1. folded NORMAL；2. folded DIM；3. away；4. disconnected；5. allin；
6. 多状态冲突；7. 无 Status；8. 空白；9. 非 Status 灰字；10. equity 胶囊负例；
11. 1876 dispatch；12. 1419 regression；13. UNKNOWN propagation。
+ 22 GT exact（16 folded / 2 away / 1 disconnected / 3 allin，FP=0/FN=0）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.render_class import (  # noqa: E402
    EVIDENCE_LIMITED)
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    StatusObserver)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_status import (  # noqa: E402
    FOLDED_TEAL_MIN, StatusRecognizer1876, allin_signature,
    disconnected_signature, teal_coverage)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))

TAG = {"S1": (138, 325, 104, 41), "S2": (816, 211, 103, 40),
       "S3": (1494, 325, 103, 41), "S4": (101, 891, 103, 40),
       "S5": (1531, 891, 103, 40), "S6": (816, 1149, 103, 40)}
ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}


def _frame(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


def _blank():
    fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
    fb[..., :3] = (6, 35, 27)
    fb[..., 3] = 255
    return fb


# ── 1. 22 GT exact（主验证）──────────────────────────────────────────
GT_SEATS = {
    "f01": {"S3": "away", "S5": "folded", "S6": "folded"},
    "f02": {"S2": "folded", "S3": "away", "S4": "folded"},
    "f03": {"S1": "folded", "S2": "allin", "S3": "folded", "S4": "folded",
            "S5": "folded"},
    "f04": {"S1": "allin", "S2": "allin", "S3": "folded", "S4": "folded",
            "S5": "folded", "S6": "folded"},
    "f05": {"S2": "folded", "S3": "disconnected", "S6": "folded"},
    "f06": {"S3": "folded", "S6": "folded"},
}
FID = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
}


def test_22_gt_exact_match():
    """22 GT：16 folded / 2 away / 1 disconnected / 3 allin 全对。"""
    rec = StatusRecognizer1876()
    total = exact = 0
    fps, fns = [], []
    for fr, gts in GT_SEATS.items():
        arr = _frame(FID[fr])
        out = rec.recognize(arr)
        for seat in sorted(gts):
            got = out["seats"][seat]["status"]
            if got == gts[seat]:
                exact += 1
            else:
                (fns if got == "UNKNOWN" else fps).append((fr, seat, gts[seat],
                                                           got))
            total += 1
        # 未标注座不得产生状态（FP 负例）
        for seat in TAG:
            if seat in gts:
                continue
            assert out["seats"][seat]["status"] == "UNKNOWN", (fr, seat)
    assert total == 22
    assert exact == 22, f"exact={exact} fps={fps} fns={fns}"
    assert not fps and not fns


# ── 2. folded NORMAL ────────────────────────────────────────────────
def test_folded_normal():
    arr = _frame(FID["f01"])
    out = StatusRecognizer1876().recognize(arr)
    assert out["seats"]["S5"]["status"] == "folded"
    assert out["seats"]["S6"]["status"] == "folded"


def test_folded_dim():
    """暗场景（f03/f04 全局亮度 p50≈15-17）判据不变。"""
    arr = _frame(FID["f03"])
    out = StatusRecognizer1876().recognize(arr)
    for seat in ("S1", "S3", "S4", "S5"):
        assert out["seats"][seat]["status"] == "folded", seat


# ── 3. away ─────────────────────────────────────────────────────────
def test_away_detection():
    arr = _frame(FID["f01"])
    out = StatusRecognizer1876().recognize(arr)
    assert out["seats"]["S3"]["status"] == "away"


# ── 4. disconnected ─────────────────────────────────────────────────
def test_disconnected_detection():
    arr = _frame(FID["f05"])
    out = StatusRecognizer1876().recognize(arr)
    assert out["seats"]["S3"]["status"] == "disconnected"


# ── 5. allin ────────────────────────────────────────────────────────
def test_allin_detection():
    arr = _frame(FID["f04"])
    out = StatusRecognizer1876().recognize(arr)
    assert out["seats"]["S1"]["status"] == "allin"
    assert out["seats"]["S2"]["status"] == "allin"


# ── 6. 多状态冲突 → UNKNOWN ─────────────────────────────────────────
def test_conflicting_evidence_unknown():
    """合成帧同座命中 folded-teal 与 allin → UNKNOWN（不猜、不仲裁）。"""
    fb = _blank().copy()
    fb[..., 3] = 255
    # 1) folded 徽章：TAG ROI teal 填充 + 深色字形（teal≈0.71 ∈ [0.5,0.95]）
    x0, y0, w, h = TAG["S2"]
    fb[y0:y0 + h, x0:x0 + w, :3] = (28, 53, 54)
    fb[y0 + 6:y0 + 30, x0 + 20:x0 + 70, :3] = (13, 16, 23)
    # 2) allin：白卡 + 红胶囊带白字（flanked-white 行 ≥20）
    #    布局避开 TAG ROI（tag x 816-919, y 211-251）：卡片放窗口右侧
    ax, ay = ANCH["S2"]
    fb[ay - 140:ay + 40, ax + 12:ax + 148, :3] = (246, 245, 245)
    by0 = ay + 95
    fb[by0:by0 + 60, ax - 60:ax + 120, :3] = (215, 49, 46)
    for sx in range(ax - 50, ax + 110, 12):        # 白竖条 → 每行 flanked
        fb[by0 + 6:by0 + 54, sx:sx + 6, :3] = (255, 255, 255)
    rec = StatusRecognizer1876()
    out = rec.recognize(fb)
    assert out["seats"]["S2"]["status"] == "UNKNOWN"
    assert out["seats"]["S2"]["reason"] == "conflicting_evidence"


# ── 7/8. 无 Status / 空白帧 ─────────────────────────────────────────
def test_no_status_blank_frame():
    fb = _blank()
    out = StatusRecognizer1876().recognize(fb)
    assert out["status"] in ("NOT_OBSERVED", "RECOGNIZED")
    for seat in TAG:
        assert out["seats"][seat]["status"] == "UNKNOWN", seat


def test_blank_no_crash_observer():
    out = StatusObserver().observe(_blank().tobytes(), 1876, 1400)
    for seat, v in out["seats"].items():
        assert v["status"] == "UNKNOWN"


# ── 9. 非 Status 灰字（wolf 头像灰声 f06 S1）─────────────────────────
def test_avatar_gray_noise_not_disconnected():
    arr = _frame(FID["f06"])
    out = StatusRecognizer1876().recognize(arr)
    assert out["seats"]["S1"]["status"] == "UNKNOWN"      # 狼毛灰字负例


# ── 10. equity 胶囊负例（过牌绿标签 f02 S1 无白卡面）─────────────────
def test_check_green_tag_not_allin():
    arr = _frame(FID["f02"])
    out = StatusRecognizer1876().recognize(arr)
    # f02 S1 = 过牌 绿标签（P25-C 目视确认）——非 allin
    assert out["seats"]["S1"]["status"] != "allin"


def test_hero_own_cards_not_allin():
    """hero 自身白卡（f03 S6 白卡 15.5k 无胶囊）→ 非 allin。"""
    arr = _frame(FID["f03"])
    out = StatusRecognizer1876().recognize(arr)
    assert out["seats"]["S6"]["status"] != "allin"


# ── 11. 1876 dispatch ───────────────────────────────────────────────
def test_dispatch_1876_real_frame():
    out = StatusObserver().observe(_frame(FID["f01"]).tobytes(), 1876, 1400)
    assert out["window"] == "1876"
    assert out["status"] != EVIDENCE_LIMITED
    assert out["seats"]["S5"]["status"] == "folded"


def test_dispatch_unknown_geometry_evidence_limited():
    fb = np.zeros((800, 1000, 4), dtype=np.uint8)
    out = StatusObserver().observe(fb.tobytes(), 1000, 800)
    assert out["status"] == EVIDENCE_LIMITED


# ── 12. 1419 regression ─────────────────────────────────────────────
def test_1419_path_unchanged():
    w, h, rgba = decode_png((EV /
        "SETTLED_BOARD_CHECK_961_1787653653866_full.png").read_bytes())
    out = StatusObserver().observe(rgba, w, h)
    assert out["window"] == "1419"
    assert out["status"] in ("NOT_OBSERVED", "RECOGNIZED", "UNKNOWN",
                             "EVIDENCE_LIMITED")


# ── 13. UNKNOWN propagation ─────────────────────────────────────────
def test_unknown_not_inherited():
    obs = StatusObserver()
    obs.observe(_frame(FID["f01"]).tobytes(), 1876, 1400)
    fb = _blank()
    out = obs.observe(fb.tobytes(), 1876, 1400)
    for seat, v in out["seats"].items():
        assert v["status"] == "UNKNOWN"


# ── 单元原语（阈值即测量值，不调低）─────────────────────────────────
def test_thresholds_match_measurements():
    """阈值来自 P25-B.3/P25-C 实测分布（正例底线 / 负例顶线之间）。"""
    assert FOLDED_TEAL_MIN == 0.5
    # 正例 0.81-0.82，负例 ≤0.25 → away NCC 0.6
    arr = _frame(FID["f01"])
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_status import (
        away_ncc, AWAY_NCC_MIN)
    assert AWAY_NCC_MIN == 0.6
    assert away_ncc(arr, *ANCH["S3"]) >= AWAY_NCC_MIN
    assert away_ncc(arr, *ANCH["S1"]) < AWAY_NCC_MIN


def test_teal_cov_separates():
    arr = _frame(FID["f01"])
    assert teal_coverage(arr, TAG["S5"]) >= FOLDED_TEAL_MIN
    assert teal_coverage(arr, TAG["S1"]) < FOLDED_TEAL_MIN


def test_disc_signature_separates():
    d = disconnected_signature(_frame(FID["f05"]), TAG["S3"])
    assert d["gray_n"] >= 250 and d["max_run"] >= 12
    w = disconnected_signature(_frame(FID["f06"]), TAG["S1"])
    assert w["max_run"] < 12        # 狼毛散斑


def test_allin_signature_separates():
    assert allin_signature(_frame(FID["f03"]), *ANCH["S2"])["pill_rows"] >= 20
    assert allin_signature(_frame(FID["f01"]), *ANCH["S5"])["pill_rows"] < 20
    # hero 自己亮牌：白卡多但无胶囊
    s = allin_signature(_frame(FID["f03"]), *ANCH["S6"])
    assert s["white_n"] >= 10000 and s["pill_rows"] < 20
