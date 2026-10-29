"""V3F-P17-A — 数据记录层 + GT loader 单元测试。

覆盖：TriggeredEvidence 每帧记录 frame_geometry/render_class；
GT loader 严格校验（render_class=1876 / frame 可追溯 / geometry 一致 /
字段白名单 / null≠0 / Action 词禁作 status / 确定性）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2.pixel.realclient.annotate import (  # noqa: E402
    TriggeredEvidence, _render_class_for)
from poker_decision_ai.vision2.pixel.realclient.gt_loader import (  # noqa: E402
    ALLOWED_FIELDS, gt_summary, validate_gt)

EV = Path("data/vision2/coinpoker_evidence")
REAL_FRAME = "BUTTON_CHANGED_167_1787992682749_full.png"   # 1876×1400 真实帧


# ── 数据记录层：TriggeredEvidence ─────────────────────────────────────
def test_triggered_evidence_records_geometry_and_render_class():
    ev = TriggeredEvidence(frame_id=1, timestamp=123.0, trigger="X",
                           trigger_roi="r", digest_before=None,
                           digest_after="d", full_frame_path="p",
                           frame_geometry=[1876, 1400], render_class="1876")
    d = ev.to_dict()
    assert d["frame_geometry"] == [1876, 1400]
    assert d["render_class"] == "1876"


def test_render_class_for_mapping():
    assert _render_class_for(1876, 1400) == "1876"
    assert _render_class_for(1419, 1066) == "1419"
    assert _render_class_for(1280, 779) == "1280"
    assert _render_class_for(1848, 1388) == "1848"
    assert _render_class_for(100, 100) is None          # 未知几何 → None


# ── GT loader ────────────────────────────────────────────────────────
def _write_gt(tmp_path, gt):
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(gt), encoding="utf-8")
    return p


def _valid_gt():
    return {
        "schema_version": "v1",
        "render_class": "1876",
        "annotations": [
            {"frame_id": REAL_FRAME, "geometry": [1876, 1400],
             "fields": {
                 "bet": {"S1": 21600, "S2": None},
                 "pot": 23280,
                 "stack": {"S1": 100000, "S6": 1300000},
                 "hero_cards": ["Ks", "9h"],
                 "status": {"S2": "folded"},
                 "position": {"dealer": "S3",
                              "occupied": {"S1": "occupied", "S3": "occupied",
                                           "S4": "occupied", "S5": "occupied",
                                           "S6": "occupied"},
                              "hero": "S6"},
             }},
        ],
    }


def test_valid_gt_passes():
    gt = _valid_gt()
    errors = validate_gt(gt, EV)
    assert errors == []
    s = gt_summary(gt)
    assert s["bet"]["annotated"] == 1          # S1=21600 计 1
    assert s["bet"]["invisible"] == 1          # S2=None 计 invisible
    assert s["pot"]["annotated"] == 1
    assert s["hero_cards"]["annotated"] == 1


def test_wrong_render_class_rejected():
    gt = _valid_gt()
    gt["render_class"] = "1419"
    errors = validate_gt(gt, EV)
    assert any("render_class" in e for e in errors)


def test_missing_frame_rejected():
    gt = _valid_gt()
    gt["annotations"][0]["frame_id"] = "NOPE_123_full.png"
    errors = validate_gt(gt, EV)
    assert any("不存在" in e for e in errors)


def test_geometry_mismatch_rejected():
    gt = _valid_gt()
    gt["annotations"][0]["geometry"] = [100, 100]
    errors = validate_gt(gt, EV)
    assert any("不一致" in e for e in errors)


def test_unknown_field_rejected():
    gt = _valid_gt()
    gt["annotations"][0]["fields"]["action"] = {"S1": "fold"}
    errors = validate_gt(gt, EV)
    assert any("不是允许字段" in e for e in errors)
    assert "action" not in ALLOWED_FIELDS


def test_null_not_interpreted_as_zero():
    gt = _valid_gt()
    # bet S2=None 合法且计 invisible（非 0）
    errors = validate_gt(gt, EV)
    assert errors == []
    s = gt_summary(gt)
    assert s["bet"]["invisible"] == 1


def test_action_word_rejected_as_status():
    gt = _valid_gt()
    gt["annotations"][0]["fields"]["status"] = {"S2": "fold"}   # Action 词
    errors = validate_gt(gt, EV)
    assert any("fold" in e for e in errors)


def test_negative_bet_rejected():
    gt = _valid_gt()
    gt["annotations"][0]["fields"]["bet"] = {"S1": -5}
    errors = validate_gt(gt, EV)
    assert any("非负" in e for e in errors)


def test_empty_annotations_rejected():
    gt = {"render_class": "1876", "annotations": []}
    errors = validate_gt(gt, EV)
    assert any("非空" in e for e in errors)


def test_deterministic_repeat():
    gt = _valid_gt()
    assert validate_gt(gt, EV) == validate_gt(gt, EV)
    assert gt_summary(gt) == gt_summary(gt)


# ── P17-B：实际 GT 文件 + disconnected 状态 ────────────────────────────
def test_disconnected_status_accepted():
    gt = _valid_gt()
    gt["annotations"][0]["fields"]["status"] = {"S3": "disconnected"}
    assert validate_gt(gt, EV) == []


def test_built_gt_file_valid_and_counts():
    """实际 GT 文件通过校验，且字段覆盖达门槛（P21 视觉修正后）。"""
    gt = json.loads(Path("data/vision2/coinpoker_1876_gt.json")
                    .read_text(encoding="utf-8"))
    assert validate_gt(gt, EV) == []
    s = gt_summary(gt)
    assert s["pot"]["annotated"] >= 5
    assert s["stack"]["annotated"] >= 5
    assert s["position"]["annotated"] >= 3
    assert s["status"]["annotated"] >= 3
    assert s["bet"]["annotated"] >= 5          # P21 修正后 ≥5（S6=21600 加入）
    assert s["hero_cards"]["annotated"] == 1   # 仍缺 2 个独立 hero 手牌
    # null 语义：hero_cards 5 帧 null（不可见），非 0
    assert s["hero_cards"]["invisible"] >= 5
