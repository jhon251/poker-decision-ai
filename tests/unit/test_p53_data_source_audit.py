"""V3F-P53 — Data Source Discovery & Audit 测试（只读审计；不接入生产）。

覆盖：审计注册表结构一致性；ground-truth 类型合法；P52 contract 映射
（pluribus 候选 → decision_grade；handhq pseudo-hero → behavior_grade；
wsop → 排除）；真实本地数据抽查（pluribus 行字段完备性 + handhq
pseudo-hero showdown=None 实证）；不修改任何策略数学。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from poker_decision_ai.decision_grade import (
    BEHAVIOR_GRADE,
    DECISION_GRADE,
    classify_record,
    dataset_verdict,
)

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs" / "V3F_P53_DATASET_AUDIT.json"
RAW_PARQUET_DIR = Path("C:/PokerDataset/UofT_CPRG/data")
UOFT_NORM = ROOT / "data" / "uoft" / "normalized"
GT_TYPES = {"real_observed_action", "solver_label", "human_annotation",
            "synthetic", "unknown"}


@pytest.fixture
def audit():
    return json.loads(AUDIT.read_text(encoding="utf-8"))


def _candidate(audit, cid):
    return next(c for c in audit["candidates"] if c["id"] == cid)


# ------------------------------------------------------------ 注册表一致性
def test_audit_registry_structure(audit):
    assert audit["audit_version"] == "p53-dataset-audit-v1"
    ids = [c["id"] for c in audit["candidates"]]
    assert len(ids) == len(set(ids)) and len(ids) >= 6
    for c in audit["candidates"]:
        for key in ("license_access", "hand_count", "game_mode",
                    "hero_hole_cards", "observed_hero_action",
                    "ground_truth_type", "p52_contract_mapping",
                    "research_use_allowed", "leakage_risk"):
            assert key in c, (c["id"], key)


def test_ground_truth_types_are_real(audit):
    """solver/benchmark 标签绝不允许标成 real_observed_action。"""
    for c in audit["candidates"]:
        gt = c["ground_truth_type"]
        tokens = set(gt.split(" / ")) if isinstance(gt, str) else {gt}
        assert tokens <= GT_TYPES, (c["id"], gt)
    acpc = _candidate(audit, "acpc-competition-archives")
    assert acpc["ground_truth_type"] == "solver_label"
    bench = _candidate(audit, "pokerbench-style-solver-benchmarks")
    assert bench["ground_truth_type"] == "solver_label"
    # pluribus：人类座位 real_observed_action + AI 座位必须显式排除
    pl = _candidate(audit, "uoft-cprg::pluribus-human-seats")
    assert "real_observed_action" in pl["ground_truth_type"]
    assert "solver_label" in pl["ground_truth_type"]


def test_mode_conclusions_declared(audit):
    mc = audit["mode_conclusions"]
    assert "decision-grade 候选存在" in mc["CASH"]
    assert "NOT VALIDATABLE" in mc["PKO/KO/Mystery"]


# ------------------------------------------------------------ P52 contract 映射
def test_pluribus_candidate_maps_to_decision_grade():
    """Pluribus 行 → P52 字段语义（依据 raw parquet 实测结构）→
    cash 模式下 decision_grade。"""
    rec = {
        "hand_id": "pluribus/98/161.phh|161",
        "player_id": "MrPink",                       # 人类座位
        "hero_hole_cards": ["8s", "Tc"],             # 'd dh p2 Qh5h' 等
        "board_by_street": ["Jh", "4s", "7d"],       # 'd db Jh4s7d'
        "pot_effective_stack": {"starting_stacks": [10000.0] * 6,
                                "blinds": [50.0, 100.0]},
        "position_button": {"blinds_or_straddles": [50.0, 100.0],
                            "seat_count": 6},
        "legal_actions": "derived_from_complete_state",
        "action_sequence_sizes": ["p3 f", "p4 f", "p6 cbr 225"],
        "observed_hero_action": "cc",                # 真实人类动作
        "opponent_identity_actions": ["Pluribus", "Eddie", "MrOrange"],
        "game_mode": "cash",
        "blind_level": "50/100",
        # payout/bounty 对 cash 可选——缺省
    }
    fv = classify_record(rec, "cash")
    assert fv["hero_hole_cards"]["verdict"] == "usable"
    assert fv["observed_hero_action"]["verdict"] == "usable"
    assert dataset_verdict(fv, "cash")["verdict"] == DECISION_GRADE


def test_handhq_pseudo_hero_maps_to_behavior_grade():
    """handhq pseudo-hero（缺 stack）在 strict 契约下 = behavior_grade。"""
    rec = {
        "hand_id": "3461681234", "player_id": "qAu8LUEKtuCGlD",
        "hero_hole_cards": ["Td", "Th"],
        "board_by_street": ["7h", "7c", "4h"],
        "pot_effective_stack": {"verdict": "partially_usable",
                                "reason": "pot derivable; stack missing"},
        "position_button": {"button_seat": 1},
        "legal_actions": "derived_except_allin",
        "action_sequence_sizes": ["post 5", "post 10", "raise 25"],
        "observed_hero_action": "raise",
        "opponent_identity_actions": ["+hDnS3qG", "E/t2lYei"],
        "game_mode": "cash", "blind_level": "NL1000",
    }
    fv = classify_record(rec, "cash")
    assert fv["pot_effective_stack"]["verdict"] == "partially_usable"
    assert dataset_verdict(fv, "cash")["verdict"] == BEHAVIOR_GRADE


# ------------------------------------------------------------ 真实本地抽查
@pytest.mark.skipif(not RAW_PARQUET_DIR.exists(),
                    reason="raw parquet not present")
def test_pluribus_rows_field_completeness():
    """raw parquet pluribus 行实测：10,000 手、6 座、5 人类、栈/盲注/
    发牌/动作字段完备（只读）。"""
    import glob

    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(RAW_PARQUET_DIR / "train-*.parquet")))
    t = pq.read_table(files[0], columns=["source_file", "actions",
                                         "players", "starting_stacks",
                                         "blinds_or_straddles"])
    starts = pc.starts_with(t.column("source_file"), "pluribus")
    starts_list = starts.to_pylist()
    n = sum(1 for b in starts_list if b)
    assert n == 10000
    i0 = starts_list.index(True)
    players = t.column("players")[i0].as_py()
    stacks = t.column("starting_stacks")[i0].as_py()
    blinds = t.column("blinds_or_straddles")[i0].as_py()
    actions = t.column("actions")[i0].as_py()
    assert len(players) == 6 and sum(1 for p in players if p != "Pluribus") == 5
    assert stacks and all(s == 10000.0 for s in stacks)
    assert blinds[:2] == [50.0, 100.0]
    assert any(a.startswith("d dh") for a in actions)      # 发牌行存在
    assert any(a.startswith("d db") for a in actions)      # 公共牌行存在


@pytest.mark.skipif(not UOFT_NORM.exists(),
                    reason="UofT normalized data not present")
def test_handhq_pseudo_hero_showdown_none_proof():
    """handhq 单牌主手牌实测：牌从发牌即知（showdown=None）→ 真实
    player-view 残留，非摊牌揭示。"""
    found = 0
    for cf in sorted(UOFT_NORM.glob("chunk_0[0-1]*.jsonl")):
        with open(cf, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                c = r.get("canonical") or {}
                hc = [p for p in (c.get("players") or [])
                      if isinstance(p, dict) and p.get("hole_cards")]
                if len(hc) == 1:
                    found += 1
                    assert c.get("showdown") is None
                    pid = str(hc[0].get("player_id"))
                    acts = [a for a in (c.get("preflop_actions") or [])
                            if a.get("player_id") == pid
                            and a.get("action") != "post"]
                    assert acts                            # 牌主有真实行动
                    if found >= 50:
                        return
    assert found >= 50


def test_audit_file_not_touched_by_runtime(tmp_path):
    """审计产物是纯文档——运行时模块不 import 它（纪律：发现≠接入）。"""
    import poker_decision_ai.decision_grade as dg
    import poker_decision_ai.real_validation as rv
    src_dg = Path(dg.__file__).read_text(encoding="utf-8")
    src_rv = Path(rv.__file__).read_text(encoding="utf-8")
    assert "V3F_P53_DATASET_AUDIT" not in src_dg
    assert "V3F_P53_DATASET_AUDIT" not in src_rv
