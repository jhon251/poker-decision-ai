"""M6 治理层 + 工件守卫 + wiring 风险 回归测试。

覆盖：UNKNOWN fail-closed、dealer/hero/occupied 缺失、mode isolation、
manifest mismatch、stale artifact、research→production 禁止路径、B6 限制声明完整性。
不修改任何旧验收测试定义。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from poker_decision_ai.research_unfreeze import governance as gov

ROOT = Path(__file__).resolve().parents[2]


# ── B6：受治理 RESULT 必须内嵌六项限制 ────────────────────────────────
def test_governed_result_embeds_all_six_limitations():
    src = json.loads((ROOT / "docs/V3F_S3PRIME_RESULT_R22.json").read_text(
        encoding="utf-8"))
    doc = gov.govern_result(src)
    assert set(gov.REQUIRED_LIMITATIONS) <= set(doc["limitations"])
    assert gov.validate_governed(doc) == []


def test_governed_result_states_required_values():
    doc = gov.govern_result({"arm": "R2.2", "runner_version": "x", "run_id": "y"})
    lim = doc["limitations"]
    assert lim["independence"]["value"] == "NOT INDEPENDENT"
    assert lim["solver"]["value"] == "BLOCKED"
    assert lim["solver"]["reference"].endswith("NOT_exact_gto=true")
    assert lim["profit_unidentifiable"]["value"] == "UNIDENTIFIABLE"
    assert lim["lambda_u2"]["value"] == "BLOCKED"
    assert lim["defect_b_meaning"]["value"].startswith("CLOSED")
    assert "冻结决策集" in lim["accept_scope"]["value"]
    assert doc["claims"]["may_not_claim"]


def test_governed_result_rejects_banned_claims():
    doc = gov.govern_result({"arm": "R2.2"})
    doc["claims"]["may_claim"].append("具有盈利能力")
    assert any("禁止性表述" in v for v in gov.validate_governed(doc))


def test_governed_result_detects_missing_limitation():
    doc = gov.govern_result({"arm": "R2.2"})
    doc["limitations"].pop("lambda_u2")
    assert any("lambda_u2" in v for v in gov.validate_governed(doc))


# ── B7-2：工件守卫 fail-closed ────────────────────────────────────────
def test_guard_rejects_missing_manifest(tmp_path):
    with pytest.raises(gov.ArtifactGuardError):
        gov.ArtifactGuard(tmp_path / "nope.json")


def test_guard_rejects_unregistered_artifact():
    g = gov.ArtifactGuard(ROOT / "docs/V3F_S3PRIME_MANIFEST.json", root=ROOT)
    with pytest.raises(gov.ArtifactGuardError):
        g.require("docs/V3F_NOT_REGISTERED.json")


def test_guard_rejects_stale_artifact(tmp_path):
    p = tmp_path / "a.json"
    p.write_text("{}", encoding="utf-8")
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"code_hashes": {"a.json": "0" * 64}}),
                   encoding="utf-8")
    g = gov.ArtifactGuard(man, root=tmp_path)
    with pytest.raises(gov.ArtifactGuardError) as e:
        g.require("a.json")
    assert "stale/wrong-manifest" in str(e.value)


def test_guard_accepts_matching_hash_and_detects_drift(tmp_path):
    p = tmp_path / "a.json"
    p.write_text("{}", encoding="utf-8")
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"code_hashes": {"a.json": gov.sha256_file(p)}}),
                   encoding="utf-8")
    g = gov.ArtifactGuard(man, root=tmp_path)
    assert g.require("a.json") == p
    assert g.verify_sealed_code() == {"checked": 1, "drift": {}}
    p.write_text('{"changed":1}', encoding="utf-8")
    with pytest.raises(gov.ArtifactGuardError):
        g.require("a.json")          # 不 fallback、不缓存旧值


def test_real_manifests_verify_without_drift():
    for rel in ("docs/V3F_S3PRIME_MANIFEST.json",
                "docs/V3F_PERF_V2_MANIFEST.json"):
        g = gov.ArtifactGuard(ROOT / rel, root=ROOT)
        assert g.verify_sealed_code()["drift"] == {}


# ── research → production 禁止路径 ────────────────────────────────────
def test_research_modules_not_imported_by_production_package():
    pkg = ROOT / "poker_decision_ai"
    allowed = {"perf_profile.py", "perf_optimize.py", "perf_runner_v2.py",
               "s3prime_runner.py"}
    offenders = []
    for p in pkg.rglob("*.py"):
        if "research_unfreeze" in p.parts or p.name in allowed:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        for n in ast.walk(tree):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                    [(n.module or "")] if isinstance(n, ast.ImportFrom) else [])
            if any("research_unfreeze" in m for m in mods):
                offenders.append(str(p.relative_to(pkg)))
    assert offenders == [], f"生产包导入研究模块: {offenders}"


def test_r22_implementation_is_not_wired_into_any_decision_path():
    """R2.2 实现只在 benchmark/research 中被调用，不得出现在生产决策入口。"""
    pkg = ROOT / "poker_decision_ai"
    callers = []
    for p in pkg.rglob("*.py"):
        if p.name in ("multistreet_v3.py",):
            continue
        t = p.read_text(encoding="utf-8", errors="ignore")
        if "evaluate_v3(" in t:
            callers.append(str(p.relative_to(pkg)))
    assert all(("benchmark" in c) or ("research_unfreeze" in c)
               for c in callers), f"非基准模块调用 evaluate_v3: {callers}"


# ── mode isolation（CASH 不隐式调用 MTT/PKO） ─────────────────────────
def test_cash_mode_never_applies_tournament_transform():
    """CASH 路径不得隐式套用 MTT/PKO 变换（k_p=k_m=1, bounty=None）。"""
    from poker_decision_ai.multistreet_v2 import mode_transform_available
    r = mode_transform_available({"game_type": "cash",
                                  "players": [{"player_id": "a"}]})
    assert r["available"] is True
    assert r["k_p"] == 1.0 and r["k_m"] == 1.0
    assert r["bounty_gain"] is None
    # 非 cash 模式的可用性由冻结私有助手决定，与 CASH 路径无关；
    # 只需确认它们不会把 k_p/k_m 之外的 CASH 语义写回（此处仅做存在性检查）。
    for mode in ("mtt", "pko", "mystery"):
        rr = mode_transform_available({"game_type": mode, "players": []})
        assert isinstance(rr, dict) and "available" in rr


# ── UNKNOWN / fallback fail-closed ────────────────────────────────────
def test_unknown_hero_cards_do_not_default_to_a_seat():
    from poker_decision_ai.research_unfreeze.pi_reference_complete import (
        UNAVAILABLE_INCOMPLETE, reference_complete)
    h = {"hand_id": "g|1", "players": ["Gogo", "Pluribus"],
         "hole_cards": {"Gogo": ["As", "Kd"]},      # AI 座手牌未知
         "board_by_street": {}, "streets": [], "bb": 100.0}
    ref = reference_complete(hand=h, hero_pid="Gogo", folded=set(),
                             hero_cards=["As", "Kd"], board=["2h", "7c", "9s"],
                             pot_bb=10.0, to_call_bb=0.0, hero_stack_bb=100.0,
                             candidates=[{"action": "check", "amount_bb": 0.0}])
    assert ref["available"] is False
    assert UNAVAILABLE_INCOMPLETE in ref["unavailable_reason"]


def test_missing_hero_seat_is_not_defaulted():
    """hero 不在座位表中 → 不得回退到任意座位（fail closed）。"""
    from poker_decision_ai.research_unfreeze.pi_reference_complete import (
        build_villain_hands)
    h = {"hand_id": "g|2", "players": ["MrBlue", "Pluribus"],
         "hole_cards": {"MrBlue": ["Qh", "Qs"], "Pluribus": ["2c", "7d"]},
         "board_by_street": {}, "streets": [], "bb": 100.0}
    facts = build_villain_hands(h, "Gogo", set())
    # hero 不在座位中：不得把任一玩家当作 hero 自己而漏算
    assert facts["live_count"] == 2
    assert facts["villains_used"] == 2


def test_solver_remains_blocked_and_not_claimed_available():
    from poker_decision_ai.research_unfreeze.unfreeze_core import SolverReference
    s = SolverReference()
    assert s.is_configured() is False
    assert "POSTFLOP_SOLVER_PATH" in str(s.unavailable_reason())
