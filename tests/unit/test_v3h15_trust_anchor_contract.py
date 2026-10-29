# -*- coding: utf-8 -*-
"""V3H-15 测试 — CASH 外部信任锚接口契约（只读接入前检查）。

覆盖：CASH-only fail closed / 非 dict / 缺 attestations / 缺 anchor_state /
**六类断言彼此独立**（无蕴含矩阵；一类不得代偿另一类）/
A5（内容未被篡改）**不**蕴含 A6（真实发生）/ A1 **不**蕴含 A2 /
SHA-256 **只**表达内容指纹（`sha256_alone_sufficient` / `digest_only` 一律拒绝）/
**自签与内部主体一律拒绝**（`self_issued_attestation` /
`attesting_party_is_this_project` / `attesting_party_kind_not_external`）/
状态机拒绝态优先且 fail-closed / **`readiness` 恒为 False** /
**本项目不验签、不验真**（`signature_*` / `authenticity_*` 恒 False）/
断言记录 schema 12 字段逐一强制 + 非法值拒绝 /
清单全部 blocking 且 `ready_for_real_intake` 恒 False /
**不消费 V3H-14 判定**（`dependency_manifest_consumed = False`）/
零结果零执行零统计（AST：无 Mult/Div/Pow/FloorDiv/Mod）/ 定输入复现 /
state 不变异 / AST·import 隔离（零 V3G、零任何层、无 IO）/ 生产 import 图零新增 /
frozen·sealed drift / Solver 未被调用。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"

from poker_decision_ai.cash_strategy_trust_anchor_contract import (  # noqa: E402
    ANCHOR_STATES, ASSERTION_CLASSES, ASSERTION_CLASS_SPECS,
    ATTESTATION_FIELDS, ATTESTATION_REFUSAL_REASONS,
    CASH_STRATEGY_TRUST_ANCHOR_CONTRACT_VERSION, CLAIM_BOUNDARY,
    EXTERNAL_PARTY_KINDS, FORBIDDEN_TRUST_ASSUMPTIONS,
    INTERNAL_PARTY_MARKERS, PROJECT_CANNOT_DO, PROJECT_CAN_DO,
    attestation_schema, evaluate_anchor_state, pre_intake_checklist,
    trust_anchor_contract, validate_attestation)

# 测试侧允许 import V3G/V3H 层（生产侧不允许，由 AST 测试强制）
from poker_decision_ai.cash_solver_interface import solver_status  # noqa: E402
from poker_decision_ai.cash_strategy_external_readiness import (  # noqa: E402
    audit_external_readiness)
from poker_decision_ai.cash_strategy_experiment_eligibility import (  # noqa: E402
    ENVIRONMENT_FACT_KEYS)

NEW_MODULE = "cash_strategy_trust_anchor_contract"
MODULE_PATH = PKG / f"{NEW_MODULE}.py"

FROZEN = {
    "poker_decision_ai/domain/poker/strategy_core.py": "378a68671ff09e3f",
    "poker_decision_ai/domain/poker/models.py": "6775c06050319a85",
    "poker_decision_ai/domain/poker/conditional_model.py": "686c9f58b71d9408",
    "poker_decision_ai/agents/judge/agent.py": "239436fae7299c47",
    "poker_decision_ai/realtime.py": "3400a20ff857880d",
    "poker_decision_ai/research_unfreeze/pi_ev_contract.py": "52e6267db49a2e11",
    "poker_decision_ai/research_unfreeze/unfreeze_core.py": "17236d4ed8070d01",
}

V3G_LAYERS = ("cash_state_integrity", "cash_integrity_context",
              "cash_preflop_decision", "cash_multistreet", "cash_opponent_model",
              "cash_range_model", "cash_action_tree", "cash_unknown_e2e",
              "cash_solver_interface", "cash_profit_interface")
V3H_LAYERS = tuple(f.stem for f in PKG.glob("cash_strategy_*.py"))

ALL_FACTS = {k: True for k in ENVIRONMENT_FACT_KEYS}


def _sha(rel: str) -> str:
    p = ROOT / rel
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"


def cash_state(**over) -> dict:
    st = {"game_type": "cash", "street": "flop",
          "board": ["Qs", "7d", "2c"], "pot": 10.0, "to_call": 0.0,
          "bb": 1.0, "sb": 0.5, "hero_id": "hero", "table_size": 3,
          "synthetic_test_only": True, "players": [], "action_history": []}
    st.update(over)
    return st


def att(cls: str, i: int = 0, **over) -> dict:
    """一条**结构完整**的外部断言（测试合成；未做任何真实验签）。

    未知 `cls` 时退回一个通用断言类型，以便测试**无效类别**必须被拒绝。
    """
    spec = next((s for s in ASSERTION_CLASS_SPECS if s["class"] == cls), None)
    atype = (spec["allowed_attestation_types"][0] if spec
             else "external_party_attestation")
    a = {
        "attestation_id": f"att-{cls.split('_')[0]}-{i:04d}",
        "assertion_class": cls,
        "attestation_type": atype,
        "attesting_party": "acme-external-auditor-ltd",
        "attesting_party_kind": "external_auditor",
        "subject_ref": f"subject-{i:04d}",
        "subject_content_digest": "a" * 64,
        "issued_at_epoch_ms": 1_700_000_000_000 + i,
        "signature_ref": "https://external.example/attestations/att-0001",
        "signature_scheme": "ed25519-external",
        "revocation_ref": None,
        "scope_note": "covers exactly one assertion class",
    }
    a.update(over)
    return a


def all_six() -> list:
    return [att(c, i) for i, c in enumerate(ASSERTION_CLASSES)]


# ── 1. CASH-only fail closed ─────────────────────────────────────────
@pytest.mark.parametrize("mode", ["mtt", "pko", "ko", "mystery", "spin",
                                 "MTT", "PKO", "", "   ", None])
def test_non_cash_fail_closed_all_entry_points(mode):
    st = cash_state(game_type=mode)
    for r in (validate_attestation(st, att(ASSERTION_CLASSES[0])),
              evaluate_anchor_state(st, attestations=all_six()),
              pre_intake_checklist(st, anchor_state={"present": True})):
        assert r["present"] is False
        assert r["reason"] == "not_cash_mode_fail_closed"
        assert r["state"] == "ANCHOR_UNSATISFIED"
        assert r["readiness"] is False
        assert r["classes_satisfied"] == []
        assert r["classes_missing"] == list(ASSERTION_CLASSES)
        assert r["executed"] is False
        assert r["result"] is None
        assert r["evidence_level"] == "UNAVAILABLE"


@pytest.mark.parametrize("bad", [None, [], "cash", 42, 3.5, True])
def test_non_dict_state_fail_closed(bad):
    assert evaluate_anchor_state(bad, attestations=[])["present"] is False
    assert validate_attestation(bad, att(ASSERTION_CLASSES[0]))["present"] \
        is False
    assert pre_intake_checklist(bad, anchor_state={})["present"] is False


@pytest.mark.parametrize("bad", [None, "x", 42, {"a": 1}])
def test_missing_attestations_fail_closed(bad):
    r = evaluate_anchor_state(cash_state(), attestations=bad)
    assert r["present"] is False
    assert r["reason"] == "attestations_required"
    assert r["readiness"] is False


@pytest.mark.parametrize("bad", [None, [], "x", 42, True])
def test_missing_anchor_state_fail_closed(bad):
    r = pre_intake_checklist(cash_state(), anchor_state=bad)
    assert r["present"] is False
    assert r["reason"] == "anchor_state_required"


@pytest.mark.parametrize("bad", [None, "", "   ", 42, []])
def test_missing_attestation_fail_closed(bad):
    r = validate_attestation(cash_state(), bad)
    assert r["present"] is False
    assert r["reason"] == "attestation_required"


# ── 2. 六类断言彼此独立 ──────────────────────────────────────────────
def test_six_assertion_classes_are_fixed_and_ordered():
    assert ASSERTION_CLASSES == (
        "A1_EXTERNAL_RESOURCE_EXISTS",
        "A2_SOURCE_IDENTITY_TRUSTED",
        "A3_RECORD_COMPLETENESS",
        "A4_RECORD_LINKAGE",
        "A5_RECORD_INTEGRITY",
        "A6_RECORD_ACTUALLY_OCCURRED",
    )
    assert len(ASSERTION_CLASS_SPECS) == 6
    assert [s["class"] for s in ASSERTION_CLASS_SPECS] == list(ASSERTION_CLASSES)


def test_no_class_implies_any_other_class():
    for s in ASSERTION_CLASS_SPECS:
        assert s["implied_by_other_classes"] == [], s["class"]
        assert s["sha256_alone_is_sufficient"] is False, s["class"]
    c = trust_anchor_contract()
    assert c["independence_rule"]
    matrix = c["no_class_implication_matrix"]
    assert set(matrix) == set(ASSERTION_CLASSES)
    assert all(v == [] for v in matrix.values()), matrix


def test_satisfying_one_class_does_not_satisfy_the_others():
    r = evaluate_anchor_state(cash_state(),
                              attestations=[att(ASSERTION_CLASSES[0])])
    assert r["state"] == "ANCHOR_STRUCTURE_VALID"
    assert r["classes_satisfied"] == ["A1_EXTERNAL_RESOURCE_EXISTS"]
    assert r["classes_missing"] == list(ASSERTION_CLASSES[1:])
    assert r["structural_fully_anchored"] is False
    assert r["readiness"] is False
    assert r["class_satisfaction_counts"]["A1_EXTERNAL_RESOURCE_EXISTS"] == 1
    assert r["class_satisfaction_counts"]["A6_RECORD_ACTUALLY_OCCURRED"] == 0


def test_integrity_does_not_imply_occurrence():
    """A5（内容未被篡改）**不**蕴含 A6（真实发生）——核心反例。"""
    r = evaluate_anchor_state(cash_state(),
                              attestations=[att("A5_RECORD_INTEGRITY", 0),
                                            att("A3_RECORD_COMPLETENESS", 1),
                                            att("A4_RECORD_LINKAGE", 2)])
    assert "A5_RECORD_INTEGRITY" in r["classes_satisfied"]
    assert "A6_RECORD_ACTUALLY_OCCURRED" in r["classes_missing"]
    assert r["state"] == "ANCHOR_STRUCTURE_VALID"
    assert r["readiness"] is False
    spec = next(s for s in ASSERTION_CLASS_SPECS
                if s["class"] == "A5_RECORD_INTEGRITY")
    assert "不**蕴含" in spec["note"] or "does not" in spec["note"].lower()


def test_existence_does_not_imply_identity_trust():
    r = evaluate_anchor_state(cash_state(),
                              attestations=[att("A1_EXTERNAL_RESOURCE_EXISTS")])
    assert "A2_SOURCE_IDENTITY_TRUSTED" in r["classes_missing"]


def test_completeness_and_linkage_do_not_imply_occurrence():
    r = evaluate_anchor_state(cash_state(),
                              attestations=[att("A3_RECORD_COMPLETENESS", 0),
                                            att("A4_RECORD_LINKAGE", 1)])
    assert "A6_RECORD_ACTUALLY_OCCURRED" in r["classes_missing"]
    assert "A1_EXTERNAL_RESOURCE_EXISTS" in r["classes_missing"]


def test_claiming_implication_between_classes_is_refused():
    r = evaluate_anchor_state(
        cash_state(),
        attestations=[att("A5_RECORD_INTEGRITY",
                          implies_classes=["A6_RECORD_ACTUALLY_OCCURRED"])])
    assert r["state"] == "ANCHOR_REFUSED_CLASS_MERGE"
    assert "claims_imply_other_classes" in r["refusal_tokens"]
    assert r["classes_satisfied"] == []
    assert r["structural_fully_anchored"] is False


def test_all_six_classes_gives_content_consistent_but_not_verified():
    r = evaluate_anchor_state(cash_state(), attestations=all_six())
    assert r["state"] == "ANCHOR_CONTENT_CONSISTENT"
    assert r["structural_fully_anchored"] is True
    assert r["classes_missing"] == []
    assert r["n_structure_valid"] == 6
    # **结构齐备 ≠ 真实性成立**
    assert r["readiness"] is False
    assert "NOT implemented" in r["readiness_reason"]
    assert r["signature_verified"] is False
    assert r["authenticity_verified"] is False
    assert r["result"] is None
    assert r["executed"] is False


# ── 3. SHA-256 边界 ─────────────────────────────────────────────────
def test_sha256_role_is_content_fingerprint_only():
    c = trust_anchor_contract()
    role = c["sha256_role"]
    assert role["expresses"] == "content_fingerprint_only"
    assert role["sha256_alone_is_sufficient_for_any_class"] is False
    assert role["digest_only_attestations_are_refused"] is True
    for cannot in ("authenticity", "certification", "provenance",
                   "source_identity", "actual_occurrence"):
        assert cannot in role["cannot_prove"], cannot
    assert c["attestations_issued_by_this_project"] is False
    assert c["issuer_of_attestations"] == "must_be_external"


def test_digest_only_attestation_is_refused():
    r = evaluate_anchor_state(
        cash_state(),
        attestations=[att("A5_RECORD_INTEGRITY", signature_ref=None)])
    assert r["state"] == "ANCHOR_REFUSED_DIGEST_AS_PROOF"
    assert "digest_only_attestation" in r["refusal_tokens"]
    assert r["classes_satisfied"] == []
    assert r["digest_used_as_proof"] is False


def test_sha256_alone_sufficiency_claim_is_refused():
    r = evaluate_anchor_state(
        cash_state(),
        attestations=[att("A6_RECORD_ACTUALLY_OCCURRED",
                          sha256_alone_sufficient=True)])
    assert r["state"] == "ANCHOR_REFUSED_DIGEST_AS_PROOF"
    assert "sha256_alone_claimed_sufficient" in r["refusal_tokens"]
    assert r["classes_satisfied"] == []


def test_digest_field_never_sets_readiness():
    r = evaluate_anchor_state(cash_state(),
                              attestations=[att(c, i) for i, c in
                                            enumerate(ASSERTION_CLASSES)])
    assert all(a["assertion_class"] for a in r["per_attestation"])
    assert r["readiness"] is False
    assert r["digest_used_as_proof"] is False
    assert CLAIM_BOUNDARY["digest_treated_as_provenance_proof"] is False


# ── 4. 自签 / 内部主体一律拒绝 ───────────────────────────────────────
@pytest.mark.parametrize("marker", list(INTERNAL_PARTY_MARKERS))
def test_internal_party_markers_are_refused(marker):
    r = validate_attestation(cash_state(),
                             att(ASSERTION_CLASSES[0],
                                 attesting_party=f"{marker}-signer"))
    assert r["valid"] is False
    assert "attesting_party_is_this_project" in r["issues"]
    assert "self_issued_attestation" in r["issues"]


@pytest.mark.parametrize("kind", ["internal_signer", "self", "local_process",
                                  "same_project", "", None])
def test_non_external_party_kind_is_refused(kind):
    r = validate_attestation(cash_state(),
                             att(ASSERTION_CLASSES[0],
                                 attesting_party_kind=kind))
    assert r["valid"] is False
    assert "attesting_party_kind_not_external" in r["issues"]


@pytest.mark.parametrize("kind", list(EXTERNAL_PARTY_KINDS))
def test_external_party_kinds_are_accepted(kind):
    r = validate_attestation(cash_state(),
                             att(ASSERTION_CLASSES[0],
                                 attesting_party_kind=kind))
    assert r["valid"] is True, r["issues"]


def test_self_issued_attestation_drives_refused_state():
    r = evaluate_anchor_state(
        cash_state(),
        attestations=[att(c, i) for i, c in enumerate(ASSERTION_CLASSES)] +
        [att("A1_EXTERNAL_RESOURCE_EXISTS", 9,
             attesting_party_kind="internal_signer")])
    assert r["state"] == "ANCHOR_REFUSED_SELF_ISSUED"
    assert "attesting_party_kind_not_external" in r["refusal_tokens"]
    assert "self_issued_attestation" in r["refusal_tokens"]


def test_project_never_issues_an_attestation():
    assert "issue_an_attestation" in PROJECT_CANNOT_DO
    assert "fabricate_a_trust_anchor" in PROJECT_CANNOT_DO
    assert "prefill_attestation_fields" in PROJECT_CANNOT_DO
    assert "verify_authenticity_itself" in PROJECT_CANNOT_DO
    assert "verify_signatures_itself" in PROJECT_CANNOT_DO
    assert "declare_an_external_resource_satisfied" in PROJECT_CANNOT_DO
    assert CLAIM_BOUNDARY["issues_attestations"] is False
    assert CLAIM_BOUNDARY["fabricates_trust_anchor"] is False
    assert CLAIM_BOUNDARY["performs_signature_verification"] is False
    assert CLAIM_BOUNDARY["verifies_authenticity_itself"] is False
    assert CLAIM_BOUNDARY["declares_external_resource_satisfied"] is False


def test_project_capability_boundary_is_honest_on_both_sides():
    assert len(PROJECT_CANNOT_DO) == 10
    assert len(PROJECT_CAN_DO) == 6
    assert "access_network_wallet_client_or_external_process" in PROJECT_CANNOT_DO
    assert "invoke_a_solver" in PROJECT_CANNOT_DO
    assert "validate_attestation_structure_fail_closed" in PROJECT_CAN_DO
    assert "report_that_signature_verification_is_not_implemented_here" \
        in PROJECT_CAN_DO


# ── 5. schema 与逐字段强制 ──────────────────────────────────────────
def test_schema_shape_and_zero_prefill():
    s = attestation_schema()
    assert s["fields"] == list(ATTESTATION_FIELDS)
    assert s["n_fields"] == len(ATTESTATION_FIELDS) == 12
    assert set(s["field_kinds"]) == set(ATTESTATION_FIELDS)
    assert s["prefilled_values"] == []
    assert s["example_provided"] is False
    assert s["result"] is None
    assert s["executed"] is False
    assert s["external_party_kinds"] == list(EXTERNAL_PARTY_KINDS)


@pytest.mark.parametrize("field", ATTESTATION_FIELDS)
def test_each_required_field_is_enforced(field):
    a = att(ASSERTION_CLASSES[0])
    del a[field]
    v = validate_attestation(cash_state(), a)
    assert v["valid"] is False
    assert f"missing_required_field:{field}" in v["issues"]


@pytest.mark.parametrize("field,bad", [
    ("attestation_id", ""), ("attestation_id", "   "), ("attestation_id", None),
    ("assertion_class", None), ("assertion_class", "A9_UNKNOWN"),
    ("assertion_class", "a1_external_resource_exists"),
    ("attestation_type", None), ("attestation_type", "self_declaration"),
    ("attesting_party", None), ("subject_ref", None), ("subject_ref", ""),
    ("issued_at_epoch_ms", "1700000000000"), ("issued_at_epoch_ms", 1.5),
    ("issued_at_epoch_ms", None), ("issued_at_epoch_ms", True),
    ("signature_ref", None), ("signature_scheme", None),
    ("scope_note", None),
])
def test_bad_field_values_are_rejected(field, bad):
    v = validate_attestation(cash_state(),
                             att(ASSERTION_CLASSES[0], **{field: bad}))
    assert v["valid"] is False, (field, bad)
    assert v["assertion_class"] is None


def test_unknown_assertion_class_never_contributes_a_class():
    r = evaluate_anchor_state(cash_state(),
                              attestations=[att("A9_UNKNOWN_CLASS")])
    assert r["classes_satisfied"] == []
    assert "assertion_class_not_in_closed_enum" in r["refusal_tokens"]
    assert r["state"] == "ANCHOR_UNSATISFIED"


def test_attestation_type_must_match_its_class():
    a = att("A1_EXTERNAL_RESOURCE_EXISTS", attestation_type=(
        "notarised_occurrence_attestation"))
    v = validate_attestation(cash_state(), a)
    assert v["valid"] is False
    assert "attestation_type_invalid_for_class" in v["issues"]


def test_duplicate_attestation_ids_are_detected_not_collapsed():
    a = att("A1_EXTERNAL_RESOURCE_EXISTS", 0)
    r = evaluate_anchor_state(cash_state(), attestations=[a, dict(a)])
    assert r["n_attestations"] == 2
    assert "attestation_duplicate" in r["refusal_tokens"]
    assert r["n_refused"] == 1
    assert len(r["per_attestation"]) == 2


def test_non_dict_attestations_marked_not_dropped():
    r = evaluate_anchor_state(cash_state(),
                              attestations=[att(ASSERTION_CLASSES[0]), None,
                                            "x", 7])
    assert r["n_attestations"] == 4
    assert len(r["per_attestation"]) == 4
    assert r["n_refused"] == 3
    for x in r["per_attestation"][1:]:
        assert x["issues"] == ["attestation_not_a_dict"]


def test_refusal_reason_enum_is_closed_and_documented():
    assert len(ATTESTATION_REFUSAL_REASONS) == 18
    for x in ("digest_only_attestation", "sha256_alone_claimed_sufficient",
              "self_issued_attestation", "attesting_party_is_this_project",
              "claims_imply_other_classes", "scope_note_invalid",
              "assertion_class_not_in_closed_enum"):
        assert x in ATTESTATION_REFUSAL_REASONS


def test_no_assertion_is_recorded_when_structure_is_invalid():
    a = att("A6_RECORD_ACTUALLY_OCCURRED", attesting_party="this_project")
    r = evaluate_anchor_state(cash_state(), attestations=[a])
    assert r["classes_satisfied"] == []
    assert r["class_satisfaction_counts"]["A6_RECORD_ACTUALLY_OCCURRED"] == 0
    assert r["state"] in ("ANCHOR_UNSATISFIED", "ANCHOR_REFUSED_SELF_ISSUED")


# ── 6. 不验签 / 不验真 ──────────────────────────────────────────────
def test_signature_and_authenticity_are_never_verified():
    for r in (validate_attestation(cash_state(), att(ASSERTION_CLASSES[0])),
              evaluate_anchor_state(cash_state(), attestations=all_six())):
        assert r["signature_verified"] is False
        assert r["signature_verification_performed"] is False
        assert r["authenticity_verified"] is False
        assert r["authenticity_verification_performed"] is False


def test_contract_states_signature_verification_is_out_of_band():
    sv = trust_anchor_contract()["signature_verification"]
    assert sv["implemented_here"] is False
    assert sv["performed_here"] is False
    assert sv["must_be_done_out_of_band"] is True
    assert "不" in sv["note"]


def test_unverified_items_are_declared_in_uncertainty():
    r = evaluate_anchor_state(cash_state(), attestations=all_six())
    u = r["uncertainty"]
    assert u["measured"] is False
    assert set(u["unverified_by_this_layer"]) == {
        "signature", "authenticity", "source_trust", "actual_occurrence"}


# ── 7. 接入前检查清单 ───────────────────────────────────────────────
def test_checklist_items_are_all_blocking_and_not_done_here():
    anchor = evaluate_anchor_state(cash_state(), attestations=all_six())
    c = pre_intake_checklist(cash_state(), anchor_state=anchor)
    assert c["n_checks"] == 6
    assert c["n_satisfied"] == 6
    assert c["n_outstanding"] == 0
    assert c["all_checks_satisfied"] is True
    for it in c["items"]:
        assert it["blocking"] is True
        assert it["done_by_this_project"] is False
        assert it["must_be_done_by"] == "external_party"
    # 即使全部结构性满足，也**不**等于可接入
    assert c["ready_for_real_intake"] is False
    assert "out of band" in c["ready_for_real_intake_reason"]
    assert c["external_resources_declared_satisfied_by_this_layer"] == []
    assert c["network_wallet_client_external_process_or_solver_used"] is False


def test_checklist_reflects_unsatisfied_classes():
    anchor = evaluate_anchor_state(cash_state(),
                                   attestations=[att(ASSERTION_CLASSES[0])])
    c = pre_intake_checklist(cash_state(), anchor_state=anchor)
    assert c["n_satisfied"] == 1
    assert c["n_outstanding"] == 5
    assert c["all_checks_satisfied"] is False
    assert c["ready_for_real_intake"] is False
    assert f"outstanding:{ASSERTION_CLASSES[1]}" in c["unknown"]


def test_checklist_never_declares_readiness_from_structure_alone():
    for atts in ([], [att(ASSERTION_CLASSES[0])], all_six()):
        anchor = evaluate_anchor_state(cash_state(), attestations=atts)
        c = pre_intake_checklist(cash_state(), anchor_state=anchor)
        assert c["ready_for_real_intake"] is False
        assert c["executed"] is False
        assert c["result"] is None


# ── 8. 不重复 V3H-14 / 不消费其判定 ─────────────────────────────────
def test_does_not_consume_or_recompute_v3h14_verdict():
    r = evaluate_anchor_state(cash_state(), attestations=all_six())
    assert r["dependency_manifest_consumed"] is False
    assert r["v3h14_verdict_not_recomputed"] is True
    assert r["external_resources_declared_satisfied"] == []
    src = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("audit_external_readiness", "external_dependency_manifest",
                   "EXTERNAL_REQUIREMENTS", "RESOLUTION_CLASSES",
                   "READINESS_VERDICTS", "FORBIDDEN_SUBSTITUTIONS"):
        assert banned not in src, banned


def test_readiness_is_never_true_regardless_of_input():
    cases = [
        [],
        [att(ASSERTION_CLASSES[0])],
        all_six(),
        all_six() + [att("A1_EXTERNAL_RESOURCE_EXISTS", 9,
                         signature_ref=None)],
        [att(c, i, attesting_party_kind="internal_signer")
         for i, c in enumerate(ASSERTION_CLASSES)],
        [None, "x", 7],
    ]
    for atts in cases:
        r = evaluate_anchor_state(cash_state(), attestations=atts)
        assert r["readiness"] is False, atts
        assert r["result"] is None
        assert r["executed"] is False
        assert r["analysis_executed"] is False


def test_forbidden_trust_assumptions_are_declared_and_unused():
    r = evaluate_anchor_state(cash_state(), attestations=all_six())
    assert r["forbidden_trust_assumptions"] == list(FORBIDDEN_TRUST_ASSUMPTIONS)
    assert r["assumptions_used"] == []
    assert len(FORBIDDEN_TRUST_ASSUMPTIONS) == 13
    for x in ("digest_as_proof", "self_issued_trust_anchor",
              "internal_attester", "class_merge", "mock_solver",
              "approximated_equilibrium", "declared_exploitability",
              "model_internal_as_external"):
        assert x in FORBIDDEN_TRUST_ASSUMPTIONS, x


# ── 9. 零结果 / 零执行 / 零统计 ────────────────────────────────────
def test_never_produces_results_or_executes():
    for r in (trust_anchor_contract(), attestation_schema(),
              validate_attestation(cash_state(), att(ASSERTION_CLASSES[0])),
              evaluate_anchor_state(cash_state(), attestations=all_six()),
              pre_intake_checklist(
                  cash_state(),
                  anchor_state=evaluate_anchor_state(
                      cash_state(), attestations=all_six()))):
        assert r["executed"] is False
        assert r["result"] is None
        assert r.get("results_available", False) is False
        assert r.get("analysis_executed", False) is False


def test_claim_boundary_all_false_23_flags():
    assert len(CLAIM_BOUNDARY) == 23
    assert all(v is False for v in CLAIM_BOUNDARY.values())
    for k in ("issues_attestations", "performs_signature_verification",
              "verifies_authenticity_itself",
              "declares_external_resource_satisfied",
              "merges_assertion_classes", "digest_treated_as_provenance_proof",
              "generates_execution_or_profit_or_statistical_results"):
        assert CLAIM_BOUNDARY[k] is False, k
    r = evaluate_anchor_state(cash_state(), attestations=all_six())
    assert all(v is False for v in r["claim_boundary"].values())


def test_flat_key_scan_has_no_result_or_statistic_field():
    r = evaluate_anchor_state(cash_state(), attestations=all_six())
    keys: set[str] = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                keys.add(k)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(r)
    for banned in ("p_value", "effect_size", "confidence_interval", "win_rate",
                   "roi", "pnl", "profit_estimate", "exploitability_value",
                   "equilibrium", "gto", "gto_exact"):
        assert banned not in keys, banned


def test_source_has_no_multiplication_division_or_power():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    ops = {type(n.op).__name__ for n in ast.walk(tree)
           if isinstance(n, ast.BinOp)}
    for banned in ("Mult", "Div", "FloorDiv", "Pow", "Mod"):
        assert banned not in ops, banned


def test_source_has_no_network_io_or_process_tokens():
    src = MODULE_PATH.read_text(encoding="utf-8")
    lower = src.lower()
    for banned in ("scipy", "numpy", "p_value", "effect_size", "socket",
                   "requests", "urllib", "http", "subprocess", "open(",
                   "pathlib", "sqlite"):
        assert banned not in lower, banned
    # "wallet" 只允许出现在**禁止/能力边界声明**中：
    # 去掉三处声明字符串与模块 docstring 之后，正文必须一次都不剩。
    residue = lower
    for decl in ("access_network_wallet_client_or_external_process",
                 "network_wallet_client_external_process_or_solver_used"):
        residue = residue.replace(decl, "")
    header = '"""v3h-15'
    start = residue.find(header)
    end = residue.find('"""', start + 3) if start != -1 else -1
    body = residue[end + 3:] if end != -1 else residue
    assert "wallet" not in body
    assert lower.count("wallet") == 2      # 禁令 1 处 + docstring 1 处


# ── 10. 定输入复现 / 不变异 / 可序列化 ──────────────────────────────
def test_deterministic_same_input_same_output():
    st = cash_state()
    atts = all_six()
    outs = [evaluate_anchor_state(st, attestations=atts) for _ in range(5)]
    assert all(o == outs[0] for o in outs)
    outs2 = [trust_anchor_contract() for _ in range(3)]
    assert all(o == outs2[0] for o in outs2)


def test_refusal_tokens_are_sorted_for_reproducibility():
    r = evaluate_anchor_state(cash_state(),
                              attestations=[None, att("A9_UNKNOWN")])
    assert r["refusal_tokens"] == sorted(r["refusal_tokens"])
    assert r["unknown"] == sorted(r["unknown"])


def test_state_and_attestations_not_mutated():
    st = cash_state()
    atts = all_six()
    st_before = json.dumps(st, sort_keys=True, ensure_ascii=False)
    atts_before = json.dumps(atts, sort_keys=True, ensure_ascii=False)
    evaluate_anchor_state(st, attestations=atts)
    assert json.dumps(st, sort_keys=True, ensure_ascii=False) == st_before
    assert json.dumps(atts, sort_keys=True, ensure_ascii=False) == atts_before


def test_outputs_json_serialisable():
    st = cash_state()
    anchor = evaluate_anchor_state(st, attestations=all_six())
    for r in (trust_anchor_contract(), attestation_schema(), anchor,
              validate_attestation(st, att(ASSERTION_CLASSES[0])),
              pre_intake_checklist(st, anchor_state=anchor)):
        assert json.loads(json.dumps(r, ensure_ascii=False)) == r


# ── 11. AST · import 隔离 ─────────────────────────────────────────
def test_declares_version_constant():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    consts = {n.targets[0].id for n in tree.body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert any(c.endswith("_VERSION") for c in consts)
    assert CASH_STRATEGY_TRUST_ANCHOR_CONTRACT_VERSION.startswith("v3h15-")


def test_top_level_imports_are_stdlib_only():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    mods = [n.module for n in tree.body if isinstance(n, ast.ImportFrom)]
    assert mods == ["__future__", "typing"], mods
    plain = [a.name for n in tree.body if isinstance(n, ast.Import)
             for a in n.names]
    assert plain == [], plain


def test_zero_v3g_imports_anywhere_including_function_bodies():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            assert not any(v in (n.module or "") for v in V3G_LAYERS), n.module
        if isinstance(n, ast.Import):
            for a in n.names:
                assert not any(v in a.name for v in V3G_LAYERS), a.name


def test_zero_layer_imports_anywhere():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.ImportFrom):
            mods.append(n.module or "")
        elif isinstance(n, ast.Import):
            mods += [a.name for a in n.names]
        for m in mods:
            assert not any(v in m for v in V3G_LAYERS + V3H_LAYERS), m


def test_no_random_import():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name.split(".")[0] != "random" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert (n.module or "").split(".")[0] != "random"


def test_no_frozen_multistreet_or_sample_runouts_name():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all("cash_multistreet" not in a.name for a in n.names)
        if isinstance(n, ast.Name):
            assert n.id != "sample_runouts"


def test_no_client_automation_path_anywhere():
    src = MODULE_PATH.read_text(encoding="utf-8")
    for banned in ("click", "pyautogui", "keyboard.", "screenshot", "input(",
                   "advapi32", "win32"):
        assert banned not in src, banned


# ── 12. 生产 import 图零新增 ─────────────────────────────────────
def _production_importers() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for p in PKG.rglob("*.py"):
        if p.name.startswith("cash_") or "__pycache__" in str(p):
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8-sig"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.ImportFrom) and n.module:
                mods.append(n.module.split(".")[-1])
            elif isinstance(n, ast.Import):
                mods += [a.name.split(".")[-1] for a in n.names]
            for m in mods:
                if m in V3G_LAYERS or m in V3H_LAYERS:
                    found.setdefault(m, set()).add(p.name)
    return found


def test_no_new_production_imports():
    found = _production_importers()
    assert set(found) == {"cash_integrity_context", "cash_unknown_e2e"}, found


def test_new_module_not_imported_in_production():
    assert NEW_MODULE not in _production_importers()


def test_all_v3h_layers_remain_unwired():
    found = _production_importers()
    for layer in V3H_LAYERS:
        assert layer not in found, layer
    required = {
        "cash_strategy_action_policy", "cash_strategy_action_ranking",
        "cash_strategy_belief_conditioned_eval", "cash_strategy_belief_eval",
        "cash_strategy_belief_update", "cash_strategy_decision_assembly",
        "cash_strategy_dynamic_belief", "cash_strategy_evidence_closure",
        "cash_strategy_execution_intake", "cash_strategy_experiment_eligibility",
        "cash_strategy_external_readiness", "cash_strategy_likelihood_model",
        "cash_strategy_multistreet_eval", "cash_strategy_policy_evaluator",
        "cash_strategy_preregistration", "cash_strategy_search_eval",
        "cash_strategy_search_tree", "cash_strategy_shadow_evaluation",
        "cash_strategy_trust_anchor_contract",
    }
    assert required <= set(V3H_LAYERS), sorted(required - set(V3H_LAYERS))


# ── 13. frozen / sealed drift · Solver ───────────────────────────
@pytest.mark.parametrize("rel,prefix", sorted(FROZEN.items()))
def test_frozen_files_unchanged(rel, prefix):
    assert _sha(rel).startswith(prefix), rel


def test_v3g_manifest_drift_unchanged():
    man = json.loads((ROOT / "docs" / "V3G_11_FREEZE_MANIFEST.json")
                     .read_text(encoding="utf-8"))
    drift = []
    for block in ("frozen_modules", "frozen_tests", "frozen_closeouts",
                  "production_baseline"):
        for rel, h in man[block].items():
            if h == "__SELF__":
                continue
            if _sha(rel) != h:
                drift.append(rel)
    assert drift in ([], ["docs/V3G_11_EVALUATION_FREEZE_CLOSEOUT.md"]), drift


def test_solver_still_blocked_and_never_invoked_here():
    s = solver_status(cash_state())
    assert s["solver_status"] == "BLOCKED"
    assert s["available"] is False
    assert s["equilibrium"] is False
    assert s["gto_exact"] is False
    assert s["discovery_state"] == "UNAVAILABLE"
    src = MODULE_PATH.read_text(encoding="utf-8")
    assert "solver_status" not in src
    assert "mock_solver" in src            # 仅作为**禁止项**出现
    assert "approximated_equilibrium" in src
    assert "declared_exploitability" in src


# ── 14. 与 V3H-14 的关系（只读、不复制判定） ─────────────────────
def test_readiness_state_does_not_flip_v3h14_verdict():
    st = cash_state()
    v3h14 = audit_external_readiness(
        st, closure={"present": True, "closure_possible": False,
                     "closure_test": {"established_blockers": ["B1", "B3",
                                                               "B4", "B5",
                                                               "solver"]}},
        preregistration={"present": True, "analysis_ready": False,
                         "result_generation_refused": True,
                         "blockers_from_upstream": ["B1", "B3", "B4", "B5",
                                                    "solver"]},
        solver=solver_status(st), environment=ALL_FACTS)
    assert v3h14["verdict"] == "BLOCKED"
    anchor = evaluate_anchor_state(st, attestations=all_six())
    assert anchor["readiness"] is False
    # 状态机不触碰 V3H-14 结论
    assert anchor["dependency_manifest_consumed"] is False
    v3h14_after = audit_external_readiness(
        st, closure={"present": True, "closure_possible": False,
                     "closure_test": {"established_blockers": ["B1", "B3",
                                                               "B4", "B5",
                                                               "solver"]}},
        preregistration={"present": True, "analysis_ready": False,
                         "result_generation_refused": True,
                         "blockers_from_upstream": ["B1", "B3", "B4", "B5",
                                                    "solver"]},
        solver=solver_status(st), environment=ALL_FACTS)
    assert v3h14_after["verdict"] == "BLOCKED"
    assert v3h14_after["n_blocked_external"] == 5


def test_contract_and_schema_declare_they_are_not_verdicts():
    for r in (trust_anchor_contract(), attestation_schema()):
        assert r["evidence_level"] == "INTERNAL_DECLARATION"
        assert r["evidence_grade"] == "NONE"
        assert r["result"] is None
        assert r["claim_boundary"]["claims_experiment_ready"] is False
