"""M7 / B7-2 + B7-8：benchmark 真实 artifact 读取路径必须经过 ArtifactGuard。

覆盖：
  * 正向：每次真实读取都触发一次 `require(rel)`，且 rel 与读取目标一致；
  * 负向（fail closed）：未登记 / 缺失 / stale / manifest 缺失 / manifest 损坏
    → ArtifactGuardError，**绝不**回退到 `{"present": False}` 或旧值；
  * 登记完整性：所有被读取的历史工件都在 manifest 中登记。
不修改任何历史封存工件；不修改任何旧验收测试定义。
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from poker_decision_ai import multistreet_benchmark as mb
from poker_decision_ai import quality_benchmark as qb
from poker_decision_ai.artifact_guard import (ArtifactGuard,
                                              ArtifactGuardError, sha256_file)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_MANIFEST = ROOT / "docs" / "V3F_ARTIFACT_MANIFEST.json"


@pytest.fixture()
def scratch():
    """本测试专用临时目录（工作区内；不依赖 pytest tmp_path）。

    独立于 `tmp_path` 的原因：本机沙箱禁止删除 %TEMP% 下的目录，导致
    pytest 的 tmp_path 工厂在 setup 阶段即报 PermissionError。与本任务
    要验证的语义无关，故改用自管理的临时目录。
    """
    d = ROOT / ".tmp" / "m7_guard_tests" / uuid.uuid4().hex
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


#: 本模块覆盖的真实读取路径 → 其必须校验的工件（先校验后读）
QUALITY_READS = {
    "build_manifest": ("data/uoft/manifest.json",),
    "reference_metrics": tuple(qb.REFERENCE_SNAPSHOTS.values()),
}
MULTISTREET_READS = ("docs/V3F_R1_QUALITY_BASELINE.json",)


class _Recorder:
    """包装真实 ArtifactGuard，记录被判定的请求（不改变校验语义）。"""

    def __init__(self, inner: ArtifactGuard):
        self._inner = inner
        self.calls: list[str] = []

    def require(self, rel: str):
        self.calls.append(rel)
        return self._inner.require(rel)

    def __getattr__(self, name):            # 其余 API 原样透传
        return getattr(self._inner, name)


def _arm(monkeypatch, module) -> _Recorder:
    inner = ArtifactGuard(ARTIFACT_MANIFEST, root=ROOT)
    rec = _Recorder(inner)
    monkeypatch.setattr(module, "_guard", lambda: rec)
    return rec


# ── B7-8：真实读取路径确实经过 Guard ────────────────────────────────────
def test_quality_manifest_read_goes_through_guard(monkeypatch):
    rec = _arm(monkeypatch, qb)
    out = qb.build_manifest()
    assert rec.calls == ["data/uoft/manifest.json"]
    # 校验通过了才读；产物仍来自真实文件
    assert out["datasets"]["cash_uoft_normalized"]["hand_count"] == 2740778


def test_quality_reference_reads_go_through_guard(monkeypatch):
    rec = _arm(monkeypatch, qb)
    refs = qb.reference_metrics()
    assert rec.calls == list(qb.REFERENCE_SNAPSHOTS.values())
    assert all(refs[k]["present"] is True for k in qb.REFERENCE_SNAPSHOTS)
    assert refs["P54_decision_grade"]["fold_calibration"]["arms"]["REAL"][
        "brier"] == 0.125886


def test_multistreet_r1_reference_read_goes_through_guard(monkeypatch):
    rec = _arm(monkeypatch, mb)
    out = mb.r1_reference()
    assert rec.calls == list(MULTISTREET_READS)
    assert out["present"] is True
    assert out["fold_calibration"]["arms"]["REAL"]["brier"] == 0.125886


def test_every_read_path_is_registered_and_declared():
    man = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    registered = set(man["code_hashes"]) | set(
        man.get("readonly_dependency_hashes") or {})
    read = set(QUALITY_READS["build_manifest"]) \
        | set(QUALITY_READS["reference_metrics"]) | set(MULTISTREET_READS)
    assert read <= registered, f"读取未登记工件: {sorted(read - registered)}"
    # manifest 显式声明被接入的读取路径（可审计）
    for entry in man["guard_read_paths"]:
        mod, _, func = entry.partition("::")
        src = ROOT / (mod[:-3].replace(".", "/") + ".py")
        assert src.exists(), entry
        assert f"def {func}(" in src.read_text(encoding="utf-8"), entry


def test_reference_snapshots_are_the_declared_targets():
    """防漂移：读取集合一旦变化，必须同步登记（否则上面测试失败）。"""
    assert set(qb.REFERENCE_SNAPSHOTS) == {
        "P54_decision_grade", "P55_divergence", "P56_continuum_v31",
        "P51_uoft_real"}


# ── B7-2：fail closed（无 fallback）────────────────────────────────────
def test_guard_rejects_missing_manifest(scratch, monkeypatch):
    monkeypatch.setattr(qb, "_guard",
                        lambda: ArtifactGuard(scratch / "nope.json", root=ROOT))
    with pytest.raises(ArtifactGuardError):
        qb.reference_metrics()


def test_guard_rejects_corrupt_manifest(scratch, monkeypatch):
    bad = scratch / "m.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(qb, "_guard", lambda: ArtifactGuard(bad, root=ROOT))
    with pytest.raises(ArtifactGuardError):
        qb.reference_metrics()


def test_guard_rejects_unregistered_artifact(scratch, monkeypatch):
    (scratch / "docs").mkdir()
    (scratch / "docs" / "x.json").write_text("{}", encoding="utf-8")
    man = scratch / "m.json"
    man.write_text(json.dumps({"code_hashes": {}}), encoding="utf-8")
    g = ArtifactGuard(man, root=scratch)
    with pytest.raises(ArtifactGuardError) as e:
        g.require("docs/x.json")
    assert "未在 manifest 中登记" in str(e.value)


def test_guard_rejects_missing_artifact(scratch, monkeypatch):
    """已登记但文件不存在 → 拒绝（不回退到「present: False」）。"""
    reg = {rel: "0" * 64 for rel in qb.REFERENCE_SNAPSHOTS.values()}
    man = scratch / "m.json"
    man.write_text(json.dumps({"code_hashes": reg}), encoding="utf-8")
    monkeypatch.setattr(qb, "_guard", lambda: ArtifactGuard(man, root=scratch))
    with pytest.raises(ArtifactGuardError) as e:
        qb.reference_metrics()
    assert "缺失" in str(e.value)


def test_guard_rejects_stale_artifact(scratch, monkeypatch):
    """文件存在但哈希与登记不一致 → 拒绝（stale/wrong-manifest）。"""
    reg = {}
    for rel in qb.REFERENCE_SNAPSHOTS.values():
        p = scratch / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
        reg[rel] = "0" * 64                      # 故意登记错误哈希
    man = scratch / "m.json"
    man.write_text(
        json.dumps({"readonly_dependency_hashes": reg}), encoding="utf-8")
    monkeypatch.setattr(qb, "_guard", lambda: ArtifactGuard(man, root=scratch))
    with pytest.raises(ArtifactGuardError) as e:
        qb.reference_metrics()
    assert "stale/wrong-manifest" in str(e.value)


def test_guard_has_no_stale_hash_cache(monkeypatch):
    """篡改后立刻拒绝：绝不使用缓冲的旧哈希（无 fallback）。"""
    g = ArtifactGuard(ARTIFACT_MANIFEST, root=ROOT)
    rel = "docs/V3F_P55_DIVERGENCE_DIAGNOSIS.json"
    assert g.require(rel) == ROOT / rel
    original = (ROOT / rel).read_bytes()
    try:
        (ROOT / rel).write_bytes(original + b"\n")
        with pytest.raises(ArtifactGuardError):
            g.require(rel)
    finally:
        (ROOT / rel).write_bytes(original)
    assert g.require(rel) == ROOT / rel        # 复原后重新可用


def test_revoked_authorization_blocks_read(monkeypatch, scratch):
    """授权被撤销（登记集为空）→ 读取路径必须拒绝，而非降级读取。"""
    man = scratch / "m.json"
    man.write_text(json.dumps({"code_hashes": {}}), encoding="utf-8")
    monkeypatch.setattr(qb, "_guard", lambda: ArtifactGuard(man, root=ROOT))
    with pytest.raises(ArtifactGuardError):
        qb.build_manifest()
    monkeypatch.setattr(mb, "_guard", lambda: ArtifactGuard(man, root=ROOT))
    with pytest.raises(ArtifactGuardError):
        mb.r1_reference()


def test_require_returns_path_under_trusted_root(scratch):
    """批准读取使用 root/rel（不采用工件自报路径）。"""
    p = scratch / "a.json"
    p.write_text("{}", encoding="utf-8")
    man = scratch / "m.json"
    man.write_text(json.dumps({"code_hashes": {"a.json": sha256_file(p)}}),
                   encoding="utf-8")
    g = ArtifactGuard(man, root=scratch)
    assert g.require("a.json") == scratch / "a.json"


# ── B7-2：历史调用点语义不变（re-export）───────────────────────────────
def test_governance_reexports_guard_without_changing_semantics(scratch):
    from poker_decision_ai.research_unfreeze import governance as gov
    assert gov.ArtifactGuard is ArtifactGuard
    assert gov.ArtifactGuardError is ArtifactGuardError
    p = scratch / "a.json"
    p.write_text("{}", encoding="utf-8")
    man = scratch / "m.json"
    man.write_text(
        json.dumps({"code_hashes": {"a.json": gov.sha256_file(p)}}),
        encoding="utf-8")
    g = gov.ArtifactGuard(man, root=scratch)
    assert g.require("a.json") == p
    assert g.verify_sealed_code() == {"checked": 1, "drift": {}}
    p.write_text('{"changed":1}', encoding="utf-8")
    with pytest.raises(gov.ArtifactGuardError):
        g.require("a.json")


def test_artifact_manifest_covers_all_readonly_dependencies():
    man = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    assert man["registered_count"] == 10          # 历史封存项数量不变
    assert man["readonly_dependency_count"] == len(
        man["readonly_dependency_hashes"]) == 5
    assert not (set(man["code_hashes"]) & set(man["readonly_dependency_hashes"]))
