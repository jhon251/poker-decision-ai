"""M10 — SEALED-BENCH-EXEC 契约 C2/C4 的 fail-closed 硬证据。

契约条款（`docs/V3F_M10_SEALED_BENCH_EXEC_CONTRACT.md`）：
  C2  校验未通过 → 抛 `ArtifactGuardError`，且**被调用的 stage 函数不得执行**
  C4  三种写入情形 → 抛 `SealedWriteRefusedError`，且**stage 不得执行**
  C6  拒绝必须是**声明的异常类型**，不得是其它异常（防"catch 后继续"）

每项都以"探针函数"断言 stage **根本未被调用**（不是口头承诺）。
不修改任何 sealed 文件 / manifest / 验收定义。
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from poker_decision_ai import sealed_benchmark_entry as sbe
from poker_decision_ai.artifact_guard import ArtifactGuard, ArtifactGuardError

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_MANIFEST = ROOT / "docs" / "V3F_ARTIFACT_MANIFEST.json"


@pytest.fixture()
def scratch():
    """工作区内自管理临时目录（本机沙箱禁止删除 %TEMP%，故不用 tmp_path）。"""
    d = ROOT / ".tmp" / "m10_contract" / uuid.uuid4().hex
    d.mkdir(parents=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _real_hashes() -> dict:
    man = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    reg = dict(man["code_hashes"])
    reg.update(man.get("readonly_dependency_hashes") or {})
    return reg


class _Probe:
    """若被调用即失败 —— 用于证明 stage 函数未被调用。"""

    def __init__(self):
        self.called = False

    def __call__(self, *a, **kw):
        self.called = True
        raise AssertionError("契约 C2/C4 被违反：stage 函数被执行了")


def _arm(monkeypatch, manifest: Path, stage: str) -> _Probe:
    """把入口 guard 指向测试 manifest，并把 sealed 函数换成探针。

    注意：这是**测试侧**的调用探针，只替换测试进程内的模块属性；不改任何
    sealed 文件字节，也不构成 M9/M10 契约中的 monkeypatch 规避手段
    （入口自身不含任何 patch，已由 test_c7 断言）。
    """
    import importlib
    monkeypatch.setattr(
        sbe, "_guard",
        lambda mp=None: ArtifactGuard(manifest, root=ROOT))
    meta = sbe.get_stage(stage)
    probe = _Probe()
    monkeypatch.setattr(importlib.import_module(meta["module"]),
                        meta["func"], probe)
    return probe


def _manifest(path: Path, hashes: dict) -> Path:
    path.write_text(json.dumps({"code_hashes": hashes}), encoding="utf-8")
    return path


# ── C2：校验未通过 → 不执行（4 项）─────────────────────────────────────
def test_c2_unregistered_artifact_never_executes_stage(scratch, monkeypatch):
    man = _manifest(scratch / "m.json", {})
    probe = _arm(monkeypatch, man, "v3.b1")
    with pytest.raises(ArtifactGuardError):
        sbe.run_stage("v3.b1", out_path=scratch / "out.json")
    assert probe.called is False


def test_c2_missing_artifact_never_executes_stage(scratch, monkeypatch):
    reg = {rel: "0" * 64 for rel in sbe.required_artifacts("v3.b1")}
    man = _manifest(scratch / "m.json", reg)
    probe = _arm(monkeypatch, man, "v3.b1")
    with pytest.raises(ArtifactGuardError):
        sbe.run_stage("v3.b1", out_path=scratch / "out.json")
    assert probe.called is False


def test_c2_stale_artifact_never_executes_stage(scratch, monkeypatch):
    reg = _real_hashes()
    for rel in sbe.required_artifacts("v3.b1"):
        reg[rel] = "0" * 64
    man = _manifest(scratch / "m.json", reg)
    probe = _arm(monkeypatch, man, "v3.b1")
    with pytest.raises(ArtifactGuardError):
        sbe.run_stage("v3.b1", out_path=scratch / "out.json")
    assert probe.called is False


def test_c2_missing_manifest_never_executes_stage(scratch, monkeypatch):
    probe = _arm(monkeypatch, scratch / "does_not_exist.json", "v2.b1")
    with pytest.raises(ArtifactGuardError):
        sbe.run_stage("v2.b1", out_path=scratch / "out.json")
    assert probe.called is False


# ── C4：写入保护 → 不执行（3 项）───────────────────────────────────────
def test_c4_unredirectable_stage_never_executes(monkeypatch):
    probe = _arm(monkeypatch, ARTIFACT_MANIFEST, "v2.b2")
    with pytest.raises(sbe.SealedWriteRefusedError):
        sbe.run_stage("v2.b2", out_path=ROOT / "anywhere.json")
    assert probe.called is False


def test_c4_sealed_output_target_never_executes(monkeypatch):
    probe = _arm(monkeypatch, ARTIFACT_MANIFEST, "v3.b1")
    with pytest.raises(sbe.SealedWriteRefusedError):
        sbe.run_stage("v3.b1", out_path=ROOT / "docs" / "V3F_R22_B1.json")
    assert probe.called is False


def test_c4_missing_out_path_never_executes(monkeypatch):
    probe = _arm(monkeypatch, ARTIFACT_MANIFEST, "v3.b1")
    with pytest.raises(sbe.SealedWriteRefusedError):
        sbe.run_stage("v3.b1", out_path=None)
    assert probe.called is False


# ── C6：拒绝必须是声明的异常类型（两条独立断言）─────────────────────────
def test_c6_refusals_are_the_declared_types(scratch):
    """C2 拒绝 == ArtifactGuardError；C4 拒绝 == SealedWriteRefusedError。

    两者必须是**互不相同**的类型，避免调用方用 except 一类异常把写入保护
    连同校验一起吞掉。
    """
    assert issubclass(ArtifactGuardError, RuntimeError)
    assert issubclass(sbe.SealedWriteRefusedError, RuntimeError)
    assert ArtifactGuardError is not sbe.SealedWriteRefusedError

    # C4 拒绝不得是 ArtifactGuardError（否则可被 except ArtifactGuardError 吞掉）
    with pytest.raises(sbe.SealedWriteRefusedError) as e1:
        sbe.run_stage("v2.b2", out_path=scratch / "out.json")
    assert not isinstance(e1.value, ArtifactGuardError)

    # C2 拒绝不得是 SealedWriteRefusedError
    man = _manifest(scratch / "m.json", {})
    import poker_decision_ai.sealed_benchmark_entry as m
    real_guard = m._guard
    m._guard = lambda mp=None: ArtifactGuard(man, root=ROOT)
    try:
        with pytest.raises(ArtifactGuardError) as e2:
            sbe.run_stage("v3.b1", out_path=scratch / "out.json")
        assert not isinstance(e2.value, sbe.SealedWriteRefusedError)
    finally:
        m._guard = real_guard


def test_c2_no_fallback_result_on_failure(scratch, monkeypatch):
    """失败时**不得**返回任何结果对象（禁止 fallback / 降级占位）。"""
    man = _manifest(scratch / "m.json", {})
    _arm(monkeypatch, man, "v3.b1")
    try:
        out = sbe.run_stage("v3.b1", out_path=scratch / "out.json")
    except ArtifactGuardError:
        out = "RAISED"
    assert out == "RAISED", f"校验失败却返回了结果：{out!r}"
    assert not (scratch / "out.json").exists(), "失败路径不得产生输出文件"
