"""M9 — sealed benchmark 独立入口的最小测试。

验证三件事（且不夸大覆盖面）：
  1. **读取前真实校验生效**：未登记 / 缺失 / stale → 抛错，且 sealed stage
     函数**根本不被调用**（fail closed 的硬证据）；
  2. **历史证据写入保护**：输出不可重定向（v2.b2）、或输出指向已登记历史
     工件 → 拒绝执行，stage 函数**根本不被调用**；
  3. **边界如实**：sealed 模块逐字节未改；本入口**不是** in-path guard，
     直接调用 sealed `run_b1()` 仍然绕过校验（以测试固定该事实）。

不修改任何 sealed 文件 / manifest / 验收定义。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path

import pytest

from poker_decision_ai import sealed_benchmark_entry as sbe
from poker_decision_ai.artifact_guard import ArtifactGuardError

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_MANIFEST = ROOT / "docs" / "V3F_ARTIFACT_MANIFEST.json"

_SEALED_MODULES = ("poker_decision_ai/multistreet_v2_benchmark.py",
                   "poker_decision_ai/multistreet_v3_benchmark.py")


@pytest.fixture()
def scratch():
    """工作区内自管理临时目录（本机沙箱禁止删除 %TEMP%，故不用 tmp_path）。"""
    d = ROOT / ".tmp" / "m9_sealed_entry" / uuid.uuid4().hex
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


def _write_manifest(path: Path, hashes: dict) -> Path:
    path.write_text(json.dumps({"code_hashes": hashes}), encoding="utf-8")
    return path


def _arm(monkeypatch, manifest: Path) -> None:
    """把入口的 guard 指向测试 manifest（不改任何 sealed 文件）。"""
    monkeypatch.setattr(sbe, "_guard",
                        lambda mp=None: __import__(
                            "poker_decision_ai.artifact_guard", fromlist=["x"]
                        ).ArtifactGuard(manifest, root=ROOT))


class _Explode:
    """若被调用即失败：用于证明 stage 函数未被调用。"""

    def __init__(self):
        self.called = False

    def __call__(self, *a, **kw):
        self.called = True
        raise AssertionError("stage 函数不应被调用")


def _arm_guard_and_module(monkeypatch, manifest: Path, stage: str):
    """同时接管 guard 与 sealed 函数（后者只做调用探针，不改文件）。"""
    _arm(monkeypatch, manifest)
    import importlib
    meta = sbe.get_stage(stage)
    mod = importlib.import_module(meta["module"])
    probe = _Explode()
    monkeypatch.setattr(mod, meta["func"], probe)
    return probe


# ── 1. 读取前真实校验生效（fail closed）────────────────────────────────
def test_unregistered_artifact_aborts_before_execution(scratch, monkeypatch):
    """登记集为空 → 未登记 → 抛错，且 stage 从未被调用。"""
    man = _write_manifest(scratch / "m.json", {})
    probe = _arm_guard_and_module(monkeypatch, man, "v3.b1")
    with pytest.raises(ArtifactGuardError) as e:
        sbe.run_stage("v3.b1", out_path=scratch / "out" / "b1.json")
    assert "未在 manifest 中登记" in str(e.value)
    assert probe.called is False


def test_missing_artifact_aborts_before_execution(scratch, monkeypatch):
    """已登记但文件缺失 → 抛错，且 stage 从未被调用。"""
    reg = {rel: "0" * 64 for rel in sbe.required_artifacts("v3.b1")}
    man = _write_manifest(scratch / "m.json", reg)
    probe = _arm_guard_and_module(monkeypatch, man, "v3.b1")
    with pytest.raises(ArtifactGuardError):
        sbe.run_stage("v3.b1", out_path=scratch / "out" / "b1.json")
    assert probe.called is False


def test_stale_artifact_aborts_before_execution(scratch, monkeypatch):
    """哈希不一致（stale/wrong-manifest）→ 抛错，且 stage 从未被调用。"""
    reg = _real_hashes()
    for rel in sbe.required_artifacts("v3.b1"):
        reg[rel] = "0" * 64                      # 故意登记错误哈希
    man = _write_manifest(scratch / "m.json", reg)
    probe = _arm_guard_and_module(monkeypatch, man, "v3.b1")
    with pytest.raises(ArtifactGuardError) as e:
        sbe.run_stage("v3.b1", out_path=scratch / "out" / "b1.json")
    assert "stale/wrong-manifest" in str(e.value)
    assert probe.called is False


def test_real_manifest_passes_preflight_for_every_stage():
    """对照：真实 manifest 下，所有 stage 的读取前校验必须通过。"""
    for stage in sbe.list_stages():
        ev = sbe.validate_before_read(stage)
        assert ev["checked"] == list(sbe.required_artifacts(stage))
        assert ev["in_path_guarded"] is False      # 如实声明：非 in-path


def test_preflight_does_not_read_artifact_content():
    """校验只做哈希，不解析工件内容 —— 因此不改变任何读取语义。"""
    src = (ROOT / "poker_decision_ai/sealed_benchmark_entry.py").read_text(
        encoding="utf-8")
    assert "read_text" not in src.replace("_write_manifest", "")
    assert "json.loads" not in src
    assert "_load_registered" in src                # 只读登记集（哈希表）


# ── 2. 历史证据写入保护 ────────────────────────────────────────────────
def test_unredirectable_stage_is_refused(monkeypatch):
    """v2.b2 输出不可重定向 → 拒绝执行，stage 从未被调用。"""
    probe = _arm_guard_and_module(monkeypatch, ARTIFACT_MANIFEST, "v2.b2")
    import pytest as _pytest
    with _pytest.raises(sbe.SealedWriteRefusedError) as e:
        sbe.run_stage("v2.b2", out_path=ROOT / "elsewhere.json")
    assert "不可重定向" in str(e.value)
    assert probe.called is False
    assert sbe.get_stage("v2.b2")["redirectable"] is False


def test_output_pointing_at_sealed_artifact_is_refused(scratch, monkeypatch):
    """输出路径指向已登记历史工件 → 拒绝（不覆盖历史证据）。"""
    probe = _arm_guard_and_module(monkeypatch, ARTIFACT_MANIFEST, "v3.b1")
    sealed_out = ROOT / "docs" / "V3F_R22_B1.json"
    with pytest.raises(sbe.SealedWriteRefusedError) as e:
        sbe.run_stage("v3.b1", out_path=sealed_out)
    assert "已登记历史工件" in str(e.value)
    assert probe.called is False


def test_missing_out_path_is_refused(monkeypatch):
    """未显式指定 out_path → 拒绝（绝不默认写回 docs/）。"""
    probe = _arm_guard_and_module(monkeypatch, ARTIFACT_MANIFEST, "v3.b1")
    with pytest.raises(sbe.SealedWriteRefusedError):
        sbe.run_stage("v3.b1", out_path=None)
    assert probe.called is False


def test_passing_run_delegates_with_original_signature(scratch, monkeypatch):
    """通过校验后按原签名委派，返回值原样透传（无包装改写）。"""
    seen = {}

    def fake_run_b1(**kw):
        seen.update(kw)
        return {"snapshot_version": "r22-bench-b1-v1", "ok": True}

    import poker_decision_ai.multistreet_v3_benchmark as v3
    monkeypatch.setattr(v3, "run_b1", fake_run_b1)
    out = scratch / "out" / "b1.json"
    res = sbe.run_stage("v3.b1", out_path=out)
    assert seen == {"snapshot": out}
    assert res["result"] == {"snapshot_version": "r22-bench-b1-v1", "ok": True}
    assert res["preflight"]["checked_count"] == 3
    assert res["preflight"]["in_path_guarded"] is False


# ── 3. 边界如实（不夸大）───────────────────────────────────────────────
def test_sealed_modules_are_byte_identical_to_seal():
    """sealed 模块逐字节未改（M9 未触碰历史封存证据）。"""
    for mod in _SEALED_MODULES:
        man_rel = ("docs/V3F_R21_MANIFEST.json" if "v2_benchmark" in mod
                   else "docs/V3F_R22_MANIFEST.json")
        man = json.loads((ROOT / man_rel).read_text(encoding="utf-8"))
        now = hashlib.sha256((ROOT / mod).read_bytes()).hexdigest()
        assert now == man["code_hashes"][mod], f"{mod} 偏离 {man_rel}"


def test_sealed_read_paths_remain_unguarded_in_path():
    """本入口不冒充 in-path guard：sealed 模块内仍无 guard 调用。"""
    for mod in _SEALED_MODULES:
        src = (ROOT / mod).read_text(encoding="utf-8")
        assert "ArtifactGuard" not in src
        assert "require(" not in src
        assert "def _read(" in src          # 宽松读取仍在模块内部


def test_entry_point_does_not_patch_sealed_modules():
    """入口不 monkeypatch：**可执行代码**中无 setattr/delattr，也无对 sealed
    模块符号赋值（按 AST 检查，不受 docstring 措辞影响）。"""
    import ast
    src = (ROOT / "poker_decision_ai/sealed_benchmark_entry.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (node.func.id if isinstance(node.func, ast.Name)
                    else getattr(node.func, "attr", ""))
            assert name not in ("setattr", "delattr", "monkeypatch"), name
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for t in targets:
                nm = getattr(t, "id", None) or getattr(t, "attr", "")
                assert nm != "_read", "禁止给 _read 赋值（会造成语义分叉）"

    # 运行期证据：sealed 模块的函数仍是其自身定义，未被替换
    import poker_decision_ai.multistreet_v3_benchmark as v3
    assert v3.run_b1.__module__ == "poker_decision_ai.multistreet_v3_benchmark"


def test_direct_call_still_bypasses_the_entry_point():
    """如实固定事实：直接调用 sealed 模块的函数**不经过**本入口。

    这是 M9 无法消除的残余缺口——消除它必须修改 sealed 文件本身。
    """
    import inspect
    import poker_decision_ai.multistreet_v3_benchmark as v3
    src = inspect.getsource(v3.run_b1)
    assert "sealed_benchmark_entry" not in src
    assert "validate_before_read" not in src


def test_registered_artifacts_cover_every_stage_requirement():
    """登记侧无缺口：所有 stage 需要的工件都已在 manifest 中登记。"""
    reg = set(_real_hashes())
    for stage in sbe.list_stages():
        missing = [r for r in sbe.required_artifacts(stage) if r not in reg]
        assert missing == [], (stage, missing)


def test_guard_read_paths_metadata_still_reflects_reality():
    """manifest 的 guard_read_paths 不得声称 4 处 sealed 路径已接线。"""
    man = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    declared = " ".join(man.get("guard_read_paths") or [])
    assert "multistreet_v2_benchmark" not in declared
    assert "multistreet_v3_benchmark" not in declared
