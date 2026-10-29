"""M11 — `sealed-benchmark-exec-contract-v1` 的治理验收（C1–C7 真实性验证）。

与 M10 测试的分工：
  * `test_m10_contract.py` / `test_m10_fail_closed.py` 验证**契约要素存在**；
  * 本文件验证 C1–C7 在**真实的 production import graph、真实执行入口与
    真实失败路径**中成立，并固定两条决定性的边界事实：
      (a) 不存在**任何**非入口的生产代码访问 sealed benchmark
          （静态 import、动态 import、字符串字面量三条路径全查）；
      (b) sealed `_read()` 的结果是**纯元数据、无任何消费者** —— 因此
          "绕过入口"这一残余缺口的风险被限定在快照的参考区块，
          **不进入**任何指标/决策计算。

不修改 sealed 文件、历史 manifest、历史结果、验收定义或阈值。
"""
from __future__ import annotations

import ast
import json
import shutil
import uuid
from pathlib import Path

import pytest

from poker_decision_ai import sealed_benchmark_entry as sbe
from poker_decision_ai.artifact_guard import ArtifactGuard, ArtifactGuardError

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"
CONTRACT_DOC = ROOT / "docs" / "V3F_M10_SEALED_BENCH_EXEC_CONTRACT.md"
ARTIFACT_MANIFEST = ROOT / "docs" / "V3F_ARTIFACT_MANIFEST.json"

SEALED_STEMS = ("multistreet_v2_benchmark", "multistreet_v3_benchmark")
ENTRY = "sealed_benchmark_entry.py"
EXCLUDED = {ENTRY, "multistreet_v2_benchmark.py", "multistreet_v3_benchmark.py"}


def _read(p: Path) -> str:
    """源码读取：容忍 BOM（仓库内部分脚本带 UTF-8 BOM）。"""
    return p.read_text(encoding="utf-8-sig", errors="ignore")


def _production_modules():
    return [p for p in sorted(PKG.rglob("*.py")) if p.name not in EXCLUDED]


def _scripts():
    return sorted((ROOT / "scripts").rglob("*.py"))


@pytest.fixture()
def scratch_pair():
    """(manifest 路径, 输出路径)，位于工作区内（本机沙箱限制 %TEMP%）。"""
    d = ROOT / ".tmp" / "m11_audit" / uuid.uuid4().hex
    d.mkdir(parents=True)
    try:
        yield d / "m.json", d / "out.json"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ══ C3：真实 production import graph ═══════════════════════════════════
def test_c3_no_static_import_outside_entry():
    offenders = []
    for p in _production_modules():
        tree = ast.parse(_read(p))
        for n in ast.walk(tree):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""] if isinstance(n, ast.ImportFrom)
                    else [])
            if any(stem in m for m in mods for stem in SEALED_STEMS):
                offenders.append(str(p.relative_to(ROOT)))
    assert offenders == [], f"非入口模块静态 import 了 sealed benchmark: {offenders}"


def test_c3_no_dynamic_import_outside_entry():
    """动态 import 只能加载 STAGES 声明的模块；非入口模块不得出现。"""
    importers = []
    for p in _production_modules():
        src = _read(p)
        if "import_module" in src or "__import__" in src:
            if any(stem in src for stem in SEALED_STEMS):
                importers.append(str(p.relative_to(ROOT)))
    assert importers == [], f"非入口模块动态 import 了 sealed benchmark: {importers}"


def test_c3_no_string_literal_bypass():
    """字符串字面量里也不得出现 sealed 模块名（防 importlib(name) 式规避）。"""
    offenders = []
    for p in _production_modules():
        tree = ast.parse(_read(p))
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                if any(stem in n.value for stem in SEALED_STEMS):
                    offenders.append((str(p.relative_to(ROOT)), n.lineno))
    assert offenders == [], f"字符串字面量引用 sealed 模块: {offenders}"


def test_c3_entry_declares_all_sealed_modules_it_may_load():
    """入口可加载的模块集合必须**恰好**是 STAGES 声明的两个（无越权面）。"""
    declared = {m["module"] for m in sbe.STAGES.values()}
    assert declared == {"poker_decision_ai.multistreet_v2_benchmark",
                        "poker_decision_ai.multistreet_v3_benchmark"}


# ══ C1：真实执行入口 ═══════════════════════════════════════════════════
def test_c1_no_script_executes_stages():
    for p in _scripts():
        assert "run_stage(" not in _read(p), \
            f"脚本直接执行 stage（绕过治理语义）: {p}"


def test_c1_entry_api_requires_explicit_out_path():
    import inspect
    sig = inspect.signature(sbe.run_stage)
    assert "out_path" in sig.parameters
    assert sig.parameters["out_path"].default is inspect.Parameter.empty


def test_c1_no_production_code_executes_sealed_stages():
    """现状如实登记：**没有任何生产模块/脚本**调用 sealed stage 函数。

    即 C1 在当前仓库是"空真"（vacuously true）：不存在可绕过入口的
    生产执行路径。若未来新增执行路径，本测试会失败并强制其走入口。
    """
    offenders = []
    for p in list(_production_modules()) + list(_scripts()):
        tree = ast.parse(_read(p))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                nm = (n.func.id if isinstance(n.func, ast.Name)
                      else getattr(n.func, "attr", ""))
                if nm in ("run_b1", "run_b2", "run_b3"):
                    offenders.append(f"{p.relative_to(ROOT)}:{n.lineno}:{nm}")
    assert offenders == [], f"生产代码直接执行 sealed stage: {offenders}"


# ══ 决定性边界事实：sealed `_read()` 是纯元数据、无消费者 ══════════════
def test_sealed_reads_are_metadata_only():
    """4 处 sealed `_read()` 的结果只写进快照的参考区块（不参与计算）。"""
    for stem, keys in (("multistreet_v2_benchmark",
                        ("r1_reference", "r2_reference")),
                       ("multistreet_v3_benchmark", ("references",))):
        src = _read(PKG / f"{stem}.py")
        lines = src.splitlines()
        calls = [n for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "id", "") == "_read"]
        assert calls, f"{stem} 的 _read 调用点缺失（契约面变化）"
        for c in calls:
            ctx = "\n".join(lines[max(0, c.lineno - 3):c.lineno + 2])
            assert any(f'"{k}"' in ctx for k in keys), ctx


def test_sealed_reference_keys_have_no_consumers():
    """生产代码中：sealed 的 r1_reference / r2_reference **零计算消费者**。

    这限定了"绕过入口"残余缺口的风险面：sealed `_read()` 的
    `{"present": False}` 降级不会进入任何指标/决策计算。

    精确边界：全仓唯一的 `["r1_reference"]` 下标访问位于
    `tests/unit/test_r2_multistreet.py::test_r2_snapshot_separate_and_additive`，
    它读取的是**冻结的历史快照 JSON 文件**（非 sealed `_read()` 的产物），
    属只读断言。生产侧（`poker_decision_ai/` 与 `scripts/`）为 0 处。
    """
    consumers = []
    for p in list(_production_modules()) + list(_scripts()):
        tree = ast.parse(_read(p))
        for n in ast.walk(tree):
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant):
                if n.slice.value in ("r1_reference", "r2_reference"):
                    consumers.append(f"{p.relative_to(ROOT)}:{n.lineno}")
    assert consumers == [], f"生产代码出现 r1/r2_reference 消费者: {consumers}"


def test_wired_path_fails_closed_while_sealed_path_degrades():
    """两条 API 必须保持区分：已接线者抛错，未接线者降级（且无消费者）。"""
    mb = _read(PKG / "multistreet_benchmark.py")
    assert "_guard().require(" in mb
    assert 'return {"present": False}' not in mb
    for stem in SEALED_STEMS:
        assert 'return {"present": False}' in _read(PKG / f"{stem}.py")


# ══ C2 / C4：真实失败路径 ═════════════════════════════════════════════
def _arm(monkeypatch, manifest: Path, stage: str):
    """测试侧探针：接管入口 guard 与 stage 函数（不改任何 sealed 文件）。"""
    import importlib
    monkeypatch.setattr(sbe, "_guard",
                        lambda mp=None: ArtifactGuard(manifest, root=ROOT))
    meta = sbe.get_stage(stage)
    calls = []
    monkeypatch.setattr(importlib.import_module(meta["module"]), meta["func"],
                        lambda *a, **kw: calls.append(kw) or {"reached": True})
    return calls


def test_c2_and_c4_are_not_mutually_catchable(scratch_pair):
    """端到端：用 C4 的 except 抓不住 C2 的失败，反之亦然。"""
    man, out = scratch_pair
    man.write_text(json.dumps({"code_hashes": {}}), encoding="utf-8")

    real = sbe._guard
    sbe._guard = lambda mp=None: ArtifactGuard(man, root=ROOT)
    try:
        caught_by_c4 = None
        try:
            sbe.run_stage("v3.b1", out_path=out)
        except sbe.SealedWriteRefusedError as e:     # 用 C4 类型抓 C2 失败
            caught_by_c4 = e
        except ArtifactGuardError:
            pass
        assert caught_by_c4 is None, "C2 失败被 C4 的 except 吞掉了"
    finally:
        sbe._guard = real

    caught_by_c2 = None
    try:
        sbe.run_stage("v2.b2", out_path=out)          # C4 失败
    except ArtifactGuardError as e:                   # 用 C2 类型抓 C4 失败
        caught_by_c2 = e
    except sbe.SealedWriteRefusedError:
        pass
    assert caught_by_c2 is None, "C4 失败被 C2 的 except 吞掉了"


def test_c2_failure_produces_no_output_and_no_stage_call(scratch_pair,
                                                         monkeypatch):
    """真实失败路径：既不调用 stage，也不落任何输出文件。"""
    man, out = scratch_pair
    man.write_text(json.dumps({"code_hashes": {}}), encoding="utf-8")
    calls = _arm(monkeypatch, man, "v3.b1")
    with pytest.raises(ArtifactGuardError):
        sbe.run_stage("v3.b1", out_path=out)
    assert calls == []
    assert not out.exists()


def test_c4_failure_produces_no_output_and_no_stage_call(monkeypatch,
                                                         scratch_pair):
    """C4 不可重定向路径：同样不得调用 stage、不得落输出。"""
    _man, out = scratch_pair
    calls = _arm(monkeypatch, ARTIFACT_MANIFEST, "v2.b2")
    with pytest.raises(sbe.SealedWriteRefusedError):
        sbe.run_stage("v2.b2", out_path=out)
    assert calls == []
    assert not out.exists()


# ══ C6 / CHECK-A / CHECK-B ════════════════════════════════════════════
def test_check_a_sealed_reads_still_unguarded_and_not_counted_as_wired():
    for stem in SEALED_STEMS:
        src = _read(PKG / f"{stem}.py")
        assert "ArtifactGuard" not in src
        assert "require(" not in src
        assert "def _read(" in src
    man = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    declared = " ".join(man.get("guard_read_paths") or [])
    for stem in SEALED_STEMS:
        assert stem not in declared


def test_c6_contract_states_three_distinctions():
    doc = CONTRACT_DOC.read_text(encoding="utf-8")
    for phrase in ("仍是 unguarded", "执行前校验", "禁止覆盖历史工件",
                   "不等价于 in-path guard"):
        assert phrase in doc, phrase


def test_c7_sealed_modules_unchanged():
    import hashlib
    for rel, man_rel in (("poker_decision_ai/multistreet_v2_benchmark.py",
                          "docs/V3F_R21_MANIFEST.json"),
                         ("poker_decision_ai/multistreet_v3_benchmark.py",
                          "docs/V3F_R22_MANIFEST.json")):
        man = json.loads((ROOT / man_rel).read_text(encoding="utf-8"))
        now = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert now == man["code_hashes"][rel], rel
