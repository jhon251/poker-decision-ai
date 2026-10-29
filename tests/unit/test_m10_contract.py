"""M10 — Sealed Benchmark 执行治理契约的机器可核查部分。

契约：`docs/V3F_M10_SEALED_BENCH_EXEC_CONTRACT.md`
（`sealed-benchmark-exec-contract-v1`，条款 C1–C7）

本文件把契约里"可核查"的条款变成断言，使下列事实**无法被无声改写**：
  * CHECK-A：两个 sealed 模块内仍**没有** guard 调用（缺口如实存在）；
  * CHECK-B：登记侧**未**声称这 4 处读取已接线；
  * C3：生产包内**仅**入口 import sealed 模块；
  * C5：两个 sealed 模块 SHA-256 == 其封存值；
  * C6：§1.1 三条陈述必须同时出现在契约文档中（禁止只引用其中一条）。

不修改任何 sealed 文件 / manifest / 验收定义。
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from poker_decision_ai import sealed_benchmark_entry as sbe

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "poker_decision_ai"
CONTRACT_DOC = ROOT / "docs" / "V3F_M10_SEALED_BENCH_EXEC_CONTRACT.md"
ARTIFACT_MANIFEST = ROOT / "docs" / "V3F_ARTIFACT_MANIFEST.json"

SEALED_MODULES = {
    "poker_decision_ai/multistreet_v2_benchmark.py": "docs/V3F_R21_MANIFEST.json",
    "poker_decision_ai/multistreet_v3_benchmark.py": "docs/V3F_R22_MANIFEST.json",
}
ENTRY_MODULE = "sealed_benchmark_entry.py"


# ── CHECK-A：缺口如实存在（sealed 模块内无 guard）─────────────────────
def test_check_a_sealed_modules_have_no_guard():
    for rel in SEALED_MODULES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "ArtifactGuard" not in src, rel
        assert "require(" not in src, rel
        assert "def _read(" in src, rel       # 宽松读取仍在模块内部


# ── CHECK-B：登记侧未声称这 4 处已接线 ─────────────────────────────────
def test_check_b_registry_does_not_claim_sealed_paths():
    man = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    declared = " ".join(man.get("guard_read_paths") or [])
    for rel in SEALED_MODULES:
        stem = Path(rel).stem
        assert stem not in declared, (
            f"登记侧不得声称 {stem} 的读取路径已接入 guard（契约 C6）")


def test_check_b_only_unsealed_paths_are_declared():
    """已声明的接线路径必须**全部**位于未封存文件内（正反对照）。"""
    man = json.loads(ARTIFACT_MANIFEST.read_text(encoding="utf-8"))
    sealed_srcs = set(SEALED_MODULES)
    for entry in man.get("guard_read_paths") or []:
        mod = entry.partition("::")[0]
        src = mod[:-3].replace(".", "/") + ".py"
        assert src not in sealed_srcs, entry
        assert (ROOT / src).exists(), entry


# ── C3：生产包内仅入口 import sealed 模块 ──────────────────────────────
def _imported_names(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            names.add(base)
            names |= {f"{base}.{a.name}" for a in n.names}
    return names


def test_c3_only_entry_point_imports_sealed_modules():
    targets = {"poker_decision_ai.multistreet_v2_benchmark",
               "poker_decision_ai.multistreet_v3_benchmark"}
    offenders = []
    for p in sorted(PKG.rglob("*.py")):
        if p.name in (ENTRY_MODULE, "multistreet_v2_benchmark.py",
                      "multistreet_v3_benchmark.py"):
            continue
        hit = {t for t in targets
               if t in _imported_names(p)
               or f"from .{t.rsplit('.', 1)[-1]} import" in
               p.read_text(encoding="utf-8", errors="ignore")}
        if hit:
            offenders.append((str(p.relative_to(ROOT)), sorted(hit)))
    assert offenders == [], f"生产包内非入口模块 import 了 sealed benchmark: {offenders}"


def test_c3_entry_point_is_the_sanctioned_importer():
    """入口本身确实通过 STAGES 声明引用 sealed 模块（非硬编码散落）。"""
    for meta in sbe.STAGES.values():
        assert meta["module"].startswith("poker_decision_ai.multistreet_v")
    src = (PKG / ENTRY_MODULE).read_text(encoding="utf-8")
    assert "importlib.import_module" in src


# ── C5：封存不变 ───────────────────────────────────────────────────────
def test_c5_sealed_modules_match_their_seal():
    for rel, man_rel in SEALED_MODULES.items():
        man = json.loads((ROOT / man_rel).read_text(encoding="utf-8"))
        assert rel in man["code_hashes"], rel
        now = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        assert now == man["code_hashes"][rel], f"{rel} 偏离 {man_rel}"


# ── C1：入口逐 stage 覆盖（与本契约 §2 表一致）─────────────────────────
def test_c1_contract_surface_covers_every_stage():
    doc = CONTRACT_DOC.read_text(encoding="utf-8")
    for stage in sbe.list_stages():
        assert f"`{stage}`" in doc, f"契约文档未声明 stage {stage}"
    assert set(sbe.list_stages()) == {"v2.b1", "v2.b2", "v3.b1", "v3.b2", "v3.b3"}


def test_contract_surface_table_is_accurate():
    """契约 §2 的「可重定向」必须与实现一致，且不可重定向者显式标注。"""
    doc = CONTRACT_DOC.read_text(encoding="utf-8")
    for stage, meta in sbe.STAGES.items():
        assert meta["redirectable"] == (meta["out_kwarg"] is not None), stage
        assert f"`{stage}`" in doc
    assert doc.count("**不可**") >= 1


# ── C6：§1.1 三条陈述必须同时出现（禁止只引用其中一条）─────────────────
def test_c6_contract_states_all_three_distinctions():
    doc = CONTRACT_DOC.read_text(encoding="utf-8")
    # (1) 内部直接调用仍 unguarded
    assert "仍是 unguarded" in doc
    # (2) 入口：执行前校验 + 禁止覆盖
    assert "执行前校验" in doc and "禁止覆盖历史工件" in doc
    # (3) 入口不等价于 in-path guard
    assert "不等价于 in-path guard" in doc
    # 并且显式禁止相反的表述
    assert "不得**表述为\"4 处 `_read()` 已接入 ArtifactGuard\"" in doc


def test_c6_contract_declares_version_and_status():
    doc = CONTRACT_DOC.read_text(encoding="utf-8")
    assert "sealed-benchmark-exec-contract-v1" in doc
    assert "生效" in doc


# ── C7：禁止规避（手段未出现在任何新增工件中）──────────────────────────
def test_c7_no_forbidden_workaround_in_entry_module():
    src = (PKG / ENTRY_MODULE).read_text(encoding="utf-8")
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
                assert nm != "_read", "禁止给 _read 赋值（语义分叉）"


def test_c7_no_second_benchmark_implementation_exists():
    """不存在复制的/带 guarded 后缀的第二份 benchmark 实现（防语义分叉）。

    现有三个 multistreet benchmark 模块均为历史既有产物（R2 / R2.1 / R2.2），
    其中 R2.1/R2.2 属封存面；本契约**不新增**任何一份。
    """
    names = sorted(p.name for p in PKG.rglob("*.py")
                   if p.name.endswith("benchmark.py")
                   and p.name.startswith("multistreet"))
    assert names == ["multistreet_benchmark.py",
                     "multistreet_v2_benchmark.py",
                     "multistreet_v3_benchmark.py"], names
    assert [p.name for p in PKG.rglob("*_guarded*.py")] == []
    assert [p.name for p in PKG.rglob("*benchmark_m9*.py")] == []
