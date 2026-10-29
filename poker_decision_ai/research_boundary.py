"""M7 隔离边界：research 实现不得从 production decision path 到达。

附加式模块（不修改任何既有文件）。提供静态检查器，供测试与运行期审计使用。
"""
from __future__ import annotations

import ast
from pathlib import Path

BOUNDARY_VERSION = "v3f-research-production-boundary-v1"
PKG = Path(__file__).resolve().parents[0]          # poker_decision_ai/
RESEARCH_PKG = "research_unfreeze"

# research 侧实现（禁止被生产决策路径调用）
RESEARCH_ENTRYPOINTS = ("evaluate_v3", "reference_complete", "run_s3prime",
                        "run_core", "process_hands")

# 允许调用 research 实现的模块类别（显式白名单，必须是 benchmark/research/测试）
ALLOWED_CALLER_MARKERS = ("benchmark", RESEARCH_PKG, "test")


class BoundaryViolation(RuntimeError):
    pass


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="ignore"))
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out.add(n.module or "")
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


def production_modules() -> list:
    return [p for p in PKG.rglob("*.py")
            if RESEARCH_PKG not in p.parts and p.name != "__init__.py"]


def find_research_imports() -> list:
    """生产包中导入 research_unfreeze 的模块（应为空）。"""
    bad = []
    for p in production_modules():
        if any(RESEARCH_PKG in m for m in _imports(p)):
            bad.append(str(p.relative_to(PKG)))
    return bad


def find_research_entrypoint_callers() -> list:
    """调用 research 入口（如 evaluate_v3）但不在白名单类别的模块。"""
    bad = []
    for p in PKG.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        called = [e for e in RESEARCH_ENTRYPOINTS if f"{e}(" in text]
        if not called:
            continue
        # 定义者自身不算"调用方"（如 multistreet_v3.py 定义 evaluate_v3）
        if all(f"def {e}(" in text for e in called):
            continue
        rel = str(p.relative_to(PKG))
        if any(m in rel or m in p.parts for m in ALLOWED_CALLER_MARKERS):
            continue
        bad.append(f"{rel} -> {called}")
    return bad


def assert_production_isolation() -> dict:
    """fail closed：发现越界即抛异常。"""
    imports = find_research_imports()
    callers = find_research_entrypoint_callers()
    if imports or callers:
        raise BoundaryViolation(
            f"research/production 边界被突破: imports={imports} callers={callers}")
    return {"boundary_version": BOUNDARY_VERSION,
            "checked_production_modules": len(production_modules()),
            "research_imports": [], "unallowed_entrypoint_callers": []}
