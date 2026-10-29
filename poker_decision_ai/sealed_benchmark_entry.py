"""M9 — sealed benchmark 的「读取前校验 + 证据写入保护」独立入口。

背景（M7/M8 已记录的缺口）
----------------------------
`multistreet_v2_benchmark.py` 与 `multistreet_v3_benchmark.py` 被 R21/R22 历史
manifest 逐字节封存，因此**不能修改**（改动即便于历史封存证据）。这两个模块内
各有自带的 `_read()`（缺失即静默返回 `{"present": False}`）用于读取历史快照，
共 4 处读取路径未被 `ArtifactGuard` 覆盖。

本模块提供的东西（**精确边界，不得夸大**）
------------------------------------------
1. **读取前真实校验（pre-flight）**：在任何 stage 执行之前，用
   `ArtifactGuard.require()` 对"该 stage 直接读取的历史工件"逐一做真实
   SHA-256 校验；未登记 / 缺失 / stale / 哈希不一致 → `ArtifactGuardError`，
   **stage 根本不会被调用**（fail closed，无 fallback）。
2. **历史证据写入保护**：拒绝把输出写到任何已登记历史工件路径上。
   本项不是锦上添花——`multistreet_v2_benchmark.run_b2()` 的输出路径
   **不可重定向**（硬编码写 `B2_SNAPSHOT`），调用它会**直接覆盖已登记的历史
   工件**。因此该 stage 在本入口中被明确**拒绝执行**。

本模块**不**提供的东西（必须如实声明）
--------------------------------------
* 它**不是**那 4 处 `_read()` 的 in-path guard。sealed 模块内部的 `_read()`
  仍然是"缺失即 `{"present": False}"` 的宽松实现——因为没有修改那些文件
  就不可能改变这一点。
* 因此"直接调用 sealed 模块的 `run_b1()`"仍然**绕过**本入口。
* **不得**把本模块描述为"4 处读取路径已接入 guard"。正确表述是：
  「经本入口执行时，读取前有真实校验且不会破坏历史证据；
    绕过本入口直接调用时，仍无任何校验」。

与 M8 已否决方案的关系（均未使用）
----------------------------------
* 未 monkeypatch：不修改任何模块的全局符号，`_read` 行为原样；
* 未复制模块：没有第二份实现，不存在语义分叉；
* 非"外层假 guard"：校验失败时 stage **确实不执行**（有测试断言），
  且本模块在 docstring 中明示其覆盖范围有限；
* 未改 manifest 哈希：`V3F_ARTIFACT_MANIFEST.json` / R21 / R22 均未改动。
"""
from __future__ import annotations

from pathlib import Path

from .artifact_guard import ArtifactGuard, ArtifactGuardError

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_MANIFEST = ROOT / "docs" / "V3F_ARTIFACT_MANIFEST.json"

__all__ = ["SealedWriteRefusedError", "get_stage", "list_stages",
           "required_artifacts", "validate_before_read", "run_stage", "STAGES"]


class SealedWriteRefusedError(RuntimeError):
    """拒绝执行：该 stage 会写入历史封存工件，或输出路径未被重定向。"""


def _guard(manifest_path=None) -> ArtifactGuard:
    return ArtifactGuard(Path(manifest_path or ARTIFACT_MANIFEST), root=ROOT)


def _registered(manifest_path=None) -> set:
    """运行期登记集（code_hashes ∪ readonly_dependency_hashes）。"""
    man = _guard(manifest_path)._load_registered()
    return set(man)


# ── sealed 模块中被读取的历史工件（= M8 §4 的 4 处读取路径）─────────────
V2_READS = ("docs/V3F_R1_QUALITY_BASELINE.json",
            "docs/V3F_R2_MULTISTREET_BASELINE.json")
V3_READS = ("docs/V3F_R1_QUALITY_BASELINE.json",
            "docs/V3F_R2_MULTISTREET_BASELINE.json",
            "docs/V3F_R21_B1_BENCH_pass2.json")

#: stage → 元数据。`redirectable=False` 表示该 stage 的输出**无法重定向**，
#: 执行它必然覆盖历史工件 → 本入口拒绝执行。
STAGES: dict[str, dict] = {
    "v2.b1": {
        "module": "poker_decision_ai.multistreet_v2_benchmark",
        "func": "run_b1",
        "reads": V2_READS,
        "out_kwarg": "snapshot_path",
        "redirectable": True,
        "in_path_guarded": False,
    },
    "v2.b2": {
        "module": "poker_decision_ai.multistreet_v2_benchmark",
        "func": "run_b2",
        "reads": V2_READS,
        "out_kwarg": None,
        "redirectable": False,
        "in_path_guarded": False,
        "refuse_reason": ("run_b2() 的输出路径不可重定向（硬编码写 "
                          "B2_SNAPSHOT=docs/V3F_R21_B2_CROSS_CORPUS.json）；"
                          "执行会覆盖已登记历史工件，故拒绝。"),
    },
    "v3.b1": {
        "module": "poker_decision_ai.multistreet_v3_benchmark",
        "func": "run_b1",
        "reads": V3_READS,
        "out_kwarg": "snapshot",
        "redirectable": True,
        "in_path_guarded": False,
    },
    "v3.b2": {
        "module": "poker_decision_ai.multistreet_v3_benchmark",
        "func": "run_b2",
        "reads": V3_READS,
        "out_kwarg": "snapshot",
        "redirectable": True,
        "in_path_guarded": False,
    },
    "v3.b3": {
        "module": "poker_decision_ai.multistreet_v3_benchmark",
        "func": "run_b3",
        "reads": (),                      # b3 只做再分层，不读历史快照
        "out_kwarg": "snapshot",
        "redirectable": True,
        "in_path_guarded": False,
    },
}


def list_stages() -> list:
    return sorted(STAGES)


def get_stage(stage: str) -> dict:
    if stage not in STAGES:
        raise KeyError(f"未知 stage: {stage!r}；可用: {list_stages()}")
    return STAGES[stage]


def required_artifacts(stage: str) -> tuple:
    """该 stage 直接读取的历史工件（= 需要读取前校验的对象）。"""
    return tuple(get_stage(stage)["reads"])


def validate_before_read(stage: str, *, manifest_path=None) -> dict:
    """在任何读取/执行之前，对 stage 所需的全部历史工件做真实校验。

    通过 → 返回可审计记录；任一不通过 → `ArtifactGuardError`（fail closed）。
    本函数**不读取工件内容**，因此不改变任何读取语义。
    """
    meta = get_stage(stage)
    guard = _guard(manifest_path)
    registered = _registered(manifest_path)
    checked = []
    for rel in meta["reads"]:
        guard.require(rel)                      # 真实 SHA-256 校验；失败即抛
        checked.append(rel)
    return {
        "stage": stage,
        "format_declared": "pre-flight（非 in-path guard）",
        "in_path_guarded": bool(meta["in_path_guarded"]),
        "checked": checked,
        "checked_count": len(checked),
        "registered_count": len(registered),
        "claim": ("读取前已真实校验；未承诺 sealed 模块内部 _read() 亦被拦截"),
    }


def run_stage(stage: str, *, out_path, manifest_path=None, **kwargs) -> dict:
    """校验后委派给 sealed stage 函数；只做校验与写入保护，不改语义。

    保证（有测试断言）：
      1. 未通过校验 → 抛 `ArtifactGuardError`，**stage 函数根本不被调用**；
      2. 输出路径不可重定向、或指向已登记历史工件 → `SealedWriteRefusedError`，
         **stage 函数根本不被调用**（这是保护历史证据的第一道闸）；
      3. 通过后按原签名委派，返回值原样透传（无包装、无改写）。
    """
    meta = get_stage(stage)

    # ── 闸 1：证据写入保护（先于校验，因为"不可重定向"是硬拒绝）─────────
    if not meta["redirectable"]:
        raise SealedWriteRefusedError(
            f"{stage}: {meta['refuse_reason']}")
    if out_path is None:
        raise SealedWriteRefusedError(
            f"{stage}: 必须显式指定 out_path（绝不默认写回 docs/ 历史工件）")
    out = Path(out_path)
    if out in {ROOT / rel for rel in _registered(manifest_path)}:
        raise SealedWriteRefusedError(
            f"{stage}: 输出路径指向已登记历史工件（拒绝覆盖）: {out}")

    # ── 闸 2：读取前真实校验 ────────────────────────────────────────────
    evidence = validate_before_read(stage, manifest_path=manifest_path)

    # ── 委派：原函数、原签名、原返回值 ──────────────────────────────────
    import importlib
    mod = importlib.import_module(meta["module"])
    func = getattr(mod, meta["func"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result = func(**{meta["out_kwarg"]: out}, **kwargs)
    return {"stage": stage, "out_path": str(out),
            "preflight": evidence, "result": result}
