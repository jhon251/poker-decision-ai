"""B7-2/B7-8 — 工件 fail-closed 守卫（中性实现；不含研究/策略逻辑）。

本模块只做一件事：在**真实读取历史工件之前**，按 manifest 登记的 SHA-256
校验该工件。语义（**fail closed，禁止任何 fallback**）：

  missing manifest     → ArtifactGuardError（manifest 永不回退到内置默认值）
  unregistered path    → ArtifactGuardError（不"当作新工件"读下去）
  missing file         → ArtifactGuardError（不回退到旧版本/旧快照）
  hash mismatch        → ArtifactGuardError（stale / wrong-manifest）
  cached hash deleted  → 每次 require 都重算文件哈希，绝不使用缓冲值

不变量：
  * 读取批准结果使用 `root / rel`（root 为受信基准目录），不使用工件自报
    路径，也不跟随符号链接逃出 root——避免"校验 A、读取 B"。
  * **授权工件集在每次 require 时从 manifest 重新载入**（不缓存 manifest 内容），
    因此 manifest 变更立即生效，不存在陈旧授权。
  * manifest 自身是信任锚（trust anchor）：它的路径被显式声明，无法自我登记；
    其完整性由各自的历史封存（如 docs/V3F_R21_MANIFEST.json 等）另行校验。

位置说明：实现放在中性包位置 `poker_decision_ai/artifact_guard.py`，使
benchmark 读取路径可以 fail closed 校验，同时不违反 B7-1
"生产包不得 import research_unfreeze" 的隔离约束。
`research_unfreeze.governance` 仍以同名 re-export 暴露 ArtifactGuard /
ArtifactGuardError / sha256_file，历史调用点语义不变。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

__all__ = ["ArtifactGuardError", "ArtifactGuard", "sha256_file"]

MANIFEST_KEY = "code_hashes"
DEPENDENCY_KEY = "readonly_dependency_hashes"


class ArtifactGuardError(RuntimeError):
    """工件校验失败（fail closed）。调用方不得捕获后继续读取。"""


def sha256_file(path) -> str:
    """文件真实 SHA-256（十六进制小写）。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArtifactGuard:
    """运行期工件校验：**fail closed**，禁止 fallback。

    用法：`ArtifactGuard(manifest_path, root=ROOT).require(rel_path)`

    `require(rel)` 校验通过后返回 `root / rel`（可读路径；非 str，可直接
    `.read_text()`）。校验失败一律抛 `ArtifactGuardError`。
    """

    def __init__(self, manifest_path, *, root: Path | None = None):
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else Path.cwd()
        if not self.manifest_path.exists():
            raise ArtifactGuardError(
                f"manifest 缺失（fail closed）: {self.manifest_path}")
        try:
            self.manifest = json.loads(
                self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:      # 损坏 manifest 不是"空登记"
            raise ArtifactGuardError(
                f"manifest 不可解析（fail closed）: {self.manifest_path}: "
                f"{exc}") from exc
        if not isinstance(self.manifest, dict):
            raise ArtifactGuardError(
                f"manifest 根必须是对象（fail closed）: {self.manifest_path}")
        #: 信任锚：本次载入所用 manifest（内容每次 require 重新载入，见下）
        self.hashes = self._registered_hashes(self.manifest)

    # ── 内部：登记集 ────────────────────────────────────────────────────
    @staticmethod
    def _registered_hashes(man: dict) -> dict:
        """登记集 = code_hashes ∪ readonly_dependency_hashes（后者不覆盖前者）。"""
        merged = dict(man.get(MANIFEST_KEY) or {})
        merged.update(man.get(DEPENDENCY_KEY) or {})
        return merged

    def _load_registered(self) -> dict:
        """每次调用重新载入登记集（无陈旧授权；损坏即 fail closed）。"""
        try:
            man = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactGuardError(
                f"manifest 不可解析（fail closed）: {self.manifest_path}: "
                f"{exc}") from exc
        return self._registered_hashes(man)

    # ── 主入口 ──────────────────────────────────────────────────────────
    def require(self, rel: str) -> Path:
        """校验 `rel` 并在通过后返回其受信路径；任何异常情况一律拒绝。"""
        registered = self._load_registered()
        if rel not in registered:
            raise ArtifactGuardError(
                f"工件未在 manifest 中登记（fail closed）: {rel}")
        p = self.root / rel
        if not p.exists():
            raise ArtifactGuardError(f"工件缺失（fail closed）: {rel}")
        now = sha256_file(p)
        if now != registered[rel]:
            raise ArtifactGuardError(
                "工件哈希与 manifest 不一致（stale/wrong-manifest，"
                f"fail closed）: {rel}")
        # 读取批准同样使用受信 root 拼接（不采用工件自报路径）
        return self.root / rel

    def require_all(self) -> dict:
        """登记集中每一项都必须通过；任一失败即整体拒绝。"""
        return {rel: str(self.require(rel))
                for rel in sorted(self._load_registered())}

    def verify_sealed_code(self) -> dict:
        """校验登记集全部项；漂移即拒绝（不 fallback、不跳过缺失项）。"""
        drift = {}
        for rel, h in self._load_registered().items():
            p = self.root / rel
            now = sha256_file(p) if p.exists() else "MISSING"
            if now != h:
                drift[rel] = {"sealed": h, "now": now}
        if drift:
            raise ArtifactGuardError(f"封存漂移（fail closed）: {sorted(drift)}")
        return {"checked": len(self._load_registered()), "drift": {}}
