# -*- coding: utf-8 -*-
"""P30-9-AW 回归：求解器子进程**必须被回收**（超时/异常路径），且不得留孤儿。

背景（用户任务管理器实证）：7 个 `postflop-bench-cli.exe` 孤儿 / ~4.5GB，
根因是 `solve()` 超时只抛异常、从不 kill 子进程。
"""
from __future__ import annotations

import asyncio
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pytest                                                            # noqa: E402


@pytest.mark.timeout if False else (lambda f: f)                          # noqa: E731
def _noop(f):                                                            # pragma: no cover
    return f


def test_solver_timeout_kills_child_process():
    """真实起一个长sleep进程 ⇒ 1 秒超时 ⇒ **必须被 kill 且 wait 回收**。"""
    from poker_decision_ai.core.providers import solver as S

    async def run():
        orig_exec = asyncio.create_subprocess_exec
        spawned: list = []

        async def fake_exec(*a, **k):
            # 起一个真实的长驻进程（与求解器同构：stdin/stdout 管道）
            p = await orig_exec(
                "powershell", "-NoProfile", "-Command", "Start-Sleep -Seconds 120",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE)
            spawned.append(p)
            return p

        # is_configured() 需要真实存在的路径（实际不会执行，被 fake_exec 拦截）
        real = r"<repo>\tools\bin\postflop-bench-cli.exe"
        if not pathlib.Path(real).exists():
            pytest.skip("求解器 CLI 不存在，跳过")
        prov = S.PostflopSolverCLIProvider(real)
        prov.build_cli_query = lambda q: {"stub": True}          # 绕过入参校验
        asyncio.create_subprocess_exec = fake_exec
        try:
            with pytest.raises(Exception):
                await prov.solve({"timeout_s": 1})
        finally:
            asyncio.create_subprocess_exec = orig_exec
        assert spawned, "未成功起子进程（测试无效）"
        p = spawned[0]
        try:
            await asyncio.wait_for(p.wait(), timeout=5)
        except asyncio.TimeoutError:                             # pragma: no cover
            pass
        return p.returncode

    rc = asyncio.run(run())
    assert rc is not None, "子进程未被回收（仍是孤儿）"


def test_default_solver_timeout_is_interactive():
    """默认超时必须远小于原 600s（交互工具）。"""
    from poker_decision_ai.core.providers import solver as S
    assert S.DEFAULT_SOLVE_TIMEOUT_S <= 60.0
    assert S.DEFAULT_SOLVE_TIMEOUT_S > 0


def test_solver_leak_fix_present_in_source():
    src = (ROOT / "poker_decision_ai/core/providers/solver.py").read_text(
        encoding="utf-8")
    assert "proc.kill()" in src and "await proc.wait()" in src
    assert "finally:" in src
    assert "DEFAULT_SOLVE_TIMEOUT_S" in src
