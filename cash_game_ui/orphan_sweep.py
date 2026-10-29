# -*- coding: utf-8 -*-
"""P30-9-AX：启动清扫——回收**父进程已死**的求解器孤儿进程。

场景（用户实证）：GUI/脚本被强杀时，正在求解的 `postflop-bench-cli.exe`
会成为孤儿并永久占用内存（实测 7 个 / ~4.5GB）。
本模块在 GUI 启动时做一次只读扫描 + **仅杀父进程已死的**求解器进程。
"""
from __future__ import annotations

import os
import subprocess
import sys


def _run_ps(script: str) -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                              capture_output=True, text=True, timeout=30)
        return (out.stdout or "").strip()
    except Exception:                                          # noqa: BLE001
        return ""


def sweep_orphan_solvers(*, dry: bool = False,
                          name: str = "postflop-bench-cli") -> dict:
    """回收父进程已退出的求解器进程（只动这一种名字的进程，绝不碰其它）。"""
    if sys.platform != "win32":
        return {"supported": False, "killed": 0}
    script = (
        f"$procs = Get-CimInstance Win32_Process -Filter \"Name LIKE '{name}%'\" "
        f"-ErrorAction SilentlyContinue; "
        "foreach ($p in $procs) { "
        "  $par = Get-Process -Id $p.ParentProcessId -ErrorAction SilentlyContinue; "
        "  if (-not $par) { "
        f"    if ('{int(dry)}' -eq '0') {{ Stop-Process -Id $p.ProcessId -Force "
        "      -ErrorAction SilentlyContinue }; "
        "    Write-Output \"$($p.ProcessId)\" } }")
    dead = [ln.strip() for ln in _run_ps(script).splitlines() if ln.strip()]
    return {"supported": True, "dry": dry, "orphans": dead,
            "killed": 0 if dry else len(dead)}


def main() -> int:
    r = sweep_orphan_solvers(dry="--dry" in sys.argv)
    print(f"孤儿求解器: {r.get('orphans') or '无'} | 已回收: {r.get('killed')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
