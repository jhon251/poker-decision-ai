"""EXE Preview Build 启动器（仅打包用——不属于运行时核心）。

职责（Preview ≠ Final Product）：
  1. frozen 模式下把 CWD 切到 EXE 所在目录——configs/ logs/ data/
     等相对路径全部解析到 EXE 旁边（外置、用户可改）；
  2. 清掉可能污染冻结运行环境的 PYTHONPATH/PYTHONHOME；
  3. stderr 打印 Preview 诚实声明（stdout 保持纯 JSON——脚本可解析）；
  4. 转调既有 CLI 入口 poker_decision_ai.cli:main（零逻辑复制）。

红线：不自动下注/点击；不新增 exploit strategy；不伪造 REAL 能力
（solver/LLM/vision 未配置时按 ProviderStatus 诚实报告 mock/unavailable）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _app_dir() -> Path:
    """EXE 目录（frozen）或仓库根（源码运行）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _preview_banner() -> None:
    """stderr 诚实声明（不污染 stdout JSON）。"""
    print(
        "[poker-ai EXE Preview Build]\n"
        "  - Preview ≠ Final Product；仅供核心 decision pipeline 演示。\n"
        "  - 不自动下注/点击：本程序只输出建议（Decision Envelope JSON）。\n"
        "  - 能力诚实：LLM/Vision/Solver 未配置 REAL 时报告 mock/unavailable，"
        "绝不伪造 REAL（REAL Solver 未随 Preview 分发）。\n"
        "  - configs/ 与 data/、logs/ 位于本 EXE 旁边，可直接编辑。",
        file=sys.stderr,
    )


def main() -> int:
    if getattr(sys, "frozen", False):
        app_dir = _app_dir()
        os.chdir(app_dir)                     # 相对路径 → EXE 目录
        for var in ("PYTHONPATH", "PYTHONHOME"):
            os.environ.pop(var, None)         # 干净环境守卫
    _preview_banner()

    from poker_decision_ai.cli import main as cli_main

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
