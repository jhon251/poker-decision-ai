"""EXE-1 产品构建启动器 — CLI（poker-ai.exe）。

与 Preview 的差别：完整子命令（analyze/play/replay/...）+ 决策引擎全量
（solver 锚定 spot 库 / 频率先验便携索引 / 池 bot 对打）。

职责（与 preview_launch 同纪律）：
  1. frozen 模式下 chdir 到 EXE 目录——configs/ data/ tools/ logs/ 相对路径
     全部解析到 EXE 旁（便携布局，用户可编辑）；
  2. 清掉可能污染冻结环境的 PYTHONPATH/PYTHONHOME；
  3. stderr 打印诚实声明（stdout 保持纯 JSON——脚本可管道）；
  4. 转调 poker_decision_ai.cli:main（零逻辑复制）。

红线：不自动下注/点击/键鼠控制；--screenshot 只做**只读** BitBlt 像素
读取（无任何自动操作）；provider_status 如实（mock 标 mock）；spot 库
缺失时 solver 走诚实降级路径。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _banner() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    print(
        "[poker-ai 产品版 CLI]\n"
        "  - 形态：离线分析 + 本地对打（非实时辅助；无屏幕捕捉/自动操作）。\n"
        "  - 能力诚实：LLM/Vision 未配置 REAL 时报告 mock/unavailable；\n"
        "    Solver = postflop-bench-cli（EXE 旁 tools/bin/，真实 CFR）+\n"
        "    spot 查表库（EXE 旁 data/solver/）；缺失时诚实降级并标注。\n"
        "  - 对打：poker-ai play --stake nl2 --hands 3 [--seed N]\n"
        "  - configs/ data/ logs/ 在 EXE 旁，可直接查看/编辑。",
        file=sys.stderr,
    )


def main() -> int:
    if getattr(sys, "frozen", False):
        os.chdir(_app_dir())
        for var in ("PYTHONPATH", "PYTHONHOME"):
            os.environ.pop(var, None)
    _banner()
    from poker_decision_ai.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
