"""EXE-1 — 组装产品分发布局（PyInstaller onedir 产出 + 便携外置数据）。

用法：.venv/Scripts/python.exe scripts/assemble_product.py [--out DIR]
缺省输出 dist/PokerAI-Product/：
    poker-ai.exe / poker-ai-gui.exe / _internal/     ← PyInstaller COLLECT
    configs/           ← 配置（用户可编辑）
    data/solver/       ← spot 查表库（spots_r1 + spots_fb；不含 r1backup）
    data/cache/frequency_prior_portable.json   ← 频率先验便携索引
    tools/bin/postflop-bench-cli.exe           ← 真实求解器 CLI
    README_PRODUCT.txt ← 布局/红线说明
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "dist" / "poker-ai-product"

README = """poker-ai 产品版（EXE-1/EXE-2）—— 离线分析 + 本地对打 + 一键识别
================================================================

形态声明（诚实边界）：
  * 无自动操作：本软件不含自动点击/下注/键盘鼠标控制路径；"一键识别"
    （analyze --screenshot / GUI 识别按钮）只做**只读** BitBlt 屏幕像素
    读取（CoinPoker 窗口）用于视觉识别，绝不代打。
  * 能力如实：LLM/Vision 未配置 REAL 时按 mock/unavailable 报告；
    Solver 为真实 postflop CFR（tools/bin/postflop-bench-cli.exe）+
    spot 查表库（data/solver/）；任一缺失时 solver 路径诚实降级并标注
    （provenance: solver_miss_fallback），绝不伪造。
  * 识别三态：OK → 建议；字段缺失 → 识别不到（原因+缺失清单）；
    部分识别 → 已识别+缺失清单（不编造）。

布局（全部相对本目录）：
  poker-ai.exe        命令行：analyze / play / replay / agents / ...
  poker-ai-gui.exe    对打界面（图形界面）
  configs/            配置文件（可编辑）
  data/solver/        solver spot 查表库
  data/cache/         frequency_prior_portable.json（对手频率先验）
  tools/bin/          postflop-bench-cli.exe
  data/play_history/  对打训练数据（JSONL，自动生成）
  logs/               运行日志

上手（图形界面请看 使用说明.txt；升级见 升级说明.txt / 升级.bat）：
  poker-ai.exe play --stake nl2 --hands 3        # 终端对打 3 手（带 AI 参考）
  poker-ai.exe play --seed 12345 --hands 10      # 确定性种子（教学复盘）
  poker-ai-gui.exe                               # 图形界面对打
  poker-ai.exe analyze --text "<手牌历史文本>"     # 离线分析一手

对打时你的每次决策与 AI 参考建议（含决策 provenance）都会落盘到
data/play_history/*.jsonl——"越打越懂你"的训练数据底座。
"""


def _copytree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"FATAL: missing {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist" / "PokerAI-Product"))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if not BUILD_DIR.exists():
        print(f"FATAL: {BUILD_DIR} 不存在——先跑 pyinstaller --clean "
              f"--noconfirm packaging/poker_ai_product.spec", file=sys.stderr)
        return 1
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # 1) PyInstaller onedir 产出（两个 EXE + _internal）
    for item in BUILD_DIR.iterdir():
        dst = out / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    # 2) 外置便携数据
    _copytree(ROOT / "configs", out / "configs")
    (out / "data" / "cache").mkdir(parents=True, exist_ok=True)
    # R13-EXE-2：一键识别链视觉模板资产（只读模板；不含任何帧语料）
    import re
    vis_dst = out / "data" / "vision2"
    vis_dst.mkdir(parents=True, exist_ok=True)
    asset_refs = set()
    for py in (ROOT / "poker_decision_ai").rglob("*.py"):
        asset_refs.update(re.findall(
            r'data/vision2/coinpoker_assets_[A-Za-z0-9_]+',
            py.read_text(encoding="utf-8", errors="ignore")))
    asset_refs.add("data/vision2/coinpoker_assets_v3b_header")
    staged = 0
    for d in sorted(asset_refs):
        src = ROOT / d
        if src.is_dir():
            shutil.copytree(src, vis_dst / src.name,
                            ignore=shutil.ignore_patterns("__pycache__"))
            staged += 1
    for jf in ("hero_card_layout_v1.json", "stack_text_layout_v1.json"):
        if (ROOT / "data" / "vision2" / jf).exists():
            shutil.copy2(ROOT / "data" / "vision2" / jf, vis_dst / jf)
    print(f"vision assets staged: {staged} dirs")
    solver_dst = out / "data" / "solver"
    if solver_dst.exists():
        shutil.rmtree(solver_dst)
    solver_dst.mkdir(parents=True)
    for sub in ("spots_r1", "spots_fb"):
        _copytree(ROOT / "data" / "solver" / sub, solver_dst / sub)
    portable = ROOT / "data" / "cache" / "frequency_prior_portable.json"
    if not portable.exists():
        print("FATAL: frequency_prior_portable.json 缺失——先跑 "
              "scripts/build_portable_freq_index.py", file=sys.stderr)
        return 1
    shutil.copy2(portable, out / "data" / "cache" / portable.name)
    tools_dst = out / "tools" / "bin"
    tools_dst.mkdir(parents=True, exist_ok=True)
    solver_bin = ROOT / "tools" / "bin" / "postflop-bench-cli.exe"
    if not solver_bin.exists():
        print(f"FATAL: {solver_bin} 缺失", file=sys.stderr)
        return 1
    shutil.copy2(solver_bin, tools_dst / solver_bin.name)
    (out / "README_PRODUCT.txt").write_text(README, encoding="utf-8")
    # 玩家文档 + 升级脚本
    for fname in ("使用说明.txt", "升级说明.txt", "升级.bat"):
        src = ROOT / "packaging" / fname
        if src.exists():
            shutil.copy2(src, out / fname)
        else:
            print(f"WARN: {src} 缺失", file=sys.stderr)
    n = sum(1 for _ in out.rglob("*"))
    print(f"assembled: {out}  ({n} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
