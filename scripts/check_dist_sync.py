#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30-9-FM：**发布包一致性守卫**（源 ↔ dist 漂移检测）。

背景：第 37 轮发现 `dist/configs/config.yaml` **缺少** `solver_timeout_ms: 1200`
（第 6/7 轮加的实时求解放宽）⇒ 成品 EXE 会退回默认超时、实时求解器可能降级 ✗。
本脚本对比 configs/ 与 data/solver/ 的源与 dist 副本，缺失/不一致即非零退出。
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = pathlib.Path(__file__).resolve().parents[1]
# 运行期**真正读取**的路径（见 configs/config.yaml 与 provider 默认值）；
# 构建期目录（*_backup_*、spots_r3*/spots_fb2..5 等实验输出）不参与发布一致性。
KINDS = ("configs",)
RUNTIME_DATA = ("spots_r1", "spots_fb", "spots_6max")
# P30-9-IC：**运行时可变**的配置 ⇒ 排除出同步检查 ✓
#   实测：`configs/gui_blinds.json` 是 GUI 的**盲注输入偏好** ✓，程序运行时会改写它 ✓
#   ⇒ 每次检查都报"漂移" ✗（且会级联让审计一致性 FAIL ✗）。
#   这些文件**本就该由运行实例各自持有** ✓，不属于"发布包与源一致"的范畴 ✓。
RUNTIME_MUTABLE = ("gui_blinds.json",)


def _hash(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _compare(src: pathlib.Path, dst: pathlib.Path, label: str,
             problems: list[str]) -> None:
    skipped: list[str] = []
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        if f.name in RUNTIME_MUTABLE:
            skipped.append(str(rel))
            continue                                          # 运行时可变 ⇒ 不参与 ✓
        d = dst / rel
        if not d.exists():
            problems.append(f"缺失: dist/{label}/{rel}")
        elif _hash(f) != _hash(d):
            problems.append(f"不一致: {label}/{rel}")
    if skipped:
        print(f"  （已跳过运行时可变配置 {len(skipped)} 个: {', '.join(sorted(skipped))} ✓）")


def main() -> int:
    problems: list[str] = []
    for kind in KINDS:
        src, dst = ROOT / kind, ROOT / "dist" / kind
        if src.exists():
            _compare(src, dst, kind, problems)
    for lib in RUNTIME_DATA:
        src, dst = ROOT / "data" / "solver" / lib, ROOT / "dist" / "data" / "solver" / lib
        if src.exists():
            _compare(src, dst, f"data/solver/{lib}", problems)
    # EXE 存在性
    for exe in ("poker-ai-gui-preview.exe", "poker-ai-preview.exe"):
        p = ROOT / "dist" / exe
        if not p.exists():
            problems.append(f"缺失: dist/{exe}")
    out = ROOT / "docs" / "dist_sync.json"
    out.write_text(json.dumps({"problems": problems, "checked": list(KINDS) + list(RUNTIME_DATA)},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    if problems:
        print("[FAIL] 发布包漂移：")
        for p in problems[:20]:
            print("  -", p)
        print(f"（共 {len(problems)} 项；详见 docs/dist_sync.json）")
        return 1
    print("[PASS] 发布包与源一致（configs + data/solver 全部同步）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
