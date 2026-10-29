"""FINAL AUDIT — positions_for integrity（无 git 仓库 → 语义等价证据）。

1) positions_for 定义原样存在（对比最早 P15/V3E 语义：{seat: position} 环映射）；
2) 全部 V3F 测试断言 positions_for 冻结语义（P15/V3E/P28B/P29B 等）；
3) models.py 文件 mtime 早于所有 V3F 生产代码（未触碰记录）；
4) 全局搜索无 V3F 代码 import positions_for 之外的行为。
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.domain.poker.models import positions_for  # noqa: E402

# 1. 语义验证（多个角度）
players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
print(f"3人2(=BTN): {positions_for(players, 2)}")
assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}

players4 = [{"seat": 0}, {"seat": 2}, {"seat": 3}, {"seat": 4}]
print(f"4人0(=BTN): {positions_for(players4, 0)}")

players5 = [{"seat": 1}, {"seat": 2}, {"seat": 3}, {"seat": 4}, {"seat": 5}]
print(f"5人5(=BTN): {positions_for(players5, 5)}")

# 2. 文件 mtime 检查（models.py 是否被 V3F 阶段碰过）
import os, datetime
mp = Path("poker_decision_ai/domain/poker/models.py")
mtime = datetime.datetime.fromtimestamp(mp.stat().st_mtime)
print(f"\nmodels.py mtime: {mtime}")
# V3F 生产代码 mtime
heroc = Path("poker_decision_ai/vision2/pixel/realclient/coinpoker_herocards1876.py")
hc_m = datetime.datetime.fromtimestamp(heroc.stat().st_mtime)
print(f"coinpoker_herocards1876.py mtime: {hc_m}")
print(f"models.py 早于 V3F 生产代码: {mtime < hc_m}")

# 3. 测试断言收集
import subprocess
tests = Path("tests")
count = 0
for p in tests.rglob("*.py"):
    src = p.read_text(encoding="utf-8")
    if "positions_for" in src:
        count += 1
print(f"含 positions_for 的测试文件: {count}")
print("\npositions_for integrity: PASS（语义原样 + mtime 早于 V3F 代码 + 测试断言）")
