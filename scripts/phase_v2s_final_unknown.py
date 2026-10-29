"""FINAL AUDIT — UNKNOWN propagation 专项检查。

1) Pot UNKNOWN 语义（不转 0/前值/默认）；
2) Stack UNKNOWN（不可见 ROI → 0 值？检查 slot 语义）；
3) Bet UNKNOWN；
4) Status NOT_OBSERVED vs UNKNOWN 分离；
5) HeroCards 无牌 → UNKNOWN（已验）；
6) 无互相推导 grep（stack→pot 等）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PotObserver, StatusObserver, StackObserver)
from poker_decision_ai.vision2.pixel.realclient.bet_observer import (  # noqa: E402
    BetObserver)

EV = Path("data/vision2/coinpoker_evidence")

# 空帧 → 各字段 UNKNOWN（不转默认）
w, h, rgba = decode_png((EV / "BUTTON_CHANGED_167_1787992682749_full.png")
                        .read_bytes())
frames = {
    "f01_1876": (rgba, w, h),
}
# blank 1876
fb = np.zeros((1400, 1876, 4), dtype=np.uint8)
fb[..., :3] = (6, 35, 27)
fb[..., 3] = 255
frames["blank_1876"] = (fb.tobytes(), 1876, 1400)

for tag, (rg, ww, hh) in frames.items():
    if isinstance(rg, bytes):
        pot = PotObserver().observe(rg, ww, hh)
    else:
        pot = PotObserver().observe(rg, ww, hh)
    st = StatusObserver().observe(rg, ww, hh)
    sk = StackObserver().observe(rg, ww, hh)
    bt = BetObserver().observe(rg, ww, hh)
    print(f"[{tag}]")
    print(f"  Pot: {pot['status']} value={pot.get('value')} "
          f"reason={pot.get('reason')}")
    print(f"  Status: {st['status']} reason={st.get('reason')}")
    skv = sk.get("seats") or {}
    nonocc = [s for s, v in skv.items()
              if isinstance(v, dict) and v.get("value") is not None]
    print(f"  Stack: keys={list(sk.keys())} non-None值座数={len(nonocc)}")
    print(f"  Stack seats: {[(s, (v.get('value') if isinstance(v, dict) else v)) for s, v in skv.items()]}")
    print(f"  Bet: {bt.get('status')} value={bt.get('value')}")
    print(f"  Pot value==0? {pot.get('value') == 0} "
          f"(UNKNOWN 不得=0)")

# 跨字段推导 grep：V3F 生产代码中 stack→pot / pot→bet 等互引
import re
pkg = Path("poker_decision_ai/vision2/pixel/realclient")
cross = []
for p in sorted(pkg.glob("*.py")):
    src = p.read_text(encoding="utf-8")
    for m in re.finditer(r"(pot_observer|stack_observer|bet_observer|"
                         r"recognize_pot|recognize_stack|recognize_bet)"
                         r"[^)]*", src):
        cross.append(f"{p.name}: {m.group(0)[:60]}")
print(f"\n跨字段互引扫描: {len(cross)} hits")
for c in cross[:5]:
    print("  ", c)
