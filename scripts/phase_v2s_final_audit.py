"""FINAL AUDIT — 1876 runtime 端到端组合测试（只读检查，不改生产）。

用真实帧（f03 POT_CHANGED_102 — hero 亮牌 9c3d + D 钮 + folded/allin status）
验证 1876 全字段接线：table_topology / occupied / dealer / hero /
hero_position / pot / bet / stack / status / hero_cards。
仅调用 observer，检查输出形状与 UNKNOWN 语义。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.seat_position import (  # noqa: E402
    PositionObserver, PotObserver, StatusObserver, StackObserver,
    HeroCardsObserver)
from poker_decision_ai.vision2.pixel.realclient.bet_observer import (  # noqa: E402
    BetObserver)

EV = Path("data/vision2/coinpoker_evidence")


def load(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    return rgba, w, h


rgba, w, h = load("POT_CHANGED_102_1787992505794_full.png")
print(f"frame {w}x{h} (1876? {w==1876 and h==1400})")

pos = PositionObserver().observe(rgba, w, h)
rr = pos["runtime_result"]
print(f"\n[Position] hero_seat={rr.get('hero_seat')} "
      f"dealer_seat={rr.get('dealer_seat')} "
      f"uncertainty={rr.get('uncertainty')}")
occ = rr.get("occupied_seats", {})
print(f"  occupied: {[(s, v.get('status') if isinstance(v, dict) else v) for s, v in occ.items()]}")
print(f"  hero_position={pos.get('hero_position')} "
      f"positions_ring={'SET' if pos.get('positions_ring') else None}")

pot = PotObserver().observe(rgba, w, h)
print(f"\n[Pot] status={pot['status']} value={pot.get('value')}")

bet = BetObserver().observe(rgba, w, h)
print(f"[Bet] status={bet.get('status')} value={bet.get('value')}")

st = StackObserver().observe(rgba, w, h)
print(f"[Stack] status={st.get('status')} seats={list((st.get('seats') or {}).keys()) if st.get('seats') else None}")

bs = StatusObserver().observe(rgba, w, h)
print(f"[Status] status={bs['status']} seats={sorted((bs.get('seats') or {}).keys())}")

hc = HeroCardsObserver().observe(rgba, w, h)
print(f"[HeroCards] status={hc['status']} value={hc.get('value')}")

# UNKNOWN 语义：folded 帧 f01（无 hero 亮牌 / 无 pot?）
rgba2, w2, h2 = load("BUTTON_CHANGED_167_1787992682749_full.png")
hc2 = HeroCardsObserver().observe(rgba2, w2, h2)
print(f"\n[f01 folded] HeroCards status={hc2['status']} value={hc2.get('value')} "
      f"reason={hc2.get('reason')}")
