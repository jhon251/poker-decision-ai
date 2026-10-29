"""FINAL AUDIT — hardcode 扫描（hero/dealer/table_size/frame-id 特判）。"""
from __future__ import annotations
import re
from pathlib import Path

pkg = Path("poker_decision_ai/vision2/pixel/realclient")
patterns = {
    "S6-hero-fallback": r"hero_seat\s*=\s*[\"']S6[\"']|return\s*\{[\"']hero_seat[\"']\s*:\s*[\"']S6[\"']",
    "fixed-dealer": r"dealer_seat\s*=\s*[\"']S\d[\"']|dealer_seat\"\s*:\s*[\"']S\d[\"']",
    "table_size6": r"table_size\s*=\s*6|[\"']table_size[\"']\s*:\s*6\s*\}|max_players\s*=\s*6",
    "frame-id-specialcase": r"1787\d{10}|BUTTON_CHANGED_\d+|POT_CHANGED_\d+|SCENE_CHANGED_\d+",
    "previous/cache": r"previous|prev_result|self\.cache|frame_buffer|history",
}
for name, pat in patterns.items():
    hits = []
    for p in sorted(pkg.glob("*.py")):
        src = p.read_text(encoding="utf-8")
        for m in re.finditer(pat, src):
            line = src[:m.start()].count("\n") + 1
            hits.append(f"{p.name}:{line}: {m.group(0)[:70]}")
    print(f"=== {name}: {len(hits)} hits ===")
    for h_ in hits[:6]:
        print("  ", h_)
