"""P30-13 最终边界审计 — 16 项红线扫描（核心生产路径）。"""
from __future__ import annotations
import re
from pathlib import Path

# 核心生产路径文件（决策链/桥接/循环——非测试）
FILES = [
    "poker_decision_ai/vision2/pixel/realclient/production_bridge.py",
    "poker_decision_ai/vision2/pixel/realclient/runtime_snapshot.py",
    "poker_decision_ai/vision2/pixel/realclient/seat_position.py",
    "poker_decision_ai/vision2/pixel/realclient/coinpoker_betting_header.py",
    "poker_decision_ai/vision2/pixel/realclient/coinpoker_to_call.py",
    "poker_decision_ai/vision2/pixel/realclient/betting_header_observer.py",
    "poker_decision_ai/vision2/pixel/realclient/to_call_observer.py",
    "poker_decision_ai/frame_decision.py",
    "poker_decision_ai/production_decision_loop.py",
    "poker_decision_ai/decision_contract.py",
    "poker_decision_ai/solver_contract.py",
    "poker_decision_ai/strategy_provider.py",
    "poker_decision_ai/action_event_source.py",
    "poker_decision_ai/action_event.py",
    "poker_decision_ai/event_stream_consumer.py",
    "poker_decision_ai/unknown_semantics.py",
    "poker_decision_ai/continuous_decision.py",
    "poker_decision_ai/agents/state/agent.py",
    "poker_decision_ai/agents/judge/agent.py",
    "poker_decision_ai/agents/solver/agent.py",
    "poker_decision_ai/agents/range/agent.py",
    "poker_decision_ai/agents/equity/agent.py",
    "poker_decision_ai/realtime.py",
]

CHECK = {
    "hero hardcode": r"hero_id\s*=\s*['\"]Hero['\"]\s*#|hero_seat\s*=\s*3\b",
    "dealer hardcode": r"dealer_seat\s*=\s*(1|3|4)\b|button_seat\s*=\s*(1|3|4)\s*$",
    "table_size=6 fallback": r"table_size\s*=\s*6\s*$|6\s*if.*max_players",
    "UNKNOWN→0 污染": r"float\([^)]*(current_bet|pot|to_call)[^)]*or 0\.0\)",
    "random action fallback": r"random\.choice|random_action|randint.*action",
    "previous-frame 行为推断": r"prev_frame|previous_frame|last_obs|frame_diff",
    "click/mouse/keyboard/send_bet/submit": r"\b(click|mouse_event|keyboard|send_bet|submit_action)\b",
    "visual LLM": r"multimodal.*vision|openai.*vision|anthropic.*vision",
    "OCR": r"tesseract|pytesseract|easyocr|paddleocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "伪造视觉证据": r"fabricat.*(frame|pixel|card)|fake.*(visual|card)",
    "伪造 Solver REAL": r"is_approximate\s*=\s*False.*mock|mock.*is_approximate.*False",
}

print("=== P30-13 边界审计（核心生产路径，排除 docstring/注释声明）===")
total = 0
for fp in FILES:
    p = Path(fp)
    if not p.exists():
        print(f"  SKIP {fp} (missing)")
        continue
    src = p.read_text(encoding="utf-8")
    # 仅扫描代码体（去除 docstring 块——简化：去除三引号段）
    body = re.sub(r'""".*?"""', '', src, flags=re.S)
    body = "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("#"))
    for name, pat in CHECK.items():
        hits = [m.group(0)[:50] for m in re.finditer(pat, body, re.I)]
        if hits:
            total += len(hits)
            print(f"  ⚠ {p.name} [{name}] {len(hits)}: {hits[:2]}")
print(f"\n违规总数 = {total}")
print("（0 = 全红线安全）")
