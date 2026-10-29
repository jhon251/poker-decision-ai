"""P30-2B — hardcode / forbidden-tech 扫描（header detector + observer + bridge diff）。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/vision2/pixel/realclient/coinpoker_betting_header.py",
    "poker_decision_ai/vision2/pixel/realclient/betting_header_observer.py",
    "poker_decision_ai/vision2/pixel/realclient/production_bridge.py",
]
patterns = {
    "硬编码 SB/BB": r"sb\s*=\s*1000|bb\s*=\s*2000|ante\s*=\s*320\s*(#|$)",
    "OCR": r"tesseract|pytesseract|easyocr|paddleocr",
    "CV-ML": r"cv2|skimage|torch|tensorflow|onnx",
    "视觉LLM": r"multimodal|openai|anthropic|claude|gemini",
    "网络": r"requests|httpx|urllib|http://|https://",
    "poker-rule 推导": r"current_bet.*to_call|blinds.*derived|sb.*=\s*bb",
    "previous/cache": r"previous|cache|prev_frame",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"\n=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [f"{m.group(0)[:50]}" for m in re.finditer(pat, src, re.I)]
        # 过滤 docstring（#### 位置之前含 docstring 但保留报告）
        print(f"[{name}] {len(hits)}: {hits[:3]}")
