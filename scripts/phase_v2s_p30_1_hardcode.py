"""P30-1 — bridge 模块 hardcode / 禁线扫描。"""
from __future__ import annotations
import re
from pathlib import Path

src = Path("poker_decision_ai/vision2/pixel/realclient/"
           "production_bridge.py").read_text(encoding="utf-8")
print("=== 禁线扫描（production_bridge.py）===")
patterns = {
    "hero=S6 硬编码": r"hero_seat\s*=\s*[\"']S6[\"']|hero_seat\s*=\s*5\s*(#|$)",
    "dealer 固定": r"dealer_seat\s*=\s*[\"']S\d[\"']",
    "table_size=6": r"table_size\s*=\s*6\s*(#|$)",
    "UNKNOWN→0": r"float\(.*\bor\s+0(\.0)?\)|int\(.*\bor\s+0\)",
    "UNKNOWN→EMPTY": r"UNKNOWN.*EMPTY\s*$",
    "previous/cache": r"previous|cache|prev_frame",
    "OCR": r"ocr|tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2|skimage|torch|tensorflow|onnx",
    "视觉LLM": r"multimodal|openai|anthropic|claude|gemini",
    "网络": r"requests|httpx|urllib|http://|https://",
}
for name, pat in patterns.items():
    hits = [(m.group(0)[:50]) for m in re.finditer(pat, src, re.IGNORECASE)]
    print(f"[{name}] {len(hits)}: {hits[:3]}")
print("\n=== seat mapping 来源 ===")
print("uses ring_number import:", "ring_number" in src)
print("CANONICAL_RING_MAP reference:", "CANONICAL_RING_MAP" in src)
