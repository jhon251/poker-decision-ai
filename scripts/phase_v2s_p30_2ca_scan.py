"""P30-2C-A — hardcode / forbidden-tech 扫描（to_call 相关文件）。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/vision2/pixel/realclient/coinpoker_to_call.py",
    "poker_decision_ai/vision2/pixel/realclient/to_call_observer.py",
]
patterns = {
    "current_bet=raise": r"current_bet\s*=\s*(to_call|raise)|to_call\s*=\s*raise",
    "OCB": r"ocr|tesseract",
    "CV-ML": r"cv2|skimage|torch|tensorflow",
    "视觉LLM": r"multimodal|openai|anthropic",
    "网络": r"requests|httpx|http://",
    "previous/cache": r"previous|cache",
    "硬编码 2000/124163": r"to_call\s*=\s*(2000|124163)\s*(#|$)",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:50] for m in re.finditer(pat, src, re.I)]
        print(f"[{name}] {len(hits)}: {hits[:3]}")
