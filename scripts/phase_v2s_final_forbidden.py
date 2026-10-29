"""FINAL AUDIT — 禁用技术导入扫描（OCR/CV-ML/LLM/网络/外部服务）。"""
from __future__ import annotations
import re
from pathlib import Path

pkg = Path("poker_decision_ai/vision2/pixel/realclient")
forbidden = {
    "OCR": r"pytesseract|easyocr|tesseract|ocr\.|paddleocr",
    "CV-ML": r"import cv2|cv2\.|skimage|scikit-image|opencv|torch|tensorflow|onnx",
    "视觉LLM": r"openai|anthropic|claude|gpt-4|vision_llm|multimodal|gemini|extract_vision|send_image",
    "网络/外部推理": r"requests\.|urllib|httpx|http://|https://|socket|grpc|api_key",
}
for name, pat in forbidden.items():
    hits = []
    for p in sorted(pkg.glob("*.py")):
        src = p.read_text(encoding="utf-8")
        for m in re.finditer(pat, src, re.IGNORECASE):
            line = src[:m.start()].count("\n") + 1
            hits.append(f"{p.name}:{line}: {m.group(0)[:60]}")
    print(f"=== {name}: {len(hits)} hits ===")
    for h_ in hits[:8]:
        print("  ", h_)

# 额外：跨包违规扫描（整个 vision2）
print("\n=== vision2 全包扫描（更广） ===")
root = Path("poker_decision_ai/vision2")
for name, pat in forbidden.items():
    hits = []
    for p in sorted(root.rglob("*.py")):
        src = p.read_text(encoding="utf-8")
        for m in re.finditer(pat, src, re.IGNORECASE):
            line = src[:m.start()].count("\n") + 1
            hits.append(f"{p.relative_to(root)}:{line}: {m.group(0)[:50]}")
    print(f"-- {name}: {len(hits)} hits")
    for h_ in hits[:6]:
        print("  ", h_)
