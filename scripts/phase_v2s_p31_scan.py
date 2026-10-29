"""P31 — 修改文件 forbidden-tech / hardcode / 伪造 / fixed-identity 扫描。"""
from __future__ import annotations
import re
from pathlib import Path

files = [
    "poker_decision_ai/player_history.py",
    "poker_decision_ai/opponent_profile.py",
]
patterns = {
    "OCR": r"tesseract|pytesseract|easyocr",
    "CV-ML": r"cv2\.|skimage|torch|tensorflow|onnx",
    "LLM": r"multimodal|openai|anthropic|gemini",
    "网络": r"requests\.|httpx|socket",
    "固定 seat→player": r"S6\s*->\s*['\"]dualang|seat.*=\s*['\"]Ivan|player_id\s*=\s*['\"]dualang",
    "伪造统计": r"vpip\s*=\s*0\.\d+\s*#.*fake|fake.*vpip|fabricat.*stat",
    "小样本伪装": r"small.*PLAYER_BASED|hands.*<.*PLAYER_BASED_MIN_HANDS.*PLAYER_BASED",
    "random fallback": r"random\.choice|random_action",
    "previous-frame": r"prev_frame|previous_frame|last_obs",
}
for fp in files:
    src = Path(fp).read_text(encoding="utf-8")
    body = re.sub(r'""".*?"""', '', src, flags=re.S)
    print(f"=== {Path(fp).name} ===")
    for name, pat in patterns.items():
        hits = [m.group(0)[:60] for m in re.finditer(pat, body, re.I)]
        print(f"  [{name}] {len(hits)}: {hits[:2]}")
