"""P33 — 红线扫描扩展版（剥 docstring 后扫 P31/P32/P33 五模块）。

与 scripts/phase_v2s_p33_scan.py 互补：
- STRICT（信号层 P33 新代码）——零动作词/零 TAG/LAG/零 random；
- MAPPING（解析器层 P31/P32 冻结代码）——禁 deviation→action 映射
  （解析器中出现 call/raise/bet 属合法手历史 action 字符串分类）。
"""
import re
from pathlib import Path

files = [
    "poker_decision_ai/deviation_signal.py",
    "poker_decision_ai/population.py",
    "poker_decision_ai/population_prior.py",
    "poker_decision_ai/player_history.py",
    "poker_decision_ai/opponent_profile.py",
]
pats_strict = {  # 信号层（P33 新代码）——零动作词
    "bare action verbs": r"\b(call|raise|fold|bet)\b",
    "TAG/LAG/NIT": r"\b(TAG|LAG|NIT)\b",
    "random": r"\brandom\b",
    "click/mouse/bet-submit": r"\b(click|mouse|keyboard|send_bet|submit)\b",
    "OCR/CV/LLM": r"(tesseract|cv2\.|torch|multimodal|openai|anthropic)",
    "hardcoded player names": r"(dualang|Ivan|Lucy)",
    "fake REAL": r"data_kind\s*=\s*['\"]REAL",
}
pats_mapping = {  # 解析器层（P31/P32 冻结代码）——禁 deviation→action 映射
    "deviation→action mapping": r"deviation.*(=>|->)\s*(call|raise|fold)"
                                r"|should\s+raise",
    "TAG/LAG fallback": r"(unknown.*TAG|TAG.*fallback|LAG.*fallback)",
    "random": r"\brandom\b",
    "hardcoded player names": r"(dualang|Ivan|Lucy)",
}
strict_files = ["poker_decision_ai/deviation_signal.py"]
parser_files = [
    "poker_decision_ai/population.py",
    "poker_decision_ai/population_prior.py",
    "poker_decision_ai/player_history.py",
    "poker_decision_ai/opponent_profile.py",
]
for f in strict_files:
    body = re.sub(r'""".*?"""', "",
                  Path(f).read_text(encoding="utf-8"), flags=re.S)
    hits = {k: re.findall(p, body, re.I) for k, p in pats_strict.items()}
    hits = {k: v for k, v in hits.items() if v}
    print(f"[STRICT {Path(f).name}]",
          ("VIOLATION: " + str(hits)) if hits else "clean (0)")
for f in parser_files:
    body = re.sub(r'""".*?"""', "",
                  Path(f).read_text(encoding="utf-8"), flags=re.S)
    hits = {k: re.findall(p, body, re.I) for k, p in pats_mapping.items()}
    hits = {k: v for k, v in hits.items() if v}
    print(f"[MAPPING {Path(f).name}]",
          ("VIOLATION: " + str(hits)) if hits else "clean (0)")
