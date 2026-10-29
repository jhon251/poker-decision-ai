"""界面文案双语言框架（中/英）。

覆盖静态界面框架（菜单/按钮/面板/对话框标签）；动态内容（日志、报告、
分析文本）当前仍为中文，随版本逐步接入。语言选择持久化于
%APPDATA%/poker-ai/settings.json，切换后重启生效。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_APP_DIR = Path(os.environ.get("APPDATA") or Path.home()) / "poker-ai"
_SETTINGS = _APP_DIR / "settings.json"

STRINGS: dict[str, dict[str, str]] = {
    "zh": {
        "app_title": "poker-ai 对打模式（产品 v1）",
        "view": "视图", "zoom": "缩放", "language": "语言",
        "new_session": "▶ 新会话", "replay": "⟲ 复盘", "report": "📊 弱点报告",
        "sound": "提示音", "not_started": "● 未开局",
        "action_bar": "你的动作", "view_ai": "查看 AI 参考",
        "fold": "弃牌", "check": "过牌", "call": "跟注", "raise": "加注",
        "bet": "下注", "allin": "全下", "min": "最小", "half_pot": "½池",
        "threeq_pot": "¾池", "pot_full": "满池", "raise_to": "加注到",
        "advice_panel": "AI 参考建议（独立于你的动作）",
        "log_panel": "逐手解析 / 会话日志",
        "ready": "READY — 点「新会话」开始对打（非实时辅助；纯本地）",
        "dlg_title": "新会话", "dlg_stake": "档位", "dlg_table": "桌型",
        "dlg_stack": "起始筹码", "dlg_focus": "训练定向",
        "dlg_hands": "手数", "dlg_seed": "种子（空=随机）",
        "dlg_difficulty": "AI 难度", "dlg_advice": "AI 参考建议",
        "dlg_advice_on": "启用（judge 链：solver 锚定+频率先验）",
        "dlg_start": "开始",
        "import_analysis": "📥 导入分析", "dlg_import_title": "导入手牌分析",
        "report_external": "🌐 外部牌谱分析",
        "dlg_paste_hint": "粘贴手牌历史文本（或点「从文件加载」），AI 将给出建议",
        "dlg_load_file": "📄 从文件加载", "dlg_analyze": "🔍 分析",
        "dlg_close": "关闭",
    },
    "en": {
        "app_title": "poker-ai Play Mode (v1)",
        "view": "View", "zoom": "Zoom", "language": "Language",
        "new_session": "▶ New Session", "replay": "⟲ Replay",
        "report": "📊 Reports", "sound": "Sound", "not_started": "● Idle",
        "action_bar": "Your Action", "view_ai": "AI Advisor",
        "fold": "Fold", "check": "Check", "call": "Call", "raise": "Raise",
        "bet": "Bet", "allin": "All-in", "min": "Min", "half_pot": "½ Pot",
        "threeq_pot": "¾ Pot", "pot_full": "Pot", "raise_to": "Raise to",
        "advice_panel": "AI Reference (independent of your action)",
        "log_panel": "Hand Review / Session Log",
        "ready": "READY — click 「New Session」 to start (offline; local only)",
        "dlg_title": "New Session", "dlg_stake": "Stakes", "dlg_table": "Table",
        "dlg_stack": "Stack", "dlg_focus": "Focus Training",
        "dlg_hands": "Hands", "dlg_seed": "Seed (blank = random)",
        "dlg_difficulty": "AI Difficulty", "dlg_advice": "AI Advisor",
        "dlg_advice_on": "Enable (judge chain: solver + frequency prior)",
        "dlg_start": "Start",
        "import_analysis": "📥 Import", "dlg_import_title": "Import Hand Analysis",
        "report_external": "🌐 External Hands",
        "dlg_paste_hint": "Paste hand history text (or load from file) — AI will advise",
        "dlg_load_file": "📄 Load File", "dlg_analyze": "🔍 Analyze",
        "dlg_close": "Close",
    },
}


def current_lang() -> str:
    try:
        lang = json.loads(_SETTINGS.read_text(encoding="utf-8")).get("lang")
    except Exception:                             # noqa: BLE001
        lang = None
    return lang if lang in STRINGS else "zh"


def set_lang(lang: str) -> None:
    if lang not in STRINGS:
        return
    _APP_DIR.mkdir(parents=True, exist_ok=True)
    state = {}
    try:
        state = json.loads(_SETTINGS.read_text(encoding="utf-8"))
    except Exception:                             # noqa: BLE001
        pass
    state["lang"] = lang
    _SETTINGS.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                         encoding="utf-8")


def tr(key: str) -> str:
    """取当前语言的界面文案；缺失键回落中文。"""
    return STRINGS.get(current_lang(), STRINGS["zh"]).get(key,
                                                          STRINGS["zh"].get(key, key))
