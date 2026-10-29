"""本地许可证 + 试用期（纯离线；HMAC 完整性校验 + 三处冗余防重置）。

诚实边界：签名密钥内嵌于本地程序（发行时编译为 .pyd），属于
**防误用/轻度防篡改**，不是密码学强保护。试用期时钟做三处冗余
（许可文件 / 注册表 / 隐藏文件），加载时取**最早**的首次运行日期——
删除或改某个文件都无法重置试用。二期升级路径：签名迁到服务端
（Ed25519 非对称），密钥彻底离库。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from pathlib import Path

_SECRET = b"poer-ai-play-v1-license-2026"       # 内嵌签名密钥（混淆级）
TRIAL_DAYS = 3

#: 订阅档位与特性门控（tier 名内嵌于激活码 payload 第三段）。
#: base=基础版 / pro=专业版 / max=大师版 / trial=试用期（全功能）。
#: 兼容：v1 旧码（无 tier 段）按 base 处理。
TIERS = ("trial", "base", "pro", "max")
_TIER_ALIASES = {"": "base", "basic": "base"}
FEATURES_BY_TIER: dict[str, set[str]] = {
    "trial": {"play", "advice", "replay", "import", "report", "export",
              "focus", "stake_high", "stake_premium", "difficulty_hard",
              "difficulty_max", "cli"},
    "base": {"play", "advice", "replay", "focus"},
    "pro": {"play", "advice", "replay", "focus", "import", "report",
            "export", "stake_high", "difficulty_hard"},
    "max": {"play", "advice", "replay", "focus", "import", "report",
            "export", "stake_high", "stake_premium", "difficulty_hard",
            "difficulty_max", "cli"},
}
FEATURE_LABELS = {
    "import": "导入分析", "report": "弱点报告", "export": "会话导出",
    "stake_high": "nl10/nl25 盲注档位", "stake_premium": "nl50 盲注档位",
    "difficulty_hard": "高手桌难度", "difficulty_max": "大师桌难度",
    "cli": "CLI 批量分析",
}
APP_DIR = Path(os.environ.get("APPDATA") or Path.home()) / "poker-ai"
STATE_FILE = APP_DIR / "license.json"
ALT_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "poker-ai"
ALT_FILE = ALT_DIR / ".trial_state"
_REG_KEY = r"Software\POER\poker-ai"


def machine_id() -> str:
    """机器指纹（MAC + 主机名 + 系统 → 16 位 hex）。"""
    import platform
    import uuid
    raw = f"{uuid.getnode()}|{platform.node()}|{platform.system()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16].upper()


def _hmac(payload: str) -> str:
    return hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:40]


def make_license_key(machine: str, days: int | None = 365,
                     tier: str = "base", ref: str = "") -> str:
    """按机器签发激活码（发行方工具 scripts/gen_license.py 调用）。

    payload = 机器码|到期日|档位|代理标记（v3；v2 无第四段 → ref 空；
    v1 无第三段 → 兼容为基础版）。ref 用于分销分账，随签名防篡改。"""
    tier = _norm_tier(tier)
    ref = _norm_ref(ref)
    expires = ("" if days is None else
               time.strftime("%Y-%m-%d", time.gmtime(time.time() + days * 86400)))
    payload = f"{machine}|{expires}|{tier}|{ref}"
    return base64.urlsafe_b64encode(payload.encode()).decode() + "." + _hmac(payload)


def _norm_ref(ref: str) -> str:
    """代理标记：A-Z0-9_-，≤16 字符；非法输入丢弃（不拒绝发码）。"""
    import re
    r = str(ref or "").strip().upper()
    return r if re.fullmatch(r"[A-Z0-9_-]{1,16}", r) else ""


def _norm_tier(tier: str) -> str:
    t = str(tier or "").strip().lower()
    t = _TIER_ALIASES.get(t, t)
    return t if t in TIERS else "base"


def _parse_key(key: str) -> tuple[str, str, str, str] | None:
    """激活码 → (机器码, 到期日, 档位, 代理标记)；无效 → None。

    兼容：v1（2 段）、v2（3 段）、v3（4 段，含 ref）。"""
    try:
        payload_b64, sig = str(key).strip().split(".", 1)
        payload = base64.urlsafe_b64decode(payload_b64.encode()).decode()
        parts = payload.split("|")
        if len(parts) == 2:                       # v1：无 tier 段
            parts += ["base", ""]
        elif len(parts) == 3:                     # v2：无 ref 段
            parts.append("")
        if len(parts) != 4:
            return None
        mid, expires, tier, ref = parts
        if not hmac.compare_digest(sig, _hmac(payload)):
            return None
        return mid, expires, _norm_tier(tier), _norm_ref(ref)
    except Exception:                             # noqa: BLE001
        return None


# ────────────────────── 试用期时钟：三处冗余 ──────────────────────
def _signed(obj: dict) -> dict:
    obj = dict(obj)
    obj.pop("_sig", None)
    obj["_sig"] = _hmac(json.dumps(obj, sort_keys=True, ensure_ascii=False))
    return obj


def _verified(obj: dict, first_run_field: str = "first_run") -> str | None:
    """签名校验通过 → 返回 obj 里的日期；否则 None（防手改）。"""
    if not isinstance(obj, dict):
        return None
    sig = obj.pop("_sig", None)
    try:
        if not sig or not hmac.compare_digest(
                _hmac(json.dumps(obj, sort_keys=True, ensure_ascii=False)), sig):
            return None
        date = str(obj.get(first_run_field) or "")
        time.strptime(date, "%Y-%m-%d")             # 格式校验
        return date
    except Exception:                             # noqa: BLE001
        return None


def _reg_read_first_run() -> str | None:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _REG_KEY) as k:
            raw = str(winreg.QueryValueEx(k, "first_run")[0])
        date, _, sig = raw.partition("|")
        # 注册表的签名是对裸日期做的（与 JSON 文件格式不同）
        return date if hmac.compare_digest(sig, _hmac(date)) else None
    except Exception:                             # noqa: BLE001
        return None


def _reg_write_first_run(date: str) -> None:
    try:
        import winreg
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _REG_KEY) as k:
            winreg.SetValueEx(k, "first_run", 0, winreg.REG_SZ,
                              f"{date}|{_hmac(date)}")
    except OSError:                               # noqa: BLE001（非 Windows 等）
        pass


def _earliest_first_run(state: dict) -> str:
    """三处冗余里取**最早**的首次运行日期（全部无效 → 今天）。"""
    candidates: list[str] = []
    date = _verified(state)
    if date:
        candidates.append(date)
    reg = _reg_read_first_run()
    if reg:
        candidates.append(reg)
    try:
        alt = json.loads(ALT_FILE.read_text(encoding="utf-8"))
        date = _verified(alt)
        if date:
            candidates.append(date)
    except Exception:                             # noqa: BLE001
        pass
    if not candidates:
        return time.strftime("%Y-%m-%d")
    return min(candidates)


def _persist_first_run(date: str) -> None:
    """把首次运行日期写满三处（带签名）。"""
    _reg_write_first_run(date)
    try:
        ALT_DIR.mkdir(parents=True, exist_ok=True)
        ALT_FILE.write_text(json.dumps(_signed({"first_run": date}),
                                       ensure_ascii=False), encoding="utf-8")
    except OSError:                               # noqa: BLE001
        pass


# ────────────────────── 状态读写与校验 ──────────────────────
def _load_state() -> dict:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:                             # noqa: BLE001
        return {}
    # 主状态文件带签名；被手改 → 只采信未篡改字段里的 key（激活路径
    # 每次启动都会重新验 key 签名与机器绑定，手改 "activated" 无收益）
    if _verified(state, "first_run") is None:
        state.pop("first_run", None)
    return state


def _save_state(state: dict) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(_signed(state), ensure_ascii=False,
                                     indent=1), encoding="utf-8")


def _trial_left(state: dict) -> int:
    first_run = _earliest_first_run(state)
    if not state.get("first_run"):
        state["first_run"] = first_run            # 回填，下次保存时带上
        _save_state(state)
    elif str(state.get("first_run")) != first_run:
        state["first_run"] = first_run            # 以最早值修正
        _save_state(state)
    _persist_first_run(first_run)
    try:
        started = time.mktime(time.strptime(first_run, "%Y-%m-%d"))
    except ValueError:
        return TRIAL_DAYS
    return TRIAL_DAYS - int((time.time() - started) / 86400)


def check_license() -> dict:
    """启动校验 → {"ok", "mode": licensed|trial|expired|invalid_key, "days_left"}。"""
    state = _load_state()
    if not state:
        state = {"first_run": time.strftime("%Y-%m-%d")}
        _save_state(state)
        _persist_first_run(state["first_run"])
    if state.get("activated"):
        parsed = _parse_key(str(state.get("key") or ""))
        if parsed is not None:
            mid, expires, tier, ref = parsed
            if (mid == machine_id()
                    and (not expires
                         or expires >= time.strftime("%Y-%m-%d"))):
                return {"ok": True, "mode": "licensed", "days_left": None,
                        "tier": tier, "ref": ref}
        return {"ok": False, "mode": "invalid_key", "days_left": 0,
                "tier": "base"}
    left = _trial_left(state)
    return {"ok": left > 0, "mode": "trial" if left > 0 else "expired",
            "days_left": max(0, left), "tier": "trial"}


def activate(key: str) -> bool:
    """校验并持久化激活码（绑定当前机器 + 有效期 + 档位）。"""
    parsed = _parse_key(key)
    if parsed is None:
        return False
    mid, expires, _tier, _ref = parsed
    if mid != machine_id():
        return False
    if expires and expires < time.strftime("%Y-%m-%d"):
        return False
    state = _load_state()
    state["activated"] = True
    state["key"] = str(key).strip()
    _save_state(state)
    return True


# ────────────────────── 档位特性门控（GUI 调用） ──────────────────────
_CURRENT_TIER: str | None = None


def set_current_tier(tier: str) -> None:
    global _CURRENT_TIER
    _CURRENT_TIER = _norm_tier(tier if tier else "trial")


def current_tier() -> str:
    if _CURRENT_TIER is None:
        st = check_license()
        set_current_tier(str(st.get("tier") or "base")
                         if st.get("ok") else
                         ("trial" if st.get("mode") == "trial" else "base"))
    return _CURRENT_TIER


def has_feature(feature: str) -> bool:
    """当前授权是否含某特性（未激活/过期 → 仅 base 特性不对齐，统一 False）。"""
    st = check_license()
    if not st.get("ok"):
        return False
    return feature in FEATURES_BY_TIER.get(current_tier(), set())


def tier_label(tier: str) -> str:
    return {"trial": "试用版", "base": "基础版",
            "pro": "专业版", "max": "大师版"}.get(_norm_tier(tier), "基础版")
