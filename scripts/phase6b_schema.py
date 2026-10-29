"""Phase 6 Step 6B.1：Feature Schema v2 定义（17 维，语义元数据）。

17 维保持冻结不变，不新增不删除。每特征声明：
  - name / label / definition / gate（机会门控条件）
  - category: behavioral | outcome | derived | alias
  - derived_from（position_sensitivity → rfi_ep, rfi_lp）
  - alias_of（ep_open → rfi_ep，定义相同，避免重复语义）
  - conditional（位置条件化子特征）
  - suff_key（对应 full_pool_suff_stats 的字段名）

Tier 不在本步分配（由 6B.3 代码自动判定）。

输出 data/uoft/models/phase6/feature_schema.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scripts.phase5b_common import ensure_dir  # noqa: E402

FEATURE_SCHEMA = [
    # ---- 全手分母特征 ----
    {"name": "vpip", "label": "VPIP", "definition": "自愿入池频率",
     "gate": "all_ring_hands", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "vpip"},
    {"name": "pfr", "label": "PFR", "definition": "翻前加注频率",
     "gate": "all_ring_hands", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "pfr"},
    {"name": "three_bet", "label": "3Bet", "definition": "3bet 频率（第 2 次加注）",
     "gate": "all_ring_hands", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "three_bet"},
    {"name": "four_bet", "label": "4Bet", "definition": "4bet 频率（第 3 次加注）",
     "gate": "all_ring_hands", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "four_bet", "note": "极稀事件（全量 ~1170）"},
    {"name": "won_hand", "label": "won_hand", "definition": "赢下底池频率（结果型）",
     "gate": "all_ring_hands", "category": "outcome",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "won_hand", "note": "结果型，非行为选择"},
    # ---- 机会门控特征 ----
    {"name": "cbet", "label": "CBet", "definition": "翻牌持续下注频率",
     "gate": "cbet_opportunity", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "cbet"},
    {"name": "fold_to_3bet", "label": "Fold-to-3Bet", "definition": "面对 3bet 弃牌频率",
     "gate": "faced_3bet", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "fold_to_3bet"},
    {"name": "fold_to_cbet", "label": "Fold-to-CBet", "definition": "面对 cbet 弃牌频率",
     "gate": "faced_cbet", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "fold_to_cbet"},
    {"name": "wtsd", "label": "WTSD", "definition": "入池后进入摊牌频率",
     "gate": "saw_flop", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "wtsd"},
    {"name": "wsd", "label": "W$SD", "definition": "摊牌后获胜频率",
     "gate": "went_to_showdown", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "wsd"},
    {"name": "agg_freq", "label": "Aggression Frequency",
     "definition": "下注/加注占主动行动（下注+跟注）比例",
     "gate": "bets+calls>0", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "agg_freq"},
    {"name": "cold_call", "label": "Cold Call",
     "definition": "非盲位被动入池（vpip 且非 pfr）频率",
     "gate": "non_blind_position", "category": "behavioral",
     "conditional": False, "alias_of": None, "derived_from": None,
     "suff_key": "cold_call"},
    # ---- 位置特征（分量/派生/别名）----
    {"name": "rfi_ep", "label": "RFI@EP", "definition": "EP 位置 raise-first-in 频率",
     "gate": "position=EP & rfi_opp", "category": "behavioral",
     "conditional": True, "alias_of": None, "derived_from": None,
     "suff_key": "rfi_ep"},
    {"name": "rfi_lp", "label": "RFI@LP", "definition": "LP/BTN 位置 raise-first-in 频率",
     "gate": "position in {LP,BTN} & rfi_opp", "category": "behavioral",
     "conditional": True, "alias_of": None, "derived_from": None,
     "suff_key": "rfi_lp"},
    {"name": "btn_steal", "label": "BTN Steal", "definition": "BTN 位置 raise-first-in 频率",
     "gate": "position=BTN & rfi_opp", "category": "behavioral",
     "conditional": True, "alias_of": None, "derived_from": "rfi_lp",
     "suff_key": "btn_steal", "note": "rfi_lp 的 BTN 子集"},
    {"name": "ep_open", "label": "EP Open",
     "definition": "EP 位置 raise-first-in 频率（与 rfi_ep 定义相同）",
     "gate": "position=EP & rfi_opp", "category": "alias",
     "conditional": True, "alias_of": "rfi_ep", "derived_from": "rfi_ep",
     "suff_key": "ep_open"},
    {"name": "position_sensitivity", "label": "Position Sensitivity",
     "definition": "RFI_LP − RFI_EP（位置敏感度差量）",
     "gate": "rfi_ep & rfi_lp 均有机会", "category": "derived",
     "conditional": False, "alias_of": None,
     "derived_from": ["rfi_ep", "rfi_lp"], "suff_key": None},
]

# 校验：17 维、无重名、alias/derived 引用有效
_NAMES = [f["name"] for f in FEATURE_SCHEMA]
assert len(_NAMES) == 17, f"expected 17 features, got {len(_NAMES)}"
assert len(set(_NAMES)) == 17, "duplicate feature names"
for f in FEATURE_SCHEMA:
    if f["alias_of"]:
        assert f["alias_of"] in _NAMES, f"alias target missing: {f['alias_of']}"
    if f["derived_from"]:
        targets = f["derived_from"] if isinstance(f["derived_from"], list) \
            else [f["derived_from"]]
        for t in targets:
            assert t in _NAMES, f"derived target missing: {t}"


def main() -> int:
    out = ensure_dir("data/uoft/models/phase6")
    (out / "feature_schema.json").write_text(
        json.dumps({"schema_version": "6.0", "n_features": len(FEATURE_SCHEMA),
                    "features": FEATURE_SCHEMA},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"feature_schema written: {len(FEATURE_SCHEMA)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
