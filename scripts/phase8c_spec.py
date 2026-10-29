"""Phase 8C 预注册规格（冻结常量，全部一次定义，实验期只读）。

所有参数在本文件冻结；detection/EV/calibration 阶段禁止修改、禁止按结果调整。
"""
from __future__ import annotations

from pathlib import Path
import json

_ROOT = Path(__file__).parent.parent
OUT_DIR = _ROOT / "data/uoft/models/phase8/phase8c"

SEED = 42                       # 全局实验种子
N_SEATS = 6                     # 6-max
SB, BB, STACK = 0.5, 1.0, 100.0  # 0.5/1 盲注、100bb 深筹码（EV 直接以 bb 计）
HERO_SEAT = "S1"                # hero 固定座位（预注册）
VILLAIN_SEAT = "S2"             # 目标 villain 固定座位
N_FILLERS = 4                   # 陪练（V0 baseline）

# 难度层级（完全冻结，禁止按结果修改）
TIERS = {"strong": 0.20, "moderate": 0.10, "near": 0.02}

# 可偏差特征 = 7B validated 三特征（three_bet/cbet 保持 baseline，如实呈现不推荐）
DEVIABLE_FEATURES = ("vpip", "pfr", "fold_to_3bet")
DIRECTIONS = ("over", "under")
N_VILLAINS_PER_CELL = 15

H_HISTORY = 200                 # history 手数（7C 画像输入）
E_EVAL = 100                    # eval 手数（两臂 EV）
HORIZONS = (50, 100, 200)       # 检测 horizon（冻结，禁止增减）

R_ROLLOUTS_PRIMARY = 20         # 主臂 rollout 数
R_ROLLOUTS_OTHER = 10           # 其余 5 臂 rollout 数（预注册折中）
BOOT_N, BOOT_SEED = 1000, 42
PRACTICAL_BB100 = 1.0           # convention（预注册，非统计判据）

# V3 orthogonal-deviation（adversarial control）：只在 7C 不可观测行为上偏离
V3_DONK_MULT = 1.5              # 非翻前加注者的 postflop bet 概率乘子（donk 倾向）
# 结构性证明：donk 乘子只修改 (非翻前 aggressor 的 postflop bet/check 混合)，
# 该 spot 对 5 个决策特征的 suff stats 无任何贡献（cbet opportunity 要求
# 翻前 aggressor 身份；vpip/pfr/three_bet 是翻前特征；fold_to_3bet 与 postflop 无关）

# calibration 参数（一次性、预实验、seed=42、结果无关；实验阶段前冻结）
CAL_SIM_HANDS = 6000            # 每轮校准模拟手数（faced_3bet 等稀有机会需更多手稳定）
CAL_MAX_ITERS = 20
CAL_TOL = 0.005

EV_LABEL = "simulated EV under frozen synthetic response model"
PROHIBITED_WORDS = ["exploit_utility_proven", "EV proven", "ROI proven",
                    "profitable", "real-world profitability", "winnings"]


def load_frozen_priors() -> dict:
    """冻结 train-only priors（hb_fit 4 特征 + fold_to_3bet 方案 B 口径）。

    μ = a/(a+b) 精确重算（与 8B 口径一致；7A 一致性硬断言要求）。
    """
    hb = json.load(open(_ROOT / "data/uoft/models/mvp/hb_fit.json", encoding="utf-8"))
    assert hb["config"]["fit_on"] == "task_a_train_players"
    priors = {}
    for f in ("vpip", "pfr", "three_bet", "cbet"):
        fit = hb["probes"][f]
        a, b = fit["a"], fit["b"]
        priors[f] = {"a": a, "b": b, "mu": a / (a + b), "provenance": "hb_fit"}
    # fold_to_3bet 方案 B（train-only，与 8B 相同口径）——从 8B 的 PAM 元数据不可得，
    # 由 phase8c_calibrate 使用冻结 fit_fold_to_3bet_prior 重新拟合（只读冻结函数，
    # 测试玩家跳过硬断言）并写入 calibration artifact。
    return priors
