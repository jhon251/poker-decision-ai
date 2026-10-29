# Poker Decision AI（德州扑克决策研究框架）

一个**本地运行、可解释、诚实地标注边界**的德州扑克决策辅助与研究工作台：
把"屏幕识别 → 状态规范化 → 求解器/启发式决策 → 审计留痕"串成一条可复现的链路，
并对每一步都保留**来源、置信与样本量**。

> ⚠️ **请先读这两条**
> 1. **不做效果承诺**：本项目**不宣称**任何"更有水平/更赚/达到 GTO"的结论。仓库里所有指标都带
>    口径与样本量；样本不足的地方会**显式标注"不可作结论"**。
> 2. **合规风险自负**：在**真钱**牌局中使用任何第三方辅助工具，**可能违反该平台的服务条款**并带来
>    封号等后果。本项目仅用于**个人研究/学习**；是否使用、如何使用，由你自行判断并承担后果。
>    本项目**不会**自动点击、自动下注或操作任何客户端。

---

## 这是什么 / 不是什么

**是**：一个模块化的决策框架 —— 多 Agent 编排、可插拔 Provider（LLM / 视觉 / 求解器 / 规则）、
可审计的决策记录、以及一套**门禁（gates）**用来防止"报告漂移"和"口径膨胀"。

**不是**：
* 不是"必赢系统" ✗，也不是"GTO 求解器" ✗（求解器是**外部依赖**，本项目只做接线与校验）；
* 不含**社区牌库/对手数据库** ✗；
* 不含**真实客户端截图与其识别模板** ✗（见下"你需要自备什么"）。

## 核心设计

```
┌──────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│  Capture │──▶│  Vision      │──▶│  Orchestration │──▶│  Providers   │
│ (可选)   │   │  (可插拔)    │   │  (多 Agent)    │   │ LLM/Solver/… │
└──────────┘   └──────────────┘   └───────────────┘   └──────────────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ 决策 + 审计    │
                                  │ 来源/置信/样本 │
                                  └───────────────┘
```

关键原则（贯穿全仓）：

| 原则 | 落地方式 |
|---|---|
| **UNKNOWN-first** | 拿不到的证据就写 `UNKNOWN`，**绝不猜**；相应字段在决策里被标为阻断 |
| **不伪造数据** | 缺失字段写 `NULL`；测试/回放数据打来源标记（`runtime` / `test` / `replay`）|
| **可审计** | 每个决策带 `basis`、`confidence`、`evidence`（原始数字），可逐条复核 |
| **可回退** | 关键优化都有环境变量开关（如 `VISION_CACHE=0`、`GLYPH_BOUNDS=0`）|
| **口径诚实** | 报告里所有比例都附样本量；样本不足显式写"不可作结论" |

## 目录结构

```
poker_decision_ai/     核心库（编排、Provider、领域模型、审计）
cash_game_ui/          GUI（一键分析）与点击链路接线
configs/               配置（环境变量占位，不含任何密钥）
scripts/               开发/验证/审计脚本（含门禁）
tests/                 单元与集成测试
packaging/             PyInstaller 打包描述
docs/                  架构与边界说明
```

## 安装

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env      # 按需填写；全空则降级到 mock
```

## 你需要自备什么（重要）

本仓库**刻意不包含**以下内容，因为它们要么含第三方版权，要么含真实牌桌截图与**玩家昵称**：

| 缺什么 | 为什么 | 怎么补 |
|---|---|---|
| 视觉识别模板资产（卡面/数字/按钮）| 由真实客户端截图标注而来，含玩家昵称与客户端 UI ✗ | 自行截图并按 `docs/ARCHITECTURE.md` 的资产清单标注 |
| 求解器二进制（postflop-solver 封装）| 第三方程序，许可与分发需自行确认 ✗ | 自行获取并在 `configs/config.yaml` 里配置路径 |
| 基准数据集（如 PokerBench）| 第三方数据，体积大且有各自许可 ✗ | 自行下载到 `data/external/` |
| 预解牌面库 | 体积巨大（GB 级）| 用仓库内脚本自行批量求解生成 |

缺少这些时，程序会**降级**并明确标注（`UNAVAILABLE` / `UNKNOWN`），**不会**编造结果。

## 快速开始

```bash
# 1) 跑测试
pytest -q

# 2) 无界面跑一次完整链（打印 JSON，便于脚本化）
python -m poker_decision_ai.cli --help

# 3) 起 GUI（需要自备视觉资产与求解器才有真实输出）
python cash_game_main.py
```

## 测试与门禁

```bash
pytest -q                       # 单元 + 集成
python scripts/verify_all.py    # 汇总门禁（依你本地资产与数据而定，缺资产会自动跳过/降级）
```

部分测试依赖"真实帧夹具"，夹具**未随仓库分发**（含玩家昵称 ✗）——
这类测试会自动 **skip**（而不是失败），这是刻意设计。

## 已知限制（确实存在的）

* 屏幕识别对**窗口尺寸/DPI/动画**敏感；本项目做了几何归一化，但**不是**万能；
* `to_call`（跟注额）在某些面板样式下读不准 ⇒ 程序会**拒绝**给出可能"被支配"的动作
  （例如无法确认面对下注时，**不会**建议弃牌/跟注）；
* 求解器真解覆盖率受**算力**限制（预解 N 个翻牌 ≈ N/22,100 的均匀覆盖率）；
* 第三方数据集的对比**不可当作能力评价**（范围/抽象无法对齐）。

## 许可

代码：**MIT**（见 [LICENSE](LICENSE)）。
第三方组件与数据集：见 [NOTICE.md](NOTICE.md) —— 它们**不随本仓库分发**。

## 隐私

本仓库**不含**任何个人数据（不含真实牌桌截图、抓包、日志、对手库）。
详见 [PRIVACY.md](PRIVACY.md)。若你自己运行并保存截图，请自行注意其中可能包含的**玩家昵称**。

---

## English (short)

A local, auditable Texas Hold'em decision *research* framework: capture → vision → orchestration →
providers (LLM/solver/heuristics) → decision **with provenance and confidence**, plus gates to keep
reports honest. **No performance claims.** Using third-party assistance in real-money games may
violate the platform's terms of service — your responsibility. Vision assets, solver binaries and
benchmark datasets are intentionally **not** included (third-party licensing / personal data).

Code licensed **MIT**; see [NOTICE.md](NOTICE.md) for third-party components.
