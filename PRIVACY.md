# 隐私说明（PRIVACY）

## 本仓库不含个人数据

发布前做过逐项核查，**本仓库不包含**：

* ✗ 真实牌桌截图 / 录屏（含玩家昵称）
* ✗ 抓包、运行日志（`logs/`、`dist/logs/`）
* ✗ 对手数据库（`*.db`）
* ✗ 任何 API Key / Token / 凭据（`.env` 已被 `.gitignore` 排除；`.env.example` 只有空占位）
* ✗ 本地绝对路径与用户名（已统一替换为 `<repo>` 占位）

## 程序运行时会产生什么（仅在本机）

| 数据 | 位置（默认）| 是否上传 |
|---|---|---|
| 决策/事件日志 | `logs/`、`dist/logs/` | **否** ✗ |
| 失败帧截图（便于你排查识别问题）| `logs/frames/` | **否** ✗ |
| 对手档案库（如启用）| `data/opponents.db` | **否** ✗ |
| 现场解缓存 | `data/solver/spots_6max_livecache/` | **否** ✗ |

**没有遥测、没有自动上传**。唯一可能的对外网络请求来自你**显式配置**的 LLM / 求解器
Provider（例如你自己填的 API Key），这属于你的选择与责任。

## 如果你要分享截图/日志

程序保存的失败帧**会包含牌桌上其他玩家的昵称** ✗。分享前请自行打码；
本仓库的贡献指南同样要求：**不要提交含他人个人信息的截图或日志**。

## 删除数据

```bash
rm -rf logs/ dist/logs/ data/opponents.db
# Windows: Remove-Item -Recurse -Force logs, dist\logs, data\opponents.db
```
