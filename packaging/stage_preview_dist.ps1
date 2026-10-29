# EXE Preview GUI 分发暂存脚本（幂等）——生成便携分发目录。
# 用法:  powershell -File packaging\stage_preview_dist.ps1 [-StageDir path]
# 布局:  EXE + configs\ + data\vision2 视觉资产（V3F observers 所需）
#        + examples\hands\ + sample_frame.png（可选演示帧）
param(
    [string]$StageDir = "dist_clean\poker-ai-gui-preview",
    [string]$ExePath = "dist\poker-ai-gui-preview.exe"
)
$ErrorActionPreference = "Stop"
if (Test-Path $StageDir) { Remove-Item -Recurse -Force $StageDir }
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

Copy-Item $ExePath $StageDir\
Copy-Item -Recurse configs $StageDir\configs
New-Item -ItemType Directory -Force -Path "$StageDir\data\vision2" | Out-Null
# V3F observers 读取的视觉资产（CWD 相对——必须外置于 EXE 旁）
Copy-Item data\vision2\coinpoker_assets_* "$StageDir\data\vision2\" -Recurse
Copy-Item data\vision2\hero_card_layout_v1.json "$StageDir\data\vision2\"
Copy-Item data\vision2\stack_text_layout_v1.json "$StageDir\data\vision2\"
if (Test-Path examples\hands) {
    New-Item -ItemType Directory -Force -Path "$StageDir\examples" | Out-Null
    Copy-Item -Recurse examples\hands "$StageDir\examples\hands"
}
if (Test-Path data\vision2\coinpoker_evidence\8bf82f134651db0add626be0304a2ff1.png) {
    Copy-Item data\vision2\coinpoker_evidence\8bf82f134651db0add626be0304a2ff1.png `
        "$StageDir\sample_frame.png"
}
$sz = (Get-ChildItem $StageDir -Recurse -File | Measure-Object Length -Sum).Sum
"staged: $StageDir  ($([math]::Round($sz/1MB,2)) MB)"
Get-ChildItem $StageDir | Select-Object -ExpandProperty Name
