@echo off
title poker-ai 升级脚本
echo ============================================
echo   poker-ai 升级脚本
echo   保留：对局历史 / 配置 / 日志 / 激活状态
echo ============================================
echo.
set "OLD="
set /p OLD=请把【旧版本文件夹】拖入本窗口后按回车: 
if "%OLD%"=="" (
  echo 未输入路径，退出。
  pause
  exit /b 1
)
set OLD=%OLD:"=%
if "%OLD:~-1%"=="\" set OLD=%OLD:~0,-1%
if not exist "%OLD%\poker-ai-gui.exe" (
  echo [错误] 在该路径下没找到 poker-ai-gui.exe —— 请确认拖入的是旧版本文件夹
  pause
  exit /b 1
)
if exist "%OLD%\data\play_history" (
  robocopy "%OLD%\data\play_history" "data\play_history" /E /NFL /NDL /NJH /NJS /NP >nul
  echo [OK] 对局历史已保留
)
if exist "%OLD%\configs" (
  robocopy "%OLD%\configs" "configs" /E /NFL /NDL /NJH /NJS /NP >nul
  echo [OK] 配置已保留
)
if exist "%OLD%\logs" (
  robocopy "%OLD%\logs" "logs" /E /NFL /NDL /NJH /NJS /NP >nul
  echo [OK] 日志已保留
)
echo.
echo ============================================
echo   升级完成！双击 poker-ai-gui.exe 继续
echo ============================================
pause
