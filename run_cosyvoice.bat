@echo off
REM 显示可用的启动选项

echo ======================================================
echo CosyVoice 启动选项
echo ======================================================
echo 请选择要启动的服务:
echo.
echo 1. 标准Web界面 (Gradio)
echo 2. 高性能Web界面 (Gradio + FP16)
echo 3. 标准API服务器 (FastAPI)
echo 4. 高性能API服务器 (FastAPI + FP16)
echo 5. 查看README
echo 0. 退出
echo ======================================================
echo.

set /p choice=请输入选项 (0-5): 

if "%choice%"=="1" (
    start scripts\start_cosyvoice_with_azure.bat
) else if "%choice%"=="2" (
    start scripts\start_cosyvoice_with_azure_highperf.bat
) else if "%choice%"=="3" (
    start scripts\start_cosyvoice_azure_api.bat
) else if "%choice%"=="4" (
    start scripts\start_cosyvoice_azure_api_highperf.bat
) else if "%choice%"=="5" (
    type scripts\README.md | more
    pause
    %0
) else if "%choice%"=="0" (
    exit
) else (
    echo 无效的选项，请重新选择
    timeout /t 2 >nul
    %0
) 