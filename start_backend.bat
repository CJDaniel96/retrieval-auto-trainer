@echo off
chcp 65001 >nul
echo ================================================================
echo           AI 自動化訓練系統 - 後端服務啟動
echo ================================================================
echo.

:: 檢查 Python 是否安裝
echo [1/4] 檢查 Python 環境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: 未找到 Python，請確保 Python 3.10+ 已安裝並加入環境變數
    pause
    exit /b 1
)
echo ✅ Python 環境檢查通過

:: 切換到專案目錄
echo.
echo [2/4] 切換到專案目錄...
cd /d "%~dp0"
echo ✅ 目前目錄: %CD%

:: 檢查並安裝依賴
echo.
echo [3/4] 檢查 Python 依賴套件...
if exist requirements.txt (
    echo 📦 安裝/更新依賴套件...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ 依賴安裝失敗，請檢查網路連線或 pip 設定
        pause
        exit /b 1
    )
    echo ✅ 依賴套件安裝完成
) else (
    echo ⚠️  未找到 requirements.txt 檔案
)

:: 啟動後端 API 服務
echo.
echo [4/4] 啟動後端 API 服務...
echo 🚀 正在啟動 API 服務，請稍候...
echo.
echo ================================================================
echo    API 服務將在 http://localhost:8000 上運行
echo    按 Ctrl+C 可停止服務
echo ================================================================
echo.

python -m backend.api_service

echo.
echo 後端服務已停止
pause