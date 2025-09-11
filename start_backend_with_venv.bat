@echo off
chcp 65001 >nul
echo ================================================================
echo       AI 自動化訓練系統 - 後端服務啟動 (虛擬環境版本)
echo ================================================================
echo.

:: 設定變數
set VENV_NAME=venv
set PYTHON_MIN_VERSION=3.10

:: 檢查 Python 版本
echo [1/6] 檢查 Python 環境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 錯誤: 未找到 Python，請確保 Python %PYTHON_MIN_VERSION%+ 已安裝並加入環境變數
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ 找到 Python %PYTHON_VERSION%

:: 切換到專案目錄
echo.
echo [2/6] 切換到專案目錄...
cd /d "%~dp0"
echo ✅ 目前目錄: %CD%

:: 檢查/創建虛擬環境
echo.
echo [3/6] 檢查虛擬環境...
if exist %VENV_NAME%\Scripts\activate.bat (
    echo ✅ 找到現有虛擬環境: %VENV_NAME%
) else (
    echo 📦 創建新的虛擬環境: %VENV_NAME%
    python -m venv %VENV_NAME%
    if errorlevel 1 (
        echo ❌ 虛擬環境創建失敗
        pause
        exit /b 1
    )
    echo ✅ 虛擬環境創建完成
)

:: 啟動虛擬環境
echo.
echo [4/6] 啟動虛擬環境...
call %VENV_NAME%\Scripts\activate.bat
echo ✅ 虛擬環境已啟動

:: 檢查並安裝依賴
echo.
echo [5/6] 檢查 Python 依賴套件...
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
echo [6/6] 啟動後端 API 服務...
echo 🚀 正在啟動 API 服務，請稍候...
echo.
echo ================================================================
echo    API 服務將在 http://localhost:8000 上運行
echo    虛擬環境: %VENV_NAME%
echo    按 Ctrl+C 可停止服務
echo ================================================================
echo.

python -m backend.api_service

echo.
echo 後端服務已停止
echo 虛擬環境將保持啟動狀態
pause