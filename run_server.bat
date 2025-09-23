@echo off
REM ==========================================
REM 啟動自動化影像檢索模型訓練系統
REM ==========================================

echo Starting Auto Training System...

REM 檢查前端是否已構建
if not exist "frontend\out" (
    echo ERROR: Frontend not built. Please run deploy.bat first.
    echo Or manually build with: cd frontend && npm run build
    pause
    exit /b 1
)

REM 檢查虛擬環境
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Python virtual environment not found. Please run deploy.bat first.
    pause
    exit /b 1
)

REM 激活虛擬環境
echo Activating Python virtual environment...
call venv\Scripts\activate.bat

REM 設置環境變量
set PYTHONPATH=%CD%

REM 啟動後端服務
echo.
echo ==========================================
echo Starting backend server...
echo Application will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo ==========================================
echo.

python -m backend.api.api_service

pause