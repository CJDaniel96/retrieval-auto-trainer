@echo off
REM ==========================================
REM 自動化影像檢索模型訓練系統 - Windows 部署腳本
REM ==========================================

echo Starting deployment process...

REM 檢查 Node.js 是否安裝
echo.
echo [1/4] Checking Node.js...
node --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/
    pause
    exit /b 1
)

REM 檢查 Python 是否安裝
echo.
echo [2/4] Checking Python...
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed. Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

REM 安裝前端依賴並構建
echo.
echo [3/4] Building frontend...
cd frontend
if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install frontend dependencies
        pause
        exit /b 1
    )
)

echo Building frontend for production...
set NODE_ENV=production
npm run build
if %ERRORLEVEL% neq 0 (
    echo ERROR: Frontend build failed
    pause
    exit /b 1
)
cd ..

REM 檢查 Python 虛擬環境
echo.
echo [4/4] Setting up Python environment...
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM 激活虛擬環境並安裝依賴
call venv\Scripts\activate.bat
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

REM 創建必要的目錄
if not exist "datasets" mkdir datasets
if not exist "outputs" mkdir outputs
if not exist "rawdata" mkdir rawdata
if not exist "modules" mkdir modules
if not exist "logs" mkdir logs
if not exist "temp_uploads" mkdir temp_uploads

echo.
echo ==========================================
echo Deployment completed successfully!
echo ==========================================
echo.
echo To start the application:
echo   run_server.bat
echo.
echo The application will be available at:
echo   http://localhost:8000
echo.
pause