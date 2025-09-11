@echo off
chcp 65001 >nul
title AI 自動化訓練系統 - 後端服務

:: 切換到專案目錄
cd /d "%~dp0"

:: 快速啟動後端服務
echo 🚀 啟動後端 API 服務...
echo 服務地址: http://localhost:8000
echo.

python -m backend.api_service

echo.
echo 服務已停止
pause