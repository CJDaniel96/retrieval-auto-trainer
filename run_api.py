#!/usr/bin/env python3
"""
API Server Launcher - 快速啟動API服務器
"""

if __name__ == "__main__":
    from backend.main import run_api_server

    print("Starting Retrieval Auto Trainer API Server...")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")

    run_api_server(
        host="0.0.0.0",
        port=8000,
        reload=True  # 開發模式
    )