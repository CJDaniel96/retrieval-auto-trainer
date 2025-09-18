"""
Main Entry Point - 主要應用程式入口
"""

import argparse
import logging
import uvicorn
from pathlib import Path


def setup_logging():
    """設置日誌配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def run_cli():
    """運行CLI模式"""
    from .core.auto_training_system import main as cli_main
    cli_main()


def run_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """運行API服務器"""
    setup_logging()

    uvicorn.run(
        "backend.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


def run_training_module():
    """運行訓練模組"""
    from .train import main as train_main
    train_main()


def main():
    """主入口函數"""
    parser = argparse.ArgumentParser(description="Retrieval Auto Trainer")
    parser.add_argument(
        "--mode",
        choices=["api", "cli", "train"],
        default="cli",
        help="Run mode: api (API server), cli (command line), train (training module)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="API server host address")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development mode)")

    # 解析已知參數，剩餘參數傳遞給CLI或訓練模組
    args, unknown = parser.parse_known_args()

    if args.mode == "api":
        print(f"Starting API server on {args.host}:{args.port}")
        run_api_server(args.host, args.port, args.reload)
    elif args.mode == "cli":
        print("Starting CLI mode")
        run_cli()
    elif args.mode == "train":
        print("Starting training module")
        run_training_module()


if __name__ == '__main__':
    main()