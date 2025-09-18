"""
Logging Middleware - 日誌記錄中間件
"""

import time
import logging
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    請求日誌中間件
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # 記錄請求開始
        logger.info(f"請求開始: {request.method} {request.url}")

        # 處理請求
        response = await call_next(request)

        # 計算處理時間
        process_time = time.time() - start_time

        # 記錄請求完成
        logger.info(
            f"請求完成: {request.method} {request.url} - "
            f"狀態碼: {response.status_code} - "
            f"處理時間: {process_time:.4f}s"
        )

        # 添加處理時間到響應頭
        response.headers["X-Process-Time"] = str(process_time)

        return response


def setup_logging(app: FastAPI) -> None:
    """
    設置日誌中間件
    """
    app.add_middleware(LoggingMiddleware)