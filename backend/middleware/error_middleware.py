"""
Error Handling Middleware - 錯誤處理中間件
"""

import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from ..views.response_formatter import ResponseFormatter

logger = logging.getLogger(__name__)


def setup_error_handling(app: FastAPI) -> None:
    """
    設置全域錯誤處理中間件
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """HTTP例外處理"""
        logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")

        return JSONResponse(
            status_code=exc.status_code,
            content=ResponseFormatter.error(
                message=str(exc.detail),
                error_code="HTTP_ERROR",
                status_code=exc.status_code
            )
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """請求驗證錯誤處理"""
        logger.error(f"Validation error: {exc.errors()}")

        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })

        return JSONResponse(
            status_code=422,
            content=ResponseFormatter.error(
                message="請求驗證失敗",
                error_code="VALIDATION_ERROR",
                status_code=422,
                details=error_details
            )
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """一般例外處理"""
        logger.error(f"Unexpected error: {type(exc).__name__} - {str(exc)}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content=ResponseFormatter.error(
                message="內部伺服器錯誤",
                error_code="INTERNAL_SERVER_ERROR",
                status_code=500
            )
        )