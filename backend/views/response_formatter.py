"""
統一響應格式化器
"""

from typing import Any, Dict, Optional
from datetime import datetime
from fastapi.responses import JSONResponse
import json


class ResponseFormatter:
    """
    統一API響應格式化器
    """

    @staticmethod
    def success(
        data: Any = None,
        message: str = "Success",
        status_code: int = 200,
        meta: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        成功響應格式

        Args:
            data: 響應數據
            message: 響應訊息
            status_code: HTTP狀態碼
            meta: 元數據 (分頁信息等)

        Returns:
            JSONResponse
        """
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

        if data is not None:
            response["data"] = data

        if meta is not None:
            response["meta"] = meta

        return JSONResponse(
            status_code=status_code,
            content=response
        )

    @staticmethod
    def error(
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 400,
        details: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """
        錯誤響應格式

        Args:
            message: 錯誤訊息
            error_code: 錯誤代碼
            status_code: HTTP狀態碼
            details: 錯誤詳情

        Returns:
            JSONResponse
        """
        response = {
            "success": False,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

        if error_code:
            response["error_code"] = error_code

        if details:
            response["details"] = details

        return JSONResponse(
            status_code=status_code,
            content=response
        )

    @staticmethod
    def paginated(
        data: list,
        total: int,
        page: int,
        page_size: int,
        message: str = "Success"
    ) -> JSONResponse:
        """
        分頁響應格式

        Args:
            data: 當前頁數據
            total: 總記錄數
            page: 當前頁碼
            page_size: 每頁大小
            message: 響應訊息

        Returns:
            JSONResponse
        """
        total_pages = (total + page_size - 1) // page_size

        meta = {
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }

        return ResponseFormatter.success(
            data=data,
            message=message,
            meta=meta
        )


class CustomJSONEncoder(json.JSONEncoder):
    """
    自定義JSON編碼器，處理特殊類型
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        # 可以添加其他類型的處理
        return super().default(obj)