"""
基礎控制器類 - 提供共用的控制器功能
"""

from abc import ABC
from typing import Dict, Any, Optional
from fastapi import HTTPException
from fastapi.responses import JSONResponse
import logging


class BaseController(ABC):
    """
    基礎控制器類，提供共用功能
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def success_response(self, data: Any = None, message: str = "Success") -> Dict[str, Any]:
        """
        成功響應格式

        Args:
            data: 響應數據
            message: 響應訊息

        Returns:
            標準成功響應格式
        """
        response = {
            "success": True,
            "message": message
        }
        if data is not None:
            response["data"] = data
        return response

    def error_response(self, message: str, error_code: str = None, status_code: int = 400) -> HTTPException:
        """
        錯誤響應格式

        Args:
            message: 錯誤訊息
            error_code: 錯誤代碼
            status_code: HTTP狀態碼

        Returns:
            HTTPException
        """
        detail = {
            "success": False,
            "message": message
        }
        if error_code:
            detail["error_code"] = error_code

        return HTTPException(status_code=status_code, detail=detail)

    def handle_exception(self, e: Exception, operation: str) -> HTTPException:
        """
        統一異常處理

        Args:
            e: 異常對象
            operation: 操作描述

        Returns:
            HTTPException
        """
        self.logger.error(f"{operation}時發生錯誤: {str(e)}", exc_info=True)
        return self.error_response(
            message=f"{operation}失敗: {str(e)}",
            error_code="INTERNAL_ERROR",
            status_code=500
        )

    def validate_required_fields(self, data: Dict[str, Any], required_fields: list) -> Optional[HTTPException]:
        """
        驗證必填欄位

        Args:
            data: 要驗證的數據
            required_fields: 必填欄位列表

        Returns:
            如果驗證失敗返回HTTPException，否則返回None
        """
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]

        if missing_fields:
            return self.error_response(
                message=f"缺少必填欄位: {', '.join(missing_fields)}",
                error_code="MISSING_REQUIRED_FIELDS",
                status_code=400
            )
        return None