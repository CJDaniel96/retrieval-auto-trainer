"""
基礎Service抽象類別
"""

from abc import ABC
from typing import Optional, Callable, Any
import logging


class BaseService(ABC):
    """
    基礎Service抽象類別，提供共用功能
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._progress_callback: Optional[Callable[[str, float], None]] = None

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """設置進度回調函數"""
        self._progress_callback = callback

    def update_progress(self, step: str, progress: float, **kwargs):
        """更新進度"""
        if self._progress_callback:
            self._progress_callback(step, progress)

        # 記錄進度到日誌
        self.logger.info(f"進度更新: {step} ({progress*100:.1f}%)")

    def handle_error(self, error: Exception, operation: str) -> Exception:
        """統一錯誤處理"""
        self.logger.error(f"{operation}時發生錯誤: {str(error)}", exc_info=True)
        return error

    def validate_required_params(self, **params) -> None:
        """驗證必要參數"""
        missing_params = [key for key, value in params.items() if value is None]
        if missing_params:
            raise ValueError(f"缺少必要參數: {', '.join(missing_params)}")


class AsyncBaseService(BaseService):
    """
    異步基礎Service類別
    """

    async def async_update_progress(self, step: str, progress: float, **kwargs):
        """異步更新進度"""
        self.update_progress(step, progress, **kwargs)

    async def async_handle_error(self, error: Exception, operation: str) -> Exception:
        """異步錯誤處理"""
        return self.handle_error(error, operation)