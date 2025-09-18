"""
Common通用的Pydantic模型定義
"""

from typing import Any, Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel


class SuccessResponse(BaseModel):
    """成功響應基礎模型"""
    success: bool = True
    message: str
    data: Optional[Any] = None
    timestamp: datetime = datetime.now()


class ErrorResponse(BaseModel):
    """錯誤響應基礎模型"""
    success: bool = False
    message: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()


class PaginationMeta(BaseModel):
    """分頁元數據模型"""
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedResponse(BaseModel):
    """分頁響應模型"""
    success: bool = True
    message: str
    data: List[Any]
    meta: PaginationMeta
    timestamp: datetime = datetime.now()


class HealthCheckResponse(BaseModel):
    """健康檢查響應模型"""
    status: str
    service: str
    version: str
    timestamp: datetime = datetime.now()


class FileDownloadInfo(BaseModel):
    """檔案下載資訊模型"""
    file_type: str
    file_size: int
    download_url: str
    expires_at: Optional[datetime] = None


class TaskProgress(BaseModel):
    """任務進度模型"""
    task_id: str
    current_step: str
    progress: float  # 0.0 to 1.0
    estimated_time_remaining: Optional[int] = None  # seconds
    details: Optional[Dict[str, Any]] = None