"""
Download相關的Pydantic模型定義
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class DownloadRequest(BaseModel):
    """下載請求模型"""
    site: str = Field(..., description="工廠名稱", pattern="^(HPH|JQ|ZJ|NK|HZ)$")
    line_id: str = Field(..., description="線別ID")
    start_date: str = Field(..., description="開始日期", pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., description="結束日期", pattern=r"^\d{4}-\d{2}-\d{2}$")
    part_number: str = Field(..., description="料號")
    limit: Optional[int] = Field(None, description="限制數量", ge=1, le=10000)


class PartInfo(BaseModel):
    """料號資訊模型"""
    part_number: str
    path: str
    image_count: int
    download_time: str


class ClassifyRequest(BaseModel):
    """分類請求模型"""
    part_number: str = Field(..., description="料號")
    classifications: Dict[str, str] = Field(..., description="影像分類結果 {filename: 'OK'|'NG'}")


class DownloadEstimate(BaseModel):
    """下載預估模型"""
    estimated_count: int
    estimated_size_mb: float
    time_range: str
    site: str
    line_id: str
    part_number: str


class DownloadResult(BaseModel):
    """下載結果模型"""
    success: bool
    downloaded_count: int
    total_size_mb: float
    download_path: str
    part_number: str
    errors: List[str] = []


class ClassifyResult(BaseModel):
    """分類結果模型"""
    success: bool
    moved_count: int
    total_count: int
    errors: List[str] = []


class ImageInfo(BaseModel):
    """影像資訊模型"""
    filename: str
    path: str
    size: int
    base64_data: Optional[str] = None


class ImageListResponse(BaseModel):
    """影像列表響應模型"""
    part_number: str
    total_images: int
    images: List[ImageInfo]