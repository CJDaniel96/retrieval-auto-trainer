"""
Download Service - 下載服務業務邏輯
"""

from typing import List, Dict, Optional, Any
from .base_service import BaseService
from ..infrastructure.storage import ImageDownloadService
from ..models.schemas.download import DownloadRequest, DownloadResult


class DownloadService(BaseService):
    """
    下載服務 - 封裝影像下載相關業務邏輯
    """

    def __init__(self):
        super().__init__()
        self.image_downloader = ImageDownloadService()

    async def download_rawdata(self, request: DownloadRequest) -> DownloadResult:
        """下載原始資料"""
        # TODO: 實現下載邏輯
        pass

    async def list_downloaded_parts(self) -> List[Dict[str, Any]]:
        """列出已下載的料號"""
        # TODO: 實現從實際存儲中獲取已下載料號的邏輯
        # 目前返回示例數據
        return [
            {
                "part_number": "P001",
                "name": "示例料號 1",
                "site": "HPH",
                "line_id": "V31",
                "download_date": "2025-09-18T10:00:00Z",
                "image_count": 150,
                "status": "completed"
            },
            {
                "part_number": "P002",
                "name": "示例料號 2",
                "site": "HPH",
                "line_id": "V32",
                "download_date": "2025-09-18T11:00:00Z",
                "image_count": 200,
                "status": "completed"
            }
        ]


_download_service_instance: Optional[DownloadService] = None

def get_download_service() -> DownloadService:
    """獲取下載服務實例"""
    global _download_service_instance
    if _download_service_instance is None:
        _download_service_instance = DownloadService()
    return _download_service_instance