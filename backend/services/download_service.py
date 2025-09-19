"""
Download Service - 下載服務業務邏輯
"""

from typing import List, Dict, Optional, Any
from .base_service import BaseService
from ..infrastructure.storage import ImageDownloadService
from ..models.schemas.download import DownloadRequest, DownloadResult, DownloadEstimate


class DownloadService(BaseService):
    """
    下載服務 - 封裝影像下載相關業務邏輯
    """

    def __init__(self):
        super().__init__()
        self.image_downloader = ImageDownloadService()

    async def estimate_download(self, request: DownloadRequest) -> DownloadEstimate:
        """預估下載數量和大小"""
        try:
            # 使用 ImageDownloadService 的實際方法查詢資料庫
            self.logger.info(f"調用 ImageDownloadService.estimate_data_count: site={request.site}, line_id={request.line_id}, part_number={request.part_number}")
            result = self.image_downloader.estimate_data_count(
                site=request.site,
                line_id=request.line_id,
                start_date=request.start_date,
                end_date=request.end_date,
                part_number=request.part_number
            )
            self.logger.info(f"ImageDownloadService 返回結果: {result}")

            if not result["success"]:
                raise Exception(result["message"])

            estimated_count = result["estimated_count"]

            # 預估檔案大小 (假設每張影像約 200KB)
            estimated_size_mb = (estimated_count * 200) / 1024  # Convert to MB

            time_range = f"{request.start_date} 至 {request.end_date}"

            return DownloadEstimate(
                estimated_count=estimated_count,
                estimated_size_mb=round(estimated_size_mb, 2),
                time_range=time_range,
                site=request.site,
                line_id=request.line_id,
                part_number=request.part_number
            )

        except Exception as e:
            self.logger.error(f"預估下載失敗: {e}")
            raise Exception(f"預估下載失敗: {str(e)}")

    async def download_rawdata(self, request: DownloadRequest) -> DownloadResult:
        """下載原始資料"""
        try:
            self.logger.info(f"開始下載原始資料: {request.part_number}")

            # 使用 ImageDownloadService 的實際方法進行下載
            result = self.image_downloader.download_rawdata(
                site=request.site,
                line_id=request.line_id,
                start_date=request.start_date,
                end_date=request.end_date,
                part_number=request.part_number,
                limit=request.limit
            )

            if not result["success"]:
                return DownloadResult(
                    success=False,
                    downloaded_count=0,
                    total_size_mb=0.0,
                    download_path="",
                    part_number=request.part_number,
                    errors=[result["message"]]
                )

            # 計算實際檔案大小 (假設每張影像約 200KB)
            downloaded_count = result["image_count"]
            total_size_mb = (downloaded_count * 200) / 1024  # Convert to MB

            return DownloadResult(
                success=True,
                downloaded_count=downloaded_count,
                total_size_mb=round(total_size_mb, 2),
                download_path=result["path"],
                part_number=request.part_number,
                errors=[]
            )

        except Exception as e:
            self.logger.error(f"下載原始資料失敗: {e}")
            return DownloadResult(
                success=False,
                downloaded_count=0,
                total_size_mb=0.0,
                download_path="",
                part_number=request.part_number,
                errors=[str(e)]
            )

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