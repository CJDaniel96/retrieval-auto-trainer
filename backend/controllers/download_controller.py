"""
Download Controller - 下載相關API端點
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from ..controllers.base_controller import BaseController
from ..services import get_download_service
from ..models.schemas.download import DownloadRequest, DownloadResult
from ..views.response_formatter import ResponseFormatter


class DownloadController(BaseController):
    """
    下載控制器
    """

    def __init__(self):
        super().__init__()
        self.download_service = get_download_service()
        self.router = APIRouter(prefix="/download", tags=["Download"])
        self._setup_routes()

    def _setup_routes(self):
        """設置路由"""

        @self.router.post("/rawdata")
        async def download_rawdata(request: DownloadRequest):
            """下載原始資料"""
            try:
                # 驗證請求
                validation_error = self.validate_required_fields({
                    "site": request.site,
                    "line_id": request.line_id,
                    "part_numbers": request.part_numbers,
                    "output_dir": request.output_dir
                })
                if validation_error:
                    raise validation_error

                result = await self.download_service.download_rawdata(request)

                return ResponseFormatter.success(
                    data=result.dict(),
                    message="原始資料下載完成"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="DOWNLOAD_ERROR",
                    status_code=500
                )

        @self.router.get("/parts/list")
        async def list_downloaded_parts():
            """列出已下載的料號"""
            try:
                parts = await self.download_service.list_downloaded_parts()

                return ResponseFormatter.success(
                    data=parts,
                    message="已下載料號列表獲取成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="PARTS_LIST_ERROR",
                    status_code=500
                )

        @self.router.get("/status/{download_id}")
        async def get_download_status(download_id: str):
            """查詢下載狀態"""
            try:
                # TODO: 實現下載狀態查詢
                # 這個功能需要在 DownloadService 中實現
                return ResponseFormatter.success(
                    data={"download_id": download_id, "status": "completed"},
                    message="下載狀態查詢成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="DOWNLOAD_STATUS_ERROR",
                    status_code=500
                )


def get_download_router() -> APIRouter:
    """獲取下載路由器"""
    controller = DownloadController()
    return controller.router