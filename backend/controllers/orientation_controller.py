"""
Orientation Controller - 方向確認相關API端點
"""

from typing import Dict, List
from fastapi import APIRouter, HTTPException
from ..controllers.base_controller import BaseController
from ..services import get_orientation_service
from ..models.schemas.orientation import OrientationSample
from ..views.response_formatter import ResponseFormatter


class OrientationController(BaseController):
    """
    方向確認控制器
    """

    def __init__(self):
        super().__init__()
        self.orientation_service = get_orientation_service()
        self.router = APIRouter(prefix="/orientation", tags=["Orientation"])
        self._setup_routes()

    def _setup_routes(self):
        """設置路由"""

        @self.router.get("/samples/{task_id}")
        async def get_orientation_samples(task_id: str):
            """獲取方向確認的樣本影像"""
            try:
                # 驗證任務存在
                validation_error = self.validate_required_fields(
                    {"task_id": task_id},
                    ["task_id"]
                )
                if validation_error:
                    raise validation_error

                samples = await self.orientation_service.get_orientation_samples(task_id)

                if not samples:
                    return ResponseFormatter.error(
                        message=f"找不到任務 {task_id} 的樣本影像",
                        error_code="SAMPLES_NOT_FOUND",
                        status_code=404
                    )

                # 轉換為響應格式
                sample_data = []
                for sample in samples:
                    sample_data.append({
                        "class_name": sample.class_name,
                        "image_paths": sample.image_paths,
                        "temp_urls": sample.temp_urls
                    })

                return ResponseFormatter.success(
                    data=sample_data,
                    message="樣本影像獲取成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="SAMPLES_QUERY_ERROR",
                    status_code=500
                )

        @self.router.post("/confirm/{task_id}")
        async def confirm_orientations(task_id: str, orientations: Dict[str, str]):
            """確認所有方向並繼續訓練流程"""
            try:
                # 驗證請求
                validation_error = self.validate_required_fields(
                    {"task_id": task_id, "orientations": orientations},
                    ["task_id", "orientations"]
                )
                if validation_error:
                    raise validation_error

                # 驗證方向值
                valid_orientations = {"Up", "Down", "Left", "Right"}
                for class_name, orientation in orientations.items():
                    if orientation not in valid_orientations:
                        return ResponseFormatter.error(
                            message=f"無效的方向值: {orientation}。有效值: {valid_orientations}",
                            error_code="INVALID_ORIENTATION",
                            status_code=400
                        )

                result = await self.orientation_service.confirm_all_orientations(
                    task_id, orientations
                )

                return ResponseFormatter.success(
                    data=result,
                    message="方向確認成功，訓練將繼續進行"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="ORIENTATION_CONFIRM_ERROR",
                    status_code=500
                )


def get_orientation_router() -> APIRouter:
    """獲取方向確認路由器"""
    controller = OrientationController()
    return controller.router