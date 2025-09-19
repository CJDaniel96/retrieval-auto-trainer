"""
Training Controller - 訓練相關API端點
"""

from typing import Dict, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from ..controllers.base_controller import BaseController
from ..services import get_training_service
from ..models.schemas.training import (
    TrainingRequest, TrainingStatus, TrainingResult,
    ConfigUpdateRequest, CreateModuleRequest
)
from ..views.response_formatter import ResponseFormatter


class TrainingController(BaseController):
    """
    訓練控制器
    """

    def __init__(self):
        super().__init__()
        self.training_service = get_training_service()
        self.router = APIRouter(prefix="/training", tags=["Training"])
        self._setup_routes()

    def _setup_routes(self):
        """設置路由"""

        @self.router.post("/start")
        async def start_training(
            request: TrainingRequest,
            background_tasks: BackgroundTasks
        ):
            """啟動新的訓練任務"""
            try:
                # 驗證請求
                validation_error = self.validate_required_fields(
                    {"input_dir": request.input_dir},
                    ["input_dir"]
                )
                if validation_error:
                    raise validation_error

                # 創建任務
                task = await self.training_service.create_training_task(
                    input_dir=request.input_dir,
                    site=request.site,
                    line_id=request.line_id,
                    use_database_classification=request.use_database_classification,
                    part_numbers=request.part_numbers,
                    config_overrides=self._build_config_overrides(request)
                )

                # 在背景開始訓練
                background_tasks.add_task(
                    self.training_service.start_training,
                    task.task_id
                )

                return ResponseFormatter.success(
                    data={"task_id": task.task_id},
                    message="訓練任務已啟動"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="TRAINING_START_ERROR",
                    status_code=500
                )

        @self.router.get("/status/{task_id}")
        async def get_training_status(task_id: str):
            """查詢訓練任務狀態"""
            try:
                task = await self.training_service.get_task_status(task_id)
                if not task:
                    return ResponseFormatter.error(
                        message=f"找不到任務: {task_id}",
                        error_code="TASK_NOT_FOUND",
                        status_code=404
                    )

                # 轉換為響應模型
                status = TrainingStatus(
                    task_id=task.task_id,
                    status=task.status.value,
                    start_time=task.start_time,
                    end_time=task.end_time,
                    current_step=task.progress.current_step,
                    progress=task.progress.progress,
                    error_message=task.error_message,
                    output_dir=task.output_dir,
                    input_dir=task.input_dir
                )

                return ResponseFormatter.success(
                    data=status.dict(),
                    message="任務狀態查詢成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="STATUS_QUERY_ERROR",
                    status_code=500
                )

        @self.router.get("/list")
        async def list_training_tasks():
            """列出所有訓練任務"""
            try:
                tasks = await self.training_service.list_tasks()

                task_list = []
                for task in tasks:
                    task_list.append({
                        "task_id": task.task_id,
                        "status": task.status.value,
                        "start_time": task.start_time,
                        "end_time": task.end_time,
                        "current_step": task.progress.current_step,
                        "progress": task.progress.progress,
                        "site": task.site,
                        "line_id": task.line_id
                    })

                return ResponseFormatter.success(
                    data=task_list,
                    message="任務列表查詢成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="TASK_LIST_ERROR",
                    status_code=500
                )

        @self.router.get("/result/{task_id}")
        async def get_training_result(task_id: str):
            """取得訓練結果"""
            try:
                task = await self.training_service.get_task_status(task_id)
                if not task:
                    return ResponseFormatter.error(
                        message=f"找不到任務: {task_id}",
                        error_code="TASK_NOT_FOUND",
                        status_code=404
                    )

                if not task.is_completed:
                    return ResponseFormatter.error(
                        message=f"任務尚未完成: {task.status.value}",
                        error_code="TASK_NOT_COMPLETED",
                        status_code=400
                    )

                if not task.result:
                    return ResponseFormatter.error(
                        message="找不到訓練結果",
                        error_code="RESULT_NOT_FOUND",
                        status_code=404
                    )

                result = TrainingResult(
                    task_id=task_id,
                    accuracy=task.result.accuracy,
                    total_classes=task.result.total_classes,
                    total_images=task.result.total_images,
                    model_path=task.result.model_path,
                    evaluation_csv=task.result.evaluation_csv_path,
                    confusion_matrix=task.result.confusion_matrix_path
                )

                return ResponseFormatter.success(
                    data=result.dict(),
                    message="訓練結果查詢成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="RESULT_QUERY_ERROR",
                    status_code=500
                )

        @self.router.post("/cancel/{task_id}")
        async def cancel_training(task_id: str):
            """取消訓練任務"""
            try:
                success = await self.training_service.cancel_task(task_id)

                if success:
                    return ResponseFormatter.success(
                        message=f"任務 {task_id} 已取消"
                    )
                else:
                    return ResponseFormatter.error(
                        message=f"無法取消任務 {task_id}",
                        error_code="CANCEL_FAILED",
                        status_code=400
                    )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="CANCEL_ERROR",
                    status_code=500
                )

        @self.router.get("/statistics")
        async def get_training_statistics():
            """獲取訓練統計"""
            try:
                stats = await self.training_service.get_task_statistics()

                return ResponseFormatter.success(
                    data=stats,
                    message="統計資料查詢成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="STATISTICS_ERROR",
                    status_code=500
                )

    def _build_config_overrides(self, request: TrainingRequest) -> Dict:
        """構建配置覆蓋"""
        overrides = {}

        # 處理新的配置結構
        if request.experiment_config:
            overrides['experiment'] = request.experiment_config.dict(exclude_none=True)

        if request.training_config:
            overrides['training'] = request.training_config.dict(exclude_none=True)

        if request.model_config_override:
            overrides['model'] = request.model_config_override.dict(exclude_none=True)

        if request.data_config:
            overrides['data'] = request.data_config.dict(exclude_none=True)

        if request.loss_config:
            overrides['loss'] = request.loss_config.dict(exclude_none=True)

        if request.knn_config:
            overrides['knn'] = request.knn_config.dict(exclude_none=True)

        # 處理向後兼容的舊字段
        if request.max_epochs:
            if 'training' not in overrides:
                overrides['training'] = {}
            overrides['training']['max_epochs'] = request.max_epochs

        if request.batch_size:
            if 'training' not in overrides:
                overrides['training'] = {}
            overrides['training']['batch_size'] = request.batch_size

        if request.learning_rate:
            if 'training' not in overrides:
                overrides['training'] = {}
            overrides['training']['lr'] = request.learning_rate

        if request.patience is not None:
            if 'training' not in overrides:
                overrides['training'] = {}
            overrides['training']['patience'] = request.patience

        if request.enable_early_stopping is not None:
            if 'training' not in overrides:
                overrides['training'] = {}
            overrides['training']['enable_early_stopping'] = request.enable_early_stopping

        return overrides


def get_training_router() -> APIRouter:
    """獲取訓練路由器"""
    controller = TrainingController()
    return controller.router