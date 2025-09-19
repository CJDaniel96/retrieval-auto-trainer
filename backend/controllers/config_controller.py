"""
Config Controller - 配置管理相關API端點
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from ..controllers.base_controller import BaseController
from ..services import get_config_service
from ..models.schemas.training import ConfigUpdateRequest, SystemConfigUpdateRequest
from ..views.response_formatter import ResponseFormatter


class ConfigController(BaseController):
    """
    配置管理控制器
    """

    def __init__(self):
        super().__init__()
        self.config_service = get_config_service()
        self.router = APIRouter(prefix="/config", tags=["Configuration"])
        self._setup_routes()

    def _setup_routes(self):
        """設置路由"""

        @self.router.get("/current")
        async def get_current_config():
            """取得當前訓練配置"""
            try:
                config = await self.config_service.get_current_config()

                return ResponseFormatter.success(
                    data=config,
                    message="配置獲取成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="CONFIG_GET_ERROR",
                    status_code=500
                )

        @self.router.post("/update/training")
        async def update_training_config(request: ConfigUpdateRequest):
            """更新訓練配置"""
            try:
                # 將請求轉換為字典格式
                updates = request.model_dump(exclude_none=True)

                updated_config = await self.config_service.update_training_config(
                    updates
                )

                return ResponseFormatter.success(
                    data=updated_config,
                    message="訓練配置更新成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="TRAINING_CONFIG_UPDATE_ERROR",
                    status_code=500
                )

        @self.router.post("/update/system")
        async def update_system_config(request: SystemConfigUpdateRequest):
            """更新系統配置"""
            try:
                # 將請求轉換為字典格式
                updates = request.model_dump(exclude_none=True)

                updated_config = await self.config_service.update_system_config(
                    updates
                )

                return ResponseFormatter.success(
                    data=updated_config,
                    message="系統配置更新成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="SYSTEM_CONFIG_UPDATE_ERROR",
                    status_code=500
                )

        @self.router.get("/database/sites")
        async def get_database_sites():
            """取得可用的資料庫站點"""
            try:
                sites = await self.config_service.get_database_sites()

                return ResponseFormatter.success(
                    data=sites,
                    message="資料庫站點獲取成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="DATABASE_SITES_ERROR",
                    status_code=500
                )

        @self.router.get("/default")
        async def get_default_config():
            """取得預設配置"""
            try:
                # TODO: 實現預設配置獲取邏輯
                default_config = {
                    "training": {
                        "max_epochs": 100,
                        "batch_size": 32,
                        "learning_rate": 0.001,
                        "patience": 10,
                        "enable_early_stopping": True
                    },
                    "model": {
                        "backbone": "efficientnet_v2_s",
                        "embedding_dim": 512,
                        "dropout": 0.1
                    },
                    "data": {
                        "image_size": 224,
                        "augmentation": True,
                        "val_split": 0.2
                    }
                }

                return ResponseFormatter.success(
                    data=default_config,
                    message="預設配置獲取成功"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="DEFAULT_CONFIG_ERROR",
                    status_code=500
                )

        @self.router.post("/reset")
        async def reset_to_default():
            """重置為預設配置"""
            try:
                # TODO: 實現重置邏輯
                # 這個功能需要在 ConfigService 中實現
                return ResponseFormatter.success(
                    message="配置已重置為預設值"
                )

            except Exception as e:
                return ResponseFormatter.error(
                    message=str(e),
                    error_code="CONFIG_RESET_ERROR",
                    status_code=500
                )


def get_config_router() -> APIRouter:
    """獲取配置管理路由器"""
    controller = ConfigController()
    return controller.router