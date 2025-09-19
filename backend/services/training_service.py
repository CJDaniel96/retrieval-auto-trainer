"""
Training Service - 訓練服務業務邏輯
"""

from typing import Dict, Optional, List, Any
from .base_service import BaseService
from ..models.domain import TrainingTask, TaskStatus
from ..models.repositories import get_task_repository


class TrainingService(BaseService):
    """
    訓練服務 - 重構自AutoTrainingSystem
    """

    def __init__(self):
        super().__init__()
        self.task_repository = get_task_repository()

    async def create_training_task(self, **kwargs) -> TrainingTask:
        """創建訓練任務"""
        # TODO: 實現創建邏輯
        pass

    async def start_training(self, task_id: str) -> bool:
        """開始訓練流程"""
        # TODO: 實現訓練邏輯
        pass

    async def get_task_status(self, task_id: str) -> Optional[TrainingTask]:
        """獲取任務狀態"""
        return await self.task_repository.get_by_id(task_id)

    async def list_tasks(self) -> List[TrainingTask]:
        """列出所有任務"""
        return await self.task_repository.list_all()

    async def cancel_task(self, task_id: str) -> bool:
        """取消任務"""
        # TODO: 實現取消邏輯
        task = await self.task_repository.get_by_id(task_id)
        if task and task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task.status = TaskStatus.CANCELLED
            await self.task_repository.update(task)
            return True
        return False

    async def get_task_statistics(self) -> Dict[str, Any]:
        """獲取任務統計"""
        tasks = await self.list_tasks()
        stats = {
            "total": len(tasks),
            "pending": len([t for t in tasks if t.status == TaskStatus.PENDING]),
            "running": len([t for t in tasks if t.status == TaskStatus.RUNNING]),
            "completed": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "cancelled": len([t for t in tasks if t.status == TaskStatus.CANCELLED])
        }
        return stats


_training_service_instance: Optional[TrainingService] = None

def get_training_service() -> TrainingService:
    """獲取訓練服務實例"""
    global _training_service_instance
    if _training_service_instance is None:
        _training_service_instance = TrainingService()
    return _training_service_instance