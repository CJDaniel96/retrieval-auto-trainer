"""
Training Task Repository
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from .base_repository import MemoryRepository
from ..domain.training_task import TrainingTask, TaskStatus


class TaskRepository(MemoryRepository[TrainingTask]):
    """
    訓練任務Repository
    """

    def __init__(self):
        super().__init__()

    async def get_by_status(self, status: TaskStatus) -> List[TrainingTask]:
        """根據狀態獲取任務"""
        return await self.list_all(status=status)

    async def get_running_tasks(self) -> List[TrainingTask]:
        """獲取正在執行的任務"""
        tasks = await self.list_all()
        return [task for task in tasks if task.is_running]

    async def get_completed_tasks(self) -> List[TrainingTask]:
        """獲取已完成的任務"""
        return await self.get_by_status(TaskStatus.COMPLETED)

    async def get_failed_tasks(self) -> List[TrainingTask]:
        """獲取失敗的任務"""
        return await self.get_by_status(TaskStatus.FAILED)

    async def get_tasks_by_site(self, site: str) -> List[TrainingTask]:
        """根據站點獲取任務"""
        return await self.list_all(site=site)

    async def get_tasks_by_line(self, line_id: str) -> List[TrainingTask]:
        """根據產線獲取任務"""
        return await self.list_all(line_id=line_id)

    async def get_recent_tasks(self, days: int = 7) -> List[TrainingTask]:
        """獲取最近的任務"""
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)

        tasks = await self.list_all()
        return [
            task for task in tasks
            if task.start_time and task.start_time >= cutoff_date
        ]

    async def cleanup_old_tasks(self, days: int = 30) -> int:
        """清理舊任務"""
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days)

        tasks_to_delete = []
        for task_id, task in self._storage.items():
            if (task.end_time and task.end_time < cutoff_date and
                task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]):
                tasks_to_delete.append(task_id)

        for task_id in tasks_to_delete:
            await self.delete(task_id)

        return len(tasks_to_delete)

    async def get_task_statistics(self) -> Dict[str, Any]:
        """獲取任務統計"""
        all_tasks = await self.list_all()

        total_tasks = len(all_tasks)
        if total_tasks == 0:
            return {
                'total_tasks': 0,
                'by_status': {},
                'by_site': {},
                'success_rate': 0.0,
                'average_duration': 0.0
            }

        # 按狀態統計
        status_counts = {}
        for task in all_tasks:
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # 按站點統計
        site_counts = {}
        for task in all_tasks:
            site = task.site
            site_counts[site] = site_counts.get(site, 0) + 1

        # 計算成功率
        completed_count = status_counts.get(TaskStatus.COMPLETED.value, 0)
        failed_count = status_counts.get(TaskStatus.FAILED.value, 0)
        finished_count = completed_count + failed_count
        success_rate = completed_count / finished_count if finished_count > 0 else 0.0

        # 計算平均持續時間
        durations = [
            task.duration_seconds for task in all_tasks
            if task.duration_seconds is not None
        ]
        average_duration = sum(durations) / len(durations) if durations else 0.0

        return {
            'total_tasks': total_tasks,
            'by_status': status_counts,
            'by_site': site_counts,
            'success_rate': success_rate,
            'average_duration': average_duration
        }

    def _apply_filters(self, entities: List[TrainingTask], filters: Dict[str, Any]) -> List[TrainingTask]:
        """應用過濾器"""
        filtered = entities

        for key, value in filters.items():
            if key == 'status' and isinstance(value, TaskStatus):
                filtered = [e for e in filtered if e.status == value]
            elif hasattr(TrainingTask, key):
                filtered = [e for e in filtered if getattr(e, key, None) == value]

        return filtered


# 單例模式的Repository實例
_task_repository_instance: Optional[TaskRepository] = None


def get_task_repository() -> TaskRepository:
    """獲取任務Repository實例"""
    global _task_repository_instance
    if _task_repository_instance is None:
        _task_repository_instance = TaskRepository()
    return _task_repository_instance