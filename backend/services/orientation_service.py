"""
Orientation Service - 方向確認服務業務邏輯
"""

from typing import List, Dict, Optional, Any
from .base_service import BaseService
from ..models.repositories import get_task_repository
from ..models.schemas.orientation import OrientationSample


class OrientationService(BaseService):
    """
    方向確認服務
    """

    def __init__(self):
        super().__init__()
        self.task_repository = get_task_repository()

    async def get_orientation_samples(self, task_id: str) -> List[OrientationSample]:
        """獲取方向確認的樣本影像"""
        # TODO: 實現獲取樣本邏輯
        pass

    async def confirm_all_orientations(self, task_id: str, orientations: Dict[str, str]) -> Dict[str, Any]:
        """確認所有方向並繼續訓練流程"""
        # TODO: 實現確認邏輯
        pass


_orientation_service_instance: Optional[OrientationService] = None

def get_orientation_service() -> OrientationService:
    """獲取方向確認服務實例"""
    global _orientation_service_instance
    if _orientation_service_instance is None:
        _orientation_service_instance = OrientationService()
    return _orientation_service_instance