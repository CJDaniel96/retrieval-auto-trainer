"""
Orientation相關的Pydantic模型定義
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class OrientationSample(BaseModel):
    """方向確認樣本模型"""
    class_name: str
    sample_images: List[str]  # 3張樣本影像的路徑
    current_orientation: Optional[str] = None  # 當前已保存的方向選擇


class OrientationConfirmation(BaseModel):
    """方向確認請求模型"""
    task_id: str
    orientations: Dict[str, str]  # class_name -> orientation (Up, Down, Left, Right)


class PartialOrientationSave(BaseModel):
    """部分方向保存請求模型"""
    task_id: str
    class_name: str
    orientation: str  # Up, Down, Left, Right


class OrientationStatus(BaseModel):
    """方向確認狀態模型"""
    task_id: str
    total_classes: int
    completed_classes: int
    completion_rate: float
    class_status: Dict[str, Dict[str, Any]]  # 各類別的狀態詳情