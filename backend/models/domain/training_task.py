"""
Training Task業務領域模型
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """任務狀態枚舉"""
    PENDING = "pending"
    PENDING_ORIENTATION = "pending_orientation"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OrientationType(Enum):
    """方向類型枚舉"""
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"


@dataclass
class TrainingConfig:
    """訓練配置領域模型"""
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 10
    enable_early_stopping: bool = True
    image_size: int = 224
    test_split: float = 0.1
    num_workers: int = 4

    # 模型配置
    model_structure: str = "HOAMV2"
    backbone: str = "efficientnetv2_rw_s"
    embedding_size: int = 512
    pretrained: bool = True

    # 損失函數配置
    loss_type: str = "HybridMarginLoss"
    subcenter_margin: float = 0.4
    subcenter_scale: float = 30.0
    sub_centers: int = 3
    triplet_margin: float = 0.3
    center_loss_weight: float = 0.01


@dataclass
class ClassificationInfo:
    """分類資訊"""
    class_name: str
    image_count: int
    orientation: Optional[OrientationType] = None
    confirmed: bool = False


@dataclass
class TrainingProgress:
    """訓練進度資訊"""
    current_step: str = ""
    progress: float = 0.0
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    estimated_time_remaining: Optional[int] = None


@dataclass
class TrainingResult:
    """訓練結果"""
    accuracy: float
    total_classes: int
    total_images: int
    model_path: str
    evaluation_csv_path: str
    confusion_matrix_path: str
    training_time_seconds: float
    best_epoch: int
    final_loss: float


@dataclass
class TrainingTask:
    """訓練任務領域模型"""
    task_id: str
    input_dir: str
    output_dir: Optional[str] = None
    site: str = "HPH"
    line_id: str = "V31"

    # 任務狀態
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

    # 訓練配置
    config: TrainingConfig = field(default_factory=TrainingConfig)

    # 進度追蹤
    progress: TrainingProgress = field(default_factory=TrainingProgress)

    # 分類資訊
    classifications: Dict[str, ClassificationInfo] = field(default_factory=dict)

    # 訓練結果
    result: Optional[TrainingResult] = None

    # 資料庫模式支援
    use_database_classification: bool = False
    part_numbers: List[str] = field(default_factory=list)

    def start_training(self):
        """開始訓練"""
        self.status = TaskStatus.RUNNING
        self.start_time = datetime.now()
        self.progress.current_step = "開始訓練"

    def complete_training(self, result: TrainingResult):
        """完成訓練"""
        self.status = TaskStatus.COMPLETED
        self.end_time = datetime.now()
        self.progress.progress = 1.0
        self.progress.current_step = "訓練完成"
        self.result = result

    def fail_training(self, error_message: str):
        """訓練失敗"""
        self.status = TaskStatus.FAILED
        self.end_time = datetime.now()
        self.error_message = error_message
        self.progress.current_step = "訓練失敗"

    def require_orientation_confirmation(self):
        """需要方向確認"""
        self.status = TaskStatus.PENDING_ORIENTATION
        self.progress.current_step = "等待方向確認"
        self.progress.progress = 0.2

    def update_progress(self, step: str, progress: float, **kwargs):
        """更新進度"""
        self.progress.current_step = step
        self.progress.progress = progress
        for key, value in kwargs.items():
            if hasattr(self.progress, key):
                setattr(self.progress, key, value)

    def add_classification(self, class_name: str, image_count: int):
        """添加分類資訊"""
        self.classifications[class_name] = ClassificationInfo(
            class_name=class_name,
            image_count=image_count
        )

    def confirm_orientation(self, class_name: str, orientation: OrientationType):
        """確認方向"""
        if class_name in self.classifications:
            self.classifications[class_name].orientation = orientation
            self.classifications[class_name].confirmed = True

    def all_orientations_confirmed(self) -> bool:
        """檢查是否所有方向都已確認"""
        return all(
            cls.confirmed for cls in self.classifications.values()
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """計算任務持續時間"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return None

    @property
    def is_running(self) -> bool:
        """檢查任務是否正在執行"""
        return self.status in [TaskStatus.RUNNING, TaskStatus.PENDING_ORIENTATION]

    @property
    def is_completed(self) -> bool:
        """檢查任務是否已完成"""
        return self.status == TaskStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """檢查任務是否失敗"""
        return self.status == TaskStatus.FAILED