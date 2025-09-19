"""
Training相關的Pydantic模型定義
"""

from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """實驗配置模型"""
    name: Optional[str] = None


class TrainingConfigFields(BaseModel):
    """訓練配置字段模型"""
    min_epochs: Optional[int] = None
    max_epochs: Optional[int] = None
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    batch_size: Optional[int] = None
    freeze_backbone_epochs: Optional[int] = None
    patience: Optional[int] = None
    enable_early_stopping: Optional[bool] = None
    checkpoint_dir: Optional[str] = None


class ModelConfig(BaseModel):
    """模型配置模型"""
    structure: Optional[str] = None
    backbone: Optional[str] = None
    pretrained: Optional[bool] = None
    embedding_size: Optional[int] = None


class DataConfig(BaseModel):
    """數據配置模型"""
    data_dir: Optional[str] = None
    image_size: Optional[int] = None
    num_workers: Optional[int] = None
    test_split: Optional[float] = None


class LossConfig(BaseModel):
    """損失函數配置模型"""
    type: Optional[str] = None
    subcenter_margin: Optional[float] = None
    subcenter_scale: Optional[float] = None
    sub_centers: Optional[int] = None
    triplet_margin: Optional[float] = None
    center_loss_weight: Optional[float] = None


class KnnConfig(BaseModel):
    """KNN配置模型"""
    enable: Optional[bool] = None
    threshold: Optional[float] = None
    index_path: Optional[str] = None
    dataset_pkl: Optional[str] = None


class TrainingRequest(BaseModel):
    """訓練請求模型"""
    input_dir: str = Field(..., description="輸入資料夾路徑")
    site: str = Field(default="HPH", description="地區名稱")
    line_id: str = Field(default="V31", description="產線ID")

    # 支援從資料庫分類的影像
    use_database_classification: bool = Field(default=False, description="是否使用資料庫中已分類的影像")
    part_numbers: Optional[List[str]] = Field(None, description="使用資料庫分類時的料號列表")

    # 完整的配置覆蓋支持
    experiment_config: Optional[ExperimentConfig] = Field(None, description="實驗配置")
    training_config: Optional[TrainingConfigFields] = Field(None, description="訓練配置")
    model_config_override: Optional[ModelConfig] = Field(None, description="模型配置")
    data_config: Optional[DataConfig] = Field(None, description="數據配置")
    loss_config: Optional[LossConfig] = Field(None, description="損失函數配置")
    knn_config: Optional[KnnConfig] = Field(None, description="KNN配置")

    # 保持向後兼容性的舊字段
    max_epochs: Optional[int] = Field(None, description="最大訓練輪數")
    batch_size: Optional[int] = Field(None, description="批次大小")
    learning_rate: Optional[float] = Field(None, description="學習率")
    patience: Optional[int] = Field(None, description="EarlyStopping耐心值")
    enable_early_stopping: Optional[bool] = Field(None, description="是否啟用提前停止")


class TrainingStatus(BaseModel):
    """訓練狀態模型"""
    task_id: str
    status: str  # pending, pending_orientation, running, completed, failed
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    current_step: Optional[str]
    progress: Optional[float]
    error_message: Optional[str]
    output_dir: Optional[str]
    input_dir: Optional[str]


class TrainingResult(BaseModel):
    """訓練結果模型"""
    task_id: str
    accuracy: float
    total_classes: int
    total_images: int
    model_path: str
    evaluation_csv: str
    confusion_matrix: str


class ConfigUpdateRequest(BaseModel):
    """配置更新請求模型"""
    training: Optional[dict] = Field(None, description="訓練配置")
    model: Optional[dict] = Field(None, description="模型配置")
    data: Optional[dict] = Field(None, description="數據配置")
    loss: Optional[dict] = Field(None, description="損失函數配置")


class SystemConfigUpdateRequest(BaseModel):
    """系統配置更新請求模型"""
    system: Optional[dict] = Field(None, description="系統配置")
    resources: Optional[dict] = Field(None, description="資源配置")
    storage: Optional[dict] = Field(None, description="儲存配置")
    logging: Optional[dict] = Field(None, description="日誌配置")


class CreateModuleRequest(BaseModel):
    """創建模組請求模型"""
    module_name: str = Field(..., description="模組名稱", pattern=r"^[A-Za-z0-9_-]+$")
    part_number: str = Field(..., description="料號", pattern=r"^[A-Za-z0-9_-]+$")