"""
Schemas package - Pydantic模型定義
"""

# Training相關模型
from .training import (
    ExperimentConfig, TrainingConfigFields, ModelConfig, DataConfig,
    LossConfig, KnnConfig, TrainingRequest, TrainingStatus, TrainingResult,
    ConfigUpdateRequest, CreateModuleRequest
)

# Orientation相關模型
from .orientation import (
    OrientationSample, OrientationConfirmation, PartialOrientationSave,
    OrientationStatus
)

# Download相關模型
from .download import (
    DownloadRequest, PartInfo, ClassifyRequest, DownloadEstimate,
    DownloadResult, ClassifyResult, ImageInfo, ImageListResponse
)

# 通用模型
from .common import (
    SuccessResponse, ErrorResponse, PaginationMeta, PaginatedResponse,
    HealthCheckResponse, FileDownloadInfo, TaskProgress
)

__all__ = [
    # Training
    'ExperimentConfig', 'TrainingConfigFields', 'ModelConfig', 'DataConfig',
    'LossConfig', 'KnnConfig', 'TrainingRequest', 'TrainingStatus', 'TrainingResult',
    'ConfigUpdateRequest', 'CreateModuleRequest',

    # Orientation
    'OrientationSample', 'OrientationConfirmation', 'PartialOrientationSave',
    'OrientationStatus',

    # Download
    'DownloadRequest', 'PartInfo', 'ClassifyRequest', 'DownloadEstimate',
    'DownloadResult', 'ClassifyResult', 'ImageInfo', 'ImageListResponse',

    # Common
    'SuccessResponse', 'ErrorResponse', 'PaginationMeta', 'PaginatedResponse',
    'HealthCheckResponse', 'FileDownloadInfo', 'TaskProgress'
]