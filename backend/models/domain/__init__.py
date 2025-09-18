"""
Domain package - 業務領域模型
"""

from .training_task import (
    TrainingTask, TaskStatus, OrientationType, TrainingConfig,
    ClassificationInfo, TrainingProgress, TrainingResult
)
from .image_processor import (
    ImageMetadata, ImageClassification, RotationOperation,
    DatasetSplit, ProcessingStats, ImageProcessor
)
from .dataset_manager import (
    DatasetStatistics, DatasetSplit as DatasetSplitConfig,
    ClassInfo, Dataset, DatasetManager
)

__all__ = [
    # Training Task
    'TrainingTask', 'TaskStatus', 'OrientationType', 'TrainingConfig',
    'ClassificationInfo', 'TrainingProgress', 'TrainingResult',

    # Image Processing
    'ImageMetadata', 'ImageClassification', 'RotationOperation',
    'DatasetSplit', 'ProcessingStats', 'ImageProcessor',

    # Dataset Management
    'DatasetStatistics', 'DatasetSplitConfig', 'ClassInfo',
    'Dataset', 'DatasetManager'
]