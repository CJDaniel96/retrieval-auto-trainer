"""
Infrastructure package - 基礎設施層，處理外部依賴和技術實現
"""

# Database infrastructure
from .database import create_session, AmrRawData

# ML infrastructure
from .ml.models import HOAM, HOAMV2
from .ml.losses import HybridMarginLoss
from .ml.transforms import build_transforms, RobustDataset, DataStatistics

# Storage infrastructure
from .storage import ImageDownloadService, ImageMetadataManager

__all__ = [
    # Database
    'create_session', 'AmrRawData',

    # ML Models & Components
    'HOAM', 'HOAMV2', 'HybridMarginLoss',

    # ML Transforms & Data
    'build_transforms', 'RobustDataset', 'DataStatistics',

    # Storage
    'ImageDownloadService', 'ImageMetadataManager'
]