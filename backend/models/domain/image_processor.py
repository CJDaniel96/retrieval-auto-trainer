"""
Image Processing業務領域模型
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class ImageMetadata:
    """影像元數據"""
    filename: str
    timestamp: str
    component_name: str
    component_id: str
    light_type: str
    product_name: Optional[str] = None
    classification: Optional[str] = None
    file_size: int = 0
    image_path: Optional[str] = None

    @classmethod
    def from_filename(cls, filename: str) -> Optional['ImageMetadata']:
        """從檔名解析影像元數據"""
        # 範例檔名格式: 20240101120000_board123@COMP1_001_R.jpg
        pattern = r'(\d{14})_([^@]+)@([^_]+(?:_[^_]+)*)_(\d+)_(\w+)\.(?:jpg|jpeg|png)'
        match = re.match(pattern, filename, re.IGNORECASE)

        if match:
            return cls(
                filename=filename,
                timestamp=match.group(1),
                component_name=match.group(3),
                component_id=match.group(4),
                light_type=match.group(5)
            )
        return None

    def get_db_image_path(self) -> str:
        """獲取資料庫查詢用的image_path"""
        date_str = f"{self.timestamp[:4]}-{self.timestamp[4:6]}-{self.timestamp[6:8]}"
        return f"{date_str}/{self.timestamp}/{self.filename}"

    def get_class_name(self) -> str:
        """獲取類別名稱"""
        if self.product_name:
            return f"{self.product_name}_{self.component_name}_{self.light_type}"
        return f"{self.component_name}_{self.light_type}"


@dataclass
class ImageClassification:
    """影像分類結果"""
    original_path: str
    target_path: str
    classification: str  # OK, NG, Up, Down, Left, Right
    confidence: float = 1.0
    metadata: Optional[ImageMetadata] = None


@dataclass
class RotationOperation:
    """旋轉操作"""
    source_path: str
    target_path: str
    angle: int  # 90, 180, 270
    source_orientation: str
    target_orientation: str


@dataclass
class DatasetSplit:
    """資料集分割結果"""
    train_images: List[str]
    val_images: List[str]
    class_name: str
    total_count: int
    train_count: int
    val_count: int


@dataclass
class ProcessingStats:
    """處理統計資訊"""
    total_images: int = 0
    processed_images: int = 0
    error_images: int = 0
    classifications: Dict[str, int] = None

    def __post_init__(self):
        if self.classifications is None:
            self.classifications = {}

    def add_classification(self, classification: str, count: int = 1):
        """添加分類統計"""
        self.classifications[classification] = self.classifications.get(classification, 0) + count

    def increment_processed(self):
        """增加處理計數"""
        self.processed_images += 1

    def increment_error(self):
        """增加錯誤計數"""
        self.error_images += 1

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_images == 0:
            return 0.0
        return self.processed_images / self.total_images

    @property
    def error_rate(self) -> float:
        """錯誤率"""
        if self.total_images == 0:
            return 0.0
        return self.error_images / self.total_images


class ImageProcessor:
    """影像處理器領域模型"""

    def __init__(self):
        self.stats = ProcessingStats()
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.valid_orientations = {'Up', 'Down', 'Left', 'Right'}

    def validate_image_file(self, file_path: Path) -> bool:
        """驗證影像檔案"""
        return (
            file_path.exists() and
            file_path.is_file() and
            file_path.suffix.lower() in self.valid_extensions and
            file_path.stat().st_size > 100  # 最小檔案大小
        )

    def parse_filename(self, filename: str) -> Optional[ImageMetadata]:
        """解析檔名"""
        return ImageMetadata.from_filename(filename)

    def classify_image(self, image_path: str, classification: str,
                      metadata: Optional[ImageMetadata] = None) -> ImageClassification:
        """分類影像"""
        # 這裡可以添加更複雜的分類邏輯
        return ImageClassification(
            original_path=image_path,
            target_path="",  # 待設定
            classification=classification,
            metadata=metadata
        )

    def plan_rotation_operations(self, source_orientation: str,
                               images: List[str]) -> List[RotationOperation]:
        """規劃旋轉操作"""
        # 定義旋轉映射關係
        rotation_mapping = {
            'Up': {90: 'Right', 180: 'Down', 270: 'Left'},
            'Right': {90: 'Down', 180: 'Left', 270: 'Up'},
            'Down': {90: 'Left', 180: 'Up', 270: 'Right'},
            'Left': {90: 'Up', 180: 'Right', 270: 'Down'}
        }

        operations = []
        if source_orientation in rotation_mapping:
            for image_path in images:
                for angle, target_orientation in rotation_mapping[source_orientation].items():
                    operation = RotationOperation(
                        source_path=image_path,
                        target_path="",  # 待設定
                        angle=angle,
                        source_orientation=source_orientation,
                        target_orientation=target_orientation
                    )
                    operations.append(operation)

        return operations

    def split_dataset(self, images: List[str], class_name: str,
                     test_split: float = 0.1) -> DatasetSplit:
        """分割資料集"""
        import random

        total_count = len(images)
        if total_count < 2:
            return DatasetSplit(
                train_images=images,
                val_images=[],
                class_name=class_name,
                total_count=total_count,
                train_count=total_count,
                val_count=0
            )

        # 隨機分割
        images_copy = images.copy()
        random.shuffle(images_copy)

        split_idx = max(1, int(total_count * (1 - test_split)))
        train_images = images_copy[:split_idx]
        val_images = images_copy[split_idx:]

        return DatasetSplit(
            train_images=train_images,
            val_images=val_images,
            class_name=class_name,
            total_count=total_count,
            train_count=len(train_images),
            val_count=len(val_images)
        )