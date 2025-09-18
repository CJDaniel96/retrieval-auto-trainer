"""
Dataset Manager業務領域模型
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class DatasetStatistics:
    """資料集統計資訊"""
    mean: List[float]
    std: List[float]
    total_images: int
    class_distribution: Dict[str, int]
    image_size: Tuple[int, int]

    def save_to_file(self, file_path: Path):
        """保存統計資訊到檔案"""
        data = {
            'mean': self.mean,
            'std': self.std,
            'total_images': self.total_images,
            'class_distribution': self.class_distribution,
            'image_size': self.image_size
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: Path) -> 'DatasetStatistics':
        """從檔案載入統計資訊"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(
            mean=data['mean'],
            std=data['std'],
            total_images=data['total_images'],
            class_distribution=data['class_distribution'],
            image_size=tuple(data['image_size'])
        )


@dataclass
class DatasetSplit:
    """資料集分割配置"""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    min_images_per_class: int = 2
    random_seed: int = 42

    def __post_init__(self):
        """驗證分割比例"""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"分割比例總和必須為1.0，當前為{total}")


@dataclass
class ClassInfo:
    """類別資訊"""
    name: str
    image_paths: List[str] = field(default_factory=list)
    train_images: List[str] = field(default_factory=list)
    val_images: List[str] = field(default_factory=list)
    test_images: List[str] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """總影像數量"""
        return len(self.image_paths)

    @property
    def train_count(self) -> int:
        """訓練集數量"""
        return len(self.train_images)

    @property
    def val_count(self) -> int:
        """驗證集數量"""
        return len(self.val_images)

    @property
    def test_count(self) -> int:
        """測試集數量"""
        return len(self.test_images)

    def add_image(self, image_path: str):
        """添加影像"""
        self.image_paths.append(image_path)

    def split_images(self, split_config: DatasetSplit):
        """分割影像到train/val/test"""
        import random

        if self.total_count < split_config.min_images_per_class:
            # 如果影像太少，全部放到訓練集
            self.train_images = self.image_paths.copy()
            self.val_images = []
            self.test_images = []
            return

        # 設定隨機種子確保一致性
        random.seed(split_config.random_seed)
        images = self.image_paths.copy()
        random.shuffle(images)

        total = len(images)
        train_end = int(total * split_config.train_ratio)
        val_end = train_end + int(total * split_config.val_ratio)

        self.train_images = images[:train_end]
        self.val_images = images[train_end:val_end]
        self.test_images = images[val_end:]


@dataclass
class Dataset:
    """資料集領域模型"""
    name: str
    root_path: Path
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    statistics: Optional[DatasetStatistics] = None
    split_config: DatasetSplit = field(default_factory=DatasetSplit)

    def add_class(self, class_name: str) -> ClassInfo:
        """添加類別"""
        if class_name not in self.classes:
            self.classes[class_name] = ClassInfo(name=class_name)
        return self.classes[class_name]

    def get_class(self, class_name: str) -> Optional[ClassInfo]:
        """獲取類別"""
        return self.classes.get(class_name)

    def add_image_to_class(self, class_name: str, image_path: str):
        """添加影像到類別"""
        class_info = self.add_class(class_name)
        class_info.add_image(image_path)

    def split_all_classes(self):
        """分割所有類別"""
        for class_info in self.classes.values():
            class_info.split_images(self.split_config)

    def get_class_distribution(self) -> Dict[str, int]:
        """獲取類別分布"""
        return {name: cls.total_count for name, cls in self.classes.items()}

    def get_train_distribution(self) -> Dict[str, int]:
        """獲取訓練集類別分布"""
        return {name: cls.train_count for name, cls in self.classes.items()}

    def get_val_distribution(self) -> Dict[str, int]:
        """獲取驗證集類別分布"""
        return {name: cls.val_count for name, cls in self.classes.items()}

    @property
    def total_images(self) -> int:
        """總影像數量"""
        return sum(cls.total_count for cls in self.classes.values())

    @property
    def total_train_images(self) -> int:
        """總訓練影像數量"""
        return sum(cls.train_count for cls in self.classes.values())

    @property
    def total_val_images(self) -> int:
        """總驗證影像數量"""
        return sum(cls.val_count for cls in self.classes.values())

    @property
    def num_classes(self) -> int:
        """類別數量"""
        return len(self.classes)

    def validate_dataset(self) -> List[str]:
        """驗證資料集"""
        issues = []

        if self.num_classes < 2:
            issues.append("資料集至少需要2個類別")

        for class_name, class_info in self.classes.items():
            if class_info.total_count < self.split_config.min_images_per_class:
                issues.append(f"類別 {class_name} 只有 {class_info.total_count} 張影像，少於最小要求 {self.split_config.min_images_per_class}")

        if self.total_images == 0:
            issues.append("資料集中沒有影像")

        return issues

    def create_directory_structure(self):
        """創建目錄結構"""
        # 創建train/val目錄
        train_dir = self.root_path / 'train'
        val_dir = self.root_path / 'val'
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        # 為每個類別創建子目錄
        for class_name in self.classes.keys():
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)

    def save_metadata(self):
        """保存元數據"""
        metadata = {
            'name': self.name,
            'num_classes': self.num_classes,
            'total_images': self.total_images,
            'class_distribution': self.get_class_distribution(),
            'train_distribution': self.get_train_distribution(),
            'val_distribution': self.get_val_distribution(),
            'split_config': {
                'train_ratio': self.split_config.train_ratio,
                'val_ratio': self.split_config.val_ratio,
                'test_ratio': self.split_config.test_ratio,
                'min_images_per_class': self.split_config.min_images_per_class,
                'random_seed': self.split_config.random_seed
            }
        }

        metadata_path = self.root_path / 'dataset_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


class DatasetManager:
    """資料集管理器"""

    def __init__(self):
        self.datasets: Dict[str, Dataset] = {}

    def create_dataset(self, name: str, root_path: Path,
                      split_config: Optional[DatasetSplit] = None) -> Dataset:
        """創建資料集"""
        if split_config is None:
            split_config = DatasetSplit()

        dataset = Dataset(
            name=name,
            root_path=root_path,
            split_config=split_config
        )
        self.datasets[name] = dataset
        return dataset

    def get_dataset(self, name: str) -> Optional[Dataset]:
        """獲取資料集"""
        return self.datasets.get(name)

    def load_dataset_from_directory(self, name: str, directory: Path,
                                  split_config: Optional[DatasetSplit] = None) -> Dataset:
        """從目錄載入資料集"""
        dataset = self.create_dataset(name, directory, split_config)

        # 掃描目錄結構
        for class_dir in directory.iterdir():
            if class_dir.is_dir() and not class_dir.name.startswith('.'):
                class_name = class_dir.name
                # 掃描影像檔案
                for image_file in class_dir.rglob('*.jp*'):
                    if image_file.is_file():
                        dataset.add_image_to_class(class_name, str(image_file))

        return dataset