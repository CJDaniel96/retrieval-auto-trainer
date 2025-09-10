"""
Robust ImageFolder dataset that handles corrupted images gracefully
"""
import os
import logging
from PIL import Image
from torchvision.datasets import ImageFolder
from typing import Tuple, Any, Optional, Callable

logger = logging.getLogger(__name__)


class RobustImageFolder(ImageFolder):
    """
    ImageFolder that skips corrupted or truncated image files
    """
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        **kwargs
    ):
        # Initialize with default loader if not provided
        if loader is None:
            loader = self._robust_pil_loader
            
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
            **kwargs
        )
        
        # Filter out corrupted images after initialization
        self._filter_valid_samples()
    
    def _validate_image(self, path: str) -> bool:
        """Validate that an image file is readable and not corrupted"""
        try:
            # Check file exists and has reasonable size
            if not os.path.exists(path):
                return False
                
            if os.path.getsize(path) < 100:  # Files smaller than 100 bytes are likely corrupted
                return False
            
            # Try to open and verify the image
            with Image.open(path) as img:
                img.verify()
                
            # Re-open to check conversion (verify() closes the file)
            with Image.open(path) as img:
                img.convert('RGB')
                
            return True
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            logger.warning(f"Invalid image detected: {path} - {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error validating image {path}: {e}")
            return False
    
    def _filter_valid_samples(self):
        """Filter out corrupted images from the sample list"""
        original_count = len(self.samples)
        valid_samples = []
        valid_targets = []
        
        for sample_path, target in self.samples:
            if self._validate_image(sample_path):
                valid_samples.append((sample_path, target))
                valid_targets.append(target)
            else:
                logger.warning(f"Removing corrupted image from dataset: {sample_path}")
        
        self.samples = valid_samples
        self.targets = valid_targets
        
        removed_count = original_count - len(self.samples)
        if removed_count > 0:
            logger.info(f"Filtered out {removed_count} corrupted images from dataset. "
                       f"Valid images: {len(self.samples)}")
    
    def _robust_pil_loader(self, path: str) -> Image.Image:
        """PIL image loader with error handling"""
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except (OSError, IOError) as e:
            logger.error(f"Failed to load image {path}: {e}")
            raise
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Override getitem to handle any remaining edge cases gracefully
        """
        try:
            return super().__getitem__(index)
        except (OSError, IOError) as e:
            logger.error(f"Error loading image at index {index}: {e}")
            # Try to get a different sample instead of crashing
            if index + 1 < len(self.samples):
                logger.info(f"Attempting to load next image at index {index + 1}")
                return self.__getitem__(index + 1)
            elif index > 0:
                logger.info(f"Attempting to load previous image at index {index - 1}")
                return self.__getitem__(index - 1)
            else:
                raise RuntimeError("No valid images found in dataset")