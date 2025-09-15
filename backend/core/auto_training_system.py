#!/usr/bin/env python3
"""
自動化訓練系統主程式
Auto Training System for Image Retrieval Models
"""

import os
import sys
import json
import yaml

# 設置正確的編碼環境，避免中文字符錯誤
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
import shutil
import random
import logging
import argparse
import pandas as pd
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm

import torch
import torch.nn as nn

# 導入現有模組
from ..services.database.sessions import create_session
from ..services.database.amr_info import AmrRawData
from ..services.data.transforms import build_transforms
from ..services.utils import load_model

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoTrainingSystem:
    """
    自動化訓練系統主類別
    
    功能：
    1. 影像資料前處理（分類、查詢DB）
    2. 訓練資料集準備
    3. 模型訓練
    4. Golden sample生成與驗證
    5. 結果輸出
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化自動訓練系統
        
        Args:
            config_path: 系統配置檔路徑
        """
        # 自動決定配置檔路徑
        if config_path is None:
            config_path = Path(__file__).parent.parent / 'configs' / 'configs.yaml'
        
        self.config_path = str(config_path)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.progress_callback = None
        
        # 載入資料庫配置
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        with open(self.config['train_config_path'], 'r', encoding='utf-8') as f:
            self.train_config = yaml.safe_load(f)
            
        with open(self.config['database_config_path'], 'r', encoding='utf-8') as f:
            self.db_configs = json.load(f)
            
    def set_progress_callback(self, callback):
        """設置進度回調函數"""
        self.progress_callback = callback
        
    def _update_progress(self, step: str, progress: float):
        """更新進度"""
        if self.progress_callback:
            self.progress_callback(step, progress)
        
    def process_raw_images(self, input_dir: str, output_dir: str, 
                          site: str = 'HPH', line_id: str = 'HPH') -> Dict[str, int]:
        """
        處理原始影像資料，分類並查詢資料庫
        
        Args:
            input_dir: 輸入資料夾路徑（包含OK和NG子資料夾）
            output_dir: 輸出分類後的資料夾路徑
            site: 地區名稱（HPH, JQ, ZJ等）
            line_id: 產線ID
            
        Returns:
            Dict[str, int]: 各類別的影像數量統計
        """
        logger.info(f"開始處理原始影像資料: {input_dir}")
        
        # 建立資料庫連線
        proj_config = self.db_configs[site]
        session = create_session(proj_config['SSHTUNNEL'], proj_config['database'])
        
        # 處理OK資料夾中的影像
        ok_dir = Path(input_dir) / 'OK'
        if not ok_dir.exists():
            raise ValueError(f"找不到OK資料夾: {ok_dir}")
            
        stats = {}
        processed_count = 0
        error_count = 0
        
        # 遍歷OK資料夾中的所有影像
        image_files = list(ok_dir.rglob('*.jp*')) + list(ok_dir.rglob('*.JP*'))
        logger.info(f"找到 {len(image_files)} 張OK影像")
        
        for img_path in tqdm(image_files, desc="處理影像"):
            try:
                # 解析影像檔名資訊
                img_info = self._parse_image_filename(img_path.name)
                if not img_info:
                    logger.warning(f"無法解析檔名: {img_path.name}")
                    error_count += 1
                    continue
                    
                # 構建查詢用的image_path
                db_image_path = self._build_db_image_path(img_info['timestamp'], img_path.name)
                
                start_date = datetime.strptime(img_info['timestamp'], '%Y%m%d%H%M%S')
                end_date = start_date + timedelta(days=1)
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                # 查詢資料庫取得product_name
                record = session.query(AmrRawData).filter(
                    AmrRawData.site == site,
                    AmrRawData.line_id == line_id,
                    AmrRawData.create_time.between(start_str, end_str),
                    AmrRawData.image_path == db_image_path
                ).first()
                
                if not record or not record.product_name:
                    logger.warning(f"資料庫中找不到對應記錄: {db_image_path}")
                    error_count += 1
                    continue
                    
                # 組合類別名稱
                class_name = f"{record.product_name}_{img_info['comp_name']}_{img_info['light']}"
                
                # 建立目標資料夾並移動影像
                target_dir = Path(output_dir) / class_name
                target_dir.mkdir(parents=True, exist_ok=True)
                
                target_path = target_dir / img_path.name
                shutil.copy2(str(img_path), str(target_path))
                
                # 更新統計
                stats[class_name] = stats.get(class_name, 0) + 1
                processed_count += 1
                
            except Exception as e:
                logger.error(f"處理影像時發生錯誤 {img_path.name}: {str(e)}")
                error_count += 1
                
        # 處理NG資料夾（保留但不分類）
        ng_dir = Path(input_dir) / 'NG'
        if ng_dir.exists():
            ng_target = Path(output_dir) / 'NG'
            if ng_dir != ng_target:
                shutil.copytree(str(ng_dir), str(ng_target), dirs_exist_ok=True)
            stats['NG'] = len(list(ng_target.rglob('*.jp*')))
            
        session.close()
        
        logger.info(f"影像處理完成: 成功 {processed_count}, 錯誤 {error_count}")
        logger.info(f"類別統計: {json.dumps(stats, indent=2)}")
        
        return stats
        
    def _parse_image_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """解析影像檔名，提取timestamp、comp_name和light資訊"""
        import re
        
        # 範例檔名格式: 20240101120000_board123@COMP1_001_R.jpg
        # timestamp: 20240101120000
        # comp_name: COMP1
        # light: R
        
        pattern = r'(\d{14})_([^@]+)@([^_]+(?:_[^_]+)*)_(\d+)_(\w+)\.(?:jpg|jpeg|png)'
        match = re.match(pattern, filename, re.IGNORECASE)
        
        if match:
            return {
                'timestamp': match.group(1),
                'sn': match.group(2),
                'comp_name': match.group(3),
                'comp_id': match.group(4),
                'light': match.group(5)
            }
        return None
        
    def _build_db_image_path(self, timestamp: str, filename: str) -> str:
        """構建資料庫查詢用的image_path"""
        # 格式: YYYY-MM-DD/timestamp/filename
        date_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
        return f"{date_str}/{timestamp}/{filename}"
        
    def prepare_dataset(self, raw_data_dir: str, dataset_dir: str, test_split: float = 0.1) -> None:
        """
        準備訓練資料集（分割train/val） - 舊版本，保留向後相容性
        
        Args:
            raw_data_dir: 分類好的原始資料目錄
            dataset_dir: 輸出的資料集目錄
            test_split: 驗證集比例
        """
        logger.warning("使用舊版本的 prepare_dataset，建議使用新的 prepare_final_dataset")
        
        train_dir = Path(dataset_dir) / 'train'
        val_dir = Path(dataset_dir) / 'val'
        
        # 遍歷所有類別（除了NG）
        for class_dir in Path(raw_data_dir).iterdir():
            if not class_dir.is_dir() or class_dir.name == 'NG':
                continue
                
            # 取得該類別的所有影像
            images = list(class_dir.rglob('*.jp*'))
            if len(images) < 2:
                logger.warning(f"類別 {class_dir.name} 影像數量不足，跳過")
                continue
                
            # 隨機分割
            random.shuffle(images)
            split_idx = int(len(images) * (1 - test_split))
            train_images = images[:split_idx]
            val_images = images[split_idx:]
            
            # 建立目標資料夾
            train_class_dir = train_dir / class_dir.name
            val_class_dir = val_dir / class_dir.name
            train_class_dir.mkdir(parents=True, exist_ok=True)
            val_class_dir.mkdir(parents=True, exist_ok=True)
            
            # 複製影像
            for img in train_images:
                shutil.copy2(str(img), str(train_class_dir / img.name))
            for img in val_images:
                shutil.copy2(str(img), str(val_class_dir / img.name))
                
        logger.info(f"資料集準備完成: {dataset_dir}")
        
    def train_model(self, dataset_dir: str, output_dir: str) -> str:
        """
        執行模型訓練
        
        Args:
            dataset_dir: 訓練資料集目錄
            output_dir: 輸出目錄
            
        Returns:
            str: 最佳模型檔案路徑
        """
        logger.info("開始訓練模型")
        
        # 導入訓練相關模組
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging, Callback
        from lightning.pytorch.loggers import TensorBoardLogger
        
        # 準備資料模組
        from ..train import HOAMDataModule, LightningModel
        
        parent_self = self
        
        class ProgressCallback(Callback):
            def __init__(self, total_epochs: int):
                super().__init__()
                self.total_epochs = total_epochs
                
            def on_train_epoch_end(self, trainer, pl_module):
                current_epoch = trainer.current_epoch + 1
                progress = 0.35 + (current_epoch / self.total_epochs) * 0.45
                step_msg = f"訓練中 - Epoch {current_epoch}/{self.total_epochs}"
                parent_self._update_progress(step_msg, progress)
        
        # 更新配置
        self.train_config['data']['data_dir'] = dataset_dir
        
        # 建立資料模組
        data_module = HOAMDataModule(
            data_dir=dataset_dir,
            image_size=self.train_config['data']['image_size'],
            batch_size=self.train_config['training']['batch_size'],
            num_workers=self.train_config['data']['num_workers']
        )
        
        # 建立模型
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(self.train_config)
        model = LightningModel(cfg)
        
        # 設定callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        )
        
        callbacks = [checkpoint_callback]
        
        # 根據配置決定是否啟用EarlyStopping
        if self.train_config['training'].get('enable_early_stopping', True):
            early_stop_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.train_config['training']['patience'],
                mode='min'
            )
            callbacks.append(early_stop_callback)
            logger.info(f"啟用EarlyStopping，patience={self.train_config['training']['patience']}")
        else:
            logger.info("已禁用EarlyStopping，訓練將進行完整的max_epochs")
        
        if self.progress_callback:
            callbacks.append(ProgressCallback(self.train_config['training']['max_epochs']))
        
        max_epochs = self.train_config['training']['max_epochs']
        if max_epochs > 10:
            swa = StochasticWeightAveraging(
                swa_lrs=[self.train_config['training']['lr'] * 0.01, self.train_config['training']['lr'] * 0.1],
                swa_epoch_start=0.75,
                annealing_epochs=min(5, max_epochs - int(max_epochs * 0.75)),
                annealing_strategy='cos'
            )
            callbacks.append(swa)
            
        trainer_kwargs = {
            'min_epochs': self.train_config['training'].get('min_epochs', 0),
            'max_epochs': max_epochs,
            'callbacks': callbacks,
            'logger': TensorBoardLogger(save_dir=output_dir, name='logs')
        }
        
        if torch.cuda.is_available():
            trainer_kwargs['precision'] = '16-mixed'
        else:
            trainer_kwargs['precision'] = 32
            
        trainer = pl.Trainer(**trainer_kwargs)
        
        # 開始訓練
        try:
            trainer.fit(model, datamodule=data_module)
        except Exception as e:
            logger.error(f"訓練過程中發生錯誤: {str(e)}")
            if "inf checks" in str(e):
                logger.info("偵測到 inf checks 錯誤，嘗試使用替代配置...")
                
                trainer_kwargs['callbacks'] = [checkpoint_callback, early_stop_callback]
                trainer_kwargs['precision'] = 32
                
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(model, datamodule=data_module)
        
        # 儲存最終模型
        best_model_path = checkpoint_callback.best_model_path
        final_model_path = os.path.join(output_dir, 'best_model.pt')
        
        # 載入最佳檢查點並儲存模型權重
        if best_model_path and os.path.exists(best_model_path):
            best_model = LightningModel.load_from_checkpoint(best_model_path)
            torch.save(best_model.model.state_dict(), final_model_path)
        else:
            logger.warning("找不到最佳模型檢查點，儲存當前模型權重")
            torch.save(model.model.state_dict(), final_model_path)
        
        # 儲存配置
        with open(os.path.join(output_dir, 'train_config.json'), 'w') as f:
            json.dump(self.train_config, f, indent=2)
            
        # 複製mean_std.json
        mean_std_src = Path(dataset_dir) / 'mean_std.json'
        if mean_std_src.exists():
            shutil.copy2(str(mean_std_src), output_dir)
            
        logger.info(f"訓練完成，模型儲存至: {final_model_path}")
        return final_model_path
        
    def generate_golden_samples(self, dataset_dir: str, output_dir: str) -> Dict[str, str]:
        """
        從每個類別隨機選取一張作為golden sample
        
        Args:
            dataset_dir: 資料集目錄
            output_dir: 輸出目錄
            
        Returns:
            Dict[str, str]: 類別名稱到golden sample路徑的映射
        """
        logger.info("開始生成Golden Samples")
        
        golden_dir = Path(output_dir) / 'golden_samples'
        golden_dir.mkdir(exist_ok=True)
        
        golden_samples = {}
        train_dir = Path(dataset_dir) / 'train'
        
        # 遍歷所有類別（除了NG）
        for class_dir in train_dir.iterdir():
            if not class_dir.is_dir() or class_dir.name == 'NG':
                continue
                
            # 隨機選取一張影像
            images = list(class_dir.rglob('*.jp*'))
            if images:
                selected = random.choice(images)
                target_path = golden_dir / f"{class_dir.name}_{selected.name}"
                shutil.copy2(str(selected), str(target_path))
                golden_samples[class_dir.name] = str(target_path)
                
        logger.info(f"生成 {len(golden_samples)} 個Golden Samples")
        return golden_samples
        
    def evaluate_model(self, model_path: str, dataset_dir: str, 
                      golden_samples: Dict[str, str], output_dir: str) -> pd.DataFrame:
        """
        評估模型效能
        
        Args:
            model_path: 模型檔案路徑
            dataset_dir: 資料集目錄
            golden_samples: Golden sample映射
            output_dir: 輸出目錄
            
        Returns:
            pd.DataFrame: 評估結果
        """
        logger.info("開始評估模型")
        
        # 載入模型
        model = load_model(
            model_structure=self.train_config['model']['structure'],
            model_path=model_path,
            embedding_size=self.train_config['model']['embedding_size']
        )
        
        # 載入資料轉換
        with open(os.path.join(dataset_dir, 'mean_std.json'), 'r') as f:
            stats = json.load(f)
        
        transform = build_transforms(
            mode='test',
            image_size=self.train_config['data']['image_size'],
            mean=stats['mean'],
            std=stats['std']
        )
        
        # 提取golden sample特徵
        golden_features = {}
        for class_name, img_path in golden_samples.items():
            features = self._extract_features(model, img_path, transform)
            golden_features[class_name] = features
            
        # 評估所有類別
        results = []
        val_dir = Path(dataset_dir) / 'val'
        
        for class_dir in val_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            images = list(class_dir.rglob('*.jp*'))
            
            for img_path in images:
                try:
                    # 提取特徵
                    features = self._extract_features(model, str(img_path), transform)
                    
                    # 計算與所有golden sample的相似度
                    similarities = {}
                    for golden_class, golden_feat in golden_features.items():
                        sim = torch.cosine_similarity(features, golden_feat, dim=0).item()
                        similarities[golden_class] = sim
                        
                    # 找出最相似的類別
                    pred_class = max(similarities, key=similarities.get)
                    max_sim = similarities[pred_class]
                    
                    # 記錄結果
                    results.append({
                        'image': img_path.name,
                        'true_class': class_name,
                        'pred_class': pred_class,
                        'similarity': max_sim,
                        'correct': class_name == pred_class
                    })
                    
                except (OSError, IOError) as e:
                    logger.warning(f"跳過損壞的圖片 {img_path}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"處理圖片時發生未知錯誤 {img_path}: {e}")
                    continue
                
        # 轉換為DataFrame
        df_results = pd.DataFrame(results)
        
        # 計算準確率
        accuracy = df_results['correct'].mean()
        logger.info(f"整體準確率: {accuracy:.4f}")
        
        # 儲存結果
        csv_path = os.path.join(output_dir, 'evaluation_results.csv')
        df_results.to_csv(csv_path, index=False)
        
        # 生成混淆矩陣和視覺化結果
        self._generate_evaluation_plots(df_results, output_dir)
        
        return df_results
        
    def _validate_image(self, img_path: str) -> bool:
        """驗證圖片文件的完整性"""
        from PIL import Image
        import os
        
        try:
            # 檢查文件是否存在
            if not os.path.exists(img_path):
                return False
                
            # 檢查文件大小
            if os.path.getsize(img_path) < 100:  # 小於100字節的圖片通常是損壞的
                return False
                
            # 嘗試打開和驗證圖片
            with Image.open(img_path) as img:
                img.verify()  # 驗證圖片完整性
            
            # 重新打開以檢查是否能正常轉換
            with Image.open(img_path) as img:
                img.convert('RGB')
                
            return True
        except (OSError, IOError, Image.UnidentifiedImageError) as e:
            logger.warning(f"圖片驗證失敗 {img_path}: {e}")
            return False
    
    def _extract_features(self, model: nn.Module, img_path: str, transform) -> torch.Tensor:
        """提取單張影像的特徵"""
        from PIL import Image
        
        # 驗證圖片完整性
        if not self._validate_image(img_path):
            raise OSError(f"圖片文件損壞或不完整: {img_path}")
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).cuda()
        
        with torch.no_grad():
            features = model(img_tensor)
            
        return features.squeeze(0).cpu()
        
    def process_orientations(self, raw_data_dir: str, output_dir: str, 
                           orientations: Dict[str, str]) -> None:
        """
        根據用戶確認的方向，將影像移動到對應的方向資料夾
        
        Args:
            raw_data_dir: 原始分類資料目錄（按product_comp分類）
            output_dir: 輸出目錄（按方向分類）
            orientations: 類別名稱到方向的映射
        """
        logger.info("開始處理方向分類")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 建立方向資料夾
        orientation_dirs = {}
        for orientation in ['Up', 'Down', 'Left', 'Right']:
            orientation_dir = output_path / orientation
            orientation_dir.mkdir(exist_ok=True)
            orientation_dirs[orientation] = orientation_dir
        
        # 處理NG資料夾（驗證並複製）
        ng_source = Path(raw_data_dir) / 'NG'
        if ng_source.exists():
            ng_target = output_path / 'NG'
            ng_target.mkdir(exist_ok=True)
            
            # 逐個驗證NG圖片
            ng_images = list(ng_source.rglob('*.jp*'))
            valid_ng_count = 0
            
            for img in ng_images:
                if self._validate_image(str(img)):
                    target_path = ng_target / img.name
                    shutil.copy2(str(img), str(target_path))
                    valid_ng_count += 1
                else:
                    logger.warning(f"跳過損壞的NG圖片: {img}")
                    
            logger.info(f"NG資料夾: 成功複製 {valid_ng_count}/{len(ng_images)} 張有效影像")
        
        # 處理各個product_comp類別
        for class_dir in Path(raw_data_dir).iterdir():
            if not class_dir.is_dir() or class_dir.name == 'NG':
                continue
                
            class_name = class_dir.name
            if class_name not in orientations:
                logger.warning(f"類別 {class_name} 沒有方向確認，跳過")
                continue
                
            orientation = orientations[class_name]
            if orientation not in orientation_dirs:
                logger.warning(f"無效的方向 {orientation}，跳過類別 {class_name}")
                continue
                
            # 移動所有影像到對應方向資料夾
            target_dir = orientation_dirs[orientation]
            images = list(class_dir.rglob('*.jp*'))
            
            logger.info(f"移動 {len(images)} 張影像從 {class_name} 到 {orientation}")
            
            valid_count = 0
            for img in images:
                # 驗證圖片完整性
                if not self._validate_image(str(img)):
                    logger.warning(f"跳過損壞的圖片: {img}")
                    continue
                    
                # 保留原始檔名，避免衝突可以加上類別前綴
                target_name = f"{class_name}_{img.name}"
                target_path = target_dir / target_name
                shutil.copy2(str(img), str(target_path))
                valid_count += 1
                
            logger.info(f"成功移動 {valid_count}/{len(images)} 張有效影像從 {class_name} 到 {orientation}")
        
        logger.info("方向分類完成")
        
    def apply_rotation_augmentation(self, oriented_data_dir: str) -> None:
        """
        對方向資料夾中的**原始影像**進行旋轉增強（避免對旋轉後的影像重複旋轉）
        
        Args:
            oriented_data_dir: 按方向分類的資料目錄
        """
        logger.info("開始執行旋轉增強")
        
        # 定義旋轉映射關係（順時針旋轉）
        rotation_mapping = {
            # 原始方向 -> {旋轉角度: 目標方向}
            'Up': {90: 'Right', 180: 'Down', 270: 'Left'},
            'Right': {90: 'Down', 180: 'Left', 270: 'Up'},
            'Down': {90: 'Left', 180: 'Up', 270: 'Right'},
            'Left': {90: 'Up', 180: 'Right', 270: 'Down'}
        }
        
        data_path = Path(oriented_data_dir)
        
        # 第一步：收集所有原始影像（不包含已經旋轉過的影像）
        original_images = {}
        
        for orientation in ['Up', 'Down', 'Left', 'Right']:
            source_dir = data_path / orientation
            if not source_dir.exists():
                continue
                
            # 只處理原始影像（檔名不包含 _rot 的影像）
            images = [img for img in source_dir.rglob('*.jp*') 
                     if '_rot' not in img.stem]
            original_images[orientation] = images
            
            logger.info(f"在 {orientation} 資料夾中找到 {len(images)} 張原始影像")
        
        # 第二步：對每個原始影像進行旋轉增強
        for source_orientation, images in original_images.items():
            if not images:
                continue
                
            logger.info(f"對 {source_orientation} 資料夾中的 {len(images)} 張原始影像進行旋轉增強")
            
            for img_path in tqdm(images, desc=f"旋轉 {source_orientation} 原始影像"):
                try:
                    # 載入影像
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logger.warning(f"無法讀取影像: {img_path}")
                        continue
                    
                    # 對每個旋轉角度生成新影像
                    for angle, target_orientation in rotation_mapping[source_orientation].items():
                        # 執行旋轉
                        rotated_image = self._rotate_image(image, angle)
                        
                        # 確保目標資料夾存在
                        target_dir = data_path / target_orientation
                        target_dir.mkdir(exist_ok=True)
                        
                        # 生成新檔名（標記為旋轉影像）
                        base_name = img_path.stem
                        ext = img_path.suffix
                        new_name = f"{base_name}_rot{angle}{ext}"
                        target_path = target_dir / new_name
                        
                        # 儲存旋轉後的影像
                        cv2.imwrite(str(target_path), rotated_image)
                        
                except Exception as e:
                    logger.error(f"旋轉影像時發生錯誤 {img_path}: {str(e)}")
        
        logger.info("旋轉增強完成")
        
    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        旋轉影像
        
        Args:
            image: 輸入影像
            angle: 旋轉角度（順時針，90的倍數）
            
        Returns:
            旋轉後的影像
        """
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
    
    def prepare_final_dataset(self, oriented_data_dir: str, dataset_dir: str, 
                            test_split: float = 0.1) -> None:
        """
        準備最終訓練資料集（針對方向資料夾：Up, Down, Left, Right, NG）
        
        Args:
            oriented_data_dir: 按方向分類並增強後的資料目錄
            dataset_dir: 輸出的資料集目錄
            test_split: 驗證集比例
        """
        logger.info("開始準備最終訓練資料集")
        
        dataset_path = Path(dataset_dir)
        train_dir = dataset_path / 'train'
        val_dir = dataset_path / 'val'
        
        # 建立train/val目錄
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # 處理五個資料夾：Up, Down, Left, Right, NG
        orientations = ['Up', 'Down', 'Left', 'Right', 'NG']
        
        for orientation in orientations:
            source_dir = Path(oriented_data_dir) / orientation
            if not source_dir.exists():
                logger.warning(f"方向資料夾不存在: {orientation}")
                continue
                
            # 取得該方向的所有影像
            images = list(source_dir.rglob('*.jp*'))
            if len(images) == 0:
                logger.warning(f"方向資料夾 {orientation} 沒有影像，跳過")
                continue
                
            logger.info(f"處理方向 {orientation}: {len(images)} 張影像")
            
            # 隨機分割train/val
            random.shuffle(images)
            if len(images) >= 2:
                split_idx = max(1, int(len(images) * (1 - test_split)))
                train_images = images[:split_idx]
                val_images = images[split_idx:]
            else:
                # 如果影像數量少於2張，全部放入train
                train_images = images
                val_images = []
            
            # 建立目標資料夾
            train_orientation_dir = train_dir / orientation
            val_orientation_dir = val_dir / orientation
            train_orientation_dir.mkdir(exist_ok=True)
            val_orientation_dir.mkdir(exist_ok=True)
            
            # 複製影像
            for img in train_images:
                shutil.copy2(str(img), str(train_orientation_dir / img.name))
            for img in val_images:
                shutil.copy2(str(img), str(val_orientation_dir / img.name))
                
            logger.info(f"方向 {orientation}: Train {len(train_images)}, Val {len(val_images)}")
        
        # 計算並儲存資料集統計資訊
        from ..services.data.statistics import DataStatistics
        try:
            mean, std = DataStatistics.get_mean_std(
                dataset_path,
                self.train_config['data']['image_size'],
                cache_file="mean_std.json"
            )
            logger.info(f"資料集統計完成 - Mean: {mean}, Std: {std}")
        except Exception as e:
            logger.error(f"計算資料集統計時發生錯誤: {str(e)}")
        
        logger.info(f"最終資料集準備完成: {dataset_dir}")
        
    def _generate_evaluation_plots(self, df_results: pd.DataFrame, output_dir: str) -> None:
        """生成評估結果的視覺化圖表"""
        import matplotlib
        # 設置非交互式後端以支持多線程環境
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        
        # 設定樣式
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. 混淆矩陣
        true_labels = df_results['true_class'].values
        pred_labels = df_results['pred_class'].values
        classes = sorted(df_results['true_class'].unique())
        
        cm = confusion_matrix(true_labels, pred_labels, labels=classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 2. 每個類別的準確率
        class_accuracy = df_results.groupby('true_class')['correct'].mean()
        
        plt.figure(figsize=(12, 6))
        class_accuracy.plot(kind='bar')
        plt.title('Per-Class Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_accuracy.png'), dpi=300)
        plt.close()
        
        # 3. 相似度分布
        plt.figure(figsize=(10, 6))
        correct_sim = df_results[df_results['correct']]['similarity']
        incorrect_sim = df_results[~df_results['correct']]['similarity']
        
        plt.hist(correct_sim, bins=50, alpha=0.6, label='Correct', density=True)
        plt.hist(incorrect_sim, bins=50, alpha=0.6, label='Incorrect', density=True)
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.title('Similarity Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'similarity_distribution.png'), dpi=300)
        plt.close()
        
    def run_full_pipeline(self, input_dir: str, output_base_dir: str, 
                         site: str = 'HPH', line_id: str = 'V31') -> None:
        """
        執行完整的自動訓練流程
        
        Args:
            input_dir: 輸入資料夾路徑（包含OK和NG子資料夾）
            output_base_dir: 輸出基礎目錄
            site: 地區名稱（HPH, JQ, ZJ等）
            line_id: 產線ID
        """
        logger.info("=" * 50)
        logger.info("開始執行自動訓練系統")
        logger.info("=" * 50)
        
        # 建立輸出目錄結構
        output_dir = Path(output_base_dir) / f"training_{self.timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        raw_data_dir = output_dir / 'raw_data'
        dataset_dir = output_dir / 'dataset'
        model_dir = output_dir / 'model'
        results_dir = output_dir / 'results'
        
        for d in [raw_data_dir, dataset_dir, model_dir, results_dir]:
            d.mkdir(exist_ok=True)
            
        try:
            # 1. 處理原始影像
            logger.info("[步驟 1/5] 處理原始影像資料")
            self._update_progress("處理原始影像資料", 0.1)
            class_stats = self.process_raw_images(input_dir, str(raw_data_dir), site, line_id)
            self._update_progress("處理原始影像完成", 0.2)
            
            # 2. 準備資料集
            logger.info("[步驟 2/5] 準備訓練資料集")
            self._update_progress("準備訓練資料集", 0.25)
            self.prepare_dataset(str(raw_data_dir), str(dataset_dir), 
                               self.train_config['data']['test_split'])
            self._update_progress("資料集準備完成", 0.3)
            
            # 3. 訓練模型
            logger.info("[步驟 3/5] 訓練模型")
            self._update_progress("開始訓練模型", 0.35)
            model_path = self.train_model(str(dataset_dir), str(model_dir))
            self._update_progress("模型訓練完成", 0.8)
            
            # 4. 生成Golden Samples
            logger.info("[步驟 4/5] 生成Golden Samples")
            self._update_progress("生成Golden Samples", 0.82)
            golden_samples = self.generate_golden_samples(str(dataset_dir), str(results_dir))
            self._update_progress("Golden Samples生成完成", 0.85)
            
            # 5. 評估模型
            logger.info("[步驟 5/5] 評估模型效能")
            self._update_progress("評估模型效能", 0.9)
            eval_results = self.evaluate_model(model_path, str(dataset_dir), 
                                             golden_samples, str(results_dir))
            self._update_progress("評估完成", 0.95)
            
            # 生成總結報告
            self._update_progress("生成總結報告", 0.98)
            self._generate_summary_report(output_dir, class_stats, eval_results)
            self._update_progress("訓練完成", 1.0)
            
            logger.info("\n" + "=" * 50)
            logger.info("自動訓練完成！")
            logger.info(f"所有結果儲存在: {output_dir}")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"訓練過程發生錯誤: {str(e)}", exc_info=True)
            raise
            
    def _generate_summary_report(self, output_dir: Path, class_stats: Dict, 
                               eval_results: pd.DataFrame) -> None:
        """生成總結報告"""
        report = {
            'timestamp': self.timestamp,
            'class_statistics': class_stats,
            'total_classes': len(class_stats) - (1 if 'NG' in class_stats else 0),
            'total_images': sum(class_stats.values()),
            'evaluation_accuracy': eval_results['correct'].mean(),
            'training_config': self.train_config
        }
        
        with open(output_dir / 'summary_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
            
def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description='自動化訓練系統')
    parser.add_argument('--input-dir', required=True, help='輸入資料夾路徑')
    parser.add_argument('--output-dir', required=True, help='輸出資料夾路徑')
    parser.add_argument('--site', default='HPH', help='專案名稱')
    parser.add_argument('--line-id', default='V31', help='產線ID')
    parser.add_argument('--config', default=None, help='系統配置檔')
    
    args = parser.parse_args()
    
    # 建立並執行自動訓練系統
    system = AutoTrainingSystem(config_path=args.config)
    system.run_full_pipeline(
        input_dir=args.input_dir,
        output_base_dir=args.output_dir,
        site=args.site,
        line_id=args.line_id
    )


if __name__ == '__main__':
    main()