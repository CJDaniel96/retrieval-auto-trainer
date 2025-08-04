#!/usr/bin/env python3
"""
自動化訓練系統主程式
Auto Training System for Image Retrieval Models
"""

import os
import json
import yaml
import shutil
import random
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from tqdm import tqdm

import torch
import torch.nn as nn

# 導入現有模組
from database.sessions import create_session
from database.amr_info import AmrRawData
from data.transforms import build_transforms
from utils import load_model

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
    
    def __init__(self, config_path: str = 'configs/configs.yaml'):
        """
        初始化自動訓練系統
        
        Args:
            config_path: 系統配置檔路徑
        """
        self.config_path = config_path
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 載入資料庫配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        with open(self.config['train_config_path'], 'r', encoding='utf-8') as f:
            self.train_config = yaml.safe_load(f)
            
        with open(self.config['database_config_path'], 'r', encoding='utf-8') as f:
            self.db_configs = json.load(f)
        
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
        image_files = list(ok_dir.glob('*.jp*')) + list(ok_dir.glob('*.JP*'))
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
            stats['NG'] = len(list(ng_target.glob('*.jp*')))
            
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
        準備訓練資料集（分割train/val）
        
        Args:
            raw_data_dir: 分類好的原始資料目錄
            dataset_dir: 輸出的資料集目錄
            test_split: 驗證集比例
        """
        logger.info("開始準備訓練資料集")
        
        train_dir = Path(dataset_dir) / 'train'
        val_dir = Path(dataset_dir) / 'val'
        
        # 遍歷所有類別（除了NG）
        for class_dir in Path(raw_data_dir).iterdir():
            if not class_dir.is_dir() or class_dir.name == 'NG':
                continue
                
            # 取得該類別的所有影像
            images = list(class_dir.glob('*.jp*'))
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
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
        from lightning.pytorch.loggers import TensorBoardLogger
        
        # 準備資料模組
        from train import HOAMDataModule, LightningModel
        
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
        
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.train_config['training']['patience'],
            mode='min'
        )
        
        swa = StochasticWeightAveraging(
        swa_lrs=[float(self.train_config['training']['lr']) * 0.01, float(self.train_config['training']['lr']) * 0.1],
        swa_epoch_start=0.75,
        annealing_strategy='cos'
    )
        
        # 建立trainer
        trainer = pl.Trainer(
            min_epochs=self.train_config['training']['min_epochs'],
            max_epochs=self.train_config['training']['max_epochs'],
            callbacks=[checkpoint_callback, early_stop_callback, swa],
            logger=TensorBoardLogger(save_dir=output_dir, name='logs'),
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=16 if torch.cuda.is_available() else 32
        )
        
        # 開始訓練
        trainer.fit(model, datamodule=data_module)
        
        # 儲存最終模型
        best_model_path = checkpoint_callback.best_model_path
        final_model_path = os.path.join(output_dir, 'best_model.pt')
        
        # 載入最佳檢查點並儲存模型權重
        best_model = LightningModel.load_from_checkpoint(best_model_path)
        torch.save(best_model.model.state_dict(), final_model_path)
        
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
            images = list(class_dir.glob('*.jp*'))
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
        with open(os.path.join(output_dir, 'mean_std.json'), 'r') as f:
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
            images = list(class_dir.glob('*.jp*'))
            
            for img_path in images:
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
        
    def _extract_features(self, model: nn.Module, img_path: str, transform) -> torch.Tensor:
        """提取單張影像的特徵"""
        from PIL import Image
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).cuda()
        
        with torch.no_grad():
            features = model(img_tensor)
            
        return features.squeeze(0).cpu()
        
    def _generate_evaluation_plots(self, df_results: pd.DataFrame, output_dir: str) -> None:
        """生成評估結果的視覺化圖表"""
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
            logger.info("\n[步驟 1/5] 處理原始影像資料")
            class_stats = self.process_raw_images(input_dir, str(raw_data_dir), site, line_id)
            
            # 2. 準備資料集
            logger.info("\n[步驟 2/5] 準備訓練資料集")
            self.prepare_dataset(str(raw_data_dir), str(dataset_dir), 
                               self.train_config['data']['test_split'])
            
            # 3. 訓練模型
            logger.info("\n[步驟 3/5] 訓練模型")
            model_path = self.train_model(str(dataset_dir), str(model_dir))
            
            # 4. 生成Golden Samples
            logger.info("\n[步驟 4/5] 生成Golden Samples")
            golden_samples = self.generate_golden_samples(str(dataset_dir), str(results_dir))
            
            # 5. 評估模型
            logger.info("\n[步驟 5/5] 評估模型效能")
            eval_results = self.evaluate_model(model_path, str(dataset_dir), 
                                             golden_samples, str(results_dir))
            
            # 生成總結報告
            self._generate_summary_report(output_dir, class_stats, eval_results)
            
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
    parser.add_argument('--config', default='configs/configs.yaml', help='系統配置檔')
    
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