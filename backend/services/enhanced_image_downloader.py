#!/usr/bin/env python3
"""
增強版影像下載器
整合影像元資料記錄功能
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime

from backend.services.image_metadata_manager import ImageMetadataManager


class EnhancedImageDownloader:
    """
    增強版影像下載器
    在下載影像的同時記錄元資料到資料庫
    """

    def __init__(self, download_dir: str, db_path: str = "tasks.db"):
        """
        初始化增強版影像下載器

        Args:
            download_dir: 影像下載目錄
            db_path: 資料庫檔案路徑
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_manager = ImageMetadataManager(db_path)
        self.logger = logging.getLogger(__name__)

    def download_image_with_metadata(self,
                                   download_url: str,
                                   filename: str,
                                   source_site: str,
                                   source_line_id: str,
                                   remote_image_path: str = None,
                                   product_info: Dict[str, Any] = None,
                                   task_id: str = None,
                                   classification_label: str = None) -> Optional[str]:
        """
        下載影像並記錄元資料

        Args:
            download_url: 下載URL
            filename: 檔案名稱
            source_site: 來源站點
            source_line_id: 來源產線ID
            remote_image_path: 遠端影像路徑
            product_info: 產品資訊
            task_id: 關聯的任務ID
            classification_label: 預設分類標籤

        Returns:
            Optional[str]: 成功時返回影像ID，失敗時返回None
        """
        try:
            # 建立本地檔案路徑
            local_file_path = self.download_dir / filename

            # 下載影像
            self.logger.info(f"開始下載影像: {download_url}")
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()

            # 儲存影像檔案
            with open(local_file_path, 'wb') as f:
                f.write(response.content)

            self.logger.info(f"影像下載完成: {local_file_path}")

            # 記錄影像元資料
            image_id = self.metadata_manager.record_downloaded_image(
                image_path=str(local_file_path),
                original_filename=filename,
                source_site=source_site,
                source_line_id=source_line_id,
                remote_image_path=remote_image_path,
                download_url=download_url,
                product_info=product_info,
                task_id=task_id
            )

            if image_id:
                self.logger.info(f"影像元資料記錄完成: {image_id}")

                # 如果提供了預設分類標籤，進行初始分類
                if classification_label:
                    self.metadata_manager.classify_image(
                        image_id=image_id,
                        classification_label=classification_label,
                        is_manual=False,
                        notes="自動初始分類"
                    )

                return image_id
            else:
                self.logger.error(f"影像元資料記錄失敗，刪除已下載的檔案: {local_file_path}")
                # 如果元資料記錄失敗，刪除已下載的檔案
                if local_file_path.exists():
                    local_file_path.unlink()
                return None

        except requests.RequestException as e:
            self.logger.error(f"下載影像失敗: {e}")
            return None
        except Exception as e:
            self.logger.error(f"下載影像並記錄元資料時發生錯誤: {e}")
            return None

    def batch_download_with_metadata(self,
                                   download_requests: List[Dict[str, Any]],
                                   task_id: str = None) -> Dict[str, Any]:
        """
        批次下載影像並記錄元資料

        Args:
            download_requests: 下載請求列表
            task_id: 關聯的任務ID

        Returns:
            Dict[str, Any]: 下載結果統計
        """
        results = {
            'total': len(download_requests),
            'success': 0,
            'failed': 0,
            'image_ids': []
        }

        for i, request in enumerate(download_requests):
            try:
                self.logger.info(f"處理下載請求 {i+1}/{len(download_requests)}")

                image_id = self.download_image_with_metadata(
                    download_url=request['download_url'],
                    filename=request['filename'],
                    source_site=request['source_site'],
                    source_line_id=request['source_line_id'],
                    remote_image_path=request.get('remote_image_path'),
                    product_info=request.get('product_info'),
                    task_id=task_id,
                    classification_label=request.get('classification_label')
                )

                if image_id:
                    results['success'] += 1
                    results['image_ids'].append(image_id)
                else:
                    results['failed'] += 1

            except Exception as e:
                self.logger.error(f"處理下載請求時發生錯誤: {e}")
                results['failed'] += 1

        self.logger.info(f"批次下載完成: 總數 {results['total']}, 成功 {results['success']}, 失敗 {results['failed']}")
        return results

    def get_downloaded_images_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        """獲取任務相關的已下載影像"""
        return self.metadata_manager.get_images_by_task(task_id)

    def classify_downloaded_images(self,
                                 task_id: str,
                                 classifications: Dict[str, str]) -> Dict[str, Any]:
        """
        對已下載的影像進行分類

        Args:
            task_id: 任務ID
            classifications: 影像ID到分類標籤的映射

        Returns:
            Dict[str, Any]: 分類結果統計
        """
        try:
            images = self.get_downloaded_images_by_task(task_id)

            classification_list = []
            for image in images:
                image_id = image['image_id']
                if image_id in classifications:
                    classification_list.append({
                        'image_id': image_id,
                        'classification_label': classifications[image_id],
                        'is_manual': True
                    })

            success_count, failure_count = self.metadata_manager.batch_classify_images(classification_list)

            return {
                'total_images': len(images),
                'classified_count': len(classification_list),
                'success_count': success_count,
                'failure_count': failure_count
            }

        except Exception as e:
            self.logger.error(f"分類已下載影像時發生錯誤: {e}")
            return {'error': str(e)}

    def validate_downloaded_images(self, task_id: str) -> Dict[str, Any]:
        """
        驗證已下載影像的完整性

        Args:
            task_id: 任務ID

        Returns:
            Dict[str, Any]: 驗證結果
        """
        try:
            images = self.get_downloaded_images_by_task(task_id)

            valid_count = 0
            corrupted_count = 0
            corrupted_images = []

            for image in images:
                image_id = image['image_id']
                if self.metadata_manager.validate_image_integrity(image_id):
                    valid_count += 1
                else:
                    corrupted_count += 1
                    corrupted_images.append({
                        'image_id': image_id,
                        'filename': image['original_filename'],
                        'file_path': image['local_file_path']
                    })

            return {
                'total_images': len(images),
                'valid_count': valid_count,
                'corrupted_count': corrupted_count,
                'corrupted_images': corrupted_images
            }

        except Exception as e:
            self.logger.error(f"驗證已下載影像時發生錯誤: {e}")
            return {'error': str(e)}

    def get_download_statistics(self, task_id: str = None) -> Dict[str, Any]:
        """
        獲取下載統計資訊

        Args:
            task_id: 可選的任務ID過濾

        Returns:
            Dict[str, Any]: 統計資訊
        """
        try:
            if task_id:
                images = self.get_downloaded_images_by_task(task_id)
                total_images = len(images)

                # 計算各分類的數量
                classification_stats = {}
                processing_stage_stats = {}

                for image in images:
                    label = image.get('classification_label', 'unclassified')
                    stage = image.get('processing_stage', 'unknown')

                    classification_stats[label] = classification_stats.get(label, 0) + 1
                    processing_stage_stats[stage] = processing_stage_stats.get(stage, 0) + 1

                return {
                    'task_id': task_id,
                    'total_images': total_images,
                    'classification_distribution': classification_stats,
                    'processing_stage_distribution': processing_stage_stats
                }
            else:
                return self.metadata_manager.get_classification_statistics()

        except Exception as e:
            self.logger.error(f"獲取下載統計時發生錯誤: {e}")
            return {'error': str(e)}

    def export_image_metadata(self, task_id: str, output_file: str) -> bool:
        """
        匯出影像元資料到CSV檔案

        Args:
            task_id: 任務ID
            output_file: 輸出檔案路徑

        Returns:
            bool: 匯出是否成功
        """
        try:
            import csv

            images = self.get_downloaded_images_by_task(task_id)

            if not images:
                self.logger.warning(f"任務 {task_id} 沒有找到影像資料")
                return False

            # 定義CSV欄位
            fieldnames = [
                'image_id', 'original_filename', 'local_file_path',
                'source_site', 'source_line_id', 'product_name', 'component_name',
                'classification_label', 'processing_stage', 'capture_timestamp',
                'file_size', 'image_width', 'image_height', 'created_at'
            ]

            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for image in images:
                    # 只寫入指定的欄位
                    row = {field: image.get(field, '') for field in fieldnames}
                    writer.writerow(row)

            self.logger.info(f"影像元資料匯出完成: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"匯出影像元資料時發生錯誤: {e}")
            return False

# 使用範例
def example_usage():
    """使用範例"""
    # 初始化增強版影像下載器
    downloader = EnhancedImageDownloader(
        download_dir="./downloaded_images",
        db_path="./tasks.db"
    )

    # 單一影像下載範例
    image_id = downloader.download_image_with_metadata(
        download_url="http://example.com/image.jpg",
        filename="20231201123000_board1@component1_1_light1.jpg",
        source_site="HPH",
        source_line_id="V31",
        remote_image_path="/remote/path/image.jpg",
        product_info={
            'product_name': 'ProductA',
            'component_name': 'ComponentB',
            'board_info': 'Board1',
            'light_condition': 'Light1'
        },
        task_id="task_123",
        classification_label="OK"
    )

    print(f"Downloaded image ID: {image_id}")

    # 批次下載範例
    download_requests = [
        {
            'download_url': "http://example.com/image1.jpg",
            'filename': "20231201123001_board1@component1_1_light1.jpg",
            'source_site': "HPH",
            'source_line_id': "V31",
            'product_info': {'product_name': 'ProductA', 'component_name': 'ComponentB'},
            'classification_label': "OK"
        },
        {
            'download_url': "http://example.com/image2.jpg",
            'filename': "20231201123002_board1@component1_1_light1.jpg",
            'source_site': "HPH",
            'source_line_id': "V31",
            'product_info': {'product_name': 'ProductA', 'component_name': 'ComponentB'},
            'classification_label': "NG"
        }
    ]

    results = downloader.batch_download_with_metadata(download_requests, task_id="task_123")
    print(f"Batch download results: {results}")

    # 獲取統計資訊
    stats = downloader.get_download_statistics(task_id="task_123")
    print(f"Download statistics: {stats}")

    # 匯出元資料
    downloader.export_image_metadata(task_id="task_123", output_file="./image_metadata.csv")


if __name__ == "__main__":
    example_usage()