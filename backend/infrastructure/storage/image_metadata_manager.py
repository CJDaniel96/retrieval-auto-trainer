#!/usr/bin/env python3
"""
影像元資料管理器
負責管理下載影像的元資料記錄和分類
"""

import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PIL import Image
import uuid

from ...database.task_database import TaskDatabase


class ImageMetadataManager:
    """影像元資料管理器"""

    def __init__(self, db_path: str = "tasks.db"):
        """
        初始化影像元資料管理器

        Args:
            db_path: 資料庫檔案路徑
        """
        self.db = TaskDatabase(db_path)
        self.logger = logging.getLogger(__name__)

    def record_downloaded_image(self,
                              image_path: str,
                              original_filename: str,
                              source_site: str,
                              source_line_id: str,
                              remote_image_path: str = None,
                              download_url: str = None,
                              product_info: Dict[str, Any] = None,
                              task_id: str = None) -> str:
        """
        記錄下載的影像元資料

        Args:
            image_path: 本地影像檔案路徑
            original_filename: 原始檔案名稱
            source_site: 來源站點
            source_line_id: 來源產線ID
            remote_image_path: 遠端影像路徑
            download_url: 下載URL
            product_info: 產品資訊字典 (product_name, component_name, board_info, light_condition等)
            task_id: 關聯的任務ID

        Returns:
            str: 生成的影像ID
        """
        try:
            # 生成唯一的影像ID
            image_id = self._generate_image_id(original_filename, source_site, source_line_id)

            # 計算檔案雜湊值和大小
            file_hash = self._calculate_file_hash(image_path)
            file_size = os.path.getsize(image_path)

            # 獲取影像屬性
            image_width, image_height, image_format = self._get_image_properties(image_path)

            # 解析檔案名稱中的時間戳記
            capture_timestamp = self._parse_timestamp_from_filename(original_filename)

            # 準備影像元資料
            image_data = {
                'image_id': image_id,
                'original_filename': original_filename,
                'local_file_path': str(Path(image_path).resolve()),
                'file_size': file_size,
                'file_hash': file_hash,
                'source_site': source_site,
                'source_line_id': source_line_id,
                'remote_image_path': remote_image_path,
                'download_url': download_url,
                'image_width': image_width,
                'image_height': image_height,
                'image_format': image_format,
                'capture_timestamp': capture_timestamp,
                'related_task_id': task_id,
                'processing_stage': 'downloaded'
            }

            # 添加產品資訊
            if product_info:
                image_data.update({
                    'product_name': product_info.get('product_name'),
                    'component_name': product_info.get('component_name'),
                    'board_info': product_info.get('board_info'),
                    'light_condition': product_info.get('light_condition')
                })

            # 儲存到資料庫
            if self.db.save_image_metadata(image_data):
                self.logger.info(f"成功記錄影像元資料: {image_id}")
                return image_id
            else:
                self.logger.error(f"記錄影像元資料失敗: {original_filename}")
                return None

        except Exception as e:
            self.logger.error(f"記錄影像元資料時發生錯誤: {e}")
            return None

    def classify_image(self,
                      image_id: str,
                      classification_label: str,
                      confidence: float = None,
                      is_manual: bool = True,
                      notes: str = None) -> bool:
        """
        對影像進行分類

        Args:
            image_id: 影像ID
            classification_label: 分類標籤
            confidence: 分類信心度
            is_manual: 是否為手動分類
            notes: 分類備註

        Returns:
            bool: 分類是否成功
        """
        try:
            success = self.db.update_image_classification(
                image_id=image_id,
                classification_label=classification_label,
                confidence=confidence,
                is_manual=is_manual,
                notes=notes
            )

            if success:
                self.logger.info(f"成功分類影像 {image_id}: {classification_label}")

                # 更新處理階段
                self._update_processing_stage(image_id, 'classified')

            return success

        except Exception as e:
            self.logger.error(f"分類影像時發生錯誤: {e}")
            return False

    def batch_classify_images(self,
                            classifications: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        批次分類影像

        Args:
            classifications: 分類資訊列表，每個元素包含 image_id, classification_label 等

        Returns:
            Tuple[int, int]: (成功數量, 失敗數量)
        """
        success_count = 0
        failure_count = 0

        for classification in classifications:
            try:
                success = self.classify_image(
                    image_id=classification['image_id'],
                    classification_label=classification['classification_label'],
                    confidence=classification.get('confidence'),
                    is_manual=classification.get('is_manual', True),
                    notes=classification.get('notes')
                )

                if success:
                    success_count += 1
                else:
                    failure_count += 1

            except Exception as e:
                self.logger.error(f"批次分類影像失敗: {e}")
                failure_count += 1

        self.logger.info(f"批次分類完成: 成功 {success_count}, 失敗 {failure_count}")
        return success_count, failure_count

    def get_images_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        """獲取任務相關的所有影像"""
        return self.db.get_images_by_task(task_id)

    def get_images_by_classification(self,
                                   classification_label: str,
                                   site: str = None,
                                   line_id: str = None) -> List[Dict[str, Any]]:
        """根據分類標籤獲取影像"""
        return self.db.get_images_by_classification(classification_label, site, line_id)

    def get_classification_statistics(self,
                                    task_id: str = None,
                                    site: str = None,
                                    line_id: str = None) -> Dict[str, Any]:
        """
        獲取分類統計資訊

        Args:
            task_id: 可選的任務ID過濾
            site: 可選的站點過濾
            line_id: 可選的產線過濾

        Returns:
            統計資訊字典
        """
        try:
            # 取得基本統計
            stats = self.db.get_image_statistics()

            # 如果指定了過濾條件，重新計算統計
            if task_id or site or line_id:
                # 這裡可以加入更複雜的過濾邏輯
                pass

            return stats

        except Exception as e:
            self.logger.error(f"獲取分類統計時發生錯誤: {e}")
            return {}

    def mark_image_corrupted(self, image_id: str, is_corrupted: bool = True) -> bool:
        """標記影像是否損壞"""
        try:
            image_data = self.db.get_image_metadata(image_id)
            if not image_data:
                return False

            image_data['is_corrupted'] = is_corrupted
            image_data['processing_stage'] = 'corrupted' if is_corrupted else 'downloaded'

            return self.db.save_image_metadata(image_data)

        except Exception as e:
            self.logger.error(f"標記影像損壞狀態時發生錯誤: {e}")
            return False

    def _generate_image_id(self, filename: str, site: str, line_id: str) -> str:
        """生成唯一的影像ID"""
        timestamp = datetime.now().isoformat()
        unique_string = f"{site}_{line_id}_{filename}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _calculate_file_hash(self, file_path: str) -> str:
        """計算檔案的MD5雜湊值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"計算檔案雜湊值失敗: {e}")
            return None

    def _get_image_properties(self, image_path: str) -> Tuple[int, int, str]:
        """獲取影像屬性 (寬度, 高度, 格式)"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                image_format = img.format.lower() if img.format else 'unknown'
                return width, height, image_format
        except Exception as e:
            self.logger.error(f"獲取影像屬性失敗: {e}")
            return None, None, 'unknown'

    def _parse_timestamp_from_filename(self, filename: str) -> str:
        """
        從檔案名稱解析時間戳記
        預期格式: timestamp_board@component_index_light.jpg
        """
        try:
            # 移除副檔名
            name_without_ext = os.path.splitext(filename)[0]

            # 分割檔案名稱
            parts = name_without_ext.split('_')

            if len(parts) > 0:
                timestamp_str = parts[0]

                # 嘗試解析時間戳記
                if len(timestamp_str) >= 14:  # YYYYMMDDHHMMSS
                    try:
                        timestamp = datetime.strptime(timestamp_str[:14], '%Y%m%d%H%M%S')
                        return timestamp.isoformat()
                    except ValueError:
                        pass

                # 如果是Unix時間戳記
                try:
                    timestamp = datetime.fromtimestamp(int(timestamp_str))
                    return timestamp.isoformat()
                except (ValueError, OSError):
                    pass

            # 如果無法解析，返回當前時間
            return datetime.now().isoformat()

        except Exception as e:
            self.logger.error(f"解析時間戳記失敗: {e}")
            return datetime.now().isoformat()

    def _update_processing_stage(self, image_id: str, stage: str) -> bool:
        """更新影像處理階段"""
        try:
            image_data = self.db.get_image_metadata(image_id)
            if image_data:
                image_data['processing_stage'] = stage
                return self.db.save_image_metadata(image_data)
            return False

        except Exception as e:
            self.logger.error(f"更新處理階段失敗: {e}")
            return False

    def validate_image_integrity(self, image_id: str) -> bool:
        """驗證影像完整性"""
        try:
            image_data = self.db.get_image_metadata(image_id)
            if not image_data:
                return False

            file_path = image_data['local_file_path']

            # 檢查檔案是否存在
            if not os.path.exists(file_path):
                self.mark_image_corrupted(image_id, True)
                return False

            # 重新計算雜湊值並比較
            current_hash = self._calculate_file_hash(file_path)
            stored_hash = image_data.get('file_hash')

            if stored_hash and current_hash != stored_hash:
                self.mark_image_corrupted(image_id, True)
                return False

            # 嘗試開啟影像檔案
            try:
                with Image.open(file_path) as img:
                    img.verify()
                return True
            except Exception:
                self.mark_image_corrupted(image_id, True)
                return False

        except Exception as e:
            self.logger.error(f"驗證影像完整性時發生錯誤: {e}")
            return False

    def cleanup_orphaned_files(self, dry_run: bool = True) -> List[str]:
        """
        清理孤立的影像檔案 (資料庫中沒有記錄的檔案)

        Args:
            dry_run: 是否只是模擬運行，不實際刪除

        Returns:
            List[str]: 被清理的檔案列表
        """
        orphaned_files = []

        try:
            # 這個功能需要掃描本地檔案系統並與資料庫記錄比較
            # 實作時需要定義影像檔案的根目錄
            self.logger.info("清理孤立檔案功能需要根據具體需求實作")

        except Exception as e:
            self.logger.error(f"清理孤立檔案時發生錯誤: {e}")

        return orphaned_files