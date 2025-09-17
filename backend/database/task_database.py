#!/usr/bin/env python3
"""
訓練任務數據庫管理
使用 SQLite 實現任務的持久化存儲
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager


class TaskDatabase:
    """訓練任務數據庫管理類"""

    def __init__(self, db_path: str = "tasks.db"):
        """
        初始化數據庫

        Args:
            db_path: 數據庫文件路徑
        """
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._init_database()

    def _init_database(self):
        """初始化數據庫表結構"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 創建任務表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    input_dir TEXT,
                    output_dir TEXT,
                    site TEXT,
                    line_id TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    current_step TEXT,
                    progress REAL DEFAULT 0.0,
                    error_message TEXT,
                    config_override TEXT,  -- JSON格式存儲配置覆蓋
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 創建影像元資料表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS image_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    -- 影像基本信息
                    image_id TEXT UNIQUE NOT NULL,
                    original_filename TEXT NOT NULL,
                    local_file_path TEXT NOT NULL,
                    file_size INTEGER,
                    file_hash TEXT,

                    -- 來源信息
                    source_site TEXT NOT NULL,
                    source_line_id TEXT NOT NULL,
                    remote_image_path TEXT,
                    download_url TEXT,

                    -- 產品信息 (從遠端資料庫查詢得到)
                    product_name TEXT,
                    component_name TEXT,
                    board_info TEXT,
                    light_condition TEXT,

                    -- 分類信息
                    classification_label TEXT,
                    classification_confidence REAL,
                    is_manually_classified BOOLEAN DEFAULT FALSE,
                    classification_notes TEXT,

                    -- 影像屬性
                    image_width INTEGER,
                    image_height INTEGER,
                    image_format TEXT,
                    capture_timestamp DATETIME,

                    -- 任務關聯
                    related_task_id TEXT,
                    processing_stage TEXT DEFAULT 'downloaded',

                    -- 質量控制
                    is_corrupted BOOLEAN DEFAULT FALSE,
                    quality_score REAL,
                    has_annotation BOOLEAN DEFAULT FALSE,

                    -- 時間戳記
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

                    -- 外鍵約束
                    FOREIGN KEY (related_task_id) REFERENCES training_tasks(task_id) ON DELETE SET NULL
                )
            """)

            # 創建索引以提高查詢性能
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON training_tasks(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON training_tasks(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_site_line ON training_tasks(site, line_id)")

            # 影像元資料表索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_id ON image_metadata(image_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_classification_label ON image_metadata(classification_label)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_site_line ON image_metadata(source_site, source_line_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_product_component ON image_metadata(product_name, component_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_related_task ON image_metadata(related_task_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_stage ON image_metadata(processing_stage)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_img_created_at ON image_metadata(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON image_metadata(local_file_path)")

            # 創建觸發器以自動更新 updated_at 欄位
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS update_image_metadata_timestamp
                    AFTER UPDATE ON image_metadata
                BEGIN
                    UPDATE image_metadata SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                END
            """)

            conn.commit()
            self.logger.info("數據庫初始化完成")

    @contextmanager
    def _get_connection(self):
        """獲取數據庫連接的上下文管理器"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # 使結果可以通過列名訪問
        try:
            yield conn
        finally:
            conn.close()

    def save_task(self, task_data: Dict[str, Any]) -> bool:
        """
        保存或更新訓練任務

        Args:
            task_data: 任務數據字典

        Returns:
            bool: 操作是否成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                now = datetime.now().isoformat()

                # 檢查任務是否已存在
                cursor.execute("SELECT task_id FROM training_tasks WHERE task_id = ?", (task_data['task_id'],))
                exists = cursor.fetchone() is not None

                if exists:
                    # 更新現有任務
                    cursor.execute("""
                        UPDATE training_tasks SET
                            status = ?,
                            output_dir = ?,
                            end_time = ?,
                            current_step = ?,
                            progress = ?,
                            error_message = ?,
                            updated_at = ?
                        WHERE task_id = ?
                    """, (
                        task_data.get('status'),
                        task_data.get('output_dir'),
                        task_data.get('end_time'),
                        task_data.get('current_step'),
                        task_data.get('progress', 0.0),
                        task_data.get('error_message'),
                        now,
                        task_data['task_id']
                    ))
                else:
                    # 插入新任務
                    cursor.execute("""
                        INSERT INTO training_tasks (
                            task_id, status, input_dir, output_dir, site, line_id,
                            start_time, end_time, current_step, progress, error_message,
                            config_override, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task_data['task_id'],
                        task_data.get('status', 'pending'),
                        task_data.get('input_dir'),
                        task_data.get('output_dir'),
                        task_data.get('site'),
                        task_data.get('line_id'),
                        task_data.get('start_time'),
                        task_data.get('end_time'),
                        task_data.get('current_step'),
                        task_data.get('progress', 0.0),
                        task_data.get('error_message'),
                        json.dumps(task_data.get('config_override', {})),
                        now,
                        now
                    ))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"保存任務失敗: {e}")
            return False

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        獲取指定任務

        Args:
            task_id: 任務ID

        Returns:
            任務數據字典，如果不存在則返回 None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM training_tasks WHERE task_id = ?", (task_id,))
                row = cursor.fetchone()

                if row:
                    return self._row_to_dict(row)
                return None

        except Exception as e:
            self.logger.error(f"獲取任務失敗: {e}")
            return None

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        獲取所有任務

        Returns:
            任務列表
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM training_tasks ORDER BY created_at DESC")
                rows = cursor.fetchall()

                return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"獲取任務列表失敗: {e}")
            return []

    def delete_task(self, task_id: str) -> bool:
        """
        刪除任務

        Args:
            task_id: 任務ID

        Returns:
            bool: 操作是否成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM training_tasks WHERE task_id = ?", (task_id,))
                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            self.logger.error(f"刪除任務失敗: {e}")
            return False

    def get_tasks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        根據狀態獲取任務

        Args:
            status: 任務狀態

        Returns:
            任務列表
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM training_tasks WHERE status = ? ORDER BY created_at DESC", (status,))
                rows = cursor.fetchall()

                return [self._row_to_dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"獲取任務失敗: {e}")
            return []

    def cleanup_old_tasks(self, days: int = 30) -> int:
        """
        清理舊任務（可選功能）

        Args:
            days: 保留天數

        Returns:
            清理的任務數量
        """
        try:
            from datetime import timedelta
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM training_tasks
                    WHERE created_at < ? AND status IN ('completed', 'failed', 'cancelled')
                """, (cutoff_date,))
                conn.commit()
                return cursor.rowcount

        except Exception as e:
            self.logger.error(f"清理舊任務失敗: {e}")
            return 0

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """
        將數據庫行轉換為字典

        Args:
            row: 數據庫行

        Returns:
            字典格式的任務數據
        """
        data = dict(row)

        # 解析 JSON 字段
        if data.get('config_override'):
            try:
                data['config_override'] = json.loads(data['config_override'])
            except:
                data['config_override'] = {}
        else:
            data['config_override'] = {}

        return data

    def get_database_stats(self) -> Dict[str, Any]:
        """
        獲取數據庫統計信息

        Returns:
            統計信息字典
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 總任務數
                cursor.execute("SELECT COUNT(*) as total FROM training_tasks")
                total = cursor.fetchone()['total']

                # 各狀態統計
                cursor.execute("""
                    SELECT status, COUNT(*) as count
                    FROM training_tasks
                    GROUP BY status
                """)
                status_stats = {row['status']: row['count'] for row in cursor.fetchall()}

                return {
                    'total_tasks': total,
                    'status_distribution': status_stats,
                    'database_path': str(self.db_path),
                    'database_size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0
                }

        except Exception as e:
            self.logger.error(f"獲取統計信息失敗: {e}")
            return {}

    def save_image_metadata(self, image_data: Dict[str, Any]) -> bool:
        """
        保存影像元資料

        Args:
            image_data: 影像元資料字典

        Returns:
            bool: 操作是否成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 檢查影像是否已存在
                cursor.execute("SELECT image_id FROM image_metadata WHERE image_id = ?", (image_data['image_id'],))
                exists = cursor.fetchone() is not None

                if exists:
                    # 更新現有影像資料
                    cursor.execute("""
                        UPDATE image_metadata SET
                            classification_label = ?,
                            classification_confidence = ?,
                            is_manually_classified = ?,
                            classification_notes = ?,
                            processing_stage = ?,
                            is_corrupted = ?,
                            quality_score = ?,
                            has_annotation = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE image_id = ?
                    """, (
                        image_data.get('classification_label'),
                        image_data.get('classification_confidence'),
                        image_data.get('is_manually_classified', False),
                        image_data.get('classification_notes'),
                        image_data.get('processing_stage', 'downloaded'),
                        image_data.get('is_corrupted', False),
                        image_data.get('quality_score'),
                        image_data.get('has_annotation', False),
                        image_data['image_id']
                    ))
                else:
                    # 插入新影像資料
                    cursor.execute("""
                        INSERT INTO image_metadata (
                            image_id, original_filename, local_file_path, file_size, file_hash,
                            source_site, source_line_id, remote_image_path, download_url,
                            product_name, component_name, board_info, light_condition,
                            classification_label, classification_confidence, is_manually_classified, classification_notes,
                            image_width, image_height, image_format, capture_timestamp,
                            related_task_id, processing_stage,
                            is_corrupted, quality_score, has_annotation
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        image_data['image_id'],
                        image_data['original_filename'],
                        image_data['local_file_path'],
                        image_data.get('file_size'),
                        image_data.get('file_hash'),
                        image_data['source_site'],
                        image_data['source_line_id'],
                        image_data.get('remote_image_path'),
                        image_data.get('download_url'),
                        image_data.get('product_name'),
                        image_data.get('component_name'),
                        image_data.get('board_info'),
                        image_data.get('light_condition'),
                        image_data.get('classification_label'),
                        image_data.get('classification_confidence'),
                        image_data.get('is_manually_classified', False),
                        image_data.get('classification_notes'),
                        image_data.get('image_width'),
                        image_data.get('image_height'),
                        image_data.get('image_format'),
                        image_data.get('capture_timestamp'),
                        image_data.get('related_task_id'),
                        image_data.get('processing_stage', 'downloaded'),
                        image_data.get('is_corrupted', False),
                        image_data.get('quality_score'),
                        image_data.get('has_annotation', False)
                    ))

                conn.commit()
                return True

        except Exception as e:
            self.logger.error(f"保存影像元資料失敗: {e}")
            return False

    def get_image_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        獲取指定影像元資料

        Args:
            image_id: 影像ID

        Returns:
            影像元資料字典，如果不存在則返回 None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM image_metadata WHERE image_id = ?", (image_id,))
                row = cursor.fetchone()

                if row:
                    return dict(row)
                return None

        except Exception as e:
            self.logger.error(f"獲取影像元資料失敗: {e}")
            return None

    def get_images_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        """
        根據任務ID獲取影像列表

        Args:
            task_id: 任務ID

        Returns:
            影像列表
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM image_metadata WHERE related_task_id = ? ORDER BY created_at DESC", (task_id,))
                rows = cursor.fetchall()

                return [dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"獲取任務影像失敗: {e}")
            return []

    def get_images_by_classification(self, classification_label: str, site: str = None, line_id: str = None) -> List[Dict[str, Any]]:
        """
        根據分類標籤獲取影像列表

        Args:
            classification_label: 分類標籤
            site: 可選的站點過濾
            line_id: 可選的產線過濾

        Returns:
            影像列表
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM image_metadata WHERE classification_label = ?"
                params = [classification_label]

                if site:
                    query += " AND source_site = ?"
                    params.append(site)

                if line_id:
                    query += " AND source_line_id = ?"
                    params.append(line_id)

                query += " ORDER BY created_at DESC"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [dict(row) for row in rows]

        except Exception as e:
            self.logger.error(f"獲取分類影像失敗: {e}")
            return []

    def update_image_classification(self, image_id: str, classification_label: str,
                                  confidence: float = None, is_manual: bool = True,
                                  notes: str = None) -> bool:
        """
        更新影像分類

        Args:
            image_id: 影像ID
            classification_label: 分類標籤
            confidence: 分類信心度
            is_manual: 是否為手動分類
            notes: 分類備註

        Returns:
            bool: 操作是否成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE image_metadata SET
                        classification_label = ?,
                        classification_confidence = ?,
                        is_manually_classified = ?,
                        classification_notes = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE image_id = ?
                """, (classification_label, confidence, is_manual, notes, image_id))

                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            self.logger.error(f"更新影像分類失敗: {e}")
            return False

    def delete_image_metadata(self, image_id: str) -> bool:
        """
        刪除影像元資料

        Args:
            image_id: 影像ID

        Returns:
            bool: 操作是否成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM image_metadata WHERE image_id = ?", (image_id,))
                conn.commit()
                return cursor.rowcount > 0

        except Exception as e:
            self.logger.error(f"刪除影像元資料失敗: {e}")
            return False

    def get_image_statistics(self) -> Dict[str, Any]:
        """
        獲取影像統計信息

        Returns:
            統計信息字典
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 總影像數
                cursor.execute("SELECT COUNT(*) as total FROM image_metadata")
                total = cursor.fetchone()['total']

                # 各分類統計
                cursor.execute("""
                    SELECT classification_label, COUNT(*) as count
                    FROM image_metadata
                    WHERE classification_label IS NOT NULL
                    GROUP BY classification_label
                """)
                classification_stats = {row['classification_label']: row['count'] for row in cursor.fetchall()}

                # 各站點統計
                cursor.execute("""
                    SELECT source_site, COUNT(*) as count
                    FROM image_metadata
                    GROUP BY source_site
                """)
                site_stats = {row['source_site']: row['count'] for row in cursor.fetchall()}

                # 處理階段統計
                cursor.execute("""
                    SELECT processing_stage, COUNT(*) as count
                    FROM image_metadata
                    GROUP BY processing_stage
                """)
                stage_stats = {row['processing_stage']: row['count'] for row in cursor.fetchall()}

                return {
                    'total_images': total,
                    'classification_distribution': classification_stats,
                    'site_distribution': site_stats,
                    'processing_stage_distribution': stage_stats
                }

        except Exception as e:
            self.logger.error(f"獲取影像統計信息失敗: {e}")
            return {}