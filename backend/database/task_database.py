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

            # 創建索引以提高查詢性能
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON training_tasks(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON training_tasks(created_at)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_site_line ON training_tasks(site, line_id)")

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