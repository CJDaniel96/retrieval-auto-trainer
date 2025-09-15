#!/usr/bin/env python3
"""
任務管理器 - 整合內存對象和數據庫持久化
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from ..database import get_database


class TrainingStatus:
    """訓練狀態類"""

    def __init__(self, task_id: str, status: str = "pending", input_dir: str = None,
                 site: str = None, line_id: str = None, **kwargs):
        self.task_id = task_id
        self.status = status
        self.input_dir = input_dir
        self.output_dir = kwargs.get('output_dir')
        self.site = site
        self.line_id = line_id
        self.start_time = kwargs.get('start_time', datetime.now())
        self.end_time = kwargs.get('end_time')
        self.current_step = kwargs.get('current_step')
        self.progress = kwargs.get('progress', 0.0)
        self.error_message = kwargs.get('error_message')
        self.config_override = kwargs.get('config_override', {})

        # 自動保存到數據庫
        self._auto_save = True
        self._save_to_db()

    def _save_to_db(self):
        """保存到數據庫"""
        if not self._auto_save:
            return

        try:
            db = get_database()
            task_data = {
                'task_id': self.task_id,
                'status': self.status,
                'input_dir': self.input_dir,
                'output_dir': self.output_dir,
                'site': self.site,
                'line_id': self.line_id,
                'start_time': self.start_time.isoformat() if isinstance(self.start_time, datetime) else self.start_time,
                'end_time': self.end_time.isoformat() if isinstance(self.end_time, datetime) else self.end_time,
                'current_step': self.current_step,
                'progress': self.progress,
                'error_message': self.error_message,
                'config_override': self.config_override
            }
            db.save_task(task_data)
        except Exception as e:
            logging.error(f"保存任務到數據庫失敗: {e}")

    def __setattr__(self, name, value):
        """攔截屬性設置，自動同步到數據庫"""
        super().__setattr__(name, value)

        # 如果是重要屬性變更，則保存到數據庫
        if (hasattr(self, '_auto_save') and self._auto_save and
                name in ['status', 'current_step', 'progress', 'error_message', 'end_time', 'output_dir']):
            self._save_to_db()

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrainingStatus':
        """從字典創建對象"""
        # 禁用自動保存，避免重複保存
        obj = cls.__new__(cls)
        obj._auto_save = False

        obj.task_id = data['task_id']
        obj.status = data.get('status', 'pending')
        obj.input_dir = data.get('input_dir')
        obj.output_dir = data.get('output_dir')
        obj.site = data.get('site')
        obj.line_id = data.get('line_id')

        # 處理時間字段
        start_time = data.get('start_time')
        if isinstance(start_time, str):
            try:
                obj.start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            except:
                obj.start_time = datetime.now()
        else:
            obj.start_time = start_time or datetime.now()

        end_time = data.get('end_time')
        if isinstance(end_time, str) and end_time:
            try:
                obj.end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            except:
                obj.end_time = None
        else:
            obj.end_time = end_time

        obj.current_step = data.get('current_step')
        obj.progress = data.get('progress', 0.0)
        obj.error_message = data.get('error_message')
        obj.config_override = data.get('config_override', {})

        # 重新啟用自動保存
        obj._auto_save = True
        return obj

    def to_dict(self) -> Dict:
        """轉換為字典"""
        return {
            'task_id': self.task_id,
            'status': self.status,
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'site': self.site,
            'line_id': self.line_id,
            'start_time': self.start_time.isoformat() if isinstance(self.start_time, datetime) else self.start_time,
            'end_time': self.end_time.isoformat() if isinstance(self.end_time, datetime) else self.end_time,
            'current_step': self.current_step,
            'progress': self.progress,
            'error_message': self.error_message,
            'config_override': self.config_override
        }


class TaskManager:
    """任務管理器 - 統一管理內存和數據庫中的任務"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._memory_tasks: Dict[str, TrainingStatus] = {}
        self._load_tasks_from_db()

    def _load_tasks_from_db(self):
        """從數據庫加載任務到內存"""
        try:
            db = get_database()
            tasks_data = db.get_all_tasks()

            for task_data in tasks_data:
                task_obj = TrainingStatus.from_dict(task_data)
                self._memory_tasks[task_obj.task_id] = task_obj

            self.logger.info(f"從數據庫加載了 {len(tasks_data)} 個任務")

        except Exception as e:
            self.logger.error(f"從數據庫加載任務失敗: {e}")

    def create_task(self, task_id: str, **kwargs) -> TrainingStatus:
        """創建新任務"""
        task = TrainingStatus(task_id, **kwargs)
        self._memory_tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[TrainingStatus]:
        """獲取任務"""
        return self._memory_tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TrainingStatus]:
        """獲取所有任務"""
        return self._memory_tasks.copy()

    def delete_task(self, task_id: str) -> bool:
        """刪除任務"""
        try:
            # 從內存中刪除
            if task_id in self._memory_tasks:
                del self._memory_tasks[task_id]

            # 從數據庫中刪除
            db = get_database()
            return db.delete_task(task_id)

        except Exception as e:
            self.logger.error(f"刪除任務失敗: {e}")
            return False

    def task_exists(self, task_id: str) -> bool:
        """檢查任務是否存在"""
        return task_id in self._memory_tasks

    def get_tasks_by_status(self, status: str) -> List[TrainingStatus]:
        """根據狀態獲取任務"""
        return [task for task in self._memory_tasks.values() if task.status == status]

    def cleanup_old_tasks(self, days: int = 30) -> int:
        """清理舊任務"""
        try:
            db = get_database()
            deleted_count = db.cleanup_old_tasks(days)

            # 重新加載任務
            self._memory_tasks.clear()
            self._load_tasks_from_db()

            return deleted_count

        except Exception as e:
            self.logger.error(f"清理舊任務失敗: {e}")
            return 0


# 全局任務管理器實例
_task_manager_instance = None


def get_task_manager() -> TaskManager:
    """獲取任務管理器實例（單例模式）"""
    global _task_manager_instance
    if _task_manager_instance is None:
        _task_manager_instance = TaskManager()
    return _task_manager_instance