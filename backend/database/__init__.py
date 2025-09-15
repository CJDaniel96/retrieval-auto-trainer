"""
數據庫模組 - 任務持久化存儲
"""

from .task_database import TaskDatabase

# 全局數據庫實例
_db_instance = None


def get_database() -> TaskDatabase:
    """
    獲取數據庫實例（單例模式）

    Returns:
        TaskDatabase: 數據庫實例
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = TaskDatabase()
    return _db_instance


def init_database(db_path: str = "tasks.db"):
    """
    初始化數據庫

    Args:
        db_path: 數據庫文件路徑
    """
    global _db_instance
    _db_instance = TaskDatabase(db_path)


__all__ = ['TaskDatabase', 'get_database', 'init_database']