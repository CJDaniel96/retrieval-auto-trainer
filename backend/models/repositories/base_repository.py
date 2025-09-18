"""
Repository基礎類別
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, TypeVar, Generic
import logging

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Repository基礎抽象類別
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def create(self, entity: T) -> T:
        """創建實體"""
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """根據ID獲取實體"""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """更新實體"""
        pass

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """刪除實體"""
        pass

    @abstractmethod
    async def list_all(self, **filters) -> List[T]:
        """列出所有實體"""
        pass


class MemoryRepository(BaseRepository[T]):
    """
    內存Repository實現
    """

    def __init__(self):
        super().__init__()
        self._storage: Dict[str, T] = {}

    async def create(self, entity: T) -> T:
        """創建實體"""
        entity_id = self._get_entity_id(entity)
        self._storage[entity_id] = entity
        return entity

    async def get_by_id(self, entity_id: str) -> Optional[T]:
        """根據ID獲取實體"""
        return self._storage.get(entity_id)

    async def update(self, entity: T) -> T:
        """更新實體"""
        entity_id = self._get_entity_id(entity)
        if entity_id in self._storage:
            self._storage[entity_id] = entity
            return entity
        raise ValueError(f"Entity with id {entity_id} not found")

    async def delete(self, entity_id: str) -> bool:
        """刪除實體"""
        if entity_id in self._storage:
            del self._storage[entity_id]
            return True
        return False

    async def list_all(self, **filters) -> List[T]:
        """列出所有實體"""
        entities = list(self._storage.values())
        if filters:
            entities = self._apply_filters(entities, filters)
        return entities

    def _get_entity_id(self, entity: T) -> str:
        """獲取實體ID"""
        # 假設實體有id或task_id屬性
        if hasattr(entity, 'id'):
            return entity.id
        elif hasattr(entity, 'task_id'):
            return entity.task_id
        else:
            raise ValueError("Entity must have 'id' or 'task_id' attribute")

    def _apply_filters(self, entities: List[T], filters: Dict[str, Any]) -> List[T]:
        """應用過濾器"""
        filtered = entities
        for key, value in filters.items():
            filtered = [e for e in filtered if getattr(e, key, None) == value]
        return filtered

    async def exists(self, entity_id: str) -> bool:
        """檢查實體是否存在"""
        return entity_id in self._storage

    async def count(self, **filters) -> int:
        """計算實體數量"""
        entities = await self.list_all(**filters)
        return len(entities)

    async def clear(self):
        """清空所有實體"""
        self._storage.clear()


class DatabaseRepository(BaseRepository[T]):
    """
    資料庫Repository基礎類別
    """

    def __init__(self, db_session=None):
        super().__init__()
        self.db_session = db_session

    def _get_session(self):
        """獲取資料庫會話"""
        if self.db_session:
            return self.db_session
        # 這裡可以實現session獲取邏輯
        raise NotImplementedError("Database session not available")

    @abstractmethod
    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """將實體轉換為字典"""
        pass

    @abstractmethod
    def _dict_to_entity(self, data: Dict[str, Any]) -> T:
        """將字典轉換為實體"""
        pass

    @abstractmethod
    def _get_table_name(self) -> str:
        """獲取表名"""
        pass