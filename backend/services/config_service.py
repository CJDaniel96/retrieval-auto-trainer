"""
Config Service - 配置管理服務業務邏輯
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from .base_service import BaseService


class ConfigService(BaseService):
    """
    配置管理服務
    """

    def __init__(self):
        super().__init__()
        self.config_dir = "backend/configs"
        self.system_config_path = os.path.join(self.config_dir, "configs.yaml")
        self.train_config_path = os.path.join(self.config_dir, "train_configs.yaml")
        self.database_config_path = os.path.join(self.config_dir, "database_configs.json")

    async def get_current_config(self) -> Dict[str, Any]:
        """取得當前系統和訓練配置"""
        try:
            config = {}

            # 讀取系統配置
            if os.path.exists(self.system_config_path):
                with open(self.system_config_path, 'r', encoding='utf-8') as f:
                    config['system'] = yaml.safe_load(f)

            # 讀取訓練配置
            if os.path.exists(self.train_config_path):
                with open(self.train_config_path, 'r', encoding='utf-8') as f:
                    config['training'] = yaml.safe_load(f)

            return config
        except Exception as e:
            raise Exception(f"獲取配置失敗: {str(e)}")

    async def get_database_sites(self) -> Dict[str, Any]:
        """取得可用的資料庫站點列表"""
        try:
            if os.path.exists(self.database_config_path):
                with open(self.database_config_path, 'r', encoding='utf-8') as f:
                    db_configs = json.load(f)

                # 提取站點基本資訊，不包含敏感密碼資訊
                sites = {}
                for site_id, site_config in db_configs.items():
                    sites[site_id] = {
                        "id": site_id,
                        "database_name": site_config.get("database", {}).get("NAME", ""),
                        "lines": list(site_config.get("image_pool", {}).keys()) if site_config.get("image_pool") else []
                    }

                return sites
            return {}
        except Exception as e:
            raise Exception(f"獲取資料庫站點失敗: {str(e)}")

    async def update_system_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """更新系統配置"""
        try:
            # 讀取當前配置
            current_config = {}
            if os.path.exists(self.system_config_path):
                with open(self.system_config_path, 'r', encoding='utf-8') as f:
                    current_config = yaml.safe_load(f)

            # 深度合併更新
            self._deep_update(current_config, updates)

            # 寫回文件
            with open(self.system_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)

            return current_config
        except Exception as e:
            raise Exception(f"更新系統配置失敗: {str(e)}")

    async def update_training_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """更新訓練配置"""
        try:
            # 讀取當前配置
            current_config = {}
            if os.path.exists(self.train_config_path):
                with open(self.train_config_path, 'r', encoding='utf-8') as f:
                    current_config = yaml.safe_load(f)

            # 深度合併更新
            self._deep_update(current_config, updates)

            # 寫回文件
            with open(self.train_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)

            return current_config
        except Exception as e:
            raise Exception(f"更新訓練配置失敗: {str(e)}")

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """深度更新字典"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value


_config_service_instance: Optional[ConfigService] = None

def get_config_service() -> ConfigService:
    """獲取配置服務實例"""
    global _config_service_instance
    if _config_service_instance is None:
        _config_service_instance = ConfigService()
    return _config_service_instance