#!/usr/bin/env python3
"""
影像下載服務
基於 ImageDownloadCLI 的邏輯改寫
"""

import os
import json
import logging
import requests
import zipfile
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from sshtunnel import SSHTunnelForwarder
from tqdm import tqdm

from backend.services.database.sessions import create_session
from backend.services.database.amr_info import AmrRawData


class ImageDownloadService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.server = None
        self.config_path = Path("backend/configs/download_configs.json")

    def _get_configs(self) -> Dict:
        """載入下載配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)
        return configs

    def _get_ssh_tunnel_settings(self, configs: Dict, site: str) -> Dict:
        """取得 SSH tunnel 設定"""
        return configs[site]['SSHTUNNEL']

    def _get_database_settings(self, configs: Dict, site: str) -> Dict:
        """取得資料庫設定"""
        return configs[site]['database']

    def _get_download_url(self, ssh_tunnel: Dict, image_pool: Dict) -> str:
        """建立下載 URL"""
        if ssh_tunnel['ssh_address_or_host'] and ssh_tunnel['ssh_username'] and ssh_tunnel['ssh_password']:
            self.server = SSHTunnelForwarder(
                ssh_address_or_host=(ssh_tunnel['ssh_address_or_host'], 22),
                ssh_username=ssh_tunnel['ssh_username'],
                ssh_password=ssh_tunnel['ssh_password'],
                remote_bind_address=(image_pool['ip'], image_pool['port'])
            )
            self.server.start()
            local_host = '127.0.0.1'
            local_port = self.server.local_bind_port
        else:
            local_host = image_pool['ip']
            local_port = image_pool['port']

        download_url_prefix = image_pool['donwload_url_prefix']
        return f'http://{local_host}:{local_port}/{download_url_prefix}'

    def _get_image_list(
        self,
        ssh_tunnel: Dict,
        database: Dict,
        site: str,
        line_id: str,
        start_date: str,
        end_date: str,
        part_number: str,
        limit: Optional[int] = None
    ) -> List[str]:
        """從資料庫取得影像清單"""
        self.logger.info(f'正在從 {site} site {line_id} line 取得影像清單')

        with create_session(ssh_tunnel, database) as session:
            query = session.query(AmrRawData).filter(
                AmrRawData.create_time.between(start_date, end_date),
                AmrRawData.part_number == part_number,
                AmrRawData.line_id == line_id
            )

            if limit:
                query = query.limit(limit)

            data = query.all()
            self.logger.info(f'找到 {len(data)} 張影像從 {site} site {line_id} line')

        # 組建影像路徑清單
        images_list = []
        for obj in data:
            images_list.append(obj.image_path)

        return images_list

    def estimate_data_count(
        self,
        site: str,
        line_id: str,
        start_date: str,
        end_date: str,
        part_number: str
    ) -> Dict[str, Any]:
        """
        預估資料數量

        Args:
            site: 工廠名稱 (HPH, JQ 等)
            line_id: 線別 ID
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            part_number: 料號

        Returns:
            dict: 包含估計數量和狀態
        """
        try:
            configs = self._get_configs()

            # 檢查 site 是否存在
            if site not in configs:
                return {
                    "success": False,
                    "message": f"不支援的工廠: {site}",
                    "estimated_count": 0
                }

            ssh_tunnel = self._get_ssh_tunnel_settings(configs, site)
            database = self._get_database_settings(configs, site)
            image_pools = configs[site]['image_pool']

            # 檢查 line_id 是否存在
            if line_id not in image_pools:
                return {
                    "success": False,
                    "message": f"不支援的線別: {line_id}",
                    "estimated_count": 0
                }

            # 計算資料數量
            with create_session(ssh_tunnel, database) as session:
                count = session.query(AmrRawData).filter(
                    AmrRawData.create_time.between(start_date, end_date),
                    AmrRawData.part_number == part_number,
                    AmrRawData.line_id == line_id
                ).count()

                self.logger.info(f'預估 {site} site {line_id} line 有 {count} 張影像')

            return {
                "success": True,
                "message": f"找到 {count} 張符合條件的影像",
                "estimated_count": count
            }

        except Exception as e:
            self.logger.error(f"預估資料數量時發生錯誤: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"預估過程中發生錯誤: {str(e)}",
                "estimated_count": 0
            }

        finally:
            # 關閉 SSH tunnel（如果有的話）
            if hasattr(self, 'server') and self.server:
                self.server.stop()
                self.server = None

    def _download_images(
        self,
        image_list: List[str],
        download_url: str,
        output_zip_path: str
    ) -> bool:
        """下載影像並保存為 ZIP"""
        self.logger.info(f'正在下載 {len(image_list)} 張影像')
        self.logger.info(f'下載 URL: {download_url}')

        try:
            response = requests.post(download_url, json={
                "paths": image_list
            }, timeout=300)  # 5 分鐘超時

            if response.status_code == 200:
                total_size = len(response.content)

                # 寫入 ZIP 檔
                with open(output_zip_path, 'wb') as zip_file:
                    zip_file.write(response.content)

                self.logger.info(f'下載完成，檔案大小: {total_size} bytes')
                return True
            else:
                self.logger.error(f'下載失敗，狀態碼: {response.status_code}')
                return False

        except Exception as e:
            self.logger.error(f'下載過程中發生錯誤: {str(e)}')
            return False

    def _extract_zip(self, zip_path: str, extract_path: str) -> bool:
        """解壓縮 ZIP 檔案"""
        try:
            self.logger.info(f'正在解壓縮到: {extract_path}')

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            # 刪除原始 ZIP 檔
            os.remove(zip_path)
            self.logger.info('解壓縮完成，已刪除原始 ZIP 檔')
            return True

        except Exception as e:
            self.logger.error(f'解壓縮失敗: {str(e)}')
            return False

    def download_rawdata(
        self,
        site: str,
        line_id: str,
        start_date: str,
        end_date: str,
        part_number: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        下載原始資料

        Args:
            site: 工廠名稱 (HPH, JQ 等)
            line_id: 線別 ID
            start_date: 開始日期 (YYYY-MM-DD)
            end_date: 結束日期 (YYYY-MM-DD)
            part_number: 料號
            limit: 限制下載數量 (可選)

        Returns:
            dict: 包含成功狀態、訊息和結果路徑
        """
        try:
            configs = self._get_configs()

            # 檢查 site 是否存在
            if site not in configs:
                return {
                    "success": False,
                    "message": f"不支援的工廠: {site}",
                    "path": None
                }

            ssh_tunnel = self._get_ssh_tunnel_settings(configs, site)
            database = self._get_database_settings(configs, site)
            image_pools = configs[site]['image_pool']

            # 檢查 line_id 是否存在
            if line_id not in image_pools:
                return {
                    "success": False,
                    "message": f"不支援的線別: {line_id}",
                    "path": None
                }

            # 建立下載 URL
            download_url = self._get_download_url(ssh_tunnel, image_pools[line_id])

            # 強制限制最大下載數量為1000張（安全措施）
            max_download_limit = 1000
            if limit is None or limit > max_download_limit:
                actual_limit = max_download_limit
                self.logger.info(f"限制下載數量從 {limit} 調整為 {actual_limit}")
            else:
                actual_limit = limit

            # 取得影像清單
            image_list = self._get_image_list(
                ssh_tunnel, database, site, line_id,
                start_date, end_date, part_number, actual_limit
            )

            if not image_list:
                return {
                    "success": False,
                    "message": "找不到符合條件的影像",
                    "path": None
                }

            # 建立輸出路徑
            rawdata_dir = Path("rawdata")
            rawdata_dir.mkdir(exist_ok=True)

            output_dir = rawdata_dir / part_number
            output_dir.mkdir(exist_ok=True)

            # 暫時 ZIP 檔案路徑
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"{part_number}_{timestamp}.zip"
            zip_path = str(rawdata_dir / zip_filename)

            # 下載影像
            if not self._download_images(image_list, download_url, zip_path):
                return {
                    "success": False,
                    "message": "影像下載失敗",
                    "path": None
                }

            # 解壓縮
            if not self._extract_zip(zip_path, str(output_dir)):
                return {
                    "success": False,
                    "message": "解壓縮失敗",
                    "path": None
                }

            # 生成適當的成功訊息
            success_message = f"成功下載 {len(image_list)} 張影像"
            if limit is None and len(image_list) == max_download_limit:
                success_message += f" (已限制最大下載數量為 {max_download_limit} 張)"
            elif limit and limit > max_download_limit and len(image_list) == max_download_limit:
                success_message += f" (原請求 {limit} 張，已限制為 {max_download_limit} 張)"

            return {
                "success": True,
                "message": success_message,
                "path": str(output_dir),
                "image_count": len(image_list)
            }

        except Exception as e:
            self.logger.error(f"下載原始資料時發生錯誤: {str(e)}", exc_info=True)
            return {
                "success": False,
                "message": f"下載過程中發生錯誤: {str(e)}",
                "path": None
            }

        finally:
            # 關閉 SSH tunnel
            if self.server:
                self.server.stop()
                self.server = None

    def list_downloaded_parts(self) -> List[str]:
        """列出已下載的料號"""
        rawdata_dir = Path("rawdata")
        if not rawdata_dir.exists():
            return []

        part_numbers = []
        for item in rawdata_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                part_numbers.append(item.name)

        return sorted(part_numbers)

    def get_part_info(self, part_number: str) -> Optional[Dict[str, Any]]:
        """取得料號資訊"""
        rawdata_dir = Path("rawdata") / part_number
        if not rawdata_dir.exists():
            return None

        # 統計影像數量 - 使用快速掃描避免在下載期間阻塞
        image_count = 0
        try:
            import threading
            import time

            def count_images_fast(directory):
                """快速計算圖片數量，避免在下載期間阻塞"""
                count = 0
                try:
                    # 只掃描直接子目錄，避免遞歸掃描可能正在寫入的文件
                    for item in directory.iterdir():
                        if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                            count += 1
                        elif item.is_dir() and item.name in ['OK', 'NG']:
                            # 只掃描分類目錄的直接子文件
                            for subitem in item.iterdir():
                                if subitem.is_file() and subitem.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                    count += 1
                except (OSError, PermissionError):
                    # 如果目錄正在被其他進程使用，返回0
                    return 0
                return count

            # 使用線程和超時機制（跨平台解決方案）
            result = [0]  # 使用列表來儲存結果，因為內部函數需要修改外部變量
            exception_occurred = [False]

            def count_thread():
                try:
                    result[0] = count_images_fast(rawdata_dir)
                except Exception:
                    exception_occurred[0] = True

            thread = threading.Thread(target=count_thread)
            thread.daemon = True
            thread.start()
            thread.join(timeout=3.0)  # 3 秒超時

            if thread.is_alive() or exception_occurred[0]:
                # 如果線程仍在運行或發生異常，使用備用方法
                self.logger.warning(f"Directory scanning timeout for {part_number}, using fallback method")
                try:
                    # 只統計根目錄下的文件，不遞歸掃描
                    files = list(rawdata_dir.glob('*.jpg')) + list(rawdata_dir.glob('*.jpeg')) + \
                           list(rawdata_dir.glob('*.png')) + list(rawdata_dir.glob('*.bmp'))
                    image_count = len(files)
                except:
                    image_count = 0
            else:
                image_count = result[0]

        except Exception:
            # 如果掃描失敗，使用最簡單的估計方法
            try:
                # 只計算根目錄下的圖片文件
                image_count = len([f for f in rawdata_dir.iterdir()
                                 if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            except:
                image_count = 0

        # 檢查是否已分類（是否有OK/NG資料夾）
        ok_folder = rawdata_dir / "OK"
        ng_folder = rawdata_dir / "NG"
        is_classified = ok_folder.exists() and ng_folder.exists()

        # 計算分類後的影像數量 - 使用相同的快速掃描方法
        classified_count = 0
        if is_classified:
            try:
                def count_classified_thread():
                    nonlocal classified_count
                    try:
                        for folder in [ok_folder, ng_folder]:
                            if folder.exists():
                                for item in folder.iterdir():
                                    if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                        classified_count += 1
                    except:
                        pass

                thread = threading.Thread(target=count_classified_thread)
                thread.daemon = True
                thread.start()
                thread.join(timeout=2.0)  # 較短的超時時間

                if thread.is_alive():
                    classified_count = 0  # 超時則設為0
            except:
                classified_count = 0

        return {
            "part_number": part_number,
            "path": str(rawdata_dir),
            "image_count": image_count,
            "download_time": datetime.fromtimestamp(rawdata_dir.stat().st_mtime).isoformat(),
            "is_classified": is_classified,
            "classified_count": classified_count
        }