# Windows 部署指南

## 自動化影像檢索模型訓練系統 - Windows 部署

本文檔說明如何在 Windows 環境中部署和運行自動化影像檢索模型訓練系統。

## 系統需求

### 必需軟件
- **Python 3.10+** - [下載安裝](https://www.python.org/downloads/)
- **Node.js 18+** - [下載安裝](https://nodejs.org/)
- **Git** - [下載安裝](https://git-scm.com/downloads)

### 硬件需求
- **CPU**: Intel i5 或更高 (建議 i7/i9)
- **記憶體**: 至少 8GB (建議 16GB+)
- **顯卡**: NVIDIA GPU (支援 CUDA，建議 8GB+ VRAM)
- **硬碟**: 至少 50GB 可用空間

### 可選軟件
- **NVIDIA CUDA Toolkit** - [下載安裝](https://developer.nvidia.com/cuda-downloads) (支援 GPU 加速訓練)

## 快速部署

### 1. 克隆項目
```bash
git clone <repository-url>
cd retrieval_auto_trainer
```

### 2. 執行一鍵部署
```bash
deploy.bat
```

這個腳本會自動：
- 檢查系統依賴
- 創建 Python 虛擬環境
- 安裝 Python 依賴
- 安裝 Node.js 依賴
- 構建前端生產版本
- 創建必要的目錄

### 3. 配置資料庫連接（如需要）
編輯 `backend\configs\database_configs.json`：
```json
{
    "YOUR_SITE": {
        "SSHTUNNEL": {
            "ssh_address_or_host": "YOUR_SSH_HOST",
            "ssh_username": "YOUR_SSH_USERNAME",
            "ssh_password": "YOUR_SSH_PASSWORD"
        },
        "database": {
            "ENGINE": "postgresql",
            "NAME": "YOUR_DB_NAME",
            "USER": "YOUR_DB_USER",
            "PASSWORD": "YOUR_DB_PASSWORD",
            "HOST": "YOUR_DB_HOST",
            "PORT": 5432
        },
        "image_pool": {
            "YOUR_LINE_ID": {
                "ip": "YOUR_IMAGE_SERVER_IP",
                "port": 8888,
                "donwload_url_prefix": "imagesinzip"
            }
        }
    }
}
```

### 4. 啟動服務
```bash
run_server.bat
```

### 5. 訪問應用
打開瀏覽器，訪問：http://localhost:8000

## 手動部署步驟

如果自動部署失敗，可以按照以下步驟手動部署：

### 1. 準備 Python 環境
```bash
# 創建虛擬環境
python -m venv venv

# 激活虛擬環境
venv\Scripts\activate.bat

# 升級 pip
python -m pip install --upgrade pip

# 安裝依賴
pip install -r requirements.txt
```

### 2. 準備前端環境
```bash
# 進入前端目錄
cd frontend

# 安裝依賴
npm install

# 構建生產版本
set NODE_ENV=production
npm run build

# 返回根目錄
cd ..
```

### 3. 創建必要目錄
```bash
mkdir datasets
mkdir outputs
mkdir rawdata
mkdir modules
mkdir logs
mkdir temp_uploads
```

### 4. 啟動服務
```bash
# 激活虛擬環境
venv\Scripts\activate.bat

# 設置 Python 路徑
set PYTHONPATH=%CD%

# 啟動後端服務
python -m backend.api.api_service
```

## 目錄結構

部署完成後的目錄結構：

```
retrieval_auto_trainer/
├── backend/              # 後端代碼
│   ├── api/             # API 服務
│   ├── core/            # 核心訓練邏輯
│   ├── services/        # 各種服務
│   └── configs/         # 配置文件
├── frontend/            # 前端代碼
│   ├── src/            # 源代碼
│   ├── public/         # 靜態資源
│   └── out/            # 構建後的文件
├── venv/               # Python 虛擬環境
├── datasets/           # 訓練數據集
├── outputs/           # 訓練輸出
├── rawdata/           # 原始下載數據
├── modules/           # 生成的模組
├── logs/              # 日誌文件
├── temp_uploads/      # 臨時文件
├── deploy.bat         # 部署腳本
├── run_server.bat     # 啟動腳本
└── requirements.txt   # Python 依賴
```

## 配置說明

### 環境變量
可以通過設置以下環境變量來自定義配置：
- `PYTHONPATH`: Python 模組搜索路徑
- `NODE_ENV`: Node.js 環境 (development/production)

### 資料庫配置
系統支援多個站點的資料庫連接配置，每個站點包含：
- SSH tunnel 設定（用於安全連接）
- PostgreSQL 資料庫連接設定
- 影像伺服器池配置

### 模型配置
訓練參數可在以下文件中調整：
- `backend/configs/configs.yaml` - 系統配置
- `backend/configs/train_configs.yaml` - 訓練配置

## 故障排除

### 常見問題

1. **Python 版本不符**
   - 確保安裝 Python 3.10 或更高版本
   - 檢查 `python --version`

2. **Node.js 版本不符**
   - 確保安裝 Node.js 18 或更高版本
   - 檢查 `node --version`

3. **依賴安裝失敗**
   - 檢查網路連接
   - 嘗試使用國內鏡像源：
     ```bash
     pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
     npm install --registry https://registry.npm.taobao.org
     ```

4. **CUDA 相關錯誤**
   - 檢查 NVIDIA 驅動是否安裝
   - 確認 CUDA 版本與 PyTorch 版本相容
   - 運行測試：`python -c "import torch; print(torch.cuda.is_available())"`

5. **前端構建失敗**
   - 清除 Node.js 快取：`npm cache clean --force`
   - 刪除 node_modules 並重新安裝：
     ```bash
     rm -rf frontend/node_modules
     cd frontend && npm install
     ```

6. **端口被占用**
   - 檢查端口 8000 是否被其他應用占用
   - 使用 `netstat -ano | findstr :8000` 查看占用情況

### 日誌查看
- 應用日誌：`logs/` 目錄
- Python 錯誤：檢查控制台輸出
- 前端錯誤：瀏覽器開發者工具

## 生產環境建議

### 安全性
1. 修改默認端口
2. 使用 HTTPS 證書
3. 設置防火牆規則
4. 定期更新依賴

### 性能優化
1. 使用 SSD 硬碟
2. 增加記憶體
3. 使用專用 GPU
4. 配置資料庫索引

### 監控
1. 設置系統資源監控
2. 配置應用程式監控
3. 設置錯誤警報
4. 定期備份數據

## 更新與維護

### 更新系統
```bash
# 停止服務
# 拉取最新代碼
git pull

# 更新 Python 依賴
venv\Scripts\activate.bat
pip install -r requirements.txt

# 更新前端依賴並重新構建
cd frontend
npm install
npm run build
cd ..

# 重新啟動服務
run_server.bat
```

### 數據備份
定期備份以下重要目錄：
- `outputs/` - 訓練結果
- `modules/` - 生成的模組
- `backend/configs/` - 配置文件
- `rawdata/` - 原始數據（可選）

## 技術支持

如果遇到問題，請：
1. 檢查本文檔的故障排除部分
2. 查看應用日誌
3. 聯繫技術支持團隊

---

**版本**: 1.0.0
**更新日期**: 2024年1月
**維護者**: 系統開發團隊