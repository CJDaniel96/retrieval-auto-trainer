# 自動化訓練系統 - 專案結構與設置指南

## 專案目錄結構

```
auto_training_system/
├── main.py                      # 主程式入口（CLI）
├── api_service.py              # FastAPI服務
├── auto_training_system.py     # 核心訓練系統
├── requirements.txt            # 依賴套件
├── README.md                   # 專案說明文件
├── .gitignore                  # Git忽略檔案
├── configs/                    # 配置檔案目錄
│   ├── database_configs.json   # 資料庫配置
│   └── train_config.yaml       # 訓練配置模板
├── database/                   # 資料庫相關模組
│   ├── __init__.py
│   ├── amr_info.py            # 資料庫模型
│   └── sessions.py            # 資料庫連線管理
├── models/                     # 模型定義
│   ├── __init__.py
│   ├── base.py                # 基礎模型組件
│   └── hoam.py                # HOAM模型實作
├── losses/                     # 損失函數
│   ├── __init__.py
│   └── hybrid_margin.py       # 混合邊界損失
├── data/                      # 資料處理模組
│   ├── __init__.py
│   ├── transforms.py          # 資料轉換
│   └── statistics.py          # 統計計算
├── utils/                     # 工具函數
│   ├── __init__.py
│   ├── split_image.py         # 影像分類工具
│   ├── split_images_by_product_comp.py
│   └── train_test_split.py    # 資料集分割
├── train.py                   # Lightning訓練模組
├── logs/                      # 日誌目錄
├── temp_uploads/              # 臨時上傳目錄
└── outputs/                   # 輸出結果目錄
    └── training_YYYYMMDD_HHMMSS/
        ├── raw_data/          # 分類後的原始資料
        ├── dataset/           # 訓練資料集
        │   ├── train/
        │   ├── val/
        │   └── mean_std.json
        ├── model/             # 模型檔案
        │   ├── best_model.pt
        │   ├── checkpoints/
        │   └── train_config.json
        └── results/           # 評估結果
            ├── golden_samples/
            ├── evaluation_results.csv
            ├── confusion_matrix.png
            ├── class_accuracy.png
            ├── similarity_distribution.png
            └── summary_report.json
```

## 環境設置

### 1. 建立虛擬環境

```bash
# 使用 conda
conda create -n auto_train python=3.10
conda activate auto_train

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 2. 安裝依賴套件

創建 `requirements.txt`:

```txt
# Core dependencies
torch==2.1.2
torchvision==0.16.2
pytorch-lightning==2.5.1
pytorch-metric-learning==2.3.0
timm==0.9.12

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
sshtunnel==0.4.0

# API
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Data processing
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.2
Pillow==10.1.0
opencv-python==4.8.1.78

# Visualization
matplotlib==3.7.2
seaborn==0.13.0

# Configuration
hydra-core==1.3.2
omegaconf==2.3.0

# Utilities
tqdm==4.66.1
joblib==1.3.2
```

安裝：
```bash
pip install -r requirements.txt
```

### 3. 配置資料庫連線

編輯 `configs/database_configs.json`，填入您的資料庫資訊：

```json
{
    "HPH": {
        "SSHTUNNEL": {
            "ssh_address_or_host": "your_ssh_host",
            "ssh_username": "your_ssh_user",
            "ssh_password": "your_ssh_password"
        },
        "database": {
            "ENGINE": "postgresql",
            "NAME": "your_db_name",
            "USER": "your_db_user",
            "PASSWORD": "your_db_password",
            "HOST": "your_db_host",
            "PORT": 5432
        }
    }
}
```

## 使用方式

### 1. 命令列介面 (CLI)

```bash
# 基本使用
python main.py --input-dir /path/to/input --output-dir /path/to/output

# 指定專案和產線
python main.py --input-dir /path/to/input --output-dir /path/to/output --site HPH --site V31

# 使用自訂配置
python main.py --input-dir /path/to/input --output-dir /path/to/output --config custom_config.json
```

### 2. API服務

啟動API服務：
```bash
python api_service.py
```

API將在 `http://localhost:8000` 啟動

#### API端點：

- `GET /` - 健康檢查
- `POST /training/start` - 啟動新的訓練任務
- `GET /training/status/{task_id}` - 查詢任務狀態
- `GET /training/list` - 列出所有任務
- `GET /training/result/{task_id}` - 取得訓練結果
- `GET /training/download/{task_id}/{file_type}` - 下載結果檔案

#### 使用範例：

```python
import requests

# 啟動訓練
response = requests.post(
    "http://localhost:8000/training/start",
    json={
        "input_dir": "/path/to/input",
        "output_dir": "/path/to/output",
        "site": "HPH",
        "line_id": "V31",
        "max_epochs": 30
    }
)
task_id = response.json()["task_id"]

# 查詢狀態
status = requests.get(f"http://localhost:8000/training/status/{task_id}")
print(status.json())

# 下載模型
model_file = requests.get(f"http://localhost:8000/training/download/{task_id}/model")
with open("model.pt", "wb") as f:
    f.write(model_file.content)
```

### 3. Python API

```python
from auto_training_system import AutoTrainingSystem

# 建立系統實例
system = AutoTrainingSystem(config_path='configs/database_configs.json')

# 執行完整流程
system.run_full_pipeline(
    input_dir='/path/to/input',
    output_base_dir='/path/to/output',
    site='HPH',
    line_id='V31'
)
```

## 資料準備

### 輸入資料結構

輸入資料夾應包含以下結構：
```
input_folder/
├── OK/
│   ├── 20240101120000_board123@COMP1_001_R.jpg
│   ├── 20240101120001_board124@COMP2_002_G.jpg
│   └── ...
└── NG/
    ├── ng_image1.jpg
    ├── ng_image2.jpg
    └── ...
```

### 影像命名規則

OK資料夾中的影像必須遵循以下命名格式：
```
{timestamp}_{board_info}@{component_name}_{index}_{light_source}.jpg
```

- `timestamp`: 14位數字時間戳 (YYYYMMDDHHmmss)
- `board_info`: 板子資訊
- `component_name`: 元件名稱
- `index`: 序號
- `light_source`: 光源類型 (R/G/B等)

## 輸出說明

### 1. 模型檔案
- `best_model.pt`: 最佳模型權重
- `train_config.json`: 訓練配置
- `mean_std.json`: 資料集統計資訊

### 2. 評估結果
- `evaluation_results.csv`: 詳細評估結果
- `confusion_matrix.png`: 混淆矩陣
- `class_accuracy.png`: 各類別準確率
- `similarity_distribution.png`: 相似度分布圖

### 3. Golden Samples
- 每個類別的代表性樣本

### 4. 總結報告
- `summary_report.json`: 包含所有關鍵指標的JSON報告

## 進階配置

### 修改訓練參數

編輯 `auto_training_system.py` 中的 `train_config`:

```python
self.train_config = {
    'model': {
        'structure': 'HOAMV2',  # 或 'HOAM'
        'backbone': 'efficientnetv2_rw_s',
        'embedding_size': 128
    },
    'training': {
        'max_epochs': 40,
        'batch_size': 64,
        'lr': 3e-4
    }
}
```

### 新增模型

1. 在 `models/` 目錄下新增模型檔案
2. 在 `auto_training_system.py` 的 `train_model()` 方法中註冊新模型

### 自訂損失函數

1. 在 `losses/` 目錄下新增損失函數
2. 在訓練配置中指定使用新的損失函數

## 故障排除

### 常見問題

1. **資料庫連線失敗**
   - 檢查SSH隧道配置
   - 確認資料庫憑證正確
   - 檢查網路連線

2. **GPU記憶體不足**
   - 減少 batch_size
   - 使用較小的模型
   - 啟用混合精度訓練

3. **訓練過程中斷**
   - 檢查 checkpoint 目錄
   - 使用最後的 checkpoint 恢復訓練

## 效能優化建議

1. **資料載入優化**
   - 增加 num_workers
   - 使用資料快取
   - 預先計算 mean/std

2. **訓練加速**
   - 使用混合精度訓練
   - 調整 batch_size
   - 使用多GPU訓練

3. **記憶體優化**
   - 使用梯度累積
   - 減少模型大小
   - 優化資料管道

## 授權與支援

本專案使用 MIT 授權。如有問題，請聯繫技術支援團隊。
