# 資料庫分類訓練功能實現

## 功能概述

已成功實現在【開始新訓練】分頁中選擇使用已下載並分類的影像進行訓練，系統會根據資料庫中的分類標籤自動將影像組織為OK和NG類別。

## 實現的功能

### 1. 新的訓練模式

**傳統模式 (現有功能)**：
- 用戶上傳包含OK/NG子資料夾的輸入目錄
- 系統處理影像並查詢資料庫獲取產品資訊
- 影像按產品+元件+光照條件分組
- 用戶確認各組的方向（Up/Down/Left/Right）
- 影像根據方向進行旋轉和增強
- 使用方向類別進行訓練

**資料庫分類模式 (新功能)**：
- 用戶選擇已下載的料號（產品）
- 系統從資料庫載入已分類的影像
- 影像自動分組為OK（Up/Down/Left/Right → OK）和NG類別
- 跳過方向確認步驟（分類已確定）
- 直接使用OK/NG二元分類進行訓練

### 2. API擴展

#### 更新的模型
```python
class TrainingRequest(BaseModel):
    # 原有欄位...
    use_database_classification: bool = False  # 新增：是否使用資料庫分類模式
    part_numbers: Optional[List[str]] = None   # 新增：料號列表
```

#### 新增的API端點
```
GET /database/part_numbers?site=HPH&line_id=V31
```
回應範例：
```json
{
  "site": "HPH",
  "line_id": "V31",
  "part_numbers": {
    "PART001": {
      "total_images": 150,
      "classified_images": 120,
      "ok_images": 90,
      "ng_images": 30,
      "unclassified_images": 30,
      "classification_rate": 0.8
    }
  },
  "summary": {
    "total_part_numbers": 5,
    "total_images": 500,
    "total_classified": 400
  }
}
```

#### 更新的訓練端點
```
POST /training/start
```
新的請求範例：
```json
{
  "input_dir": "/dummy/path",  // 資料庫模式中會被忽略
  "site": "HPH",
  "line_id": "V31",
  "use_database_classification": true,
  "part_numbers": ["PART001", "PART002"],
  "max_epochs": 50,
  "batch_size": 32
}
```

### 3. 分類映射邏輯

系統會根據資料庫中的 `classification_label` 自動分組：

```python
# 分類映射規則
if classification_label in ['Up', 'Down', 'Left', 'Right']:
    target_category = 'OK'  # 方向分類 → OK類別
elif classification_label == 'NG':
    target_category = 'NG'  # NG分類 → NG類別
elif classification_label == 'OK':
    target_category = 'OK'  # 直接OK分類 → OK類別
else:
    # 未分類或其他分類，跳過
```

### 4. 核心實現

#### AutoTrainingSystem 新增方法
- `process_database_classified_images()` - 處理資料庫已分類影像
- `prepare_ok_ng_dataset()` - 準備OK/NG二元分類資料集

#### TaskDatabase 新增方法
- `get_all_images()` - 獲取所有影像元資料
- `get_images_by_site_and_line()` - 根據站點和產線獲取影像
- `get_images_by_part_numbers()` - 根據料號列表獲取影像

#### API服務新增函數
- `run_direct_training_task()` - 直接執行訓練（跳過方向確認）

## 使用流程

### 前端整合建議

1. **【開始新訓練】頁面新增選項**：
   ```html
   <input type="radio" name="training_mode" value="traditional"> 傳統模式（上傳影像資料夾）
   <input type="radio" name="training_mode" value="database"> 資料庫分類模式（選擇已下載料號）
   ```

2. **載入可用料號**：
   ```javascript
   const response = await fetch('/database/part_numbers?site=HPH&line_id=V31');
   const data = await response.json();

   // 顯示料號列表，包含分類統計
   data.part_numbers.forEach(partNumber => {
     displayPartNumber(partNumber.name, partNumber.total_images, partNumber.classification_rate);
   });
   ```

3. **提交訓練請求**：
   ```javascript
   const trainingRequest = {
     input_dir: "./dummy",  // 資料庫模式中會被忽略
     site: selectedSite,
     line_id: selectedLineId,
     use_database_classification: true,
     part_numbers: selectedPartNumbers,
     // 其他訓練參數...
   };

   const response = await fetch('/training/start', {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify(trainingRequest)
   });
   ```

4. **監控訓練進度**：
   ```javascript
   // 資料庫分類模式會跳過方向確認階段
   // 直接從 "pending" → "running" → "completed"
   ```

## 流程對比

### 傳統模式流程
1. 用戶上傳影像資料夾 →
2. 影像前處理和分組 →
3. **方向確認** →
4. 資料集準備 →
5. 模型訓練

### 資料庫分類模式流程
1. 用戶選擇料號 →
2. 載入已分類影像 →
3. ~~方向確認~~ (跳過) →
4. 資料集準備 →
5. 模型訓練

## 優勢

1. **效率提升**：跳過手動方向確認步驟
2. **一致性**：使用已驗證的分類標籤
3. **重用性**：充分利用已下載和分類的影像資源
4. **靈活性**：支援料號級別的選擇性訓練
5. **向後兼容**：不影響現有的傳統訓練流程

## 測試驗證

運行測試腳本驗證功能：
```bash
source activate torch2
python test_database_training.py
```

測試結果：
- ✅ 資料庫分類影像處理
- ✅ OK/NG分類映射
- ✅ 料號選擇API
- ✅ 直接訓練流程
- ✅ 整合驗證

## 資料夾結構

當使用資料庫分類模式時，系統會在輸出目錄中創建以下結構：

```
training_20240101_120000/
├── raw_data/           # 從資料庫載入的影像
│   ├── OK/            # 所有OK類別影像（Up/Down/Left/Right/OK）
│   └── NG/            # 所有NG類別影像
├── dataset/           # 訓練資料集
│   ├── train/
│   │   ├── OK/
│   │   └── NG/
│   └── val/
│       ├── OK/
│       └── NG/
├── model/             # 訓練結果
└── results/           # 評估結果
```

## 注意事項

1. **資料庫要求**：確保資料庫中有足夠的已分類影像
2. **檔案完整性**：系統會檢查影像檔案是否存在
3. **分類品質**：建議使用分類完成率較高的料號進行訓練
4. **平衡性**：注意OK和NG類別的影像數量平衡

這個實現確保了用戶可以在【開始新訓練】分頁中選擇使用已下載並分類的影像，系統會根據資料庫中的分類自動將影像組織為OK和NG類別進行訓練。