# 影像分類狀態持久化功能

## 功能概述

此功能為系統添加了影像分類狀態的持久化能力，允許用戶在分類影像時保存進度，並在下次打開分類頁面時恢復之前的選擇。

## 主要功能

### 1. 影像元資料管理
- **自動記錄影像元資料**：下載影像時自動記錄檔案資訊、來源、產品資訊等
- **檔案完整性檢查**：計算檔案雜湊值，驗證影像完整性
- **分類狀態追蹤**：記錄每張影像的分類標籤、信心度、是否手動分類等

### 2. 分類狀態持久化
- **部分保存功能**：用戶可以部分完成分類後保存進度
- **狀態恢復**：重新打開分類頁面時自動載入已保存的分類狀態
- **完成率統計**：顯示各類別的分類完成情況

### 3. 新增API端點

#### 獲取方向樣本（已更新）
```
GET /orientation/samples/{task_id}
```
回應現在包含 `current_orientation` 欄位，顯示已保存的方向選擇。

#### 部分保存方向選擇（新增）
```
POST /orientation/save/{task_id}
Content-Type: application/json

{
  "task_id": "your_task_id",
  "class_name": "ProductA_Component1",
  "orientation": "Up"
}
```

#### 獲取分類狀態（新增）
```
GET /orientation/status/{task_id}
```
回應包含：
- 總類別數
- 已完成類別數
- 完成率
- 各類別詳細狀態

#### 確認所有分類（已更新）
```
POST /orientation/confirm/{task_id}
```
現在會同時更新資料庫中的分類狀態。

## 資料庫結構

### 新增資料表：image_metadata

```sql
CREATE TABLE image_metadata (
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

    -- 產品信息
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

    FOREIGN KEY (related_task_id) REFERENCES training_tasks(task_id) ON DELETE SET NULL
);
```

## 使用方式

### 前端整合建議

1. **載入分類頁面時**：
   ```javascript
   // 獲取樣本影像和已保存的分類狀態
   const samples = await fetch(`/orientation/samples/${taskId}`);

   // 根據 current_orientation 欄位設定UI狀態
   samples.forEach(sample => {
     if (sample.current_orientation) {
       setOrientation(sample.class_name, sample.current_orientation);
     }
   });
   ```

2. **用戶選擇方向時**：
   ```javascript
   // 立即保存用戶的選擇
   await fetch(`/orientation/save/${taskId}`, {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       task_id: taskId,
       class_name: className,
       orientation: selectedOrientation
     })
   });
   ```

3. **顯示完成進度**：
   ```javascript
   // 獲取完成狀態
   const status = await fetch(`/orientation/status/${taskId}`);
   updateProgressBar(status.completion_rate);
   ```

4. **最終確認**：
   ```javascript
   // 確認所有分類並繼續訓練
   await fetch(`/orientation/confirm/${taskId}`, {
     method: 'POST',
     headers: { 'Content-Type': 'application/json' },
     body: JSON.stringify({
       task_id: taskId,
       orientations: allOrientations
     })
   });
   ```

## 核心類別

### ImageMetadataManager
負責影像元資料的管理和分類狀態的處理：
- `record_downloaded_image()` - 記錄下載影像的元資料
- `classify_image()` - 對單張影像進行分類
- `batch_classify_images()` - 批次分類多張影像
- `get_images_by_task()` - 獲取任務相關的影像
- `get_classification_statistics()` - 獲取分類統計資訊

### EnhancedImageDownloader
整合了元資料記錄功能的影像下載器：
- `download_image_with_metadata()` - 下載影像並記錄元資料
- `batch_download_with_metadata()` - 批次下載並記錄元資料
- `classify_downloaded_images()` - 對已下載影像進行分類

## 測試

運行測試腳本來驗證功能：
```bash
source activate torch2
python test_orientation_persistence_en.py
```

測試涵蓋：
- 影像元資料記錄
- 分類狀態保存和載入
- 批次分類功能
- 統計功能
- API整合

## 注意事項

1. **資料庫初始化**：新的資料表會在首次使用 `TaskDatabase` 時自動創建
2. **向後兼容**：現有功能不受影響，新功能為可選使用
3. **性能**：為資料表建立了適當的索引以優化查詢性能
4. **錯誤處理**：所有資料庫操作都包含適當的異常處理

## 擴展可能

- 支援更多分類類型（不僅限於方向）
- 添加分類信心度評估
- 實現分類歷史記錄
- 支援分類結果匯出
- 添加分類品質評分系統