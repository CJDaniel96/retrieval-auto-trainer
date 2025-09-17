# 修正後的資料庫分類訓練功能

## 重要修正

感謝您的指正！我已經修正了實現，現在資料庫訓練模式的流程與傳統模式完全一致，**也需要經過方向確認階段**。

## 修正後的流程

### 傳統模式
1. 用戶上傳包含 OK/NG 子資料夾的輸入目錄
2. 系統處理 OK 資料夾中的影像，查詢資料庫獲取產品資訊
3. 影像按 `產品名稱_元件名稱_光照條件` 分組
4. **用戶確認各組的方向** (Up/Down/Left/Right)
5. 影像根據方向進行旋轉和增強
6. 使用方向類別進行訓練

### 資料庫分類模式（修正後）
1. 用戶選擇已下載的料號（產品）
2. 系統從資料庫載入已分類的影像
3. **創建 OK/NG 資料夾結構**：將分類為 Up/Down/Left/Right/OK 的影像放入 OK 資料夾，NG 影像放入 NG 資料夾
4. **使用傳統的 `process_raw_images` 處理 OK 資料夾**，查詢資料庫獲取產品資訊
5. 影像按 `產品名稱_元件名稱_光照條件` 分組（**與傳統模式相同**）
6. **用戶確認各組的方向** (Up/Down/Left/Right) - **與傳統模式相同**
7. 後續訓練流程完全相同

## 關鍵修正點

### ✅ 修正前的錯誤理解
- ❌ 以為資料庫模式可以跳過方向確認
- ❌ 以為只需要簡單的 OK/NG 二元分類
- ❌ 實現了直接訓練功能

### ✅ 修正後的正確實現
- ✅ 資料庫模式**也需要方向確認**
- ✅ 兩種模式在得到 OK/NG 資料夾後的流程**完全相同**
- ✅ 都會按產品+元件+光照條件分組
- ✅ 都需要用戶確認方向

## 實現細節

### 修正後的 `process_database_classified_images` 方法

```python
def process_database_classified_images(self, input_dir: str, output_dir: str,
                                     site: str, line_id: str,
                                     part_numbers: List[str] = None) -> Dict[str, int]:
    """
    處理資料庫中已分類的影像，組織為OK/NG資料夾結構
    這個方法會創建與傳統模式相同的輸入結構，讓後續流程（包括方向確認）保持一致
    """

    # 第一步：建立 OK/NG 資料夾結構
    # Up/Down/Left/Right/OK → OK 資料夾
    # NG → NG 資料夾

    # 第二步：使用傳統的 process_raw_images 處理 OK 資料夾
    # 查詢資料庫獲取產品資訊，按產品+元件+光照分組
```

### API 流程修正

```python
# 兩種模式都會到達相同的狀態
task.status = "pending_orientation"
task.current_step = "等待方向確認"

# 移除了不正確的直接訓練邏輯
```

## 使用方式

### 前端無需大幅修改
原本計劃的前端邏輯基本正確，只是後續流程需要包含方向確認：

```javascript
// 1. 啟動資料庫分類訓練
const response = await fetch('/training/start', {
  method: 'POST',
  body: JSON.stringify({
    use_database_classification: true,
    part_numbers: ['PART001', 'PART002'],
    site: 'HPH',
    line_id: 'V31'
  })
});

// 2. 監控狀態 - 會到達 pending_orientation
const status = await fetch(`/training/status/${taskId}`);
// status.status === "pending_orientation" (與傳統模式相同)

// 3. 獲取方向確認樣本 (與傳統模式相同)
const samples = await fetch(`/orientation/samples/${taskId}`);

// 4. 用戶確認方向 (與傳統模式相同)
await fetch(`/orientation/confirm/${taskId}`, {
  method: 'POST',
  body: JSON.stringify({
    task_id: taskId,
    orientations: {
      "PART001_COMP1_R": "Up",
      "PART001_COMP2_G": "Down"
    }
  })
});

// 5. 訓練繼續 (與傳統模式相同)
```

## 資料夾結構對比

### 傳統模式輸入
```
user_input/
├── OK/                    # 用戶上傳
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── NG/                    # 用戶上傳
    ├── ng_image1.jpg
    └── ...
```

### 資料庫模式生成的相同結構
```
datasets/training_folder/   # 系統生成
├── OK/                     # 從資料庫分類生成
│   ├── image1.jpg         # classification_label: Up/Down/Left/Right/OK
│   ├── image2.jpg
│   └── ...
└── NG/                     # 從資料庫分類生成
    ├── ng_image1.jpg      # classification_label: NG
    └── ...
```

### 兩種模式的共同後續處理結果
```
outputs/site/line/training_xxx/
├── raw_data/              # 按產品分組的結果
│   ├── PART001_COMP1_R/   # 產品+元件+光照分組
│   ├── PART001_COMP2_G/
│   └── NG/
├── oriented_data/         # 方向確認後的結果
│   ├── Up/
│   ├── Down/
│   ├── Left/
│   ├── Right/
│   └── NG/
└── ...
```

## 優勢

1. **一致性**：兩種模式的訓練流程完全相同
2. **重用性**：可以利用已下載和分類的影像
3. **兼容性**：無需修改現有的方向確認和訓練邏輯
4. **靈活性**：支援料號級別的影像選擇

## 總結

修正後的資料庫分類訓練模式：
- ✅ 創建與傳統模式相同的 OK/NG 輸入結構
- ✅ 使用相同的產品資訊查詢和分組邏輯
- ✅ **需要經過方向確認階段**
- ✅ 後續訓練流程完全相同
- ✅ 充分利用已分類的影像資源

這個實現確保了資料庫訓練模式將實體影像正確分類到 OK 和 NG 資料夾，然後與傳統模式一樣經過完整的訓練流程，包括必要的方向確認步驟。