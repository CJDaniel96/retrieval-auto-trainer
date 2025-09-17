-- 影像元資料表結構設計
-- 用於記錄下載影像的元資料和分類信息

CREATE TABLE IF NOT EXISTS image_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- 影像基本信息
    image_id TEXT UNIQUE NOT NULL,                    -- 影像唯一識別碼
    original_filename TEXT NOT NULL,                 -- 原始檔案名稱
    local_file_path TEXT NOT NULL,                   -- 本地檔案路徑
    file_size INTEGER,                               -- 檔案大小 (bytes)
    file_hash TEXT,                                  -- 檔案MD5或SHA256雜湊值

    -- 來源信息
    source_site TEXT NOT NULL,                       -- 來源站點 (HPH, JQ, ZJ, NK, HZ)
    source_line_id TEXT NOT NULL,                    -- 來源產線ID
    remote_image_path TEXT,                          -- 遠端影像路徑
    download_url TEXT,                               -- 下載URL

    -- 產品信息 (從遠端資料庫查詢得到)
    product_name TEXT,                               -- 產品名稱
    component_name TEXT,                             -- 元件名稱
    board_info TEXT,                                 -- 板卡資訊
    light_condition TEXT,                            -- 光照條件

    -- 分類信息
    classification_label TEXT,                       -- 分類標籤 (OK/NG/Up/Down/Left/Right 等)
    classification_confidence REAL,                  -- 分類信心度 (0.0-1.0)
    is_manually_classified BOOLEAN DEFAULT FALSE,    -- 是否為手動分類
    classification_notes TEXT,                       -- 分類備註

    -- 影像屬性
    image_width INTEGER,                             -- 影像寬度
    image_height INTEGER,                            -- 影像高度
    image_format TEXT,                               -- 影像格式 (jpg, png, etc.)
    capture_timestamp DATETIME,                      -- 影像拍攝時間

    -- 任務關聯
    related_task_id TEXT,                            -- 關聯的訓練任務ID
    processing_stage TEXT DEFAULT 'downloaded',      -- 處理階段 (downloaded, classified, processed, trained)

    -- 質量控制
    is_corrupted BOOLEAN DEFAULT FALSE,              -- 是否損壞
    quality_score REAL,                              -- 影像品質分數
    has_annotation BOOLEAN DEFAULT FALSE,            -- 是否有標註

    -- 時間戳記
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- 外鍵約束
    FOREIGN KEY (related_task_id) REFERENCES training_tasks(task_id) ON DELETE SET NULL
);

-- 建立索引以提高查詢性能
CREATE INDEX IF NOT EXISTS idx_image_id ON image_metadata(image_id);
CREATE INDEX IF NOT EXISTS idx_classification_label ON image_metadata(classification_label);
CREATE INDEX IF NOT EXISTS idx_source_site_line ON image_metadata(source_site, source_line_id);
CREATE INDEX IF NOT EXISTS idx_product_component ON image_metadata(product_name, component_name);
CREATE INDEX IF NOT EXISTS idx_related_task ON image_metadata(related_task_id);
CREATE INDEX IF NOT EXISTS idx_processing_stage ON image_metadata(processing_stage);
CREATE INDEX IF NOT EXISTS idx_created_at ON image_metadata(created_at);
CREATE INDEX IF NOT EXISTS idx_file_path ON image_metadata(local_file_path);

-- 建立觸發器以自動更新 updated_at 欄位
CREATE TRIGGER IF NOT EXISTS update_image_metadata_timestamp
    AFTER UPDATE ON image_metadata
BEGIN
    UPDATE image_metadata SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;