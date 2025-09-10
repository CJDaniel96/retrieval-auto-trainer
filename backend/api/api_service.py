#!/usr/bin/env python3
"""
FastAPI服務 - 自動化訓練系統API介面
"""

import os
import json
import asyncio
import logging
import random
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from ..core.auto_training_system import AutoTrainingSystem


# Pydantic模型定義
class TrainingRequest(BaseModel):
    """訓練請求模型"""
    input_dir: str = Field(..., description="輸入資料夾路徑")
    site: str = Field(default="HPH", description="地區名稱")
    line_id: str = Field(default="V31", description="產線ID")
    
    # 可選的訓練配置覆蓋
    max_epochs: Optional[int] = Field(None, description="最大訓練輪數")
    batch_size: Optional[int] = Field(None, description="批次大小")
    learning_rate: Optional[float] = Field(None, description="學習率")
    patience: Optional[int] = Field(None, description="EarlyStopping耐心值")
    enable_early_stopping: Optional[bool] = Field(None, description="是否啟用提前停止")
    

class ConfigUpdateRequest(BaseModel):
    """配置更新請求模型"""
    training: Optional[dict] = Field(None, description="訓練配置")
    model: Optional[dict] = Field(None, description="模型配置")
    data: Optional[dict] = Field(None, description="數據配置")
    loss: Optional[dict] = Field(None, description="損失函數配置")

class TrainingStatus(BaseModel):
    """訓練狀態模型"""
    task_id: str
    status: str  # pending, pending_orientation, running, completed, failed
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    current_step: Optional[str]
    progress: Optional[float]
    error_message: Optional[str]
    output_dir: Optional[str]
    
    
class OrientationSample(BaseModel):
    """方向確認樣本模型"""
    class_name: str
    sample_images: List[str]  # 3張樣本影像的路徑


class OrientationConfirmation(BaseModel):
    """方向確認請求模型"""
    task_id: str
    orientations: Dict[str, str]  # class_name -> orientation (Up, Down, Left, Right)
    

class TrainingResult(BaseModel):
    """訓練結果模型"""
    task_id: str
    accuracy: float
    total_classes: int
    total_images: int
    model_path: str
    evaluation_csv: str
    confusion_matrix: str
    

# 全域變數
training_tasks: Dict[str, TrainingStatus] = {}
executor = ThreadPoolExecutor(max_workers=2)  # 限制同時訓練的任務數

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期管理"""
    # 應用啟動時執行的程式碼
    # 確保必要的目錄存在
    Path("temp_uploads").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    yield
    # 應用關閉時執行的程式碼
    executor.shutdown(wait=True)

# 建立FastAPI應用
app = FastAPI(
    title="自動化訓練系統API",
    description="用於影像檢索模型的自動化訓練服務",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中間件（允許瀏覽器跨域請求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=FileResponse, tags=["UI"])
async def web_ui():
    return FileResponse("static/index.html")


@app.get("/favicon.ico", response_class=FileResponse, tags=["UI"])
async def favicon():
    # Return a simple response for favicon to avoid 404 errors
    return FileResponse("static/favicon.ico", status_code=200) if Path("static/favicon.ico").exists() else JSONResponse({"status": "no favicon"}, status_code=404)


# 掛載靜態檔案 - 這必須在所有具體路由之後
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/health", tags=["Health"])
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "service": "Auto Training System API",
        "version": "1.0.0"
    }
    

@app.post("/training/start", response_model=Dict[str, str], tags=["Training"])
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    啟動新的訓練任務
    
    Returns:
        Dict containing task_id
    """
    # 生成任務ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(training_tasks)}"
    
    # 驗證輸入目錄
    if not Path(request.input_dir).exists():
        raise HTTPException(status_code=400, detail=f"輸入目錄不存在: {request.input_dir}")
    
    # 初始化任務狀態
    training_tasks[task_id] = TrainingStatus(
        task_id=task_id,
        status="pending",
        start_time=datetime.now(),
        end_time=None,
        current_step="初始化",
        progress=0.0,
        error_message=None,
        output_dir=None
    )
    
    # 在背景執行前處理（影像分類）
    background_tasks.add_task(
        run_preprocessing_task,
        task_id=task_id,
        request=request
    )
    
    return {"task_id": task_id, "message": "訓練任務已啟動"}
    

@app.get("/training/status/{task_id}", response_model=TrainingStatus, tags=["Training"])
async def get_training_status(task_id: str):
    """
    查詢訓練任務狀態
    
    Args:
        task_id: 任務ID
        
    Returns:
        TrainingStatus: 任務狀態資訊
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        
    return training_tasks[task_id]
    

@app.get("/training/list", response_model=List[TrainingStatus], tags=["Training"])
async def list_training_tasks():
    """
    列出所有訓練任務
    
    Returns:
        List[TrainingStatus]: 所有任務的狀態列表
    """
    return list(training_tasks.values())
    

@app.get("/training/result/{task_id}", response_model=TrainingResult, tags=["Training"])
async def get_training_result(task_id: str):
    """
    取得訓練結果
    
    Args:
        task_id: 任務ID
        
    Returns:
        TrainingResult: 訓練結果資訊
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        
    task = training_tasks[task_id]
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"任務尚未完成: {task.status}")
        
    # 讀取結果
    output_dir = Path(task.output_dir)
    summary_path = output_dir / "summary_report.json"
    
    if not summary_path.exists():
        raise HTTPException(status_code=500, detail="找不到訓練結果報告")
        
    with open(summary_path, 'r') as f:
        summary = json.load(f)
        
    return TrainingResult(
        task_id=task_id,
        accuracy=summary['evaluation_accuracy'],
        total_classes=summary['total_classes'],
        total_images=summary['total_images'],
        model_path=str(output_dir / "model" / "best_model.pt"),
        evaluation_csv=str(output_dir / "results" / "evaluation_results.csv"),
        confusion_matrix=str(output_dir / "results" / "confusion_matrix.png")
    )
    

@app.get("/training/download/{task_id}/{file_type}", tags=["Training"])
async def download_file(task_id: str, file_type: str):
    """
    下載訓練產生的檔案
    
    Args:
        task_id: 任務ID
        file_type: 檔案類型 (model, evaluation_csv, confusion_matrix, summary)
        
    Returns:
        檔案回應
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        
    task = training_tasks[task_id]
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"任務尚未完成: {task.status}")
        
    output_dir = Path(task.output_dir)
    
    # 根據檔案類型決定路徑
    file_paths = {
        "model": output_dir / "model" / "best_model.pt",
        "evaluation_csv": output_dir / "results" / "evaluation_results.csv",
        "confusion_matrix": output_dir / "results" / "confusion_matrix.png",
        "summary": output_dir / "summary_report.json",
        "class_accuracy": output_dir / "results" / "class_accuracy.png",
        "similarity_dist": output_dir / "results" / "similarity_distribution.png"
    }
    
    if file_type not in file_paths:
        raise HTTPException(status_code=400, detail=f"不支援的檔案類型: {file_type}")
        
    file_path = file_paths[file_type]
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"檔案不存在: {file_type}")
        
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type='application/octet-stream'
    )
    

@app.post("/training/cancel/{task_id}", tags=["Training"])
async def cancel_training(task_id: str):
    """
    取消訓練任務
    
    Args:
        task_id: 任務ID
        
    Returns:
        訊息
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        
    task = training_tasks[task_id]
    
    if task.status in ["completed", "failed"]:
        raise HTTPException(status_code=400, detail=f"任務已結束: {task.status}")
        
    # TODO: 實作取消機制
    task.status = "cancelled"
    task.end_time = datetime.now()
    
    return {"message": f"任務 {task_id} 已取消"}
    

@app.delete("/training/delete/{task_id}", tags=["Training"])
async def delete_training_task(task_id: str, delete_files: bool = False):
    """
    刪除訓練任務記錄
    
    Args:
        task_id: 任務ID
        delete_files: 是否同時刪除輸出檔案
        
    Returns:
        訊息
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        
    task = training_tasks[task_id]
    
    # 刪除輸出檔案（如果指定）
    if delete_files and task.output_dir:
        output_dir = Path(task.output_dir)
        if output_dir.exists():
            import shutil
            shutil.rmtree(str(output_dir))
            
    # 刪除任務記錄
    del training_tasks[task_id]
    
    return {"message": f"任務 {task_id} 已刪除"}


@app.get("/orientation/samples/{task_id}", response_model=List[OrientationSample], tags=["Orientation"])
async def get_orientation_samples(task_id: str):
    """
    取得方向確認的樣本影像
    
    Args:
        task_id: 任務ID
        
    Returns:
        List[OrientationSample]: 各類別的樣本影像
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        
    task = training_tasks[task_id]
    if task.status != "pending_orientation":
        raise HTTPException(status_code=400, detail=f"任務狀態錯誤，期望 pending_orientation，實際: {task.status}")
    
    # 讀取已分類的影像資料
    raw_data_dir = Path(task.output_dir) / 'raw_data'
    if not raw_data_dir.exists():
        raise HTTPException(status_code=500, detail="找不到已分類的影像資料")
    
    samples = []
    for class_dir in raw_data_dir.iterdir():
        if not class_dir.is_dir() or class_dir.name == 'NG':
            continue
            
        # 隨機選取3張影像作為樣本
        images = list(class_dir.glob('*.jp*'))
        if len(images) >= 3:
            sample_images = random.sample(images, 3)
        else:
            sample_images = images
            
        # 將影像複製到臨時資料夾供web顯示
        temp_dir = Path("temp_uploads") / task_id / class_dir.name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        sample_paths = []
        for i, img in enumerate(sample_images):
            temp_path = temp_dir / f"sample_{i}_{img.name}"
            shutil.copy2(str(img), str(temp_path))
            # 返回相對路徑用於web顯示
            sample_paths.append(f"/temp/{task_id}/{class_dir.name}/sample_{i}_{img.name}")
            
        samples.append(OrientationSample(
            class_name=class_dir.name,
            sample_images=sample_paths
        ))
    
    return samples


@app.post("/orientation/confirm/{task_id}", tags=["Orientation"])
async def confirm_orientations(
    task_id: str, 
    confirmation: OrientationConfirmation,
    background_tasks: BackgroundTasks
):
    """
    確認方向並繼續訓練流程
    
    Args:
        task_id: 任務ID
        confirmation: 方向確認資料
        
    Returns:
        訊息
    """
    if task_id not in training_tasks:
        raise HTTPException(status_code=404, detail=f"找不到任務: {task_id}")
        
    task = training_tasks[task_id]
    if task.status != "pending_orientation":
        raise HTTPException(status_code=400, detail=f"任務狀態錯誤，期望 pending_orientation，實際: {task.status}")
    
    # 儲存方向確認資料
    orientation_file = Path("temp_uploads") / task_id / "orientations.json"
    with open(orientation_file, 'w', encoding='utf-8') as f:
        json.dump(confirmation.orientations, f, ensure_ascii=False, indent=2)
    
    # 繼續執行訓練流程
    background_tasks.add_task(
        run_orientation_and_training_task,
        task_id=task_id
    )
    
    # 更新任務狀態
    task.status = "running"
    task.current_step = "處理方向確認並開始訓練"
    task.progress = 0.3
    
    return {"message": "方向確認已收到，繼續訓練流程"}


@app.get("/temp/{task_id}/{class_name}/{filename}", tags=["Static"])
async def serve_temp_file(task_id: str, class_name: str, filename: str):
    """提供臨時影像檔案"""
    file_path = Path("temp_uploads") / task_id / class_name / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="檔案不存在")
    return FileResponse(str(file_path))


# 輔助函數
def run_preprocessing_task(task_id: str, request: TrainingRequest):
    """
    在背景執行前處理任務（影像分類）
    
    Args:
        task_id: 任務ID
        request: 訓練請求
    """
    task = training_tasks[task_id]
    
    try:
        # 更新狀態為執行中
        task.status = "running"
        task.current_step = "初始化訓練系統"
        task.progress = 0.05
        
        # 建立訓練系統實例
        system = AutoTrainingSystem()
        
        # 覆蓋配置（如果有提供）
        if request.max_epochs:
            system.train_config['training']['max_epochs'] = request.max_epochs
        if request.batch_size:
            system.train_config['training']['batch_size'] = request.batch_size
        if request.learning_rate:
            system.train_config['training']['lr'] = request.learning_rate
        if request.patience is not None:
            system.train_config['training']['patience'] = request.patience
        if request.enable_early_stopping is not None:
            system.train_config['training']['enable_early_stopping'] = request.enable_early_stopping
            
        # 複製輸入資料夾到 datasets/ 目錄下
        import shutil
        input_path = Path(request.input_dir)
        folder_name = input_path.name
        datasets_dir = Path("datasets")
        datasets_dir.mkdir(exist_ok=True)
        
        # 複製輸入資料夾到 datasets/
        copied_input_dir = datasets_dir / folder_name
        if copied_input_dir.exists():
            shutil.rmtree(copied_input_dir)
        
        task.current_step = "複製輸入資料到datasets目錄"
        task.progress = 0.05
        shutil.copytree(input_path, copied_input_dir)
        logging.info(f"已複製輸入資料夾到: {copied_input_dir}")
        
        # 建立輸出目錄結構: outputs/site/line/folder_name
        output_base = Path("outputs")
        output_dir = output_base / request.site / request.line_id / folder_name
        task.output_dir = str(output_dir)
        
        # 更新進度的回調函數
        def update_progress(step: str, progress: float):
            task.current_step = step
            task.progress = min(progress, 1.0)
            logging.info(f"任務 {task_id} 進度更新: {step} ({progress * 100:.1f}%)")
            
        system.set_progress_callback(update_progress)
        
        # 建立輸出目錄結構
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_data_dir = output_dir / 'raw_data'
        raw_data_dir.mkdir(exist_ok=True)
        
        # 執行前處理（使用複製後的資料夾）
        task.current_step = "處理原始影像資料"
        task.progress = 0.1
        system.process_raw_images(str(copied_input_dir), str(raw_data_dir), request.site, request.line_id)
        
        # 更新狀態等待方向確認
        task.status = "pending_orientation"
        task.current_step = "等待方向確認"
        task.progress = 0.2
        
    except Exception as e:
        # 更新狀態為失敗
        task.status = "failed"
        task.end_time = datetime.now()
        task.error_message = str(e)
        task.current_step = "前處理失敗"
        logging.error(f"前處理任務 {task_id} 失敗: {str(e)}", exc_info=True)


def run_orientation_and_training_task(task_id: str):
    """
    執行方向處理和訓練任務
    
    Args:
        task_id: 任務ID
    """
    task = training_tasks[task_id]
    
    try:
        # 讀取方向確認資料
        orientation_file = Path("temp_uploads") / task_id / "orientations.json"
        with open(orientation_file, 'r', encoding='utf-8') as f:
            orientations = json.load(f)
        
        # 建立訓練系統實例
        system = AutoTrainingSystem()
        
        # 更新進度的回調函數
        def update_progress(step: str, progress: float):
            task.current_step = step
            task.progress = min(progress, 1.0)
            logging.info(f"任務 {task_id} 進度更新: {step} ({progress * 100:.1f}%)")
            
        system.set_progress_callback(update_progress)
        
        # 執行方向處理和旋轉增強
        raw_data_dir = Path(task.output_dir) / 'raw_data'
        oriented_data_dir = Path(task.output_dir) / 'oriented_data'
        
        update_progress("處理方向分類", 0.35)
        system.process_orientations(str(raw_data_dir), str(oriented_data_dir), orientations)
        
        update_progress("執行旋轉增強", 0.45)
        system.apply_rotation_augmentation(str(oriented_data_dir))
        
        # 準備最終資料集
        dataset_dir = Path(task.output_dir) / 'dataset'
        update_progress("準備訓練資料集", 0.55)
        system.prepare_final_dataset(str(oriented_data_dir), str(dataset_dir))
        
        # 執行訓練
        model_dir = Path(task.output_dir) / 'model'
        model_dir.mkdir(exist_ok=True)
        
        update_progress("開始訓練模型", 0.6)
        model_path = system.train_model(str(dataset_dir), str(model_dir))
        
        # 生成結果
        results_dir = Path(task.output_dir) / 'results'
        results_dir.mkdir(exist_ok=True)
        
        update_progress("生成Golden Samples", 0.85)
        golden_samples = system.generate_golden_samples(str(dataset_dir), str(results_dir))
        
        update_progress("評估模型效能", 0.9)
        eval_results = system.evaluate_model(model_path, str(dataset_dir), golden_samples, str(results_dir))
        
        # 生成總結報告
        update_progress("生成總結報告", 0.95)
        system._generate_summary_report(Path(task.output_dir), {}, eval_results)
        
        # 清理臨時檔案
        temp_dir = Path("temp_uploads") / task_id
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
        
        # 更新狀態為完成
        task.status = "completed"
        task.end_time = datetime.now()
        task.progress = 1.0
        task.current_step = "訓練完成"
        
    except Exception as e:
        # 更新狀態為失敗
        task.status = "failed"
        task.end_time = datetime.now()
        task.error_message = str(e)
        task.current_step = "訓練失敗"
        logging.error(f"訓練任務 {task_id} 失敗: {str(e)}", exc_info=True)
        

@app.post("/config/update", tags=["Configuration"])
async def update_training_config(request: ConfigUpdateRequest):
    """
    更新訓練配置
    
    Args:
        request: 配置更新請求
        
    Returns:
        更新結果
    """
    try:
        import yaml
        from pathlib import Path
        
        config_file = Path(__file__).parent.parent / "configs" / "train_configs.yaml"
        
        # 讀取當前配置
        with open(config_file, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f)
        
        # 更新配置
        updated = False
        
        if request.training:
            current_config['training'].update(request.training)
            updated = True
            
        if request.model:
            current_config['model'].update(request.model)
            updated = True
            
        if request.data:
            current_config['data'].update(request.data)
            updated = True
            
        if request.loss:
            current_config['loss'].update(request.loss)
            updated = True
        
        # 保存更新後的配置
        if updated:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(current_config, f, default_flow_style=False, allow_unicode=True)
        
        return {
            "message": "配置已更新",
            "updated": updated,
            "config": current_config
        }
        
    except Exception as e:
        logger.error(f"更新配置失敗: {e}")
        raise HTTPException(status_code=500, detail=f"更新配置失敗: {str(e)}")
    

@app.get("/config/current", tags=["Configuration"])
async def get_current_config():
    """
    取得當前訓練配置
    
    Returns:
        當前配置
    """
    system = AutoTrainingSystem()
    return system.train_config


# 錯誤處理
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """全域錯誤處理"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    # 啟動服務
    uvicorn.run(
        "backend.api.api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
