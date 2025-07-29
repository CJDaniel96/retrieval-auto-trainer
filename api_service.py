#!/usr/bin/env python3
"""
FastAPI服務 - 自動化訓練系統API介面
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from auto_training_system import AutoTrainingSystem


# Pydantic模型定義
class TrainingRequest(BaseModel):
    """訓練請求模型"""
    input_dir: str = Field(..., description="輸入資料夾路徑")
    output_dir: str = Field(..., description="輸出資料夾路徑")
    project: str = Field(default="HPH", description="專案名稱")
    site: str = Field(default="V31", description="產線ID")
    
    # 可選的訓練配置覆蓋
    max_epochs: Optional[int] = Field(None, description="最大訓練輪數")
    batch_size: Optional[int] = Field(None, description="批次大小")
    learning_rate: Optional[float] = Field(None, description="學習率")
    

class TrainingStatus(BaseModel):
    """訓練狀態模型"""
    task_id: str
    status: str  # pending, running, completed, failed
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    current_step: Optional[str]
    progress: Optional[float]
    error_message: Optional[str]
    output_dir: Optional[str]
    

class TrainingResult(BaseModel):
    """訓練結果模型"""
    task_id: str
    accuracy: float
    total_classes: int
    total_images: int
    model_path: str
    evaluation_csv: str
    confusion_matrix: str
    

# 建立FastAPI應用
app = FastAPI(
    title="自動化訓練系統API",
    description="用於影像檢索模型的自動化訓練服務",
    version="1.0.0"
)

# 全域變數
training_tasks: Dict[str, TrainingStatus] = {}
executor = ThreadPoolExecutor(max_workers=2)  # 限制同時訓練的任務數


@app.on_event("startup")
async def startup_event():
    """應用啟動事件"""
    # 確保必要的目錄存在
    Path("temp_uploads").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    

@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉事件"""
    executor.shutdown(wait=True)
    

@app.get("/", tags=["Health"])
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
    
    # 在背景執行訓練
    background_tasks.add_task(
        run_training_task,
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


# 輔助函數
def run_training_task(task_id: str, request: TrainingRequest):
    """
    在背景執行訓練任務
    
    Args:
        task_id: 任務ID
        request: 訓練請求
    """
    task = training_tasks[task_id]
    
    try:
        # 更新狀態為執行中
        task.status = "running"
        task.current_step = "初始化訓練系統"
        task.progress = 0.1
        
        # 建立訓練系統實例
        system = AutoTrainingSystem()
        
        # 覆蓋配置（如果有提供）
        if request.max_epochs:
            system.train_config['training']['max_epochs'] = request.max_epochs
        if request.batch_size:
            system.train_config['training']['batch_size'] = request.batch_size
        if request.learning_rate:
            system.train_config['training']['lr'] = request.learning_rate
            
        # 建立輸出目錄
        output_dir = Path(request.output_dir) / f"training_{task_id}"
        task.output_dir = str(output_dir)
        
        # 更新進度的回調函數
        def update_progress(step: str, progress: float):
            task.current_step = step
            task.progress = progress
            
        # 執行訓練流程
        # 注意：這裡需要修改AutoTrainingSystem以支援進度回調
        system.run_full_pipeline(
            input_dir=request.input_dir,
            output_base_dir=request.output_dir,
            project=request.project,
            site=request.site
        )
        
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
        

@app.post("/config/update", tags=["Configuration"])
async def update_training_config(config: Dict):
    """
    更新訓練配置
    
    Args:
        config: 新的配置字典
        
    Returns:
        更新後的配置
    """
    # TODO: 實作配置更新邏輯
    return {"message": "配置已更新", "config": config}
    

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
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
