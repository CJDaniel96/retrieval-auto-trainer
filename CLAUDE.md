# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated training system for image retrieval models, primarily designed for industrial quality inspection. The system:
- Processes raw image data and classifies them using database metadata
- Trains HOAM/HOAMV2 models with PyTorch Lightning
- Generates evaluation results and golden samples
- Provides both CLI and API interfaces

## Key Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Note: This project uses Python 3.10+ with CUDA support for optimal performance
```

### Running the System
```bash
# CLI interface (main entry point)
python -m backend.main --input-dir /path/to/input --output-dir /path/to/output --site HPH --line-id V31

# API service
python -m backend.api_service
# API runs on http://localhost:8000 with web UI

# Direct training module (with Hydra config)
python -m backend.train
```

### Testing
```bash
# No specific test framework detected - check for pytest or unittest in development
python -m pytest  # if pytest is available
```

## Architecture Overview

### Core Components

1. **AutoTrainingSystem** (`backend/core/auto_training_system.py`): Main orchestrator class that runs the full pipeline:
   - Image processing with database queries
   - Dataset preparation 
   - Model training coordination
   - Evaluation and golden sample generation

2. **API Service** (`backend/api/api_service.py`): FastAPI-based REST API with:
   - Asynchronous training task management
   - Web UI for monitoring progress
   - File download endpoints

3. **Training Module** (`backend/train.py`): PyTorch Lightning implementation with:
   - `HOAMDataModule`: Data loading with automatic mean/std calculation
   - `LightningModel`: Training logic with multiple loss functions
   - Hydra configuration management

### Model Architecture

- **HOAM/HOAMV2** models (`backend/services/models/hoam.py`) using EfficientNetV2 backbones
- **Orthogonal fusion** of local and global features
- **Multiple loss functions** including HybridMarginLoss, ArcFaceLoss
- **Metric learning** approach for image retrieval

### Data Pipeline

The system expects specific input structure:
```
input_folder/
├── OK/     # Images with timestamp_board@component_index_light.jpg naming
└── NG/     # Negative samples
```

**Enhanced Pipeline with Orientation Confirmation:**
1. **Initial Processing**: Images classified by product_name + component_name from database
2. **Orientation Confirmation**: Web interface shows 3 random samples per class for user to confirm orientation (Up/Down/Left/Right)
3. **Rotation Augmentation**: Images rotated 90°, 180°, 270° to generate all orientations
4. **Final Dataset**: 5 folders (Up, Down, Left, Right, NG) split into train/val

Output structure:
```
training_{timestamp}/
├── raw_data/          # Initial classified images by product_component_light
├── oriented_data/     # Images organized by confirmed orientation + rotated augmentations
│   ├── Up/
│   ├── Down/
│   ├── Left/
│   ├── Right/
│   └── NG/
├── dataset/           # Final train/val splits with mean_std.json
├── model/             # best_model.pt and training configs
└── results/           # Evaluation CSVs, plots, golden samples
```

## Database Integration

- Uses PostgreSQL with SSH tunnel support
- Queries `AmrRawData` table to map images to product names
- Configuration in `configs/database_configs.json`
- Database sessions managed in `database/sessions.py`

## Configuration Files

- `configs/configs.yaml`: System-level configuration
- `configs/train_configs.yaml`: Training hyperparameters and model settings
- `configs/database_configs.json`: Database connection details

## API Endpoints

### Core Training
- `POST /training/start` - Start new training task (returns task_id and status "pending")
- `GET /training/status/{task_id}` - Get task status
- `GET /training/list` - List all tasks

### Orientation Confirmation (New)
- `GET /orientation/samples/{task_id}` - Get sample images for orientation confirmation
- `POST /orientation/confirm/{task_id}` - Submit orientation choices and continue training
- `GET /temp/{task_id}/{class_name}/{filename}` - Serve temporary sample images

### Task States
- `pending` → `pending_orientation` → `running` → `completed`/`failed`

## Development Notes

- The project uses Chinese comments and logging messages
- Image filename parsing follows specific industrial format: `timestamp_board@component_index_light.jpg`
- Supports both Windows and Linux environments (Windows uses single worker for data loading)
- GPU training with mixed precision when available
- Progress tracking via callbacks for API integration
- **New**: Orientation confirmation workflow requires user interaction via web interface