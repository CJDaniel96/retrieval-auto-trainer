export interface TrainingRequest {
  input_dir: string;
  site?: string;
  line_id?: string;
  // Optional task-specific configuration
  experiment_config?: Partial<ExperimentConfig>;
  training_config?: Partial<TrainingConfig>;
  model_config?: Partial<ModelConfig>;
  data_config?: Partial<DataConfig>;
  loss_config?: Partial<LossConfig>;
  knn_config?: Partial<KnnConfig>;
}

export interface TrainingStatus {
  task_id: string;
  status: 'pending' | 'pending_orientation' | 'running' | 'completed' | 'failed' | 'cancelled';
  start_time?: string;
  end_time?: string;
  current_step?: string;
  progress?: number;
  error_message?: string;
  output_dir?: string;
  input_dir?: string;
}

export interface OrientationSample {
  class_name: string;
  sample_images: string[];
}

export interface OrientationConfirmation {
  task_id: string;
  orientations: Record<string, 'Up' | 'Down' | 'Left' | 'Right'>;
}

export interface TrainingResult {
  task_id: string;
  accuracy: number;
  total_classes: number;
  total_images: number;
  model_path: string;
  evaluation_csv: string;
  confusion_matrix: string;
}

export interface ApiResponse<T> {
  data?: T;
  message?: string;
  error?: string;
}

// Configuration interfaces
export interface TrainingConfig {
  min_epochs: number;
  max_epochs: number;
  lr: number;
  weight_decay: number;
  batch_size: number;
  freeze_backbone_epochs: number;
  patience: number;
  enable_early_stopping: boolean;
  checkpoint_dir: string;
}

export interface ModelConfig {
  structure: 'HOAM' | 'HOAMV2';
  backbone: string;
  pretrained: boolean;
  embedding_size: number;
}

export interface DataConfig {
  data_dir: string;
  image_size: number;
  num_workers: number;
  test_split: number;
}

export interface LossConfig {
  type: 'HybridMarginLoss' | 'ArcFaceLoss' | 'SubCenterArcFaceLoss';
  subcenter_margin: number;
  subcenter_scale: number;
  sub_centers: number;
  triplet_margin: number;
  center_loss_weight: number;
}

export interface ExperimentConfig {
  name: string;
}

export interface KnnConfig {
  enable: boolean;
  threshold: number;
  index_path: string;
  dataset_pkl: string;
}

export interface FullConfig {
  experiment: ExperimentConfig;
  training: TrainingConfig;
  model: ModelConfig;
  data: DataConfig;
  loss: LossConfig;
  knn: KnnConfig;
}

export interface ConfigUpdateRequest {
  experiment?: Partial<ExperimentConfig>;
  training?: Partial<TrainingConfig>;
  model?: Partial<ModelConfig>;
  data?: Partial<DataConfig>;
  loss?: Partial<LossConfig>;
  knn?: Partial<KnnConfig>;
}

// 下載相關類型
export interface DownloadRequest {
  site: string;
  line_id: string;
  start_date: string;
  end_date: string;
  part_number: string;
  limit?: number;
}

export interface PartInfo {
  part_number: string;
  path: string;
  image_count: number;
  download_time: string;
}

export interface ClassifyRequest {
  part_number: string;
  classifications: Record<string, 'OK' | 'NG'>;
}

export interface ImageInfo {
  filename: string;
  path: string;
  size: number;
}

export interface PartImageList {
  part_number: string;
  total_images: number;
  images: ImageInfo[];
}