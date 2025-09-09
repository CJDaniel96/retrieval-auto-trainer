export interface TrainingRequest {
  input_dir: string;
  site?: string;
  line_id?: string;
  max_epochs?: number;
  batch_size?: number;
  learning_rate?: number;
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