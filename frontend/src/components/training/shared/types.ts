import { TrainingStatus, TrainingRequest, DownloadRequest, PartInfo } from "@/lib/types";

export type { TrainingStatus, TrainingRequest, DownloadRequest, PartInfo };

export interface TrainingFormData {
  site: string;
  line_id: string;
  input_dir: string;
  output_dir: string;
  exclude_ng_from_ok: boolean;
}

export interface CreateModuleData {
  taskId: string;
  moduleName: string;
  partNumber: string;
}