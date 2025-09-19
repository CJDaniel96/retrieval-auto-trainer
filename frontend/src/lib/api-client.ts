import api from './api';
import {
  TrainingRequest,
  TrainingStatus,
  OrientationSample,
  OrientationConfirmation,
  TrainingResult,
  ApiResponse,
  FullConfig,
  ConfigUpdateRequest,
  DownloadRequest,
  PartInfo,
  ClassifyRequest,
  PartImageList,
  DownloadEstimate,
  DownloadStatus,
  DownloadResult
} from './types';

export class ApiClient {
  // Training endpoints
  static async startTraining(request: TrainingRequest): Promise<ApiResponse<{ task_id: string; message: string }>> {
    try {
      const response = await api.post('/training/start', request);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.message || error.message };
    }
  }

  static async getTrainingStatus(taskId: string): Promise<ApiResponse<TrainingStatus>> {
    try {
      const response = await api.get(`/training/status/${taskId}`);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async listTrainingTasks(): Promise<ApiResponse<TrainingStatus[]>> {
    try {
      const response = await api.get('/training/list');
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.message || error.message };
    }
  }

  static async getTrainingResult(taskId: string): Promise<ApiResponse<TrainingResult>> {
    try {
      const response = await api.get(`/training/result/${taskId}`);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async downloadFile(taskId: string, fileType: string): Promise<string> {
    try {
      const response = await api.get(`/training/download/${taskId}/${fileType}`, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      return url;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || error.message);
    }
  }

  static async cancelTraining(taskId: string): Promise<ApiResponse<{ message: string }>> {
    try {
      const response = await api.post(`/training/cancel/${taskId}`);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async deleteTraining(taskId: string, deleteFiles = false): Promise<ApiResponse<{ message: string }>> {
    try {
      const response = await api.delete(`/training/delete/${taskId}?delete_files=${deleteFiles}`);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async createModule(taskId: string, moduleName: string, partNumber: string): Promise<ApiResponse<{ message: string; module_path: string }>> {
    try {
      const response = await api.post(`/training/create-module/${taskId}`, {
        module_name: moduleName,
        part_number: partNumber
      });
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  // Orientation endpoints
  static async getOrientationSamples(taskId: string): Promise<ApiResponse<OrientationSample[]>> {
    try {
      const response = await api.get(`/orientation/samples/${taskId}`);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async confirmOrientations(confirmation: OrientationConfirmation): Promise<ApiResponse<{ message: string }>> {
    try {
      const response = await api.post(`/orientation/confirm/${confirmation.task_id}`, confirmation);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  // Configuration endpoints
  static async getCurrentConfig(): Promise<ApiResponse<FullConfig>> {
    try {
      const response = await api.get('/config/current');
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async updateSystemConfig(configUpdate: ConfigUpdateRequest): Promise<ApiResponse<FullConfig>> {
    try {
      const response = await api.post('/config/update/system', configUpdate);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async updateTrainingConfig(configUpdate: ConfigUpdateRequest): Promise<ApiResponse<FullConfig>> {
    try {
      const response = await api.post('/config/update/training', configUpdate);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async getDatabaseSites(): Promise<ApiResponse<Record<string, { id: string; database_name: string; lines: string[] }>>> {
    try {
      const response = await api.get('/config/database/sites');
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async getDefaultConfig(): Promise<ApiResponse<FullConfig>> {
    try {
      const response = await api.get('/config/default');
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  // 舊的方法保持向後兼容
  static async updateConfig(configUpdate: ConfigUpdateRequest): Promise<ApiResponse<{ message: string; updated: boolean; config: FullConfig }>> {
    try {
      const response = await api.post('/config/update/training', configUpdate);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  // Health check
  static async healthCheck(): Promise<ApiResponse<{ status: string; service: string; version: string }>> {
    try {
      const response = await api.get('/health');
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.message || error.message };
    }
  }

  // Download endpoints
  static async estimateDataCount(request: Omit<DownloadRequest, 'limit'>): Promise<ApiResponse<{ success: boolean; message: string; estimated_count: number }>> {
    try {
      const response = await api.post('/download/estimate', request);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async downloadRawdata(request: DownloadRequest): Promise<ApiResponse<{ success: boolean; message: string; path?: string; image_count?: number }>> {
    try {
      const response = await api.post('/download/rawdata', request);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async listDownloadedParts(): Promise<ApiResponse<PartInfo[]>> {
    try {
      const response = await api.get('/download/parts');
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async getPartInfo(partNumber: string): Promise<ApiResponse<PartInfo>> {
    try {
      const response = await api.get(`/download/parts/${partNumber}`);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async classifyImages(partNumber: string, request: ClassifyRequest): Promise<ApiResponse<{ success: boolean; message: string; moved_count: number; errors: string[] }>> {
    try {
      const response = await api.post(`/download/classify/${partNumber}`, request);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async listPartImages(partNumber: string, page: number = 1, pageSize: number = 50): Promise<ApiResponse<PartImageList>> {
    try {
      const response = await api.get(`/download/images/${partNumber}`, {
        params: {
          page,
          page_size: pageSize
        }
      });
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async deleteImage(partNumber: string, filename: string): Promise<ApiResponse<{ message: string }>> {
    try {
      const response = await api.delete(`/download/images/${partNumber}/${filename}`);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  // New download estimate and status methods
  static async estimateDownload(request: DownloadRequest): Promise<ApiResponse<DownloadEstimate>> {
    try {
      const response = await api.post('/download/estimate', request);
      return { data: response.data.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async downloadRawData(request: DownloadRequest): Promise<ApiResponse<DownloadResult>> {
    try {
      const response = await api.post('/download/rawdata', request);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }

  static async getDownloadStatus(downloadId: string): Promise<ApiResponse<DownloadStatus>> {
    try {
      const response = await api.get(`/download/status/${downloadId}`);
      return { data: response.data };
    } catch (error: any) {
      return { error: error.response?.data?.detail || error.message };
    }
  }
}