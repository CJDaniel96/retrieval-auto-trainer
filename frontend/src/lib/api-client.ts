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
  PartImageList
} from './types';

interface ApiError {
  response?: {
    data?: {
      message?: string;
      detail?: string;
    };
  };
  message?: string;
}

export class ApiClient {
  // Training endpoints
  static async startTraining(request: TrainingRequest): Promise<ApiResponse<{ task_id: string; message: string }>> {
    try {
      const response = await api.post('/training/start', request);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.message;
      return { error: responseError || errorMessage };
    }
  }

  static async getTrainingStatus(taskId: string): Promise<ApiResponse<TrainingStatus>> {
    try {
      const response = await api.get(`/training/status/${taskId}`);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async listTrainingTasks(): Promise<ApiResponse<TrainingStatus[]>> {
    try {
      const response = await api.get('/training/list');
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.message;
      return { error: responseError || errorMessage };
    }
  }

  static async getTrainingResult(taskId: string): Promise<ApiResponse<TrainingResult>> {
    try {
      const response = await api.get(`/training/result/${taskId}`);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async downloadFile(taskId: string, fileType: string): Promise<string> {
    try {
      const response = await api.get(`/training/download/${taskId}/${fileType}`, {
        responseType: 'blob'
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      return url;
    } catch (error: unknown) {
      const apiError = error as ApiError;
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      throw new Error(apiError?.response?.data?.detail || errorMessage);
    }
  }

  static async cancelTraining(taskId: string): Promise<ApiResponse<{ message: string }>> {
    try {
      const response = await api.post(`/training/cancel/${taskId}`);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async deleteTraining(taskId: string, deleteFiles = false): Promise<ApiResponse<{ message: string }>> {
    try {
      const response = await api.delete(`/training/delete/${taskId}?delete_files=${deleteFiles}`);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async createModule(taskId: string, moduleName: string, partNumber: string): Promise<ApiResponse<{ message: string; module_path: string }>> {
    try {
      const response = await api.post(`/training/create-module/${taskId}`, {
        module_name: moduleName,
        part_number: partNumber
      });
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  // Orientation endpoints
  static async getOrientationSamples(taskId: string): Promise<ApiResponse<OrientationSample[]>> {
    try {
      const response = await api.get(`/orientation/samples/${taskId}`);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async confirmOrientations(confirmation: OrientationConfirmation): Promise<ApiResponse<{ message: string }>> {
    try {
      const response = await api.post(`/orientation/confirm/${confirmation.task_id}`, confirmation);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  // Configuration endpoints
  static async getCurrentConfig(): Promise<ApiResponse<FullConfig>> {
    try {
      const response = await api.get('/config/current');
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async updateConfig(configUpdate: ConfigUpdateRequest): Promise<ApiResponse<{ message: string; updated: boolean; config: FullConfig }>> {
    try {
      const response = await api.post('/config/update', configUpdate);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  // Health check
  static async healthCheck(): Promise<ApiResponse<{ status: string; service: string; version: string }>> {
    try {
      const response = await api.get('/health');
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.message;
      return { error: responseError || errorMessage };
    }
  }

  // Download endpoints
  static async estimateDataCount(request: Omit<DownloadRequest, 'limit'>): Promise<ApiResponse<{ success: boolean; message: string; estimated_count: number }>> {
    try {
      const response = await api.post('/download/estimate', request);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async downloadRawdata(request: DownloadRequest): Promise<ApiResponse<{ success: boolean; message: string; path?: string; image_count?: number }>> {
    try {
      const response = await api.post('/download/rawdata', request);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async listDownloadedParts(): Promise<ApiResponse<PartInfo[]>> {
    try {
      // 使用較短的超時時間，因為這個操作應該很快
      const response = await api.get('/download/parts', { timeout: 10000 }); // 10秒超時
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async getPartInfo(partNumber: string): Promise<ApiResponse<PartInfo>> {
    try {
      // 使用較短的超時時間，因為這個操作應該很快
      const response = await api.get(`/download/parts/${partNumber}`, { timeout: 8000 }); // 8秒超時
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async classifyImages(partNumber: string, request: ClassifyRequest): Promise<ApiResponse<{ success: boolean; message: string; moved_count: number; errors: string[] }>> {
    try {
      const response = await api.post(`/download/classify/${partNumber}`, request);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
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
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }

  static async deleteImage(partNumber: string, filename: string): Promise<ApiResponse<{ message: string }>> {
    try {
      const response = await api.delete(`/download/images/${partNumber}/${filename}`);
      return { data: response.data };
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const apiError = error as ApiError;
      const responseError = apiError?.response?.data?.detail;
      return { error: responseError || errorMessage };
    }
  }
}