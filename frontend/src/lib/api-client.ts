import api from './api';
import {
  TrainingRequest,
  TrainingStatus,
  OrientationSample,
  OrientationConfirmation,
  TrainingResult,
  ApiResponse,
  FullConfig,
  ConfigUpdateRequest
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

  static async updateConfig(configUpdate: ConfigUpdateRequest): Promise<ApiResponse<{ message: string; updated: boolean; config: FullConfig }>> {
    try {
      const response = await api.post('/config/update', configUpdate);
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
}