"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { motion } from "framer-motion";
import {
  RefreshCw,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Loader2,
  Download,
  Trash2,
  Play,
  Package
} from "lucide-react";
import { TrainingStatus } from "../shared/types";
import { useTranslations } from "next-intl";
import { useRouter } from "@/i18n/routing";
import { ApiClient } from "@/lib/api-client";
import { toast } from "sonner";

interface TaskListProps {
  tasks: TrainingStatus[];
  onRefresh: () => void;
  onDeleteTask: (taskId: string, deleteFiles?: boolean) => void;
  onCancelTask: (taskId: string) => void;
  onCreateModule: (taskId: string) => void;
  loading: boolean;
}

export function TaskList({
  tasks,
  onRefresh,
  onDeleteTask,
  onCancelTask,
  onCreateModule,
  loading
}: TaskListProps) {
  const t = useTranslations();
  const router = useRouter();

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "pending":
        return <Badge variant="secondary" className="bg-yellow-100 text-yellow-800"><Clock className="w-3 h-3 mr-1" /> {t("status.pending")}</Badge>;
      case "pending_orientation":
        return <Badge variant="secondary" className="bg-blue-100 text-blue-800"><AlertCircle className="w-3 h-3 mr-1" /> 等待方向確認</Badge>;
      case "running":
        return <Badge variant="secondary" className="bg-blue-100 text-blue-800"><Loader2 className="w-3 h-3 mr-1 animate-spin" /> {t("status.running")}</Badge>;
      case "completed":
        return <Badge variant="secondary" className="bg-green-100 text-green-800"><CheckCircle className="w-3 h-3 mr-1" /> {t("status.completed")}</Badge>;
      case "failed":
        return <Badge variant="destructive"><XCircle className="w-3 h-3 mr-1" /> {t("status.failed")}</Badge>;
      case "cancelled":
        return <Badge variant="outline"><XCircle className="w-3 h-3 mr-1" /> {t("status.cancelled")}</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const handleDownload = async (taskId: string, fileType: string) => {
    try {
      const url = await ApiClient.downloadFile(taskId, fileType);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${fileType}_${taskId}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success(`${fileType} 下載成功`);
    } catch (error) {
      toast.error(`下載 ${fileType} 失敗`);
    }
  };

  return (
    <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-2xl">訓練任務</CardTitle>
            <CardDescription>檢視和管理所有訓練任務</CardDescription>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onRefresh}
            disabled={loading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            重新整理
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="w-6 h-6 animate-spin mr-2" />
            載入中...
          </div>
        ) : !Array.isArray(tasks) || tasks.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            目前沒有訓練任務
          </div>
        ) : (
          <div className="space-y-4">
            {Array.isArray(tasks) && tasks.map((task) => (
              <motion.div
                key={task.task_id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="p-4 rounded-lg border border-gray-200 bg-gray-50"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <h3 className="font-semibold text-lg">{task.task_id}</h3>
                    {getStatusBadge(task.status)}
                  </div>
                  <div className="text-sm text-gray-500">
                    {new Date(task.created_at).toLocaleString()}
                  </div>
                </div>

                {task.progress && (
                  <div className="mb-3">
                    <div className="flex justify-between text-sm mb-1">
                      <span>進度</span>
                      <span>{Math.round(task.progress)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${task.progress}%` }}
                      ></div>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                  <div>
                    <span className="font-medium">輸入目錄:</span>
                    <p className="text-gray-600">{task.input_dir}</p>
                  </div>
                  <div>
                    <span className="font-medium">輸出目錄:</span>
                    <p className="text-gray-600">{task.output_dir}</p>
                  </div>
                </div>

                {task.message && (
                  <div className="mb-4">
                    <p className="text-sm text-gray-600">
                      <span className="font-medium">訊息:</span> {task.message}
                    </p>
                  </div>
                )}

                <div className="flex items-center justify-between">
                  <div className="flex space-x-2">
                    {task.status === "pending_orientation" && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => router.push(`/orientation/${task.task_id}`)}
                      >
                        <Play className="w-4 h-4 mr-2" />
                        確認方向
                      </Button>
                    )}

                    {task.status === "completed" && (
                      <>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleDownload(task.task_id, "model")}
                        >
                          <Download className="w-4 h-4 mr-2" />
                          下載模型
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleDownload(task.task_id, "results")}
                        >
                          <Download className="w-4 h-4 mr-2" />
                          下載結果
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => onCreateModule(task.task_id)}
                        >
                          <Package className="w-4 h-4 mr-2" />
                          建立模組
                        </Button>
                      </>
                    )}

                    {task.status === "running" && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onCancelTask(task.task_id)}
                      >
                        取消
                      </Button>
                    )}
                  </div>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onDeleteTask(task.task_id, false)}
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    刪除
                  </Button>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}