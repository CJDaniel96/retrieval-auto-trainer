"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Loader2, Save } from "lucide-react";

interface SystemConfigTabProps {
  config: any;
  onUpdate: (updates: any) => void;
  loading: boolean;
}

export function SystemConfigTab({ config, onUpdate, loading }: SystemConfigTabProps) {
  const [formData, setFormData] = useState(config);

  useEffect(() => {
    setFormData(config);
  }, [config]);

  const handleSave = () => {
    onUpdate(formData);
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 系統設定 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">系統設定</h3>
          <div className="space-y-3">
            <div>
              <Label htmlFor="max_concurrent_tasks">最大併發任務數</Label>
              <Input
                id="max_concurrent_tasks"
                type="number"
                value={formData.system?.max_concurrent_tasks || 1}
                onChange={(e) => setFormData({
                  ...formData,
                  system: { ...formData.system, max_concurrent_tasks: parseInt(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="task_timeout_hours">任務超時時間 (小時)</Label>
              <Input
                id="task_timeout_hours"
                type="number"
                value={formData.system?.task_timeout_hours || 24}
                onChange={(e) => setFormData({
                  ...formData,
                  system: { ...formData.system, task_timeout_hours: parseInt(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="cleanup_completed_tasks_days">已完成任務清理天數</Label>
              <Input
                id="cleanup_completed_tasks_days"
                type="number"
                value={formData.system?.cleanup_completed_tasks_days || 7}
                onChange={(e) => setFormData({
                  ...formData,
                  system: { ...formData.system, cleanup_completed_tasks_days: parseInt(e.target.value) }
                })}
              />
            </div>
          </div>
        </div>

        {/* 資源配置 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">資源配置</h3>
          <div className="space-y-3">
            <div>
              <Label htmlFor="gpu_memory_fraction">GPU記憶體使用比例</Label>
              <Input
                id="gpu_memory_fraction"
                type="number"
                step="0.1"
                min="0.1"
                max="1.0"
                value={formData.resources?.gpu_memory_fraction || 0.8}
                onChange={(e) => setFormData({
                  ...formData,
                  resources: { ...formData.resources, gpu_memory_fraction: parseFloat(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="max_workers">最大工作程序數</Label>
              <Input
                id="max_workers"
                type="number"
                value={formData.resources?.max_workers || 4}
                onChange={(e) => setFormData({
                  ...formData,
                  resources: { ...formData.resources, max_workers: parseInt(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="batch_size">批次大小</Label>
              <Input
                id="batch_size"
                type="number"
                value={formData.resources?.batch_size || 8}
                onChange={(e) => setFormData({
                  ...formData,
                  resources: { ...formData.resources, batch_size: parseInt(e.target.value) }
                })}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 儲存配置 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">儲存配置</h3>
          <div className="space-y-3">
            <div>
              <Label htmlFor="temp_dir">暫存目錄</Label>
              <Input
                id="temp_dir"
                value={formData.storage?.temp_dir || "/tmp/auto_training"}
                onChange={(e) => setFormData({
                  ...formData,
                  storage: { ...formData.storage, temp_dir: e.target.value }
                })}
              />
            </div>
            <div>
              <Label htmlFor="results_retention_days">結果保留天數</Label>
              <Input
                id="results_retention_days"
                type="number"
                value={formData.storage?.results_retention_days || 30}
                onChange={(e) => setFormData({
                  ...formData,
                  storage: { ...formData.storage, results_retention_days: parseInt(e.target.value) }
                })}
              />
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="auto_cleanup"
                checked={formData.storage?.auto_cleanup || false}
                onCheckedChange={(checked) => setFormData({
                  ...formData,
                  storage: { ...formData.storage, auto_cleanup: checked }
                })}
              />
              <Label htmlFor="auto_cleanup">自動清理</Label>
            </div>
          </div>
        </div>

        {/* 日誌配置 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">日誌配置</h3>
          <div className="space-y-3">
            <div>
              <Label htmlFor="log_level">日誌等級</Label>
              <Select
                value={formData.logging?.level || "INFO"}
                onValueChange={(value) => setFormData({
                  ...formData,
                  logging: { ...formData.logging, level: value }
                })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="DEBUG">DEBUG</SelectItem>
                  <SelectItem value="INFO">INFO</SelectItem>
                  <SelectItem value="WARNING">WARNING</SelectItem>
                  <SelectItem value="ERROR">ERROR</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="log_file">日誌檔案</Label>
              <Input
                id="log_file"
                value={formData.logging?.file || "logs/auto_training.log"}
                onChange={(e) => setFormData({
                  ...formData,
                  logging: { ...formData.logging, file: e.target.value }
                })}
              />
            </div>
            <div>
              <Label htmlFor="max_size_mb">日誌檔案最大大小 (MB)</Label>
              <Input
                id="max_size_mb"
                type="number"
                value={formData.logging?.max_size_mb || 100}
                onChange={(e) => setFormData({
                  ...formData,
                  logging: { ...formData.logging, max_size_mb: parseInt(e.target.value) }
                })}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={loading}>
          {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
          儲存系統配置
        </Button>
      </div>
    </div>
  );
}