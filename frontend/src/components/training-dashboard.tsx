"use client";

import { useTranslations } from "next-intl";
import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { motion } from "framer-motion";
import {
  Play,
  Square,
  Download,
  Trash2,
  RefreshCw,
  Clock,
  CheckCircle,
  XCircle,
  AlertCircle,
  Loader2,
  Settings,
  Save,
  Brain,
  Database,
  Dumbbell,
  Target,
  Home,
  ListTodo,
  Cog,
  Search,
  Zap,
  FileText,
  Package,
  X,
} from "lucide-react";
import { ApiClient } from "@/lib/api-client";
import {
  TrainingStatus,
  TrainingRequest,
  DownloadRequest,
  PartInfo,
} from "@/lib/types";
import { LanguageSwitcher } from "@/components/language-switcher";
import { toast } from "sonner";
import { useRouter } from "@/i18n/routing";

// Settings Panel Component
function SettingsPanel() {
  const t = useTranslations();
  const [loading, setLoading] = useState(false);
  const [configs, setConfigs] = useState<any>({});
  const [databaseSites, setDatabaseSites] = useState<any>({});
  const [activeSettingsTab, setActiveSettingsTab] = useState("system");

  // 載入配置
  useEffect(() => {
    loadConfigs();
    loadDatabaseSites();
  }, []);

  const loadConfigs = async () => {
    try {
      setLoading(true);
      const result = await ApiClient.getCurrentConfig();
      if (result.data) {
        setConfigs(result.data);
      }
    } catch (error) {
      toast.error("載入配置失敗");
    } finally {
      setLoading(false);
    }
  };

  const loadDatabaseSites = async () => {
    try {
      const result = await ApiClient.getDatabaseSites();
      if (result.data) {
        setDatabaseSites(result.data);
      }
    } catch (error) {
      console.error("載入資料庫站點失敗:", error);
    }
  };

  const handleSystemConfigUpdate = async (updates: any) => {
    try {
      setLoading(true);
      const result = await ApiClient.updateSystemConfig({ updates });
      if (result.data) {
        setConfigs({ ...configs, system: result.data });
        toast.success("系統配置更新成功");
      }
    } catch (error) {
      toast.error("系統配置更新失敗");
    } finally {
      setLoading(false);
    }
  };

  const handleTrainingConfigUpdate = async (updates: any) => {
    try {
      setLoading(true);
      const result = await ApiClient.updateTrainingConfig({ updates });
      if (result.data) {
        setConfigs({ ...configs, training: result.data });
        toast.success("訓練配置更新成功");
      }
    } catch (error) {
      toast.error("訓練配置更新失敗");
    } finally {
      setLoading(false);
    }
  };

  if (loading && Object.keys(configs).length === 0) {
    return (
      <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin mr-2" />
          載入設定中...
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Settings className="w-6 h-6 text-blue-600" />
            <CardTitle className="text-2xl">系統設定</CardTitle>
          </div>
          <CardDescription>管理系統配置、訓練參數和資料庫連線</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeSettingsTab} onValueChange={setActiveSettingsTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="system">系統配置</TabsTrigger>
              <TabsTrigger value="training">訓練參數</TabsTrigger>
              <TabsTrigger value="database">資料庫資訊</TabsTrigger>
            </TabsList>

            {/* 系統配置分頁 */}
            <TabsContent value="system" className="space-y-4 mt-6">
              <SystemConfigTab
                config={configs.system || {}}
                onUpdate={handleSystemConfigUpdate}
                loading={loading}
              />
            </TabsContent>

            {/* 訓練參數分頁 */}
            <TabsContent value="training" className="space-y-4 mt-6">
              <TrainingConfigTab
                config={configs.training || {}}
                onUpdate={handleTrainingConfigUpdate}
                loading={loading}
              />
            </TabsContent>

            {/* 資料庫資訊分頁 */}
            <TabsContent value="database" className="space-y-4 mt-6">
              <DatabaseInfoTab sites={databaseSites} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}

// 系統配置分頁組件
function SystemConfigTab({ config, onUpdate, loading }: { config: any; onUpdate: (updates: any) => void; loading: boolean }) {
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

// 訓練配置分頁組件
function TrainingConfigTab({ config, onUpdate, loading }: { config: any; onUpdate: (updates: any) => void; loading: boolean }) {
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
        {/* 訓練參數 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">訓練參數</h3>
          <div className="space-y-3">
            <div>
              <Label htmlFor="max_epochs">最大訓練輪數</Label>
              <Input
                id="max_epochs"
                type="number"
                value={formData.training?.max_epochs || 50}
                onChange={(e) => setFormData({
                  ...formData,
                  training: { ...formData.training, max_epochs: parseInt(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="batch_size_training">訓練批次大小</Label>
              <Input
                id="batch_size_training"
                type="number"
                value={formData.training?.batch_size || 8}
                onChange={(e) => setFormData({
                  ...formData,
                  training: { ...formData.training, batch_size: parseInt(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="lr">學習率</Label>
              <Input
                id="lr"
                type="number"
                step="0.0001"
                value={formData.training?.lr || 0.0003}
                onChange={(e) => setFormData({
                  ...formData,
                  training: { ...formData.training, lr: parseFloat(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="patience">早停耐心值</Label>
              <Input
                id="patience"
                type="number"
                value={formData.training?.patience || 10}
                onChange={(e) => setFormData({
                  ...formData,
                  training: { ...formData.training, patience: parseInt(e.target.value) }
                })}
              />
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="enable_early_stopping"
                checked={formData.training?.enable_early_stopping || false}
                onCheckedChange={(checked) => setFormData({
                  ...formData,
                  training: { ...formData.training, enable_early_stopping: checked }
                })}
              />
              <Label htmlFor="enable_early_stopping">啟用早停</Label>
            </div>
          </div>
        </div>

        {/* 模型配置 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">模型配置</h3>
          <div className="space-y-3">
            <div>
              <Label htmlFor="backbone">骨幹網路</Label>
              <Select
                value={formData.model?.backbone || "efficientnetv2_rw_s"}
                onValueChange={(value) => setFormData({
                  ...formData,
                  model: { ...formData.model, backbone: value }
                })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="efficientnetv2_rw_s">EfficientNetV2-S</SelectItem>
                  <SelectItem value="efficientnetv2_rw_m">EfficientNetV2-M</SelectItem>
                  <SelectItem value="efficientnetv2_rw_l">EfficientNetV2-L</SelectItem>
                  <SelectItem value="resnet50">ResNet50</SelectItem>
                  <SelectItem value="resnet101">ResNet101</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="embedding_size">嵌入向量維度</Label>
              <Input
                id="embedding_size"
                type="number"
                value={formData.model?.embedding_size || 512}
                onChange={(e) => setFormData({
                  ...formData,
                  model: { ...formData.model, embedding_size: parseInt(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="structure">模型結構</Label>
              <Select
                value={formData.model?.structure || "HOAMV2"}
                onValueChange={(value) => setFormData({
                  ...formData,
                  model: { ...formData.model, structure: value }
                })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="HOAM">HOAM</SelectItem>
                  <SelectItem value="HOAMV2">HOAMV2</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="pretrained"
                checked={formData.model?.pretrained || false}
                onCheckedChange={(checked) => setFormData({
                  ...formData,
                  model: { ...formData.model, pretrained: checked }
                })}
              />
              <Label htmlFor="pretrained">使用預訓練模型</Label>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* 損失函數配置 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">損失函數配置</h3>
          <div className="space-y-3">
            <div>
              <Label htmlFor="loss_type">損失函數類型</Label>
              <Select
                value={formData.loss?.type || "HybridMarginLoss"}
                onValueChange={(value) => setFormData({
                  ...formData,
                  loss: { ...formData.loss, type: value }
                })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="HybridMarginLoss">HybridMarginLoss</SelectItem>
                  <SelectItem value="ArcFaceLoss">ArcFaceLoss</SelectItem>
                  <SelectItem value="TripletLoss">TripletLoss</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="triplet_margin">三元組邊界</Label>
              <Input
                id="triplet_margin"
                type="number"
                step="0.1"
                value={formData.loss?.triplet_margin || 0.3}
                onChange={(e) => setFormData({
                  ...formData,
                  loss: { ...formData.loss, triplet_margin: parseFloat(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="center_loss_weight">中心損失權重</Label>
              <Input
                id="center_loss_weight"
                type="number"
                step="0.01"
                value={formData.loss?.center_loss_weight || 0.01}
                onChange={(e) => setFormData({
                  ...formData,
                  loss: { ...formData.loss, center_loss_weight: parseFloat(e.target.value) }
                })}
              />
            </div>
          </div>
        </div>

        {/* 資料配置 */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">資料配置</h3>
          <div className="space-y-3">
            <div>
              <Label htmlFor="image_size">圖片大小</Label>
              <Input
                id="image_size"
                type="number"
                value={formData.data?.image_size || 224}
                onChange={(e) => setFormData({
                  ...formData,
                  data: { ...formData.data, image_size: parseInt(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="num_workers">資料載入程序數</Label>
              <Input
                id="num_workers"
                type="number"
                value={formData.data?.num_workers || 4}
                onChange={(e) => setFormData({
                  ...formData,
                  data: { ...formData.data, num_workers: parseInt(e.target.value) }
                })}
              />
            </div>
            <div>
              <Label htmlFor="test_split">測試集比例</Label>
              <Input
                id="test_split"
                type="number"
                step="0.1"
                min="0.1"
                max="0.5"
                value={formData.data?.test_split || 0.2}
                onChange={(e) => setFormData({
                  ...formData,
                  data: { ...formData.data, test_split: parseFloat(e.target.value) }
                })}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <Button onClick={handleSave} disabled={loading}>
          {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
          儲存訓練配置
        </Button>
      </div>
    </div>
  );
}

// 資料庫資訊分頁組件
function DatabaseInfoTab({ sites }: { sites: any }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(sites).map(([siteId, siteInfo]: [string, any]) => (
          <Card key={siteId} className="border border-gray-200">
            <CardHeader className="pb-3">
              <div className="flex items-center space-x-2">
                <Database className="w-5 h-5 text-blue-600" />
                <CardTitle className="text-lg">{siteId}</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              <div>
                <Label className="text-sm font-medium text-gray-600">資料庫名稱</Label>
                <p className="text-sm">{siteInfo.database_name}</p>
              </div>
              <div>
                <Label className="text-sm font-medium text-gray-600">可用產線</Label>
                <div className="flex flex-wrap gap-1 mt-1">
                  {siteInfo.lines.map((line: string) => (
                    <Badge key={line} variant="secondary" className="text-xs">
                      {line}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {Object.keys(sites).length === 0 && (
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            無法載入資料庫站點資訊，請檢查配置檔案。
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}

export function TrainingDashboard() {
  const t = useTranslations();
  const router = useRouter();
  const [tasks, setTasks] = useState<TrainingStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [initialLoad, setInitialLoad] = useState(true);
  const [startingTask, setStartingTask] = useState(false);
  const [activeTab, setActiveTab] = useState("new-training");
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  const [showCreateModuleDialog, setShowCreateModuleDialog] = useState(false);
  const [createModuleTaskId, setCreateModuleTaskId] = useState<string | null>(
    null
  );
  const [moduleName, setModuleName] = useState("");
  const [partNumber, setPartNumber] = useState("");
  const [creatingModule, setCreatingModule] = useState(false);

  // Download related states
  const [downloadFormData, setDownloadFormData] = useState<DownloadRequest>({
    site: "HPH",
    line_id: "V31",
    start_date: "",
    end_date: "",
    part_number: "",
    limit: undefined,
  });
  const [downloadedParts, setDownloadedParts] = useState<PartInfo[]>([]);
  const [loadingDownload, setLoadingDownload] = useState(false);
  const [loadingParts, setLoadingParts] = useState(false);
  const [loadingEstimate, setLoadingEstimate] = useState(false);
  const [estimatedCount, setEstimatedCount] = useState<number | null>(null);
  const [showDownloadSection, setShowDownloadSection] = useState(false);
  const [useExistingData, setUseExistingData] = useState(false);
  const [selectedRawdataPart, setSelectedRawdataPart] = useState<string>("");

  // Time display state to prevent hydration mismatch
  const [currentTime, setCurrentTime] = useState<string>("");
  const [mounted, setMounted] = useState(false);

  const [formData, setFormData] = useState<TrainingRequest>({
    input_dir: "",
    site: "HPH",
    line_id: "V31",
    experiment_config: {
      name: "hoam_experiment",
    },
    training_config: {
      min_epochs: 0,
      max_epochs: 50,
      lr: 0.0003,
      weight_decay: 0.0001,
      batch_size: 8,
      freeze_backbone_epochs: 10,
      patience: 10,
      enable_early_stopping: true,
      checkpoint_dir: "checkpoints",
    },
    model_config: {
      structure: "HOAMV2",
      backbone: "efficientnetv2_rw_s",
      pretrained: false,
      embedding_size: 512,
    },
    data_config: {
      image_size: 224,
      num_workers: 4,
      test_split: 0.2,
    },
    loss_config: {
      type: "HybridMarginLoss",
      subcenter_margin: 0.4,
      subcenter_scale: 30.0,
      sub_centers: 3,
      triplet_margin: 0.3,
      center_loss_weight: 0.01,
    },
    knn_config: {
      enable: false,
      threshold: 0.5,
      index_path: "knn.index",
      dataset_pkl: "dataset.pkl",
    },
  });

  useEffect(() => {
    // 首次載入任務
    fetchTasks();

    // 設定定期刷新
    const interval = setInterval(fetchTasks, 5000);

    return () => clearInterval(interval);
  }, []); // 只在組件掛載時執行一次

  // 防止無限載入的保險機制
  useEffect(() => {
    const failsafeTimeout = setTimeout(() => {
      if (loading && initialLoad) {
        console.warn("強制停止載入狀態 - 防止無限載入");
        setLoading(false);
        setError("載入超時，請手動刷新或檢查網路連接");
      }
    }, 15000); // 15秒後強制停止載入

    return () => clearTimeout(failsafeTimeout);
  }, [loading, initialLoad]);

  // Load downloaded parts when download tab or new-training tab is accessed
  useEffect(() => {
    if (activeTab === "download" || activeTab === "new-training") {
      const loadDownloadedParts = async () => {
        setLoadingParts(true);
        const result = await ApiClient.listDownloadedParts();
        if (result.data && Array.isArray(result.data)) {
          setDownloadedParts(result.data);
        } else {
          setDownloadedParts([]);
        }
        setLoadingParts(false);
      };
      loadDownloadedParts();
    }
  }, [activeTab]);

  // Handle time display to prevent hydration mismatch
  useEffect(() => {
    setMounted(true);

    const updateTime = () => {
      setCurrentTime(new Date().toLocaleString("en-US", {
        timeZone: "Asia/Taipei",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
        hour12: false,
      }));
    };

    updateTime(); // Initial update
    const interval = setInterval(updateTime, 1000); // Update every second

    return () => clearInterval(interval);
  }, []);

  const fetchTasks = async () => {
    console.log("開始獲取任務...");
    try {
      setError(null); // 清除之前的錯誤

      const response = await ApiClient.listTrainingTasks();

      if (response.data) {
        setTasks(Array.isArray(response.data) ? response.data : []);
        setError(null);
      } else if (response.error) {
        setError(response.error);
        setTasks([]);
      } else {
        // 如果沒有資料，設置為空陣列
        setTasks([]);
        setError(null);
      }
    } catch (error: any) {
      console.error("Failed to fetch tasks:", error);

      // 根據錯誤類型設置不同的錯誤訊息
      let errorMessage = "無法連接到後端服務";
      if (error.message === "請求超時") {
        errorMessage = "後端服務響應超時，請檢查服務狀態";
      } else if (
        error.code === "ECONNREFUSED" ||
        error.message.includes("Network Error")
      ) {
        errorMessage =
          "無法連接到後端服務 (http://localhost:8000)，請確認後端服務已啟動";
      } else if (error.response?.status === 404) {
        errorMessage = "API 端點不存在，請檢查後端服務版本";
      }

      setError(errorMessage);
      setTasks([]);
    } finally {
      // 確保無論什麼情況都停止載入狀態
      console.log("設置載入狀態為 false");
      setLoading(false);
      setInitialLoad(false);
    }
  };

  const handleStartTraining = async () => {
    if (!formData.input_dir) return;

    setStartingTask(true);
    const response = await ApiClient.startTraining(formData);
    if (response.data) {
      await fetchTasks();
      // Reset only the input directory, keep the configuration
      setFormData({ ...formData, input_dir: "" });
      toast.success(t("messages.training_started"));
    } else if (response.error) {
      toast.error(`${t("messages.error_occurred")}: ${response.error}`);
    }
    setStartingTask(false);
  };

  const handleCancelTask = async (taskId: string) => {
    await ApiClient.cancelTraining(taskId);
    await fetchTasks();
    toast.success(t("messages.task_cancelled"));
  };

  const handleDeleteTask = async (taskId: string) => {
    await ApiClient.deleteTraining(taskId);
    await fetchTasks();
    toast.success(t("messages.task_cancelled"));
  };

  const handleOrientationConfirm = (taskId: string) => {
    // Construct the URL based on the current origin and locale
    const currentUrl = new URL(window.location.href);
    const pathSegments = currentUrl.pathname.split("/").filter(Boolean);
    const locale = pathSegments[0] || "zh"; // Get locale from current path
    const orientationUrl = `${currentUrl.origin}/${locale}/orientation/${taskId}`;
    window.open(orientationUrl, "_blank");
  };

  const handleCreateModule = (taskId: string) => {
    setCreateModuleTaskId(taskId);
    setShowCreateModuleDialog(true);
  };

  const handleCreateModuleConfirm = async () => {
    if (!moduleName.trim() || !partNumber.trim() || !createModuleTaskId) return;

    setCreatingModule(true);
    try {
      const response = await ApiClient.createModule(
        createModuleTaskId,
        moduleName.trim(),
        partNumber.trim()
      );
      if (response.data) {
        toast.success(t("messages.module_created_successfully"));
        setShowCreateModuleDialog(false);
        setModuleName("");
        setPartNumber("");
        setCreateModuleTaskId(null);
      } else if (response.error) {
        toast.error(`${t("messages.error_occurred")}: ${response.error}`);
      }
    } catch (error: any) {
      toast.error(`${t("messages.error_occurred")}: ${error.message}`);
    } finally {
      setCreatingModule(false);
    }
  };

  const handleCreateModuleCancel = () => {
    setShowCreateModuleDialog(false);
    setModuleName("");
    setPartNumber("");
    setCreateModuleTaskId(null);
  };

  const updateTrainingField = (
    field: keyof NonNullable<TrainingRequest["training_config"]>,
    value: any
  ) => {
    setFormData((prev) => ({
      ...prev,
      training_config: {
        ...prev.training_config,
        [field]: value,
      },
    }));
  };

  const updateModelField = (
    field: keyof NonNullable<TrainingRequest["model_config"]>,
    value: any
  ) => {
    setFormData((prev) => ({
      ...prev,
      model_config: {
        ...prev.model_config,
        [field]: value,
      },
    }));
  };

  const updateDataField = (
    field: keyof NonNullable<TrainingRequest["data_config"]>,
    value: any
  ) => {
    setFormData((prev) => ({
      ...prev,
      data_config: {
        ...prev.data_config,
        [field]: value,
      },
    }));
  };

  const updateLossField = (
    field: keyof NonNullable<TrainingRequest["loss_config"]>,
    value: any
  ) => {
    setFormData((prev) => ({
      ...prev,
      loss_config: {
        ...prev.loss_config,
        [field]: value,
      },
    }));
  };

  const updateExperimentField = (
    field: keyof NonNullable<TrainingRequest["experiment_config"]>,
    value: any
  ) => {
    setFormData((prev) => ({
      ...prev,
      experiment_config: {
        ...prev.experiment_config,
        [field]: value,
      },
    }));
  };

  const updateKnnField = (
    field: keyof NonNullable<TrainingRequest["knn_config"]>,
    value: any
  ) => {
    setFormData((prev) => ({
      ...prev,
      knn_config: {
        ...prev.knn_config,
        [field]: value,
      },
    }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "pending":
        return <Clock className="w-4 h-4" />;
      case "pending_orientation":
        return <AlertCircle className="w-4 h-4" />;
      case "running":
        return <Loader2 className="w-4 h-4 animate-spin" />;
      case "completed":
        return <CheckCircle className="w-4 h-4" />;
      case "failed":
        return <XCircle className="w-4 h-4" />;
      default:
        return <Clock className="w-4 h-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "pending":
        return "bg-muted text-muted-foreground";
      case "pending_orientation":
        return "bg-yellow-500 text-yellow-50";
      case "running":
        return "bg-blue-500 text-blue-50";
      case "completed":
        return "bg-green-500 text-green-50";
      case "failed":
        return "bg-destructive text-destructive-foreground";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/20 bg-white/80 backdrop-blur-xl shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center shadow-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  {t("common.title")}
                </h1>
                <p className="text-sm text-muted-foreground">
                  Image Retrieval Model Training Platform
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-full bg-green-100 text-green-700 text-sm font-medium">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                {t("footer.powered_by")} {t("footer.ai_system")}
              </div>
              <LanguageSwitcher />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Enhanced Navigation Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="flex justify-center mb-8">
            <TabsList className="inline-flex h-14 items-center justify-center rounded-2xl bg-gradient-to-r from-white/90 to-gray-50/90 p-1.5 shadow-xl border border-white/40 backdrop-blur-lg">
              <TabsTrigger
                value="new-training"
                className="inline-flex items-center justify-center whitespace-nowrap rounded-xl px-6 py-3 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-lg hover:bg-white/60 data-[state=active]:hover:from-blue-700 data-[state=active]:hover:to-indigo-700"
              >
                <Home className="w-4 h-4 mr-2" />
                {t("training.start_new")}
              </TabsTrigger>
              <TabsTrigger
                value="task-list"
                className="inline-flex items-center justify-center whitespace-nowrap rounded-xl px-6 py-3 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-lg hover:bg-white/60 data-[state=active]:hover:from-blue-700 data-[state=active]:hover:to-indigo-700"
              >
                <ListTodo className="w-4 h-4 mr-2" />
                {t("navigation.training")}
                {tasks.length > 0 && (
                  <span className="ml-2 inline-flex items-center justify-center px-2 py-0.5 text-xs font-bold leading-none text-white bg-red-500 rounded-full">
                    {tasks.length}
                  </span>
                )}
              </TabsTrigger>
              <TabsTrigger
                value="download"
                className="inline-flex items-center justify-center whitespace-nowrap rounded-xl px-6 py-3 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-lg hover:bg-white/60 data-[state=active]:hover:from-blue-700 data-[state=active]:hover:to-indigo-700"
              >
                <Database className="w-4 h-4 mr-2" />
                {t("download.tab_name")}
              </TabsTrigger>
              <TabsTrigger
                value="settings"
                className="inline-flex items-center justify-center whitespace-nowrap rounded-xl px-6 py-3 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-lg hover:bg-white/60 data-[state=active]:hover:from-blue-700 data-[state=active]:hover:to-indigo-700"
              >
                <Settings className="w-4 h-4 mr-2" />
                {t("navigation.settings")}
              </TabsTrigger>
            </TabsList>
          </div>

          {/* New Training Tab */}
          <TabsContent value="new-training" className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="space-y-6"
            >
              <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
                <CardHeader>
                  <div className="flex items-center space-x-2">
                    <Play className="w-6 h-6 text-blue-600" />
                    <CardTitle className="text-2xl">
                      {t("training.start_new")}
                    </CardTitle>
                  </div>
                  <CardDescription>{t("form.configure_task")}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <Alert className="border-blue-200 bg-blue-50/80">
                    <AlertCircle className="h-4 w-4 text-blue-600" />
                    <AlertDescription className="text-blue-800">
                      {t("form.folder_management_info")}
                    </AlertDescription>
                  </Alert>

                  {/* Data Source Selection */}
                  <div className="space-y-4">
                    <div className="flex items-center space-x-3">
                      <Switch
                        id="use-existing-data"
                        checked={useExistingData}
                        onCheckedChange={setUseExistingData}
                      />
                      <Label
                        htmlFor="use-existing-data"
                        className="text-sm font-medium"
                      >
                        {t("download.info.rawdata_usage")}
                      </Label>
                    </div>
                    {useExistingData && (
                      <Alert className="border-green-200 bg-green-50/80">
                        <Database className="h-4 w-4 text-green-600" />
                        <AlertDescription className="text-green-800">
                          {t("download.info.rawdata_description")}
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {!useExistingData ? (
                      <div className="space-y-2">
                        <Label
                          htmlFor="input_dir"
                          className="text-sm font-medium"
                        >
                          {t("training.input_directory")}
                        </Label>
                        <Input
                          id="input_dir"
                          value={formData.input_dir}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              input_dir: e.target.value,
                            })
                          }
                          placeholder={t("form.input_placeholder")}
                          className="bg-white/70"
                        />
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <Label
                          htmlFor="rawdata_part"
                          className="text-sm font-medium"
                        >
                          {t("download.info.select_part")}
                        </Label>
                        <Select
                          value={selectedRawdataPart}
                          onValueChange={(value) => {
                            setSelectedRawdataPart(value);
                            // Update input_dir to use the rawdata path
                            setFormData({
                              ...formData,
                              input_dir: `rawdata/${value}`,
                            });
                          }}
                        >
                          <SelectTrigger className="bg-white/70">
                            <SelectValue
                              placeholder={t("download.info.select_part")}
                            />
                          </SelectTrigger>
                          <SelectContent>
                            {Array.isArray(downloadedParts) && downloadedParts.map((part) => (
                              <SelectItem
                                key={part.part_number}
                                value={part.part_number}
                              >
                                <div className="flex items-center justify-between w-full">
                                  <span>
                                    {part.part_number} ({part.image_count}{" "}
                                    {t("download.messages.image_count_suffix")})
                                  </span>
                                  {part.is_classified && (
                                    <Badge className="ml-2 text-xs bg-green-500 text-white">
                                      Classified ({part.classified_count})
                                    </Badge>
                                  )}
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    )}

                    <div className="space-y-2">
                      <Label htmlFor="site" className="text-sm font-medium">
                        {t("training.site")}
                      </Label>
                      <Select
                        value={formData.site}
                        onValueChange={(value) =>
                          setFormData({ ...formData, site: value })
                        }
                      >
                        <SelectTrigger className="bg-white/70">
                          <SelectValue placeholder={t("form.select_site")} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="HPH">HPH</SelectItem>
                          <SelectItem value="HPI">HPI</SelectItem>
                          <SelectItem value="HPM">HPM</SelectItem>
                          <SelectItem value="HPC">HPC</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="line_id" className="text-sm font-medium">
                        {t("training.line_id")}
                      </Label>
                      <Select
                        value={formData.line_id}
                        onValueChange={(value) =>
                          setFormData({ ...formData, line_id: value })
                        }
                      >
                        <SelectTrigger className="bg-white/70">
                          <SelectValue placeholder={t("form.select_line")} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="V31">V31</SelectItem>
                          <SelectItem value="V32">V32</SelectItem>
                          <SelectItem value="V33">V33</SelectItem>
                          <SelectItem value="V34">V34</SelectItem>
                          <SelectItem value="V35">V35</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() =>
                          setShowAdvancedConfig(!showAdvancedConfig)
                        }
                        className="flex items-center space-x-2"
                      >
                        <Cog className="w-4 h-4" />
                        <span>{t("config.title")}</span>
                      </Button>
                    </div>
                    <Button
                      onClick={handleStartTraining}
                      disabled={!formData.input_dir || startingTask}
                      className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 transition-all duration-200"
                      size="lg"
                    >
                      {startingTask ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          {t("messages.starting_training")}
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4 mr-2" />
                          {t("common.start")}
                        </>
                      )}
                    </Button>
                  </div>

                  {/* Advanced Configuration Panel */}
                  {showAdvancedConfig && (
                    <div className="mt-6 p-6 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg border border-gray-200">
                      <div className="flex items-center space-x-2 mb-4">
                        <Cog className="w-5 h-5 text-blue-600" />
                        <h3 className="text-lg font-semibold text-gray-800">
                          {t("config.title")}
                        </h3>
                      </div>
                      <p className="text-sm text-gray-600 mb-6">
                        這些設定僅會套用到當前的訓練任務，不會影響其他任務。所有參數都來自training_configs.yaml。
                      </p>

                      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                        {/* Experiment Configuration */}
                        <div className="space-y-4">
                          <div className="flex items-center space-x-2">
                            <FileText className="w-4 h-4 text-blue-600" />
                            <h4 className="font-medium text-gray-700">
                              {t("config.experiment.title")}
                            </h4>
                          </div>
                          <div className="space-y-2">
                            <Label
                              htmlFor="experiment_name"
                              className="text-xs"
                            >
                              {t("config.experiment.name")}
                            </Label>
                            <Input
                              id="experiment_name"
                              value={formData.experiment_config?.name || ""}
                              onChange={(e) =>
                                updateExperimentField("name", e.target.value)
                              }
                              className="h-8 text-sm"
                              placeholder="hoam_experiment"
                            />
                          </div>
                        </div>

                        {/* Training Configuration */}
                        <div className="space-y-4">
                          <div className="flex items-center space-x-2">
                            <Dumbbell className="w-4 h-4 text-blue-600" />
                            <h4 className="font-medium text-gray-700">
                              {t("config.training.title")}
                            </h4>
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <div className="space-y-2">
                              <Label htmlFor="min_epochs" className="text-xs">
                                {t("config.training.min_epochs")}
                              </Label>
                              <Input
                                id="min_epochs"
                                type="number"
                                min={0}
                                max={50}
                                value={
                                  formData.training_config?.min_epochs || ""
                                }
                                onChange={(e) =>
                                  updateTrainingField(
                                    "min_epochs",
                                    parseInt(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="max_epochs" className="text-xs">
                                {t("config.training.max_epochs")}
                              </Label>
                              <Input
                                id="max_epochs"
                                type="number"
                                min={1}
                                max={200}
                                value={
                                  formData.training_config?.max_epochs || ""
                                }
                                onChange={(e) =>
                                  updateTrainingField(
                                    "max_epochs",
                                    parseInt(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="batch_size" className="text-xs">
                                {t("config.training.batch_size")}
                              </Label>
                              <Input
                                id="batch_size"
                                type="number"
                                min={1}
                                max={256}
                                value={
                                  formData.training_config?.batch_size || ""
                                }
                                onChange={(e) =>
                                  updateTrainingField(
                                    "batch_size",
                                    parseInt(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="lr" className="text-xs">
                                {t("config.training.learning_rate")}
                              </Label>
                              <Input
                                id="lr"
                                type="number"
                                step="0.0001"
                                min={0.0001}
                                max={0.1}
                                value={formData.training_config?.lr || ""}
                                onChange={(e) =>
                                  updateTrainingField(
                                    "lr",
                                    parseFloat(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="weight_decay" className="text-xs">
                                {t("config.training.weight_decay")}
                              </Label>
                              <Input
                                id="weight_decay"
                                type="number"
                                step="0.0001"
                                min={0}
                                max={0.01}
                                value={
                                  formData.training_config?.weight_decay || ""
                                }
                                onChange={(e) =>
                                  updateTrainingField(
                                    "weight_decay",
                                    parseFloat(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="patience" className="text-xs">
                                {t("config.training.patience")}
                              </Label>
                              <Input
                                id="patience"
                                type="number"
                                min={1}
                                max={50}
                                value={formData.training_config?.patience || ""}
                                onChange={(e) =>
                                  updateTrainingField(
                                    "patience",
                                    parseInt(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label
                                htmlFor="freeze_epochs"
                                className="text-xs"
                              >
                                {t("config.training.freeze_backbone_epochs")}
                              </Label>
                              <Input
                                id="freeze_epochs"
                                type="number"
                                min={0}
                                max={20}
                                value={
                                  formData.training_config
                                    ?.freeze_backbone_epochs || ""
                                }
                                onChange={(e) =>
                                  updateTrainingField(
                                    "freeze_backbone_epochs",
                                    parseInt(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label
                                htmlFor="checkpoint_dir"
                                className="text-xs"
                              >
                                {t("config.training.checkpoint_dir")}
                              </Label>
                              <Input
                                id="checkpoint_dir"
                                value={
                                  formData.training_config?.checkpoint_dir || ""
                                }
                                onChange={(e) =>
                                  updateTrainingField(
                                    "checkpoint_dir",
                                    e.target.value
                                  )
                                }
                                className="h-8 text-sm"
                                placeholder="checkpoints"
                              />
                            </div>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Switch
                              id="early_stopping"
                              checked={
                                formData.training_config
                                  ?.enable_early_stopping ?? true
                              }
                              onCheckedChange={(checked) =>
                                updateTrainingField(
                                  "enable_early_stopping",
                                  checked
                                )
                              }
                            />
                            <Label htmlFor="early_stopping" className="text-sm">
                              {t("config.training.enable_early_stopping")}
                            </Label>
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
                        {/* Model Configuration */}
                        <div className="space-y-4">
                          <div className="flex items-center space-x-2">
                            <Brain className="w-4 h-4 text-blue-600" />
                            <h4 className="font-medium text-gray-700">
                              {t("config.model.title")}
                            </h4>
                          </div>
                          <div className="space-y-3">
                            <div className="space-y-2">
                              <Label htmlFor="structure" className="text-xs">
                                {t("config.model.structure")}
                              </Label>
                              <Select
                                value={formData.model_config?.structure || ""}
                                onValueChange={(value) =>
                                  updateModelField(
                                    "structure",
                                    value as "HOAM" | "HOAMV2"
                                  )
                                }
                              >
                                <SelectTrigger className="h-8 text-sm">
                                  <SelectValue
                                    placeholder={t("config.model.structure")}
                                  />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="HOAM">HOAM</SelectItem>
                                  <SelectItem value="HOAMV2">HOAMV2</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="backbone" className="text-xs">
                                {t("config.model.backbone")}
                              </Label>
                              <Select
                                value={formData.model_config?.backbone || ""}
                                onValueChange={(value) =>
                                  updateModelField("backbone", value)
                                }
                              >
                                <SelectTrigger className="h-8 text-sm">
                                  <SelectValue
                                    placeholder={t("config.model.backbone")}
                                  />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="efficientnetv2_rw_s">
                                    EfficientNetV2-S
                                  </SelectItem>
                                  <SelectItem value="efficientnetv2_rw_m">
                                    EfficientNetV2-M
                                  </SelectItem>
                                  <SelectItem value="resnet50">
                                    ResNet50
                                  </SelectItem>
                                  <SelectItem value="resnet101">
                                    ResNet101
                                  </SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="space-y-2">
                              <Label
                                htmlFor="embedding_size"
                                className="text-xs"
                              >
                                {t("config.model.embedding_size")}
                              </Label>
                              <Input
                                id="embedding_size"
                                type="number"
                                min={64}
                                max={2048}
                                step={64}
                                value={
                                  formData.model_config?.embedding_size || ""
                                }
                                onChange={(e) =>
                                  updateModelField(
                                    "embedding_size",
                                    parseInt(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="flex items-center space-x-2 pt-2">
                              <Switch
                                id="pretrained"
                                checked={
                                  formData.model_config?.pretrained ?? false
                                }
                                onCheckedChange={(checked) =>
                                  updateModelField("pretrained", checked)
                                }
                              />
                              <Label htmlFor="pretrained" className="text-sm">
                                {t("config.model.pretrained")}
                              </Label>
                            </div>
                          </div>
                        </div>

                        {/* Data Configuration */}
                        <div className="space-y-4">
                          <div className="flex items-center space-x-2">
                            <Database className="w-4 h-4 text-blue-600" />
                            <h4 className="font-medium text-gray-700">
                              {t("config.data.title")}
                            </h4>
                          </div>
                          <div className="space-y-3">
                            <div className="space-y-2">
                              <Label htmlFor="image_size" className="text-xs">
                                {t("config.data.image_size")}
                              </Label>
                              <Select
                                value={
                                  formData.data_config?.image_size?.toString() ||
                                  ""
                                }
                                onValueChange={(value) =>
                                  updateDataField("image_size", parseInt(value))
                                }
                              >
                                <SelectTrigger className="h-8 text-sm">
                                  <SelectValue
                                    placeholder={t("config.data.image_size")}
                                  />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="224">224x224</SelectItem>
                                  <SelectItem value="256">256x256</SelectItem>
                                  <SelectItem value="384">384x384</SelectItem>
                                  <SelectItem value="512">512x512</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="num_workers" className="text-xs">
                                {t("config.data.num_workers")}
                              </Label>
                              <Input
                                id="num_workers"
                                type="number"
                                min={0}
                                max={16}
                                value={formData.data_config?.num_workers || ""}
                                onChange={(e) =>
                                  updateDataField(
                                    "num_workers",
                                    parseInt(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="test_split" className="text-xs">
                                {t("config.data.test_split")}
                              </Label>
                              <Input
                                id="test_split"
                                type="number"
                                step="0.05"
                                min={0.1}
                                max={0.5}
                                value={formData.data_config?.test_split || ""}
                                onChange={(e) =>
                                  updateDataField(
                                    "test_split",
                                    parseFloat(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
                        {/* Loss Configuration */}
                        <div className="space-y-4">
                          <div className="flex items-center space-x-2">
                            <Target className="w-4 h-4 text-blue-600" />
                            <h4 className="font-medium text-gray-700">
                              {t("config.loss.title")}
                            </h4>
                          </div>
                          <div className="space-y-3">
                            <div className="space-y-2">
                              <Label htmlFor="loss_type" className="text-xs">
                                {t("config.loss.type")}
                              </Label>
                              <Select
                                value={formData.loss_config?.type || ""}
                                onValueChange={(value) =>
                                  updateLossField(
                                    "type",
                                    value as
                                      | "HybridMarginLoss"
                                      | "ArcFaceLoss"
                                      | "SubCenterArcFaceLoss"
                                  )
                                }
                              >
                                <SelectTrigger className="h-8 text-sm">
                                  <SelectValue
                                    placeholder={t("config.loss.type")}
                                  />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="HybridMarginLoss">
                                    HybridMarginLoss
                                  </SelectItem>
                                  <SelectItem value="ArcFaceLoss">
                                    ArcFaceLoss
                                  </SelectItem>
                                  <SelectItem value="SubCenterArcFaceLoss">
                                    SubCenterArcFaceLoss
                                  </SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                              <div className="space-y-2">
                                <Label
                                  htmlFor="subcenter_margin"
                                  className="text-xs"
                                >
                                  {t("config.loss.subcenter_margin")}
                                </Label>
                                <Input
                                  id="subcenter_margin"
                                  type="number"
                                  step="0.1"
                                  min={0.1}
                                  max={1.0}
                                  value={
                                    formData.loss_config?.subcenter_margin || ""
                                  }
                                  onChange={(e) =>
                                    updateLossField(
                                      "subcenter_margin",
                                      parseFloat(e.target.value) || 0
                                    )
                                  }
                                  className="h-8 text-sm"
                                />
                              </div>
                              <div className="space-y-2">
                                <Label
                                  htmlFor="subcenter_scale"
                                  className="text-xs"
                                >
                                  {t("config.loss.subcenter_scale")}
                                </Label>
                                <Input
                                  id="subcenter_scale"
                                  type="number"
                                  step="1"
                                  min={1}
                                  max={100}
                                  value={
                                    formData.loss_config?.subcenter_scale || ""
                                  }
                                  onChange={(e) =>
                                    updateLossField(
                                      "subcenter_scale",
                                      parseFloat(e.target.value) || 0
                                    )
                                  }
                                  className="h-8 text-sm"
                                />
                              </div>
                              <div className="space-y-2">
                                <Label
                                  htmlFor="sub_centers"
                                  className="text-xs"
                                >
                                  {t("config.loss.sub_centers")}
                                </Label>
                                <Input
                                  id="sub_centers"
                                  type="number"
                                  min={1}
                                  max={10}
                                  value={
                                    formData.loss_config?.sub_centers || ""
                                  }
                                  onChange={(e) =>
                                    updateLossField(
                                      "sub_centers",
                                      parseInt(e.target.value) || 0
                                    )
                                  }
                                  className="h-8 text-sm"
                                />
                              </div>
                              <div className="space-y-2">
                                <Label
                                  htmlFor="triplet_margin"
                                  className="text-xs"
                                >
                                  {t("config.loss.triplet_margin")}
                                </Label>
                                <Input
                                  id="triplet_margin"
                                  type="number"
                                  step="0.1"
                                  min={0.1}
                                  max={1.0}
                                  value={
                                    formData.loss_config?.triplet_margin || ""
                                  }
                                  onChange={(e) =>
                                    updateLossField(
                                      "triplet_margin",
                                      parseFloat(e.target.value) || 0
                                    )
                                  }
                                  className="h-8 text-sm"
                                />
                              </div>
                            </div>
                            <div className="space-y-2">
                              <Label
                                htmlFor="center_loss_weight"
                                className="text-xs"
                              >
                                {t("config.loss.center_loss_weight")}
                              </Label>
                              <Input
                                id="center_loss_weight"
                                type="number"
                                step="0.001"
                                min={0.001}
                                max={0.1}
                                value={
                                  formData.loss_config?.center_loss_weight || ""
                                }
                                onChange={(e) =>
                                  updateLossField(
                                    "center_loss_weight",
                                    parseFloat(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                          </div>
                        </div>

                        {/* KNN Configuration */}
                        <div className="space-y-4">
                          <div className="flex items-center space-x-2">
                            <Search className="w-4 h-4 text-blue-600" />
                            <h4 className="font-medium text-gray-700">
                              {t("config.knn.title")}
                            </h4>
                          </div>
                          <div className="space-y-3">
                            <div className="flex items-center space-x-2">
                              <Switch
                                id="knn_enable"
                                checked={formData.knn_config?.enable ?? false}
                                onCheckedChange={(checked) =>
                                  updateKnnField("enable", checked)
                                }
                              />
                              <Label htmlFor="knn_enable" className="text-sm">
                                {t("config.knn.enable")}
                              </Label>
                            </div>
                            <div className="space-y-2">
                              <Label
                                htmlFor="knn_threshold"
                                className="text-xs"
                              >
                                {t("config.knn.threshold")}
                              </Label>
                              <Input
                                id="knn_threshold"
                                type="number"
                                step="0.1"
                                min={0.1}
                                max={1.0}
                                value={formData.knn_config?.threshold || ""}
                                onChange={(e) =>
                                  updateKnnField(
                                    "threshold",
                                    parseFloat(e.target.value) || 0
                                  )
                                }
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="index_path" className="text-xs">
                                {t("config.knn.index_path")}
                              </Label>
                              <Input
                                id="index_path"
                                value={formData.knn_config?.index_path || ""}
                                onChange={(e) =>
                                  updateKnnField("index_path", e.target.value)
                                }
                                className="h-8 text-sm"
                                placeholder="knn.index"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="dataset_pkl" className="text-xs">
                                {t("config.knn.dataset_pkl")}
                              </Label>
                              <Input
                                id="dataset_pkl"
                                value={formData.knn_config?.dataset_pkl || ""}
                                onChange={(e) =>
                                  updateKnnField("dataset_pkl", e.target.value)
                                }
                                className="h-8 text-sm"
                                placeholder="dataset.pkl"
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Task List Tab */}
          <TabsContent value="task-list" className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="space-y-6"
            >
              <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
                <CardHeader className="flex flex-row items-center justify-between">
                  <div>
                    <div className="flex items-center space-x-2">
                      <ListTodo className="w-6 h-6 text-blue-600" />
                      <CardTitle className="text-2xl">
                        {t("training.title")}
                      </CardTitle>
                    </div>
                    <CardDescription>{t("form.monitor_tasks")}</CardDescription>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setLoading(true);
                      setError(null);
                      fetchTasks();
                    }}
                    disabled={loading}
                  >
                    <RefreshCw
                      className={`w-4 h-4 mr-2 ${
                        loading ? "animate-spin" : ""
                      }`}
                    />
                    {t("form.refresh")}
                  </Button>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="flex flex-col items-center justify-center py-12 space-y-3">
                      <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                      <p className="text-sm text-muted-foreground">
                        {t("messages.loading_tasks")}
                      </p>
                    </div>
                  ) : error ? (
                    <Alert
                      variant="destructive"
                      className="border-red-200 bg-red-50/80"
                    >
                      <XCircle className="h-4 w-4" />
                      <AlertTitle>{t("form.error_title")}</AlertTitle>
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  ) : tasks.length === 0 ? (
                    <Alert className="border-gray-200 bg-gray-50/80">
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>{t("form.no_training_tasks")}</AlertTitle>
                      <AlertDescription>
                        {t("form.no_tasks_description")}
                      </AlertDescription>
                    </Alert>
                  ) : (
                    <div className="space-y-4">
                      {tasks.map((task) => (
                        <Card
                          key={task.task_id}
                          className="bg-gradient-to-r from-white/90 to-gray-50/90 border border-gray-200/50"
                        >
                          <CardContent className="p-4 space-y-3">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center space-x-3">
                                {getStatusIcon(task.status)}
                                <span className="font-semibold text-lg">
                                  {task.task_id}{" "}
                                  {task.input_dir
                                    ? `(${task.input_dir
                                        .split(/[\/\\]/)
                                        .pop()})`
                                    : ""}
                                </span>
                                <Badge className={getStatusColor(task.status)}>
                                  {t(`training.${task.status}`)}
                                </Badge>
                              </div>
                              <div className="flex items-center space-x-2">
                                {task.status === "pending_orientation" && (
                                  <Button
                                    size="sm"
                                    variant="default"
                                    className="bg-yellow-600 hover:bg-yellow-700 text-white"
                                    onClick={() =>
                                      handleOrientationConfirm(task.task_id)
                                    }
                                  >
                                    <AlertCircle className="w-4 h-4 mr-2" />
                                    {t("form.confirm_orientation")}
                                  </Button>
                                )}
                                {task.status === "completed" && (
                                  <>
                                    <Button size="sm" variant="outline">
                                      <Download className="w-4 h-4 mr-2" />
                                      {t("common.download")}
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="outline"
                                      onClick={() =>
                                        handleCreateModule(task.task_id)
                                      }
                                      className="bg-green-50 hover:bg-green-100 text-green-700 border-green-200"
                                    >
                                      <Package className="w-4 h-4 mr-2" />
                                      {t("common.create_module")}
                                    </Button>
                                  </>
                                )}
                                {(task.status === "pending" ||
                                  task.status === "running") && (
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() =>
                                      handleCancelTask(task.task_id)
                                    }
                                  >
                                    <Square className="w-4 h-4 mr-2" />
                                    {t("common.cancel")}
                                  </Button>
                                )}
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => handleDeleteTask(task.task_id)}
                                >
                                  <Trash2 className="w-4 h-4 mr-2" />
                                  {t("common.delete")}
                                </Button>
                              </div>
                            </div>

                            {task.status === "pending_orientation" && (
                              <Alert className="border-yellow-200 bg-yellow-50/80">
                                <AlertCircle className="h-4 w-4 text-yellow-600" />
                                <AlertTitle className="text-yellow-800">
                                  {t("orientation.title")}
                                </AlertTitle>
                                <AlertDescription className="text-yellow-700">
                                  {t("orientation.description")}
                                </AlertDescription>
                              </Alert>
                            )}

                            {task.current_step && (
                              <p className="text-sm text-muted-foreground bg-muted/30 p-3 rounded-md border">
                                {task.current_step}
                              </p>
                            )}

                            {task.progress !== undefined && (
                              <div className="space-y-2">
                                <div className="flex justify-between text-sm">
                                  <span>{t("training.progress")}</span>
                                  <span className="font-medium">
                                    {Math.round(task.progress * 100)}%
                                  </span>
                                </div>
                                <Progress
                                  value={task.progress * 100}
                                  className="h-2"
                                />
                              </div>
                            )}

                            {task.error_message && (
                              <Alert
                                variant="destructive"
                                className="border-red-200 bg-red-50/80"
                              >
                                <XCircle className="h-4 w-4" />
                                <AlertTitle>{t("form.error_title")}</AlertTitle>
                                <AlertDescription>
                                  {task.error_message}
                                </AlertDescription>
                              </Alert>
                            )}

                            <div className="flex justify-between text-sm text-muted-foreground pt-2 border-t border-gray-200/50">
                              <span>
                                {t("training.created_at")}:{" "}
                                {task.start_time
                                  ? new Date(task.start_time).toLocaleString()
                                  : "N/A"}
                              </span>
                              {task.end_time && (
                                <span>
                                  {t("training.completed_at")}:{" "}
                                  {new Date(task.end_time).toLocaleString()}
                                </span>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Download Tab */}
          <TabsContent value="download" className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="space-y-6"
            >
              {/* Download Form */}
              <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
                <CardHeader>
                  <div className="flex items-center space-x-2">
                    <Database className="w-6 h-6 text-blue-600" />
                    <CardTitle className="text-2xl">
                      {t("download.title")}
                    </CardTitle>
                  </div>
                  <CardDescription>{t("download.description")}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="download_site">
                        {t("download.form.site")}
                      </Label>
                      <Select
                        value={downloadFormData.site}
                        onValueChange={(value) =>
                          setDownloadFormData((prev) => ({
                            ...prev,
                            site: value,
                          }))
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder={t("download.form.site")} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="HPH">HPH</SelectItem>
                          <SelectItem value="JQ">JQ</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="download_line_id">
                        {t("download.form.line_id")}
                      </Label>
                      <Select
                        value={downloadFormData.line_id}
                        onValueChange={(value) =>
                          setDownloadFormData((prev) => ({
                            ...prev,
                            line_id: value,
                          }))
                        }
                      >
                        <SelectTrigger>
                          <SelectValue
                            placeholder={t("download.form.line_id")}
                          />
                        </SelectTrigger>
                        <SelectContent>
                          {downloadFormData.site === "HPH" && (
                            <>
                              <SelectItem value="V31">V31</SelectItem>
                              <SelectItem value="V28">V28</SelectItem>
                              <SelectItem value="V27">V27</SelectItem>
                              <SelectItem value="V22">V22</SelectItem>
                              <SelectItem value="V20">V20</SelectItem>
                            </>
                          )}
                          {downloadFormData.site === "JQ" && (
                            <>
                              <SelectItem value="J12">J12</SelectItem>
                              <SelectItem value="J13">J13</SelectItem>
                              <SelectItem value="J15">J15</SelectItem>
                            </>
                          )}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="download_part_number">
                        {t("download.form.part_number")}
                      </Label>
                      <Input
                        id="download_part_number"
                        value={downloadFormData.part_number}
                        onChange={(e) =>
                          setDownloadFormData((prev) => ({
                            ...prev,
                            part_number: e.target.value,
                          }))
                        }
                        placeholder="例如: 32-500020-01"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="download_start_date">
                        {t("download.form.start_date")}
                      </Label>
                      <Input
                        id="download_start_date"
                        type="date"
                        value={downloadFormData.start_date}
                        onChange={(e) =>
                          setDownloadFormData((prev) => ({
                            ...prev,
                            start_date: e.target.value,
                          }))
                        }
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="download_end_date">
                        {t("download.form.end_date")}
                      </Label>
                      <Input
                        id="download_end_date"
                        type="date"
                        value={downloadFormData.end_date}
                        onChange={(e) =>
                          setDownloadFormData((prev) => ({
                            ...prev,
                            end_date: e.target.value,
                          }))
                        }
                      />
                    </div>
                  </div>

                  {/* Step 1: Estimate Data Count */}
                  {!showDownloadSection && (
                    <Button
                      onClick={async () => {
                        if (
                          !downloadFormData.part_number ||
                          !downloadFormData.start_date ||
                          !downloadFormData.end_date
                        ) {
                          toast.error(t("download.messages.missing_fields"));
                          return;
                        }

                        setLoadingEstimate(true);
                        const {
                          site,
                          line_id,
                          start_date,
                          end_date,
                          part_number,
                        } = downloadFormData;
                        const result = await ApiClient.estimateDataCount({
                          site,
                          line_id,
                          start_date,
                          end_date,
                          part_number,
                        });

                        if (result.error) {
                          toast.error(
                            `${t("download.messages.estimate_failed")}: ${
                              result.error
                            }`
                          );
                        } else if (result.data) {
                          if (result.data.success) {
                            setEstimatedCount(result.data.estimated_count);
                            setShowDownloadSection(true);
                            toast.success(result.data.message);
                          } else {
                            toast.error(result.data.message);
                          }
                        }
                        setLoadingEstimate(false);
                      }}
                      disabled={
                        loadingEstimate ||
                        !downloadFormData.part_number ||
                        !downloadFormData.start_date ||
                        !downloadFormData.end_date
                      }
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      {loadingEstimate ? (
                        <>
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          {t("download.messages.estimating")}
                        </>
                      ) : (
                        <>
                          <Search className="w-4 h-4 mr-2" />
                          {t("download.form.estimate_button")}
                        </>
                      )}
                    </Button>
                  )}

                  {/* Step 2: Show Estimated Count and Download Options */}
                  {showDownloadSection && estimatedCount !== null && (
                    <div className="space-y-4">
                      <Alert className="border-green-200 bg-green-50/80">
                        <AlertCircle className="h-4 w-4 text-green-600" />
                        <AlertDescription className="text-green-800">
                          {t("download.messages.found_images", {
                            count: estimatedCount,
                          })}
                        </AlertDescription>
                      </Alert>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label htmlFor="download_limit">
                            {t("download.form.limit")}
                          </Label>
                          <Input
                            id="download_limit"
                            type="number"
                            min="1"
                            max={estimatedCount}
                            value={downloadFormData.limit || ""}
                            onChange={(e) =>
                              setDownloadFormData((prev) => ({
                                ...prev,
                                limit: e.target.value
                                  ? parseInt(e.target.value)
                                  : undefined,
                              }))
                            }
                            placeholder={`不限制 (最多 ${estimatedCount} 張)`}
                          />
                        </div>
                      </div>

                      <div className="flex space-x-2">
                        <Button
                          variant="outline"
                          onClick={() => {
                            setShowDownloadSection(false);
                            setEstimatedCount(null);
                            setDownloadFormData((prev) => ({
                              ...prev,
                              limit: undefined,
                            }));
                          }}
                        >
                          {t("download.form.re_estimate")}
                        </Button>

                        <Button
                          onClick={async () => {
                            setLoadingDownload(true);
                            const result = await ApiClient.downloadRawdata(
                              downloadFormData
                            );

                            if (result.error) {
                              toast.error(`下載失敗: ${result.error}`);
                            } else if (result.data) {
                              if (result.data.success) {
                                toast.success(result.data.message);
                                // Refresh downloaded parts list
                                const partsResult =
                                  await ApiClient.listDownloadedParts();
                                if (partsResult.data && Array.isArray(partsResult.data)) {
                                  setDownloadedParts(partsResult.data);
                                } else {
                                  setDownloadedParts([]);
                                }
                                // Reset the form
                                setShowDownloadSection(false);
                                setEstimatedCount(null);
                                setDownloadFormData({
                                  site: "HPH",
                                  line_id: "V31",
                                  start_date: "",
                                  end_date: "",
                                  part_number: "",
                                  limit: undefined,
                                });
                              } else {
                                toast.error(result.data.message);
                              }
                            }
                            setLoadingDownload(false);
                          }}
                          disabled={loadingDownload}
                          className="bg-green-600 hover:bg-green-700"
                        >
                          {loadingDownload ? (
                            <>
                              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                              {t("download.form.downloading")}
                            </>
                          ) : (
                            <>
                              <Download className="w-4 h-4 mr-2" />
                              {downloadFormData.limit
                                ? t("download.form.download_limited", {
                                    limit: downloadFormData.limit,
                                  })
                                : t("download.form.download_all", {
                                    count: estimatedCount,
                                  })}
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Downloaded Parts List */}
              <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Database className="w-6 h-6 text-green-600" />
                      <CardTitle className="text-2xl">
                        {t("download.downloaded_parts.title")}
                      </CardTitle>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={async () => {
                        setLoadingParts(true);
                        const result = await ApiClient.listDownloadedParts();
                        if (result.data && Array.isArray(result.data)) {
                          setDownloadedParts(result.data);
                        } else {
                          setDownloadedParts([]);
                        }
                        setLoadingParts(false);
                      }}
                    >
                      <RefreshCw className="w-4 h-4 mr-2" />
                      {t("download.downloaded_parts.refresh")}
                    </Button>
                  </div>
                  <CardDescription>
                    {t("download.downloaded_parts.description")}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {loadingParts ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="w-6 h-6 animate-spin mr-2" />
                      載入中...
                    </div>
                  ) : !Array.isArray(downloadedParts) || downloadedParts.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                      {t("download.downloaded_parts.no_data")}
                    </div>
                  ) : (
                    <div className="grid gap-4">
                      {downloadedParts.map((part) => (
                        <div
                          key={part.part_number}
                          className="flex items-center justify-between p-4 rounded-lg border border-gray-200 bg-gray-50"
                        >
                          <div className="space-y-1">
                            <h3 className="font-semibold text-lg">
                              {part.part_number}
                            </h3>
                            <p className="text-sm text-gray-600">
                              {t("download.downloaded_parts.image_count")}:{" "}
                              {part.image_count}{" "}
                              {t("download.messages.image_count_suffix")}
                            </p>
                            <p className="text-xs text-gray-500">
                              {t("download.downloaded_parts.download_time")}:{" "}
                              {new Date(part.download_time).toLocaleString()}
                            </p>
                          </div>
                          <div className="flex space-x-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                router.push(`/classify/${part.part_number}`);
                              }}
                            >
                              {t("download.downloaded_parts.classify_images")}
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => {
                                // Switch to training tab and pre-fill with this part
                                setUseExistingData(true);
                                setSelectedRawdataPart(part.part_number);
                                setFormData({
                                  ...formData,
                                  input_dir: `rawdata/${part.part_number}`,
                                });
                                setActiveTab("new-training");
                                toast.success(
                                  t("download.messages.switch_training", {
                                    partNumber: part.part_number,
                                  })
                                );
                              }}
                            >
                              {t("download.downloaded_parts.use_for_training")}
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="space-y-6"
            >
              <SettingsPanel />
            </motion.div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer - Technical Style */}
      <footer className="border-t border-slate-800/20 bg-slate-900/95 backdrop-blur-md mt-16">
        <div className="max-w-7xl mx-auto px-6 py-12">
          {/* Main Footer Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 mb-12">
            {/* System Overview */}
            <div className="lg:col-span-2 space-y-6">
              <div className="flex items-center space-x-3">
                <div className="relative">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shadow-lg">
                    <Brain className="w-6 h-6 text-white" />
                  </div>
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse shadow-sm"></div>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white tracking-tight">
                    {t("footer.ai_system")}
                  </h3>
                  <p className="text-xs text-slate-400 font-mono">
                    {t("footer.platform_version")}
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                <p className="text-sm text-slate-300 leading-relaxed max-w-md">
                  {t("footer.description")}
                </p>

                {/* Technical Specs */}
                <div className="grid grid-cols-2 gap-4 pt-4">
                  <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                      {t("footer.architecture")}
                    </h4>
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2 text-xs text-slate-300">
                        <div className="w-1.5 h-1.5 bg-blue-400 rounded-full"></div>
                        <span className="font-mono">HOAM/HOAMV2</span>
                      </div>
                      <div className="flex items-center space-x-2 text-xs text-slate-300">
                        <div className="w-1.5 h-1.5 bg-green-400 rounded-full"></div>
                        <span className="font-mono">EfficientNetV2</span>
                      </div>
                      <div className="flex items-center space-x-2 text-xs text-slate-300">
                        <div className="w-1.5 h-1.5 bg-purple-400 rounded-full"></div>
                        <span className="font-mono">{t("footer.ml_tech")}</span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                      {t("footer.capabilities")}
                    </h4>
                    <div className="space-y-1">
                      <div className="flex items-center space-x-2 text-xs text-slate-300">
                        <Zap className="w-3 h-3 text-yellow-400" />
                        <span>{t("footer.auto_training")}</span>
                      </div>
                      <div className="flex items-center space-x-2 text-xs text-slate-300">
                        <Search className="w-3 h-3 text-cyan-400" />
                        <span>{t("footer.image_retrieval")}</span>
                      </div>
                      <div className="flex items-center space-x-2 text-xs text-slate-300">
                        <Target className="w-3 h-3 text-red-400" />
                        <span>{t("footer.quality_inspection")}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* System Status */}
            <div className="space-y-6">
              <div>
                <h4 className="text-sm font-semibold text-white mb-4 flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span>{t("footer.system_status")}</span>
                </h4>

                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                      <span className="text-sm text-slate-300 font-mono">
                        {t("footer.api_gateway")}
                      </span>
                    </div>
                    <span className="text-xs text-green-400 font-semibold">
                      {t("footer.status_online")}
                    </span>
                  </div>

                  <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                      <span className="text-sm text-slate-300 font-mono">
                        {t("footer.frontend_ui")}
                      </span>
                    </div>
                    <span className="text-xs text-blue-400 font-semibold">
                      {t("footer.status_active")}
                    </span>
                  </div>

                  <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                      <span className="text-sm text-slate-300 font-mono">
                        {t("footer.gpu_cluster")}
                      </span>
                    </div>
                    <span className="text-xs text-purple-400 font-semibold">
                      {t("footer.status_ready")}
                    </span>
                  </div>

                  <div className="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/50">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                      <span className="text-sm text-slate-300 font-mono">
                        {t("footer.active_tasks")}
                      </span>
                    </div>
                    <span className="text-xs text-orange-400 font-semibold font-mono">
                      {tasks
                        .filter((t) => t.status === "running")
                        .length.toString()
                        .padStart(2, "0")}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            {/* Technical Info */}
            <div className="space-y-6">
              <div>
                <h4 className="text-sm font-semibold text-white mb-4 flex items-center space-x-2">
                  <Cog className="w-4 h-4" />
                  <span>{t("footer.tech_stack")}</span>
                </h4>

                <div className="space-y-3">
                  <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                    <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
                      {t("footer.frontend")}
                    </div>
                    <div className="text-sm text-slate-200 font-mono">
                      {t("footer.frontend_tech")}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                    <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
                      {t("footer.backend")}
                    </div>
                    <div className="text-sm text-slate-200 font-mono">
                      {t("footer.backend_tech")}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                    <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
                      {t("footer.ml_framework")}
                    </div>
                    <div className="text-sm text-slate-200 font-mono">
                      {t("footer.ml_tech")}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
                    <div className="text-xs text-slate-400 uppercase tracking-wide mb-1">
                      {t("footer.database")}
                    </div>
                    <div className="text-sm text-slate-200 font-mono">
                      {t("footer.database_tech")}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Footer */}
          <div className="border-t border-slate-700/50 pt-8">
            <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center space-y-4 lg:space-y-0">
              <div className="space-y-2">
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-slate-400">
                    © 2024 {t("footer.company_name")}
                  </span>
                  <div className="w-1 h-1 bg-slate-600 rounded-full"></div>
                  <span className="text-sm text-slate-400">
                    {t("footer.company_subtitle")}
                  </span>
                </div>
                <p className="text-xs text-slate-500 font-mono">
                  {t("footer.specialization")}
                </p>
              </div>

              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-xs text-slate-400 font-mono">
                    {t("footer.version")} 2.0.1
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <Clock className="w-3 h-3 text-slate-500" />
                  <span className="text-xs text-slate-400 font-mono">
                    {mounted ? currentTime : "Loading..."} UTC+8
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                  <span className="text-xs text-slate-400 font-mono">
                    {t("footer.build")} #
                    {Math.floor(Math.random() * 1000) + 1000}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </footer>

      {/* Create Module Dialog */}
      {showCreateModuleDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4 shadow-xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">
                {t("common.create_module")}
              </h3>
              <Button
                variant="outline"
                size="sm"
                onClick={handleCreateModuleCancel}
                className="w-8 h-8 p-0"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="space-y-4">
              <Alert className="border-blue-200 bg-blue-50">
                <Package className="h-4 w-4 text-blue-600" />
                <AlertDescription className="text-blue-800">
                  {t("form.module_creation_info")}
                </AlertDescription>
              </Alert>

              <div className="space-y-2">
                <Label htmlFor="module_name" className="text-sm font-medium">
                  {t("form.module_name")}
                </Label>
                <Input
                  id="module_name"
                  value={moduleName}
                  onChange={(e) => setModuleName(e.target.value)}
                  placeholder={t("form.module_name_placeholder")}
                  className="w-full"
                />
                <p className="text-xs text-gray-500">
                  {t("form.module_name_example")}
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="part_number" className="text-sm font-medium">
                  料號 (Part Number)
                </Label>
                <Input
                  id="part_number"
                  value={partNumber}
                  onChange={(e) => setPartNumber(e.target.value)}
                  placeholder="例如: 32-500020-01"
                  className="w-full"
                />
                <p className="text-xs text-gray-500">
                  請輸入對應的料號，將用於配置文件中
                </p>
              </div>
            </div>

            <div className="flex justify-end space-x-2 mt-6">
              <Button
                variant="outline"
                onClick={handleCreateModuleCancel}
                disabled={creatingModule}
              >
                {t("common.cancel")}
              </Button>
              <Button
                onClick={handleCreateModuleConfirm}
                disabled={
                  !moduleName.trim() || !partNumber.trim() || creatingModule
                }
                className="bg-green-600 hover:bg-green-700"
              >
                {creatingModule ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    {t("messages.creating_module")}
                  </>
                ) : (
                  <>
                    <Package className="w-4 h-4 mr-2" />
                    {t("common.create")}
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
