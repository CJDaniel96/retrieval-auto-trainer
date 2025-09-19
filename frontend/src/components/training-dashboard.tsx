"use client";

import { useTranslations } from "next-intl";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Home,
  ListTodo,
  Database,
  Settings,
  Brain,
  Dumbbell,
  Target
} from "lucide-react";

import { ApiClient } from "@/lib/api-client";
import { TrainingStatus, TrainingRequest, DownloadRequest, PartInfo } from "@/lib/types";
import { LanguageSwitcher } from "@/components/language-switcher";
import { toast } from "sonner";
import { useRouter } from "@/i18n/routing";

// Import new modular components
import { SettingsPanel } from "./training/settings";
import { TaskList } from "./training/training-tasks";
import { TrainingForm } from "./training/new-training";
import { DownloadPanel } from "./training/data-download";
import { TrainingFormData } from "./training/shared/types";

export function TrainingDashboard() {
  const t = useTranslations();
  const router = useRouter();

  // State management
  const [tasks, setTasks] = useState<TrainingStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [initialLoad, setInitialLoad] = useState(true);
  const [startingTask, setStartingTask] = useState(false);
  const [activeTab, setActiveTab] = useState("new-training");

  // New training form state
  const [formData, setFormData] = useState<TrainingFormData>({
    site: "HPH",
    line_id: "V31",
    input_dir: "",
    output_dir: "",
    exclude_ng_from_ok: false,
  });

  // Advanced training config
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState<any>({});

  // Downloaded parts for existing data
  const [downloadedParts, setDownloadedParts] = useState<PartInfo[]>([]);
  const [useExistingData, setUseExistingData] = useState(false);
  const [selectedRawdataPart, setSelectedRawdataPart] = useState<string>("");
  const [loadingParts, setLoadingParts] = useState(false);

  // Module creation dialog state
  const [showCreateModuleDialog, setShowCreateModuleDialog] = useState(false);
  const [createModuleTaskId, setCreateModuleTaskId] = useState<string | null>(null);
  const [moduleName, setModuleName] = useState("");
  const [partNumber, setPartNumber] = useState("");
  const [creatingModule, setCreatingModule] = useState(false);

  // Load tasks and refresh
  const loadTasks = async () => {
    try {
      setLoading(true);
      const response = await ApiClient.listTrainingTasks();
      if (response.data) {
        setTasks(response.data);
        setError(null);
      } else if (response.error) {
        setError(response.error);
      }
    } catch (err) {
      setError("Failed to load tasks");
    } finally {
      setLoading(false);
      setInitialLoad(false);
    }
  };

  // Load downloaded parts
  const loadDownloadedParts = async () => {
    try {
      setLoadingParts(true);
      const result = await ApiClient.listDownloadedParts();
      if (result.data && Array.isArray(result.data)) {
        setDownloadedParts(result.data);
      } else {
        setDownloadedParts([]);
      }
    } catch (error) {
      console.error("載入已下載零件失敗:", error);
      setDownloadedParts([]);
    } finally {
      setLoadingParts(false);
    }
  };

  // Load initial data
  useEffect(() => {
    loadTasks();
    if (activeTab === "new-training") {
      loadDownloadedParts();
    }
  }, [activeTab]);

  // Auto refresh tasks
  useEffect(() => {
    if (initialLoad) return;

    const interval = setInterval(loadTasks, 5000);
    return () => clearInterval(interval);
  }, [initialLoad]);

  // Handle new training submission
  const handleStartTraining = async () => {
    if (!formData.input_dir || !formData.output_dir) {
      toast.error("請填寫所有必要欄位");
      return;
    }

    setStartingTask(true);
    try {
      const trainingRequest: TrainingRequest = {
        site: formData.site,
        line_id: formData.line_id,
        input_dir: formData.input_dir,
        output_dir: formData.output_dir,
        exclude_ng_from_ok: formData.exclude_ng_from_ok,
        config_overrides: showAdvancedConfig ? trainingConfig : undefined
      };

      const response = await ApiClient.startTraining(trainingRequest);

      if (response.data) {
        toast.success(response.data.message);
        setActiveTab("tasks");
        loadTasks();
        // Reset form
        setFormData({
          site: "HPH",
          line_id: "V31",
          input_dir: "",
          output_dir: "",
          exclude_ng_from_ok: false,
        });
        setTrainingConfig({});
        setShowAdvancedConfig(false);
      } else if (response.error) {
        toast.error(response.error);
      }
    } catch (error) {
      toast.error("啟動訓練失敗");
    } finally {
      setStartingTask(false);
    }
  };

  // Handle task deletion
  const handleDeleteTask = async (taskId: string, deleteFiles = false) => {
    try {
      const response = await ApiClient.deleteTraining(taskId, deleteFiles);
      if (response.data) {
        toast.success(response.data.message);
        loadTasks();
      } else if (response.error) {
        toast.error(response.error);
      }
    } catch (error) {
      toast.error("刪除任務失敗");
    }
  };

  // Handle task cancellation
  const handleCancelTask = async (taskId: string) => {
    try {
      const response = await ApiClient.cancelTraining(taskId);
      if (response.data) {
        toast.success(response.data.message);
        loadTasks();
      } else if (response.error) {
        toast.error(response.error);
      }
    } catch (error) {
      toast.error("取消任務失敗");
    }
  };

  // Handle module creation
  const handleCreateModule = async (taskId: string) => {
    setCreateModuleTaskId(taskId);
    setShowCreateModuleDialog(true);
  };

  const submitCreateModule = async () => {
    if (!createModuleTaskId || !moduleName || !partNumber) {
      toast.error("請填寫所有欄位");
      return;
    }

    setCreatingModule(true);
    try {
      const response = await ApiClient.createModule(createModuleTaskId, moduleName, partNumber);
      if (response.data) {
        toast.success(response.data.message);
        setShowCreateModuleDialog(false);
        setModuleName("");
        setPartNumber("");
        setCreateModuleTaskId(null);
      } else if (response.error) {
        toast.error(response.error);
      }
    } catch (error) {
      toast.error("建立模組失敗");
    } finally {
      setCreatingModule(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/70 backdrop-blur-md border-b border-white/20 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <Brain className="w-8 h-8 text-blue-600" />
                <h1 className="text-xl font-bold text-gray-900">
                  {t("title")}
                </h1>
              </div>
            </div>
            <LanguageSwitcher />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          {/* Navigation Tabs */}
          <TabsList className="grid w-full grid-cols-4 bg-white/60 backdrop-blur-sm">
            <TabsTrigger
              value="new-training"
              className="flex items-center space-x-2"
            >
              <Brain className="w-4 h-4" />
              <span>{t("navigation.new_training")}</span>
            </TabsTrigger>
            <TabsTrigger
              value="tasks"
              className="flex items-center space-x-2"
            >
              <ListTodo className="w-4 h-4" />
              <span>{t("navigation.tasks")}</span>
            </TabsTrigger>
            <TabsTrigger
              value="download"
              className="flex items-center space-x-2"
            >
              <Database className="w-4 h-4" />
              <span>{t("navigation.download")}</span>
            </TabsTrigger>
            <TabsTrigger
              value="settings"
              className="flex items-center space-x-2"
            >
              <Settings className="w-4 h-4" />
              <span>{t("navigation.settings")}</span>
            </TabsTrigger>
          </TabsList>

          {/* Tab Contents */}

          {/* New Training Tab */}
          <TabsContent value="new-training">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            >
              <TrainingForm
                formData={formData}
                onFormDataChange={setFormData}
                onSubmit={handleStartTraining}
                downloadedParts={downloadedParts}
                useExistingData={useExistingData}
                setUseExistingData={setUseExistingData}
                selectedRawdataPart={selectedRawdataPart}
                setSelectedRawdataPart={setSelectedRawdataPart}
                startingTask={startingTask}
                showAdvancedConfig={showAdvancedConfig}
                setShowAdvancedConfig={setShowAdvancedConfig}
                trainingConfig={trainingConfig}
                setTrainingConfig={setTrainingConfig}
              />
            </motion.div>
          </TabsContent>

          {/* Tasks Tab */}
          <TabsContent value="tasks">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            >
              <TaskList
                tasks={tasks}
                onRefresh={loadTasks}
                onDeleteTask={handleDeleteTask}
                onCancelTask={handleCancelTask}
                onCreateModule={handleCreateModule}
                loading={loading}
              />
            </motion.div>
          </TabsContent>

          {/* Download Tab */}
          <TabsContent value="download">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            >
              <DownloadPanel />
            </motion.div>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
            >
              <SettingsPanel />
            </motion.div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Footer */}
      <footer className="bg-white/50 backdrop-blur-sm border-t border-white/20 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="text-center text-sm text-gray-600">
            © 2024 Auto Training System. All rights reserved.
          </div>
        </div>
      </footer>

      {/* Module Creation Dialog - TODO: Move to separate component if needed */}
      {/* This would be implemented as a Dialog component */}
    </div>
  );
}