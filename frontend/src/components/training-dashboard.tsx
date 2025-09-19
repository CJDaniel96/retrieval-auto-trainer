"use client";

import { useTranslations } from "next-intl";
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Home,
  ListTodo,
  Database,
  Settings,
  Brain,
  Dumbbell,
  Target,
  Search,
  Download,
  Loader2,
  RefreshCw,
  AlertCircle
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
  });

  // Advanced training config
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState<any>({});

  // Downloaded parts for existing data
  const [downloadedParts, setDownloadedParts] = useState<PartInfo[]>([]);
  const [useExistingData, setUseExistingData] = useState(false);
  const [selectedRawdataPart, setSelectedRawdataPart] = useState<string>("");
  const [loadingParts, setLoadingParts] = useState(false);

  // Download form state
  const [downloadFormData, setDownloadFormData] = useState<DownloadRequest>({
    site: "HPH",
    line_id: "V31",
    start_date: "",
    end_date: "",
    part_number: "",
    limit: undefined,
  });
  const [showDownloadSection, setShowDownloadSection] = useState(false);
  const [estimatedCount, setEstimatedCount] = useState<number | null>(null);
  const [loadingEstimate, setLoadingEstimate] = useState(false);
  const [loadingDownload, setLoadingDownload] = useState(false);

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
      console.error("Failed to load downloaded parts:", error);
      setDownloadedParts([]);
    } finally {
      setLoadingParts(false);
    }
  };

  // Load initial data
  useEffect(() => {
    loadTasks();
    if (activeTab === "new-training" || activeTab === "download") {
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
      toast.error(t("errors.required_fields_missing"));
      return;
    }

    setStartingTask(true);
    try {
      const trainingRequest: TrainingRequest = {
        site: formData.site,
        line_id: formData.line_id,
        input_dir: formData.input_dir,
        output_dir: formData.output_dir,
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
        });
        setTrainingConfig({});
        setShowAdvancedConfig(false);
      } else if (response.error) {
        toast.error(response.error);
      }
    } catch (error) {
      toast.error(t("errors.training_start_failed"));
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
      toast.error(t("errors.task_delete_failed"));
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
      toast.error(t("errors.task_cancel_failed"));
    }
  };

  // Handle module creation
  const handleCreateModule = async (taskId: string) => {
    setCreateModuleTaskId(taskId);
    setShowCreateModuleDialog(true);
  };

  const submitCreateModule = async () => {
    if (!createModuleTaskId || !moduleName || !partNumber) {
      toast.error(t("errors.please_fill_all_fields"));
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
      toast.error(t("errors.module_create_failed"));
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
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
          {/* TypeScript Document-Style Tabs */}
          <TabsList className="relative grid w-full max-w-4xl mx-auto grid-cols-4 bg-gradient-to-br from-white to-gray-50 border border-gray-200/80 shadow-2xl rounded-lg p-2 h-20">
              {/* New Training Tab */}
              <TabsTrigger
                value="new-training"
                className="group relative flex flex-col items-center justify-center space-y-1 h-full"
              >
                <Brain className="w-5 h-5" />
                <span className="text-xs font-bold uppercase tracking-wide">
                  {t("navigation.new_training")}
                </span>
              </TabsTrigger>

              {/* Tasks Tab */}
              <TabsTrigger
                value="tasks"
                className="group relative flex flex-col items-center justify-center space-y-1 h-full"
              >
                <ListTodo className="w-5 h-5" />
                <span className="text-xs font-bold uppercase tracking-wide">
                  {t("navigation.tasks")}
                </span>
              </TabsTrigger>

              {/* Download Tab */}
              <TabsTrigger
                value="download"
                className="group relative flex flex-col items-center justify-center space-y-1 h-full"
              >
                <Database className="w-5 h-5" />
                <span className="text-xs font-bold uppercase tracking-wide">
                  {t("navigation.download")}
                </span>
              </TabsTrigger>

              {/* Settings Tab */}
              <TabsTrigger
                value="settings"
                className="group relative flex flex-col items-center justify-center space-y-1 h-full"
              >
                <Settings className="w-5 h-5" />
                <span className="text-xs font-bold uppercase tracking-wide">
                  {t("navigation.settings")}
                </span>
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
                        const result = await ApiClient.estimateDownload({
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
                          setEstimatedCount(result.data.estimated_count);
                          setShowDownloadSection(true);
                          toast.success(t("download.messages.estimate_completed"));
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
                            const result = await ApiClient.downloadRawData(
                              downloadFormData
                            );

                            if (result.error) {
                              toast.error(`下載失敗: ${result.error}`);
                            } else if (result.data) {
                              if (result.data.success) {
                                toast.success("下載成功完成");
                                // Refresh downloaded parts list
                                await loadDownloadedParts();
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
                                toast.error("下載失敗");
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
                        await loadDownloadedParts();
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

      {/* Clean Technical Footer */}
      <footer className="relative bg-white border-t border-gray-200 mt-16">
        {/* Document corner fold */}
        <div className="absolute top-0 right-0 w-8 h-8 bg-gray-100 transform rotate-45 translate-x-4 -translate-y-4 opacity-60"></div>
        <div className="absolute top-0 right-0 w-6 h-6 border-l border-b border-gray-300/50 translate-x-1 -translate-y-1"></div>

        <div className="relative max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between space-y-4 md:space-y-0">
            {/* Company Branding */}
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-lg flex items-center justify-center shadow-sm">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-gray-800">
                  Auto Training System
                </h3>
                <p className="text-xs text-gray-500">
                  Industrial AI Quality Inspection
                </p>
              </div>
            </div>

            {/* Status & Copyright */}
            <div className="flex items-center space-x-6 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-green-600 font-medium">Online</span>
              </div>
              <span className="text-gray-500">
                © 2024 Auto Training System
              </span>
            </div>
          </div>
        </div>
      </footer>

      {/* Module Creation Dialog - TODO: Move to separate component if needed */}
      {/* This would be implemented as a Dialog component */}
    </div>
  );
}