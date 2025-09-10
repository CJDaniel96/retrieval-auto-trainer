'use client';

import { useTranslations } from 'next-intl';
import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
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
  Cog
} from 'lucide-react';
import { ApiClient } from '@/lib/api-client';
import { TrainingStatus, TrainingRequest, FullConfig, ConfigUpdateRequest } from '@/lib/types';
import { LanguageSwitcher } from '@/components/language-switcher';
import { toast } from 'sonner';

export function TrainingDashboard() {
  const t = useTranslations();
  const [tasks, setTasks] = useState<TrainingStatus[]>([]);
  const [config, setConfig] = useState<FullConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [startingTask, setStartingTask] = useState(false);
  const [activeTab, setActiveTab] = useState('new-training');
  
  const [formData, setFormData] = useState<TrainingRequest>({
    input_dir: '',
    site: 'HPH',
    line_id: 'V31'
  });

  const [configData, setConfigData] = useState<ConfigUpdateRequest>({});

  useEffect(() => {
    fetchTasks();
    loadConfig();
    const interval = setInterval(fetchTasks, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchTasks = async () => {
    const response = await ApiClient.listTrainingTasks();
    if (response.data) {
      setTasks(response.data);
    }
    setLoading(false);
  };

  const loadConfig = async () => {
    const response = await ApiClient.getCurrentConfig();
    if (response.data) {
      setConfig(response.data);
      setConfigData({
        training: { ...response.data.training },
        model: { ...response.data.model },
        data: { ...response.data.data },
        loss: { ...response.data.loss }
      });
    } else if (response.error) {
      toast.error(`載入配置失敗: ${response.error}`);
    }
  };

  const handleStartTraining = async () => {
    if (!formData.input_dir) return;

    setStartingTask(true);
    const response = await ApiClient.startTraining(formData);
    if (response.data) {
      await fetchTasks();
      setFormData({ ...formData, input_dir: '' });
      toast.success('訓練任務已開始！');
    } else if (response.error) {
      toast.error(`啟動訓練失敗: ${response.error}`);
    }
    setStartingTask(false);
  };

  const handleSaveConfig = async () => {
    setSaving(true);
    const response = await ApiClient.updateConfig(configData);
    if (response.data) {
      toast.success('配置已成功更新！');
      await loadConfig();
    } else if (response.error) {
      toast.error(`更新失敗: ${response.error}`);
    }
    setSaving(false);
  };

  const handleCancelTask = async (taskId: string) => {
    await ApiClient.cancelTraining(taskId);
    await fetchTasks();
    toast.success('任務已取消');
  };

  const handleDeleteTask = async (taskId: string) => {
    await ApiClient.deleteTraining(taskId);
    await fetchTasks();
    toast.success('任務已刪除');
  };

  const updateTrainingField = (field: keyof NonNullable<ConfigUpdateRequest['training']>, value: any) => {
    setConfigData(prev => ({
      ...prev,
      training: {
        ...prev.training,
        [field]: value
      }
    }));
  };

  const updateModelField = (field: keyof NonNullable<ConfigUpdateRequest['model']>, value: any) => {
    setConfigData(prev => ({
      ...prev,
      model: {
        ...prev.model,
        [field]: value
      }
    }));
  };

  const updateDataField = (field: keyof NonNullable<ConfigUpdateRequest['data']>, value: any) => {
    setConfigData(prev => ({
      ...prev,
      data: {
        ...prev.data,
        [field]: value
      }
    }));
  };

  const updateLossField = (field: keyof NonNullable<ConfigUpdateRequest['loss']>, value: any) => {
    setConfigData(prev => ({
      ...prev,
      loss: {
        ...prev.loss,
        [field]: value
      }
    }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-4 h-4" />;
      case 'pending_orientation':
        return <AlertCircle className="w-4 h-4" />;
      case 'running':
        return <Loader2 className="w-4 h-4 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4" />;
      case 'failed':
        return <XCircle className="w-4 h-4" />;
      default:
        return <Clock className="w-4 h-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'bg-muted text-muted-foreground';
      case 'pending_orientation':
        return 'bg-yellow-500 text-yellow-50';
      case 'running':
        return 'bg-blue-500 text-blue-50';
      case 'completed':
        return 'bg-green-500 text-green-50';
      case 'failed':
        return 'bg-destructive text-destructive-foreground';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            自動化訓練系統
          </h1>
          <LanguageSwitcher />
        </div>

        {/* Main Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8 h-12 bg-white/80 backdrop-blur-sm shadow-lg border border-white/20">
            <TabsTrigger value="new-training" className="flex items-center space-x-2">
              <Home className="w-4 h-4" />
              <span>新建訓練</span>
            </TabsTrigger>
            <TabsTrigger value="task-list" className="flex items-center space-x-2">
              <ListTodo className="w-4 h-4" />
              <span>任務列表</span>
            </TabsTrigger>
            <TabsTrigger value="settings" className="flex items-center space-x-2">
              <Settings className="w-4 h-4" />
              <span>系統設定</span>
            </TabsTrigger>
            <TabsTrigger value="config" className="flex items-center space-x-2">
              <Cog className="w-4 h-4" />
              <span>訓練配置</span>
            </TabsTrigger>
          </TabsList>

          {/* New Training Tab */}
          <TabsContent value="new-training" className="space-y-6">
            <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Play className="w-6 h-6 text-blue-600" />
                  <CardTitle className="text-2xl">開始新的訓練任務</CardTitle>
                </div>
                <CardDescription>
                  配置並啟動新的模型訓練任務
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <Alert className="border-blue-200 bg-blue-50/80">
                  <AlertCircle className="h-4 w-4 text-blue-600" />
                  <AlertDescription className="text-blue-800">
                    請確保輸入資料夾中包含正確格式的圖片文件，系統會自動處理資料夾管理。
                  </AlertDescription>
                </Alert>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="input_dir" className="text-sm font-medium">輸入資料夾路徑</Label>
                    <Input
                      id="input_dir"
                      value={formData.input_dir}
                      onChange={(e) => setFormData({ ...formData, input_dir: e.target.value })}
                      placeholder="請輸入資料夾路徑"
                      className="bg-white/70"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="site" className="text-sm font-medium">產線站點</Label>
                    <Select value={formData.site} onValueChange={(value) => setFormData({ ...formData, site: value })}>
                      <SelectTrigger className="bg-white/70">
                        <SelectValue placeholder="選擇站點" />
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
                    <Label htmlFor="line_id" className="text-sm font-medium">產線ID</Label>
                    <Select value={formData.line_id} onValueChange={(value) => setFormData({ ...formData, line_id: value })}>
                      <SelectTrigger className="bg-white/70">
                        <SelectValue placeholder="選擇產線" />
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
                
                <Button
                  onClick={handleStartTraining}
                  disabled={!formData.input_dir || startingTask}
                  className="w-full md:w-auto bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 transition-all duration-200"
                  size="lg"
                >
                  {startingTask ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      啟動中...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 mr-2" />
                      開始訓練
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Task List Tab */}
          <TabsContent value="task-list" className="space-y-6">
            <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2">
                    <ListTodo className="w-6 h-6 text-blue-600" />
                    <CardTitle className="text-2xl">訓練任務列表</CardTitle>
                  </div>
                  <CardDescription>
                    監控和管理所有訓練任務的執行狀態
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={fetchTasks} disabled={loading}>
                  <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                  刷新
                </Button>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin" />
                  </div>
                ) : tasks.length === 0 ? (
                  <Alert className="border-gray-200 bg-gray-50/80">
                    <AlertCircle className="h-4 w-4" />
                    <AlertTitle>目前沒有訓練任務</AlertTitle>
                    <AlertDescription>
                      請先建立新的訓練任務來開始使用系統。
                    </AlertDescription>
                  </Alert>
                ) : (
                  <div className="space-y-4">
                    {tasks.map((task) => (
                      <Card key={task.task_id} className="bg-gradient-to-r from-white/90 to-gray-50/90 border border-gray-200/50">
                        <CardContent className="p-4 space-y-3">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3">
                              {getStatusIcon(task.status)}
                              <span className="font-semibold text-lg">{task.task_id}</span>
                              <Badge className={getStatusColor(task.status)}>
                                {t(`training.${task.status}`)}
                              </Badge>
                            </div>
                            <div className="flex items-center space-x-2">
                              {task.status === 'completed' && (
                                <Button size="sm" variant="outline">
                                  <Download className="w-4 h-4 mr-2" />
                                  下載
                                </Button>
                              )}
                              {(task.status === 'pending' || task.status === 'running') && (
                                <Button 
                                  size="sm" 
                                  variant="outline"
                                  onClick={() => handleCancelTask(task.task_id)}
                                >
                                  <Square className="w-4 h-4 mr-2" />
                                  取消
                                </Button>
                              )}
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => handleDeleteTask(task.task_id)}
                              >
                                <Trash2 className="w-4 h-4 mr-2" />
                                刪除
                              </Button>
                            </div>
                          </div>
                          
                          {task.current_step && (
                            <p className="text-sm text-muted-foreground bg-muted/30 p-3 rounded-md border">
                              {task.current_step}
                            </p>
                          )}
                          
                          {task.progress !== undefined && (
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span>訓練進度</span>
                                <span className="font-medium">{Math.round(task.progress * 100)}%</span>
                              </div>
                              <Progress value={task.progress * 100} className="h-2" />
                            </div>
                          )}
                          
                          {task.error_message && (
                            <Alert variant="destructive" className="border-red-200 bg-red-50/80">
                              <XCircle className="h-4 w-4" />
                              <AlertTitle>錯誤訊息</AlertTitle>
                              <AlertDescription>{task.error_message}</AlertDescription>
                            </Alert>
                          )}
                          
                          <div className="flex justify-between text-sm text-muted-foreground pt-2 border-t border-gray-200/50">
                            <span>建立時間: {task.start_time ? new Date(task.start_time).toLocaleString() : 'N/A'}</span>
                            {task.end_time && (
                              <span>完成時間: {new Date(task.end_time).toLocaleString()}</span>
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-6">
            <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Settings className="w-6 h-6 text-blue-600" />
                  <CardTitle className="text-2xl">系統設定</CardTitle>
                </div>
                <CardDescription>
                  管理系統相關設定和偏好
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Alert className="border-blue-200 bg-blue-50/80">
                  <AlertCircle className="h-4 w-4 text-blue-600" />
                  <AlertDescription className="text-blue-800">
                    系統設定功能正在開發中，敬請期待。
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Configuration Tab */}
          <TabsContent value="config" className="space-y-6">
            <Alert className="border-blue-200 bg-blue-50/80">
              <AlertCircle className="h-4 w-4 text-blue-600" />
              <AlertDescription className="text-blue-800">
                在這裡修改的配置會影響所有新的訓練任務。修改後請記得保存配置。
              </AlertDescription>
            </Alert>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Training Configuration */}
              <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
                <CardHeader>
                  <div className="flex items-center space-x-2">
                    <Dumbbell className="w-5 h-5 text-blue-600" />
                    <CardTitle>訓練配置</CardTitle>
                  </div>
                  <CardDescription>調整訓練相關的參數設定</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="max_epochs">最大訓練輪數</Label>
                      <Input
                        id="max_epochs"
                        type="number"
                        min={1}
                        max={200}
                        value={configData.training?.max_epochs || ''}
                        onChange={(e) => updateTrainingField('max_epochs', parseInt(e.target.value) || 0)}
                        className="bg-white/70"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="batch_size">批次大小</Label>
                      <Input
                        id="batch_size"
                        type="number"
                        min={1}
                        max={256}
                        value={configData.training?.batch_size || ''}
                        onChange={(e) => updateTrainingField('batch_size', parseInt(e.target.value) || 0)}
                        className="bg-white/70"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="lr">學習率</Label>
                      <Input
                        id="lr"
                        type="number"
                        step="0.0001"
                        min={0.0001}
                        max={0.1}
                        value={configData.training?.lr || ''}
                        onChange={(e) => updateTrainingField('lr', parseFloat(e.target.value) || 0)}
                        className="bg-white/70"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="weight_decay">權重衰減</Label>
                      <Input
                        id="weight_decay"
                        type="number"
                        step="0.0001"
                        min={0}
                        max={0.01}
                        value={configData.training?.weight_decay || ''}
                        onChange={(e) => updateTrainingField('weight_decay', parseFloat(e.target.value) || 0)}
                        className="bg-white/70"
                      />
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="early_stopping"
                      checked={configData.training?.enable_early_stopping ?? true}
                      onCheckedChange={(checked) => updateTrainingField('enable_early_stopping', checked)}
                    />
                    <Label htmlFor="early_stopping">啟用提前停止</Label>
                  </div>
                </CardContent>
              </Card>

              {/* Model Configuration */}
              <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
                <CardHeader>
                  <div className="flex items-center space-x-2">
                    <Brain className="w-5 h-5 text-blue-600" />
                    <CardTitle>模型配置</CardTitle>
                  </div>
                  <CardDescription>調整模型架構和相關參數</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="structure">模型結構</Label>
                    <Select
                      value={configData.model?.structure || ''}
                      onValueChange={(value) => updateModelField('structure', value as 'HOAM' | 'HOAMV2')}
                    >
                      <SelectTrigger className="bg-white/70">
                        <SelectValue placeholder="選擇模型結構" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="HOAM">HOAM</SelectItem>
                        <SelectItem value="HOAMV2">HOAMV2</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="backbone">骨幹網路</Label>
                    <Select
                      value={configData.model?.backbone || ''}
                      onValueChange={(value) => updateModelField('backbone', value)}
                    >
                      <SelectTrigger className="bg-white/70">
                        <SelectValue placeholder="選擇骨幹網路" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="efficientnetv2_rw_s">EfficientNetV2-S</SelectItem>
                        <SelectItem value="efficientnetv2_rw_m">EfficientNetV2-M</SelectItem>
                        <SelectItem value="resnet50">ResNet50</SelectItem>
                        <SelectItem value="resnet101">ResNet101</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Save Button */}
            <div className="flex justify-center pt-6">
              <Button 
                onClick={handleSaveConfig} 
                disabled={saving}
                className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 transition-all duration-200"
                size="lg"
              >
                {saving ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    保存中...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    保存配置
                  </>
                )}
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}