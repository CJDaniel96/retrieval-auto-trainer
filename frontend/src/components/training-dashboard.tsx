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
import { TrainingStatus, TrainingRequest } from '@/lib/types';
import { LanguageSwitcher } from '@/components/language-switcher';
import { toast } from 'sonner';

export function TrainingDashboard() {
  const t = useTranslations();
  const [tasks, setTasks] = useState<TrainingStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [startingTask, setStartingTask] = useState(false);
  const [activeTab, setActiveTab] = useState('new-training');
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  
  const [formData, setFormData] = useState<TrainingRequest>({
    input_dir: '',
    site: 'HPH',
    line_id: 'V31',
    training_config: {
      max_epochs: 50,
      batch_size: 32,
      lr: 0.001,
      weight_decay: 0.0001,
      patience: 10,
      enable_early_stopping: true,
      freeze_backbone_epochs: 0
    },
    model_config: {
      structure: 'HOAMV2',
      backbone: 'efficientnetv2_rw_s',
      pretrained: true,
      embedding_size: 512
    },
    data_config: {
      image_size: 224,
      num_workers: 4,
      test_split: 0.2
    },
    loss_config: {
      type: 'HybridMarginLoss',
      subcenter_margin: 0.5,
      subcenter_scale: 30
    }
  });

  useEffect(() => {
    fetchTasks();
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

  const handleStartTraining = async () => {
    if (!formData.input_dir) return;

    setStartingTask(true);
    const response = await ApiClient.startTraining(formData);
    if (response.data) {
      await fetchTasks();
      // Reset only the input directory, keep the configuration
      setFormData({ ...formData, input_dir: '' });
      toast.success('訓練任務已開始！此任務將使用獨立的配置設定。');
    } else if (response.error) {
      toast.error(`啟動訓練失敗: ${response.error}`);
    }
    setStartingTask(false);
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

  const updateTrainingField = (field: keyof NonNullable<TrainingRequest['training_config']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      training_config: {
        ...prev.training_config,
        [field]: value
      }
    }));
  };

  const updateModelField = (field: keyof NonNullable<TrainingRequest['model_config']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      model_config: {
        ...prev.model_config,
        [field]: value
      }
    }));
  };

  const updateDataField = (field: keyof NonNullable<TrainingRequest['data_config']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      data_config: {
        ...prev.data_config,
        [field]: value
      }
    }));
  };

  const updateLossField = (field: keyof NonNullable<TrainingRequest['loss_config']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      loss_config: {
        ...prev.loss_config,
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
          <TabsList className="grid w-full grid-cols-3 mb-8 h-12 bg-white/80 backdrop-blur-sm shadow-lg border border-white/20">
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
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => setShowAdvancedConfig(!showAdvancedConfig)}
                      className="flex items-center space-x-2"
                    >
                      <Cog className="w-4 h-4" />
                      <span>進階配置</span>
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
                        啟動中...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        開始訓練
                      </>
                    )}
                  </Button>
                </div>
                
                {/* Advanced Configuration Panel */}
                {showAdvancedConfig && (
                  <div className="mt-6 p-6 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg border border-gray-200">
                    <div className="flex items-center space-x-2 mb-4">
                      <Cog className="w-5 h-5 text-blue-600" />
                      <h3 className="text-lg font-semibold text-gray-800">任務專用配置</h3>
                    </div>
                    <p className="text-sm text-gray-600 mb-6">
                      這些設定僅會套用到當前的訓練任務，不會影響其他任務。
                    </p>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      {/* Training Configuration */}
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <Dumbbell className="w-4 h-4 text-blue-600" />
                          <h4 className="font-medium text-gray-700">訓練參數</h4>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-2">
                            <Label htmlFor="max_epochs" className="text-xs">最大訓練輪數</Label>
                            <Input
                              id="max_epochs"
                              type="number"
                              min={1}
                              max={200}
                              value={formData.training_config?.max_epochs || ''}
                              onChange={(e) => updateTrainingField('max_epochs', parseInt(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="batch_size" className="text-xs">批次大小</Label>
                            <Input
                              id="batch_size"
                              type="number"
                              min={1}
                              max={256}
                              value={formData.training_config?.batch_size || ''}
                              onChange={(e) => updateTrainingField('batch_size', parseInt(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="lr" className="text-xs">學習率</Label>
                            <Input
                              id="lr"
                              type="number"
                              step="0.0001"
                              min={0.0001}
                              max={0.1}
                              value={formData.training_config?.lr || ''}
                              onChange={(e) => updateTrainingField('lr', parseFloat(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="weight_decay" className="text-xs">權重衰減</Label>
                            <Input
                              id="weight_decay"
                              type="number"
                              step="0.0001"
                              min={0}
                              max={0.01}
                              value={formData.training_config?.weight_decay || ''}
                              onChange={(e) => updateTrainingField('weight_decay', parseFloat(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Switch
                            id="early_stopping"
                            checked={formData.training_config?.enable_early_stopping ?? true}
                            onCheckedChange={(checked) => updateTrainingField('enable_early_stopping', checked)}
                          />
                          <Label htmlFor="early_stopping" className="text-sm">啟用提前停止</Label>
                        </div>
                      </div>

                      {/* Model Configuration */}
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <Brain className="w-4 h-4 text-blue-600" />
                          <h4 className="font-medium text-gray-700">模型配置</h4>
                        </div>
                        <div className="space-y-3">
                          <div className="space-y-2">
                            <Label htmlFor="structure" className="text-xs">模型結構</Label>
                            <Select
                              value={formData.model_config?.structure || ''}
                              onValueChange={(value) => updateModelField('structure', value as 'HOAM' | 'HOAMV2')}
                            >
                              <SelectTrigger className="h-8 text-sm">
                                <SelectValue placeholder="選擇模型結構" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="HOAM">HOAM</SelectItem>
                                <SelectItem value="HOAMV2">HOAMV2</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="backbone" className="text-xs">骨幹網路</Label>
                            <Select
                              value={formData.model_config?.backbone || ''}
                              onValueChange={(value) => updateModelField('backbone', value)}
                            >
                              <SelectTrigger className="h-8 text-sm">
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
                          <div className="flex items-center space-x-2 pt-2">
                            <Switch
                              id="pretrained"
                              checked={formData.model_config?.pretrained ?? false}
                              onCheckedChange={(checked) => updateModelField('pretrained', checked)}
                            />
                            <Label htmlFor="pretrained" className="text-sm">使用預訓練權重</Label>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
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

        </Tabs>
      </div>
    </div>
  );
}