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
import { motion } from 'framer-motion';
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
  FileText
} from 'lucide-react';
import { ApiClient } from '@/lib/api-client';
import { TrainingStatus, TrainingRequest } from '@/lib/types';
import { LanguageSwitcher } from '@/components/language-switcher';
import { toast } from 'sonner';
import { useRouter } from '@/i18n/routing';

export function TrainingDashboard() {
  const t = useTranslations();
  const router = useRouter();
  const [tasks, setTasks] = useState<TrainingStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [startingTask, setStartingTask] = useState(false);
  const [activeTab, setActiveTab] = useState('new-training');
  const [showAdvancedConfig, setShowAdvancedConfig] = useState(false);
  
  const [formData, setFormData] = useState<TrainingRequest>({
    input_dir: '',
    site: 'HPH',
    line_id: 'V31',
    experiment_config: {
      name: 'hoam_experiment'
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
      checkpoint_dir: 'checkpoints'
    },
    model_config: {
      structure: 'HOAMV2',
      backbone: 'efficientnetv2_rw_s',
      pretrained: false,
      embedding_size: 512
    },
    data_config: {
      image_size: 224,
      num_workers: 4,
      test_split: 0.2
    },
    loss_config: {
      type: 'HybridMarginLoss',
      subcenter_margin: 0.4,
      subcenter_scale: 30.0,
      sub_centers: 3,
      triplet_margin: 0.3,
      center_loss_weight: 0.01
    },
    knn_config: {
      enable: false,
      threshold: 0.5,
      index_path: 'knn.index',
      dataset_pkl: 'dataset.pkl'
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
      toast.success(t('messages.training_started'));
    } else if (response.error) {
      toast.error(`${t('messages.error_occurred')}: ${response.error}`);
    }
    setStartingTask(false);
  };

  const handleCancelTask = async (taskId: string) => {
    await ApiClient.cancelTraining(taskId);
    await fetchTasks();
    toast.success(t('messages.task_cancelled'));
  };

  const handleDeleteTask = async (taskId: string) => {
    await ApiClient.deleteTraining(taskId);
    await fetchTasks();
    toast.success(t('messages.task_cancelled'));
  };

  const handleOrientationConfirm = (taskId: string) => {
    // Construct the URL based on the current origin and locale
    const currentUrl = new URL(window.location.href);
    const pathSegments = currentUrl.pathname.split('/').filter(Boolean);
    const locale = pathSegments[0] || 'zh'; // Get locale from current path
    const orientationUrl = `${currentUrl.origin}/${locale}/orientation/${taskId}`;
    window.open(orientationUrl, '_blank');
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

  const updateExperimentField = (field: keyof NonNullable<TrainingRequest['experiment_config']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      experiment_config: {
        ...prev.experiment_config,
        [field]: value
      }
    }));
  };

  const updateKnnField = (field: keyof NonNullable<TrainingRequest['knn_config']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      knn_config: {
        ...prev.knn_config,
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
                  {t('common.title')}
                </h1>
                <p className="text-sm text-muted-foreground">Image Retrieval Model Training Platform</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-full bg-green-100 text-green-700 text-sm font-medium">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
{t('footer.powered_by')} {t('footer.ai_system')}
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
{t('training.start_new')}
              </TabsTrigger>
              <TabsTrigger 
                value="task-list" 
                className="inline-flex items-center justify-center whitespace-nowrap rounded-xl px-6 py-3 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-lg hover:bg-white/60 data-[state=active]:hover:from-blue-700 data-[state=active]:hover:to-indigo-700"
              >
                <ListTodo className="w-4 h-4 mr-2" />
{t('navigation.training')}
                {tasks.length > 0 && (
                  <span className="ml-2 inline-flex items-center justify-center px-2 py-0.5 text-xs font-bold leading-none text-white bg-red-500 rounded-full">
                    {tasks.length}
                  </span>
                )}
              </TabsTrigger>
              <TabsTrigger 
                value="settings" 
                className="inline-flex items-center justify-center whitespace-nowrap rounded-xl px-6 py-3 text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-gradient-to-r data-[state=active]:from-blue-600 data-[state=active]:to-indigo-600 data-[state=active]:text-white data-[state=active]:shadow-lg hover:bg-white/60 data-[state=active]:hover:from-blue-700 data-[state=active]:hover:to-indigo-700"
              >
                <Settings className="w-4 h-4 mr-2" />
                {t('navigation.settings')}
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
                  <CardTitle className="text-2xl">{t('training.start_new')}</CardTitle>
                </div>
                <CardDescription>
{t('form.configure_task')}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <Alert className="border-blue-200 bg-blue-50/80">
                  <AlertCircle className="h-4 w-4 text-blue-600" />
                  <AlertDescription className="text-blue-800">
{t('form.folder_management_info')}
                  </AlertDescription>
                </Alert>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <Label htmlFor="input_dir" className="text-sm font-medium">{t('training.input_directory')}</Label>
                    <Input
                      id="input_dir"
                      value={formData.input_dir}
                      onChange={(e) => setFormData({ ...formData, input_dir: e.target.value })}
                      placeholder={t('form.input_placeholder')}
                      className="bg-white/70"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="site" className="text-sm font-medium">{t('training.site')}</Label>
                    <Select value={formData.site} onValueChange={(value) => setFormData({ ...formData, site: value })}>
                      <SelectTrigger className="bg-white/70">
                        <SelectValue placeholder={t('form.select_site')} />
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
                    <Label htmlFor="line_id" className="text-sm font-medium">{t('training.line_id')}</Label>
                    <Select value={formData.line_id} onValueChange={(value) => setFormData({ ...formData, line_id: value })}>
                      <SelectTrigger className="bg-white/70">
                        <SelectValue placeholder={t('form.select_line')} />
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
                      <span>{t('config.title')}</span>
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
{t('messages.starting_training')}
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
{t('common.start')}
                      </>
                    )}
                  </Button>
                </div>
                
                {/* Advanced Configuration Panel */}
                {showAdvancedConfig && (
                  <div className="mt-6 p-6 bg-gradient-to-r from-gray-50 to-blue-50 rounded-lg border border-gray-200">
                    <div className="flex items-center space-x-2 mb-4">
                      <Cog className="w-5 h-5 text-blue-600" />
                      <h3 className="text-lg font-semibold text-gray-800">{t('config.title')}</h3>
                    </div>
                    <p className="text-sm text-gray-600 mb-6">
這些設定僅會套用到當前的訓練任務，不會影響其他任務。所有參數都來自training_configs.yaml。
                    </p>
                    
                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                      {/* Experiment Configuration */}
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <FileText className="w-4 h-4 text-blue-600" />
                          <h4 className="font-medium text-gray-700">{t('config.experiment.title')}</h4>
                        </div>
                        <div className="space-y-2">
                          <Label htmlFor="experiment_name" className="text-xs">{t('config.experiment.name')}</Label>
                          <Input
                            id="experiment_name"
                            value={formData.experiment_config?.name || ''}
                            onChange={(e) => updateExperimentField('name', e.target.value)}
                            className="h-8 text-sm"
                            placeholder="hoam_experiment"
                          />
                        </div>
                      </div>

                      {/* Training Configuration */}
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <Dumbbell className="w-4 h-4 text-blue-600" />
                          <h4 className="font-medium text-gray-700">{t('config.training.title')}</h4>
                        </div>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-2">
                            <Label htmlFor="min_epochs" className="text-xs">{t('config.training.min_epochs')}</Label>
                            <Input
                              id="min_epochs"
                              type="number"
                              min={0}
                              max={50}
                              value={formData.training_config?.min_epochs || ''}
                              onChange={(e) => updateTrainingField('min_epochs', parseInt(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="max_epochs" className="text-xs">{t('config.training.max_epochs')}</Label>
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
                            <Label htmlFor="batch_size" className="text-xs">{t('config.training.batch_size')}</Label>
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
                            <Label htmlFor="lr" className="text-xs">{t('config.training.learning_rate')}</Label>
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
                            <Label htmlFor="weight_decay" className="text-xs">{t('config.training.weight_decay')}</Label>
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
                          <div className="space-y-2">
                            <Label htmlFor="patience" className="text-xs">{t('config.training.patience')}</Label>
                            <Input
                              id="patience"
                              type="number"
                              min={1}
                              max={50}
                              value={formData.training_config?.patience || ''}
                              onChange={(e) => updateTrainingField('patience', parseInt(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="freeze_epochs" className="text-xs">{t('config.training.freeze_backbone_epochs')}</Label>
                            <Input
                              id="freeze_epochs"
                              type="number"
                              min={0}
                              max={20}
                              value={formData.training_config?.freeze_backbone_epochs || ''}
                              onChange={(e) => updateTrainingField('freeze_backbone_epochs', parseInt(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="checkpoint_dir" className="text-xs">{t('config.training.checkpoint_dir')}</Label>
                            <Input
                              id="checkpoint_dir"
                              value={formData.training_config?.checkpoint_dir || ''}
                              onChange={(e) => updateTrainingField('checkpoint_dir', e.target.value)}
                              className="h-8 text-sm"
                              placeholder="checkpoints"
                            />
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Switch
                            id="early_stopping"
                            checked={formData.training_config?.enable_early_stopping ?? true}
                            onCheckedChange={(checked) => updateTrainingField('enable_early_stopping', checked)}
                          />
                          <Label htmlFor="early_stopping" className="text-sm">{t('config.training.enable_early_stopping')}</Label>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
                      {/* Model Configuration */}
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <Brain className="w-4 h-4 text-blue-600" />
                          <h4 className="font-medium text-gray-700">{t('config.model.title')}</h4>
                        </div>
                        <div className="space-y-3">
                          <div className="space-y-2">
                            <Label htmlFor="structure" className="text-xs">{t('config.model.structure')}</Label>
                            <Select
                              value={formData.model_config?.structure || ''}
                              onValueChange={(value) => updateModelField('structure', value as 'HOAM' | 'HOAMV2')}
                            >
                              <SelectTrigger className="h-8 text-sm">
                                <SelectValue placeholder={t('config.model.structure')} />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="HOAM">HOAM</SelectItem>
                                <SelectItem value="HOAMV2">HOAMV2</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="backbone" className="text-xs">{t('config.model.backbone')}</Label>
                            <Select
                              value={formData.model_config?.backbone || ''}
                              onValueChange={(value) => updateModelField('backbone', value)}
                            >
                              <SelectTrigger className="h-8 text-sm">
                                <SelectValue placeholder={t('config.model.backbone')} />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="efficientnetv2_rw_s">EfficientNetV2-S</SelectItem>
                                <SelectItem value="efficientnetv2_rw_m">EfficientNetV2-M</SelectItem>
                                <SelectItem value="resnet50">ResNet50</SelectItem>
                                <SelectItem value="resnet101">ResNet101</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="embedding_size" className="text-xs">{t('config.model.embedding_size')}</Label>
                            <Input
                              id="embedding_size"
                              type="number"
                              min={64}
                              max={2048}
                              step={64}
                              value={formData.model_config?.embedding_size || ''}
                              onChange={(e) => updateModelField('embedding_size', parseInt(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="flex items-center space-x-2 pt-2">
                            <Switch
                              id="pretrained"
                              checked={formData.model_config?.pretrained ?? false}
                              onCheckedChange={(checked) => updateModelField('pretrained', checked)}
                            />
                            <Label htmlFor="pretrained" className="text-sm">{t('config.model.pretrained')}</Label>
                          </div>
                        </div>
                      </div>

                      {/* Data Configuration */}
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <Database className="w-4 h-4 text-blue-600" />
                          <h4 className="font-medium text-gray-700">{t('config.data.title')}</h4>
                        </div>
                        <div className="space-y-3">
                          <div className="space-y-2">
                            <Label htmlFor="image_size" className="text-xs">{t('config.data.image_size')}</Label>
                            <Select
                              value={formData.data_config?.image_size?.toString() || ''}
                              onValueChange={(value) => updateDataField('image_size', parseInt(value))}
                            >
                              <SelectTrigger className="h-8 text-sm">
                                <SelectValue placeholder={t('config.data.image_size')} />
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
                            <Label htmlFor="num_workers" className="text-xs">{t('config.data.num_workers')}</Label>
                            <Input
                              id="num_workers"
                              type="number"
                              min={0}
                              max={16}
                              value={formData.data_config?.num_workers || ''}
                              onChange={(e) => updateDataField('num_workers', parseInt(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="test_split" className="text-xs">{t('config.data.test_split')}</Label>
                            <Input
                              id="test_split"
                              type="number"
                              step="0.05"
                              min={0.1}
                              max={0.5}
                              value={formData.data_config?.test_split || ''}
                              onChange={(e) => updateDataField('test_split', parseFloat(e.target.value) || 0)}
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
                          <h4 className="font-medium text-gray-700">{t('config.loss.title')}</h4>
                        </div>
                        <div className="space-y-3">
                          <div className="space-y-2">
                            <Label htmlFor="loss_type" className="text-xs">{t('config.loss.type')}</Label>
                            <Select
                              value={formData.loss_config?.type || ''}
                              onValueChange={(value) => updateLossField('type', value as 'HybridMarginLoss' | 'ArcFaceLoss' | 'SubCenterArcFaceLoss')}
                            >
                              <SelectTrigger className="h-8 text-sm">
                                <SelectValue placeholder={t('config.loss.type')} />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="HybridMarginLoss">HybridMarginLoss</SelectItem>
                                <SelectItem value="ArcFaceLoss">ArcFaceLoss</SelectItem>
                                <SelectItem value="SubCenterArcFaceLoss">SubCenterArcFaceLoss</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="grid grid-cols-2 gap-3">
                            <div className="space-y-2">
                              <Label htmlFor="subcenter_margin" className="text-xs">{t('config.loss.subcenter_margin')}</Label>
                              <Input
                                id="subcenter_margin"
                                type="number"
                                step="0.1"
                                min={0.1}
                                max={1.0}
                                value={formData.loss_config?.subcenter_margin || ''}
                                onChange={(e) => updateLossField('subcenter_margin', parseFloat(e.target.value) || 0)}
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="subcenter_scale" className="text-xs">{t('config.loss.subcenter_scale')}</Label>
                              <Input
                                id="subcenter_scale"
                                type="number"
                                step="1"
                                min={1}
                                max={100}
                                value={formData.loss_config?.subcenter_scale || ''}
                                onChange={(e) => updateLossField('subcenter_scale', parseFloat(e.target.value) || 0)}
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="sub_centers" className="text-xs">{t('config.loss.sub_centers')}</Label>
                              <Input
                                id="sub_centers"
                                type="number"
                                min={1}
                                max={10}
                                value={formData.loss_config?.sub_centers || ''}
                                onChange={(e) => updateLossField('sub_centers', parseInt(e.target.value) || 0)}
                                className="h-8 text-sm"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="triplet_margin" className="text-xs">{t('config.loss.triplet_margin')}</Label>
                              <Input
                                id="triplet_margin"
                                type="number"
                                step="0.1"
                                min={0.1}
                                max={1.0}
                                value={formData.loss_config?.triplet_margin || ''}
                                onChange={(e) => updateLossField('triplet_margin', parseFloat(e.target.value) || 0)}
                                className="h-8 text-sm"
                              />
                            </div>
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="center_loss_weight" className="text-xs">{t('config.loss.center_loss_weight')}</Label>
                            <Input
                              id="center_loss_weight"
                              type="number"
                              step="0.001"
                              min={0.001}
                              max={0.1}
                              value={formData.loss_config?.center_loss_weight || ''}
                              onChange={(e) => updateLossField('center_loss_weight', parseFloat(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                        </div>
                      </div>

                      {/* KNN Configuration */}
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          <Search className="w-4 h-4 text-blue-600" />
                          <h4 className="font-medium text-gray-700">{t('config.knn.title')}</h4>
                        </div>
                        <div className="space-y-3">
                          <div className="flex items-center space-x-2">
                            <Switch
                              id="knn_enable"
                              checked={formData.knn_config?.enable ?? false}
                              onCheckedChange={(checked) => updateKnnField('enable', checked)}
                            />
                            <Label htmlFor="knn_enable" className="text-sm">{t('config.knn.enable')}</Label>
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="knn_threshold" className="text-xs">{t('config.knn.threshold')}</Label>
                            <Input
                              id="knn_threshold"
                              type="number"
                              step="0.1"
                              min={0.1}
                              max={1.0}
                              value={formData.knn_config?.threshold || ''}
                              onChange={(e) => updateKnnField('threshold', parseFloat(e.target.value) || 0)}
                              className="h-8 text-sm"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="index_path" className="text-xs">{t('config.knn.index_path')}</Label>
                            <Input
                              id="index_path"
                              value={formData.knn_config?.index_path || ''}
                              onChange={(e) => updateKnnField('index_path', e.target.value)}
                              className="h-8 text-sm"
                              placeholder="knn.index"
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="dataset_pkl" className="text-xs">{t('config.knn.dataset_pkl')}</Label>
                            <Input
                              id="dataset_pkl"
                              value={formData.knn_config?.dataset_pkl || ''}
                              onChange={(e) => updateKnnField('dataset_pkl', e.target.value)}
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
                    <CardTitle className="text-2xl">{t('training.title')}</CardTitle>
                  </div>
                  <CardDescription>
{t('form.monitor_tasks')}
                  </CardDescription>
                </div>
                <Button variant="outline" size="sm" onClick={fetchTasks} disabled={loading}>
                  <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
{t('form.refresh')}
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
                    <AlertTitle>{t('form.no_training_tasks')}</AlertTitle>
                    <AlertDescription>
{t('form.no_tasks_description')}
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
                              {task.status === 'pending_orientation' && (
                                <Button 
                                  size="sm" 
                                  variant="default"
                                  className="bg-yellow-600 hover:bg-yellow-700 text-white"
                                  onClick={() => handleOrientationConfirm(task.task_id)}
                                >
                                  <AlertCircle className="w-4 h-4 mr-2" />
{t('form.confirm_orientation')}
                                </Button>
                              )}
                              {task.status === 'completed' && (
                                <Button size="sm" variant="outline">
                                  <Download className="w-4 h-4 mr-2" />
{t('common.download')}
                                </Button>
                              )}
                              {(task.status === 'pending' || task.status === 'running') && (
                                <Button 
                                  size="sm" 
                                  variant="outline"
                                  onClick={() => handleCancelTask(task.task_id)}
                                >
                                  <Square className="w-4 h-4 mr-2" />
{t('common.cancel')}
                                </Button>
                              )}
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => handleDeleteTask(task.task_id)}
                              >
                                <Trash2 className="w-4 h-4 mr-2" />
{t('common.delete')}
                              </Button>
                            </div>
                          </div>
                          
                          {task.status === 'pending_orientation' && (
                            <Alert className="border-yellow-200 bg-yellow-50/80">
                              <AlertCircle className="h-4 w-4 text-yellow-600" />
                              <AlertTitle className="text-yellow-800">{t('orientation.title')}</AlertTitle>
                              <AlertDescription className="text-yellow-700">
  {t('orientation.description')}
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
                                <span>{t('training.progress')}</span>
                                <span className="font-medium">{Math.round(task.progress * 100)}%</span>
                              </div>
                              <Progress value={task.progress * 100} className="h-2" />
                            </div>
                          )}
                          
                          {task.error_message && (
                            <Alert variant="destructive" className="border-red-200 bg-red-50/80">
                              <XCircle className="h-4 w-4" />
                              <AlertTitle>{t('form.error_title')}</AlertTitle>
                              <AlertDescription>{task.error_message}</AlertDescription>
                            </Alert>
                          )}
                          
                          <div className="flex justify-between text-sm text-muted-foreground pt-2 border-t border-gray-200/50">
                            <span>{t('training.created_at')}: {task.start_time ? new Date(task.start_time).toLocaleString() : 'N/A'}</span>
                            {task.end_time && (
                              <span>{t('training.completed_at')}: {new Date(task.end_time).toLocaleString()}</span>
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

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: "easeOut" }}
              className="space-y-6"
            >
            <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Settings className="w-6 h-6 text-blue-600" />
                  <CardTitle className="text-2xl">{t('navigation.settings')}</CardTitle>
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
            </motion.div>
          </TabsContent>

        </Tabs>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/20 bg-white/60 backdrop-blur-sm mt-16">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-600 to-indigo-600 flex items-center justify-center">
                  <Brain className="w-5 h-5 text-white" />
                </div>
                <span className="font-semibold text-gray-800">{t('footer.ai_system')}</span>
              </div>
              <p className="text-sm text-muted-foreground max-w-xs">
                專為影像檢索模型設計的自動化訓練平台，提供完整的訓練工作流程和任務管理功能。
              </p>
            </div>
            
            <div className="space-y-4">
              <h3 className="font-medium text-gray-800">系統功能</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-center space-x-2">
                  <Play className="w-3 h-3 text-blue-600" />
                  <span>自動化訓練管理</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Database className="w-3 h-3 text-blue-600" />
                  <span>數據處理與分類</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Target className="w-3 h-3 text-blue-600" />
                  <span>模型評估與分析</span>
                </li>
                <li className="flex items-center space-x-2">
                  <Cog className="w-3 h-3 text-blue-600" />
                  <span>彈性配置設定</span>
                </li>
              </ul>
            </div>
            
            <div className="space-y-4">
              <h3 className="font-medium text-gray-800">系統狀態</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-2 rounded-lg bg-green-50 border border-green-200">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <span className="text-sm text-green-700">API服務</span>
                  </div>
                  <span className="text-xs text-green-600 font-medium">運行中</span>
                </div>
                <div className="flex items-center justify-between p-2 rounded-lg bg-blue-50 border border-blue-200">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span className="text-sm text-blue-700">前端介面</span>
                  </div>
                  <span className="text-xs text-blue-600 font-medium">運行中</span>
                </div>
                <div className="flex items-center justify-between p-2 rounded-lg bg-gray-50 border border-gray-200">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                    <span className="text-sm text-gray-600">活動任務</span>
                  </div>
                  <span className="text-xs text-gray-500 font-medium">{tasks.filter(t => t.status === 'running').length} 個</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="border-t border-gray-200 mt-8 pt-6 flex flex-col md:flex-row justify-between items-center">
            <p className="text-sm text-muted-foreground">
              © 2024 自動化訓練系統. 專為影像檢索模型訓練設計.
            </p>
            <div className="flex items-center space-x-4 mt-4 md:mt-0">
              <span className="text-xs text-muted-foreground">Version 1.0.0</span>
              <div className="w-1 h-1 bg-gray-300 rounded-full"></div>
              <span className="text-xs text-muted-foreground">Built with Next.js & FastAPI</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}