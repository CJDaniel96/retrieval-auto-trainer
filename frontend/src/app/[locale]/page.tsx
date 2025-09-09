'use client';

import { useTranslations } from 'next-intl';
import { useState, useEffect } from 'react';
import { useRouter } from '@/i18n/routing';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
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
  Loader2
} from 'lucide-react';
import { ApiClient } from '@/lib/api-client';
import { TrainingStatus, TrainingRequest } from '@/lib/types';
import { LanguageSwitcher } from '@/components/language-switcher';

export default function HomePage() {
  const t = useTranslations();
  const router = useRouter();
  const [tasks, setTasks] = useState<TrainingStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [startingTask, setStartingTask] = useState(false);
  const [formData, setFormData] = useState<TrainingRequest>({
    input_dir: '',
    site: 'HPH',
    line_id: 'V31'
  });

  useEffect(() => {
    fetchTasks();
    const interval = setInterval(fetchTasks, 5000); // Poll every 5 seconds
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
    if (!formData.input_dir) {
      return;
    }

    setStartingTask(true);
    const response = await ApiClient.startTraining(formData);
    if (response.data) {
      await fetchTasks();
      setFormData({ ...formData, input_dir: '' });
    }
    setStartingTask(false);
  };

  const handleCancelTask = async (taskId: string) => {
    await ApiClient.cancelTraining(taskId);
    await fetchTasks();
  };

  const handleDeleteTask = async (taskId: string) => {
    await ApiClient.deleteTraining(taskId);
    await fetchTasks();
  };

  const handleOrientationConfirm = (taskId: string) => {
    router.push(`/orientation/${taskId}`);
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
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold tracking-tight">{t('common.title')}</h1>
          <LanguageSwitcher />
        </div>
        
        {/* Start Training Form */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>{t('training.start_new')}</CardTitle>
            <CardDescription>
              {t('form.configure_task')}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {t('form.folder_management_info')}
              </AlertDescription>
            </Alert>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="input_dir">{t('training.input_directory')}</Label>
                <Input
                  id="input_dir"
                  value={formData.input_dir}
                  onChange={(e) => setFormData({ ...formData, input_dir: e.target.value })}
                  placeholder={t('form.input_placeholder')}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="site">{t('training.site')}</Label>
                <Select value={formData.site} onValueChange={(value) => setFormData({ ...formData, site: value })}>
                  <SelectTrigger>
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
                <Label htmlFor="line_id">{t('training.line_id')}</Label>
                <Select value={formData.line_id} onValueChange={(value) => setFormData({ ...formData, line_id: value })}>
                  <SelectTrigger>
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
            <Button
              onClick={handleStartTraining}
              disabled={!formData.input_dir || startingTask}
              className="w-full md:w-auto"
            >
              {startingTask ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  {t('common.loading')}
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  {t('common.start')}
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        <div className="my-8">
          <Separator />
        </div>

        {/* Training Tasks List */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>{t('training.title')}</CardTitle>
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
              <div className="flex items-center justify-center py-8">
                <Loader2 className="w-8 h-8 animate-spin" />
              </div>
            ) : tasks.length === 0 ? (
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>{t('form.no_training_tasks')}</AlertTitle>
                <AlertDescription>
                  {t('form.no_tasks_description')}
                </AlertDescription>
              </Alert>
            ) : (
              <div className="space-y-4">
                {tasks.map((task) => (
                  <Card key={task.task_id}>
                    <CardContent className="p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(task.status)}
                        <span className="font-medium">{task.task_id}</span>
                        <Badge className={getStatusColor(task.status)}>
                          {t(`training.${task.status}`)}
                        </Badge>
                      </div>
                      <div className="flex items-center space-x-2">
                        {task.status === 'pending_orientation' && (
                          <Button 
                            size="sm" 
                            variant="outline"
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
                    
                    {task.current_step && (
                      <p className="text-sm text-muted-foreground bg-muted/50 p-2 rounded-md">{task.current_step}</p>
                    )}
                    
                    {task.progress !== undefined && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>{t('training.progress')}</span>
                          <span>{Math.round(task.progress * 100)}%</span>
                        </div>
                        <Progress value={task.progress * 100} />
                      </div>
                    )}
                    
                    {task.error_message && (
                      <Alert variant="destructive">
                        <XCircle className="h-4 w-4" />
                        <AlertTitle>{t('form.error_title')}</AlertTitle>
                        <AlertDescription>{task.error_message}</AlertDescription>
                      </Alert>
                    )}
                    
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">{t('training.created_at')}: {task.start_time ? new Date(task.start_time).toLocaleString() : 'N/A'}</span>
                      {task.end_time && (
                        <span className="text-muted-foreground">{t('training.completed_at')}: {new Date(task.end_time).toLocaleString()}</span>
                      )}
                    </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}