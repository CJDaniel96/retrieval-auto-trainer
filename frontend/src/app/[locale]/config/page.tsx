'use client';

import { useTranslations } from 'next-intl';
import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Settings,
  Save, 
  RefreshCw,
  Loader2,
  CheckCircle,
  AlertCircle,
  Brain,
  Database,
  Dumbbell,
  Target
} from 'lucide-react';
import { ApiClient } from '@/lib/api-client';
import { FullConfig, ConfigUpdateRequest } from '@/lib/types';
import { LanguageSwitcher } from '@/components/language-switcher';
import { Navigation } from '@/components/navigation';
import { toast } from 'sonner';

export default function ConfigPage() {
  const t = useTranslations();
  const [config, setConfig] = useState<FullConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [formData, setFormData] = useState<ConfigUpdateRequest>({});

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    setLoading(true);
    const response = await ApiClient.getCurrentConfig();
    if (response.data) {
      setConfig(response.data);
      // Initialize form with current config
      setFormData({
        training: { ...response.data.training },
        model: { ...response.data.model },
        data: { ...response.data.data },
        loss: { ...response.data.loss }
      });
    } else if (response.error) {
      toast.error(`載入配置失敗: ${response.error}`);
    }
    setLoading(false);
  };

  const handleSave = async () => {
    setSaving(true);
    const response = await ApiClient.updateConfig(formData);
    if (response.data) {
      toast.success('配置已成功更新！');
      await loadConfig();
    } else if (response.error) {
      toast.error(`更新失敗: ${response.error}`);
    }
    setSaving(false);
  };

  const updateTrainingField = (field: keyof NonNullable<ConfigUpdateRequest['training']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      training: {
        ...prev.training,
        [field]: value
      }
    }));
  };

  const updateModelField = (field: keyof NonNullable<ConfigUpdateRequest['model']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      model: {
        ...prev.model,
        [field]: value
      }
    }));
  };

  const updateDataField = (field: keyof NonNullable<ConfigUpdateRequest['data']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      data: {
        ...prev.data,
        [field]: value
      }
    }));
  };

  const updateLossField = (field: keyof NonNullable<ConfigUpdateRequest['loss']>, value: any) => {
    setFormData(prev => ({
      ...prev,
      loss: {
        ...prev.loss,
        [field]: value
      }
    }));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background p-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div className="flex items-center space-x-8">
            <div className="flex items-center space-x-3">
              <Settings className="w-8 h-8" />
              <h1 className="text-3xl font-bold tracking-tight">訓練配置管理</h1>
            </div>
            <Navigation />
          </div>
          <LanguageSwitcher />
        </div>

        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            在這裡修改的配置會影響所有新的訓練任務。修改後請記得保存配置。
          </AlertDescription>
        </Alert>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Training Configuration */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Dumbbell className="w-5 h-5" />
                <CardTitle>訓練配置</CardTitle>
              </div>
              <CardDescription>
                調整訓練相關的參數設定
              </CardDescription>
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
                    value={formData.training?.max_epochs || ''}
                    onChange={(e) => updateTrainingField('max_epochs', parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="batch_size">批次大小</Label>
                  <Input
                    id="batch_size"
                    type="number"
                    min={1}
                    max={256}
                    value={formData.training?.batch_size || ''}
                    onChange={(e) => updateTrainingField('batch_size', parseInt(e.target.value) || 0)}
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
                    value={formData.training?.lr || ''}
                    onChange={(e) => updateTrainingField('lr', parseFloat(e.target.value) || 0)}
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
                    value={formData.training?.weight_decay || ''}
                    onChange={(e) => updateTrainingField('weight_decay', parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="patience">EarlyStopping耐心值</Label>
                  <Input
                    id="patience"
                    type="number"
                    min={1}
                    max={50}
                    value={formData.training?.patience || ''}
                    onChange={(e) => updateTrainingField('patience', parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="freeze_epochs">凍結骨幹網路輪數</Label>
                  <Input
                    id="freeze_epochs"
                    type="number"
                    min={0}
                    max={20}
                    value={formData.training?.freeze_backbone_epochs || ''}
                    onChange={(e) => updateTrainingField('freeze_backbone_epochs', parseInt(e.target.value) || 0)}
                  />
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="early_stopping"
                  checked={formData.training?.enable_early_stopping ?? true}
                  onCheckedChange={(checked) => updateTrainingField('enable_early_stopping', checked)}
                />
                <Label htmlFor="early_stopping">啟用提前停止</Label>
              </div>
            </CardContent>
          </Card>

          {/* Model Configuration */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Brain className="w-5 h-5" />
                <CardTitle>模型配置</CardTitle>
              </div>
              <CardDescription>
                調整模型架構和相關參數
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="structure">模型結構</Label>
                <Select
                  value={formData.model?.structure || ''}
                  onValueChange={(value) => updateModelField('structure', value as 'HOAM' | 'HOAMV2')}
                >
                  <SelectTrigger>
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
                  value={formData.model?.backbone || ''}
                  onValueChange={(value) => updateModelField('backbone', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="選擇骨幹網路" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="efficientnetv2_rw_s">EfficientNetV2-S</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="embedding_size">嵌入向量維度</Label>
                <Input
                  id="embedding_size"
                  type="number"
                  min={64}
                  max={2048}
                  step={64}
                  value={formData.model?.embedding_size || ''}
                  onChange={(e) => updateModelField('embedding_size', parseInt(e.target.value) || 0)}
                />
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="pretrained"
                  checked={formData.model?.pretrained ?? false}
                  onCheckedChange={(checked) => updateModelField('pretrained', checked)}
                />
                <Label htmlFor="pretrained">使用預訓練權重</Label>
              </div>
            </CardContent>
          </Card>

          {/* Data Configuration */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Database className="w-5 h-5" />
                <CardTitle>數據配置</CardTitle>
              </div>
              <CardDescription>
                調整數據處理相關參數
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="image_size">圖片大小</Label>
                <Select
                  value={formData.data?.image_size?.toString() || ''}
                  onValueChange={(value) => updateDataField('image_size', parseInt(value))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="選擇圖片大小" />
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
                <Label htmlFor="num_workers">數據加載線程數</Label>
                <Input
                  id="num_workers"
                  type="number"
                  min={0}
                  max={16}
                  value={formData.data?.num_workers || ''}
                  onChange={(e) => updateDataField('num_workers', parseInt(e.target.value) || 0)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="test_split">驗證集比例</Label>
                <Input
                  id="test_split"
                  type="number"
                  step="0.05"
                  min={0.1}
                  max={0.5}
                  value={formData.data?.test_split || ''}
                  onChange={(e) => updateDataField('test_split', parseFloat(e.target.value) || 0)}
                />
              </div>
            </CardContent>
          </Card>

          {/* Loss Configuration */}
          <Card>
            <CardHeader>
              <div className="flex items-center space-x-2">
                <Target className="w-5 h-5" />
                <CardTitle>損失函數配置</CardTitle>
              </div>
              <CardDescription>
                調整損失函數相關參數
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="loss_type">損失函數類型</Label>
                <Select
                  value={formData.loss?.type || ''}
                  onValueChange={(value) => updateLossField('type', value as 'HybridMarginLoss' | 'ArcFaceLoss' | 'SubCenterArcFaceLoss')}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="選擇損失函數" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="HybridMarginLoss">HybridMarginLoss</SelectItem>
                    <SelectItem value="ArcFaceLoss">ArcFaceLoss</SelectItem>
                    <SelectItem value="SubCenterArcFaceLoss">SubCenterArcFaceLoss</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="subcenter_margin">子中心邊界</Label>
                  <Input
                    id="subcenter_margin"
                    type="number"
                    step="0.1"
                    min={0.1}
                    max={1.0}
                    value={formData.loss?.subcenter_margin || ''}
                    onChange={(e) => updateLossField('subcenter_margin', parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="subcenter_scale">子中心縮放</Label>
                  <Input
                    id="subcenter_scale"
                    type="number"
                    step="1"
                    min={1}
                    max={100}
                    value={formData.loss?.subcenter_scale || ''}
                    onChange={(e) => updateLossField('subcenter_scale', parseFloat(e.target.value) || 0)}
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        <Separator className="my-8" />

        {/* Action Buttons and Config Display */}
        <div className="space-y-6">
          <div className="flex justify-between items-center">
            <div className="flex space-x-4">
              <Button onClick={handleSave} disabled={saving}>
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
              <Button variant="outline" onClick={loadConfig} disabled={loading}>
                <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                重新載入
              </Button>
            </div>
          </div>

          {/* Current Config Display */}
          {config && (
            <Card>
              <CardHeader>
                <CardTitle>當前完整配置</CardTitle>
                <CardDescription>
                  以下是系統當前的完整配置信息
                </CardDescription>
              </CardHeader>
              <CardContent>
                <pre className="bg-muted p-4 rounded-md overflow-auto text-sm">
                  {JSON.stringify(config, null, 2)}
                </pre>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}