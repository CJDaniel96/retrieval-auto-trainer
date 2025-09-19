"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { motion } from "framer-motion";
import {
  Play,
  Cog,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Brain,
  Dumbbell,
  Target
} from "lucide-react";
import { TrainingFormData, PartInfo } from "../shared/types";
import { useTranslations } from "next-intl";

interface TrainingFormProps {
  formData: TrainingFormData;
  onFormDataChange: (data: TrainingFormData) => void;
  onSubmit: () => void;
  downloadedParts: PartInfo[];
  useExistingData: boolean;
  setUseExistingData: (value: boolean) => void;
  selectedRawdataPart: string;
  setSelectedRawdataPart: (value: string) => void;
  startingTask: boolean;
  showAdvancedConfig: boolean;
  setShowAdvancedConfig: (value: boolean) => void;
  trainingConfig: any;
  setTrainingConfig: (config: any) => void;
}

export function TrainingForm({
  formData,
  onFormDataChange,
  onSubmit,
  downloadedParts,
  useExistingData,
  setUseExistingData,
  selectedRawdataPart,
  setSelectedRawdataPart,
  startingTask,
  showAdvancedConfig,
  setShowAdvancedConfig,
  trainingConfig,
  setTrainingConfig
}: TrainingFormProps) {
  const t = useTranslations();

  const handleInputChange = (field: keyof TrainingFormData, value: string | boolean) => {
    onFormDataChange({
      ...formData,
      [field]: value
    });
  };

  return (
    <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
      <CardHeader>
        <div className="flex items-center space-x-2">
          <Brain className="w-6 h-6 text-blue-600" />
          <CardTitle className="text-2xl">{t("form.title")}</CardTitle>
        </div>
        <CardDescription>
          {t("training_form.description")}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Data Source Selection */}
        <div className="space-y-4">
          <div className="flex items-center space-x-2">
            <Switch
              id="use-existing-data"
              checked={useExistingData}
              onCheckedChange={setUseExistingData}
            />
            <Label htmlFor="use-existing-data">{t("training_form.use_existing_data")}</Label>
          </div>

          {useExistingData ? (
            <div className="space-y-2">
              <Label htmlFor="rawdata_part">{t("download.info.select_part")}</Label>
              <Select
                value={selectedRawdataPart}
                onValueChange={(value) => {
                  setSelectedRawdataPart(value);
                  handleInputChange("input_dir", `rawdata/${value}`);
                }}
              >
                <SelectTrigger className="bg-white/70">
                  <SelectValue placeholder={t("download.info.select_part")} />
                </SelectTrigger>
                <SelectContent>
                  {Array.isArray(downloadedParts) && downloadedParts.map((part) => (
                    <SelectItem key={part.part_number} value={part.part_number}>
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
          ) : (
            <div className="space-y-2">
              <Label htmlFor="input_dir">{t("form.input_label")}</Label>
              <Input
                id="input_dir"
                value={formData.input_dir}
                onChange={(e) => handleInputChange("input_dir", e.target.value)}
                placeholder={t("form.input_placeholder")}
                className="bg-white/70"
              />
            </div>
          )}
        </div>

        {/* Basic Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="site">{t("form.site_label")}</Label>
            <Select value={formData.site} onValueChange={(value) => handleInputChange("site", value)}>
              <SelectTrigger className="bg-white/70">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="HPH">HPH</SelectItem>
                <SelectItem value="JQ">JQ</SelectItem>
                <SelectItem value="ZJ">ZJ</SelectItem>
                <SelectItem value="NK">NK</SelectItem>
                <SelectItem value="HZ">HZ</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="line_id">{t("form.line_label")}</Label>
            <Input
              id="line_id"
              value={formData.line_id}
              onChange={(e) => handleInputChange("line_id", e.target.value)}
              placeholder={t("form.line_placeholder")}
              className="bg-white/70"
            />
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="output_dir">{t("form.output_label")}</Label>
          <Input
            id="output_dir"
            value={formData.output_dir}
            onChange={(e) => handleInputChange("output_dir", e.target.value)}
            placeholder={t("form.output_placeholder")}
            className="bg-white/70"
          />
        </div>


        {/* Advanced Configuration */}
        <Collapsible open={showAdvancedConfig} onOpenChange={setShowAdvancedConfig}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" className="w-full justify-between">
              <div className="flex items-center space-x-2">
                <Cog className="w-4 h-4" />
                <span>{t("training_form.advanced_settings")}</span>
              </div>
              {showAdvancedConfig ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="space-y-4">
            <Alert className="border-orange-200 bg-orange-50/80">
              <AlertCircle className="h-4 w-4 text-orange-600" />
              <AlertDescription className="text-orange-800">
                {t("training_form.advanced_info")}
              </AlertDescription>
            </Alert>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* Training Parameters */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Dumbbell className="w-4 h-4 text-purple-600" />
                  <h4 className="font-medium">{t("training_form.training_params")}</h4>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="max_epochs">{t("training_form.max_epochs")}</Label>
                  <Input
                    id="max_epochs"
                    type="number"
                    value={trainingConfig.training?.max_epochs || 50}
                    onChange={(e) => setTrainingConfig({
                      ...trainingConfig,
                      training: { ...trainingConfig.training, max_epochs: parseInt(e.target.value) }
                    })}
                    className="bg-white/70"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="batch_size">{t("training_form.batch_size")}</Label>
                  <Input
                    id="batch_size"
                    type="number"
                    value={trainingConfig.training?.batch_size || 8}
                    onChange={(e) => setTrainingConfig({
                      ...trainingConfig,
                      training: { ...trainingConfig.training, batch_size: parseInt(e.target.value) }
                    })}
                    className="bg-white/70"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="learning_rate">{t("training_form.learning_rate")}</Label>
                  <Input
                    id="learning_rate"
                    type="number"
                    step="0.0001"
                    value={trainingConfig.training?.lr || 0.0003}
                    onChange={(e) => setTrainingConfig({
                      ...trainingConfig,
                      training: { ...trainingConfig.training, lr: parseFloat(e.target.value) }
                    })}
                    className="bg-white/70"
                  />
                </div>
              </div>

              {/* Model Parameters */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Target className="w-4 h-4 text-green-600" />
                  <h4 className="font-medium">{t("training_form.model_params")}</h4>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="backbone">{t("training_form.backbone")}</Label>
                  <Select
                    value={trainingConfig.model?.backbone || "efficientnetv2_rw_s"}
                    onValueChange={(value) => setTrainingConfig({
                      ...trainingConfig,
                      model: { ...trainingConfig.model, backbone: value }
                    })}
                  >
                    <SelectTrigger className="bg-white/70">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="efficientnetv2_rw_s">EfficientNetV2-S</SelectItem>
                      <SelectItem value="efficientnetv2_rw_m">EfficientNetV2-M</SelectItem>
                      <SelectItem value="efficientnetv2_rw_l">EfficientNetV2-L</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="embedding_size">{t("training_form.embedding_size")}</Label>
                  <Input
                    id="embedding_size"
                    type="number"
                    value={trainingConfig.model?.embedding_size || 512}
                    onChange={(e) => setTrainingConfig({
                      ...trainingConfig,
                      model: { ...trainingConfig.model, embedding_size: parseInt(e.target.value) }
                    })}
                    className="bg-white/70"
                  />
                </div>
              </div>

              {/* Data Parameters */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <Target className="w-4 h-4 text-blue-600" />
                  <h4 className="font-medium">{t("training_form.data_params")}</h4>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="image_size">{t("training_form.image_size")}</Label>
                  <Input
                    id="image_size"
                    type="number"
                    value={trainingConfig.data?.image_size || 224}
                    onChange={(e) => setTrainingConfig({
                      ...trainingConfig,
                      data: { ...trainingConfig.data, image_size: parseInt(e.target.value) }
                    })}
                    className="bg-white/70"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="test_split">{t("training_form.test_split")}</Label>
                  <Input
                    id="test_split"
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="0.5"
                    value={trainingConfig.data?.test_split || 0.2}
                    onChange={(e) => setTrainingConfig({
                      ...trainingConfig,
                      data: { ...trainingConfig.data, test_split: parseFloat(e.target.value) }
                    })}
                    className="bg-white/70"
                  />
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* Submit Button */}
        <div className="flex justify-end">
          <Button
            onClick={onSubmit}
            disabled={startingTask || !formData.input_dir || !formData.output_dir}
            size="lg"
            className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
          >
            <Play className="w-4 h-4 mr-2" />
            {startingTask ? t("task_list.starting") : t("form.submit")}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}