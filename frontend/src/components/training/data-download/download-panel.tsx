"use client";

import { useState, useEffect } from "react";
import { useTranslations } from "next-intl";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Database,
  Download,
  Search,
  Calendar,
  Package,
  FileText,
  Clock,
  CheckCircle,
  AlertCircle,
  Loader2
} from "lucide-react";
import { toast } from "sonner";
import { ApiClient } from "@/lib/api-client";
import { DownloadRequest } from "@/lib/types";

interface DownloadPanelProps {
  // 可擴展的屬性接口
}

interface DownloadEstimate {
  estimated_count: number;
  estimated_size_mb: number;
  time_range: string;
  site: string;
  line_id: string;
  part_number: string;
}

interface DownloadStatus {
  download_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  downloaded_count: number;
  total_count: number;
  error_message?: string;
}

export function DownloadPanel(props: DownloadPanelProps) {
  const t = useTranslations();

  // Form state
  const [formData, setFormData] = useState<DownloadRequest>({
    site: "HPH",
    line_id: "",
    start_date: "",
    end_date: "",
    part_number: "",
    limit: undefined
  });

  // UI state
  const [isEstimating, setIsEstimating] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [estimate, setEstimate] = useState<DownloadEstimate | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<DownloadStatus | null>(null);
  const [activeDownloadId, setActiveDownloadId] = useState<string | null>(null);

  // Set default dates (last 7 days)
  useEffect(() => {
    const today = new Date();
    const lastWeek = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);

    setFormData(prev => ({
      ...prev,
      end_date: today.toISOString().split('T')[0],
      start_date: lastWeek.toISOString().split('T')[0]
    }));
  }, []);

  // Poll download status
  useEffect(() => {
    if (!activeDownloadId) return;

    const interval = setInterval(async () => {
      try {
        const response = await ApiClient.getDownloadStatus(activeDownloadId);
        if (response.data) {
          setDownloadStatus(response.data);

          if (response.data.status === 'completed' || response.data.status === 'failed') {
            setActiveDownloadId(null);
            setIsDownloading(false);

            if (response.data.status === 'completed') {
              toast.success(t("success.download_completed", { count: response.data.downloaded_count }));
            } else {
              toast.error(`${t("download_panel.download_failed")}：${response.data.error_message}`);
            }
          }
        }
      } catch (error) {
        console.error('Failed to get download status:', error);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [activeDownloadId]);

  const handleInputChange = (field: keyof DownloadRequest, value: string | number | undefined) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));

    // Reset estimate when form changes
    if (estimate) {
      setEstimate(null);
    }
  };

  const validateForm = (): string | null => {
    if (!formData.site) return t("errors.please_select_factory");
    if (!formData.line_id) return t("errors.please_enter_line_id");
    if (!formData.start_date) return t("errors.please_select_start_date");
    if (!formData.end_date) return t("errors.please_select_end_date");
    if (!formData.part_number) return t("errors.please_enter_part_number");

    if (new Date(formData.start_date) > new Date(formData.end_date)) {
      return t("errors.start_date_after_end");
    }

    return null;
  };

  const handleEstimate = async () => {
    const error = validateForm();
    if (error) {
      toast.error(error);
      return;
    }

    setIsEstimating(true);
    try {
      const response = await ApiClient.estimateDownload(formData);
      if (response.data) {
        setEstimate(response.data);
        toast.success(t("success.estimate_completed"));
      } else if (response.error) {
        toast.error(response.error);
      }
    } catch (error) {
      toast.error(t("errors.estimate_failed"));
    } finally {
      setIsEstimating(false);
    }
  };

  const handleDownload = async () => {
    const error = validateForm();
    if (error) {
      toast.error(error);
      return;
    }

    setIsDownloading(true);
    try {
      const response = await ApiClient.downloadRawData(formData);
      if (response.data) {
        setActiveDownloadId(response.data.download_id);
        setDownloadStatus({
          download_id: response.data.download_id,
          status: 'pending',
          progress: 0,
          downloaded_count: 0,
          total_count: estimate?.estimated_count || 0
        });
        toast.success(t("success.download_started"));
      } else if (response.error) {
        toast.error(response.error);
        setIsDownloading(false);
      }
    } catch (error) {
      toast.error(t("errors.download_start_failed"));
      setIsDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Main Download Form */}
      <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Database className="w-6 h-6 text-blue-600" />
            <CardTitle className="text-2xl">{t("download_panel.title")}</CardTitle>
          </div>
          <CardDescription>{t("download_panel.description")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Site and Line Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="site">{t("download_panel.factory_label")}</Label>
              <Select value={formData.site} onValueChange={(value) => handleInputChange('site', value)}>
                <SelectTrigger>
                  <SelectValue placeholder={t("download_panel.factory_placeholder")} />
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
              <Label htmlFor="line_id">{t("download_panel.line_id_label")}</Label>
              <Input
                id="line_id"
                placeholder={t("download_panel.line_id_placeholder")}
                value={formData.line_id}
                onChange={(e) => handleInputChange('line_id', e.target.value)}
              />
            </div>
          </div>

          {/* Date Range */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="start_date">{t("download_panel.start_date_label")}</Label>
              <Input
                id="start_date"
                type="date"
                value={formData.start_date}
                onChange={(e) => handleInputChange('start_date', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="end_date">{t("download_panel.end_date_label")}</Label>
              <Input
                id="end_date"
                type="date"
                value={formData.end_date}
                onChange={(e) => handleInputChange('end_date', e.target.value)}
              />
            </div>
          </div>

          {/* Part Number and Limit */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="part_number">{t("download_panel.part_number_label")}</Label>
              <Input
                id="part_number"
                placeholder={t("download_panel.part_number_placeholder")}
                value={formData.part_number}
                onChange={(e) => handleInputChange('part_number', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="limit">{t("download_panel.limit_label")}</Label>
              <Input
                id="limit"
                type="number"
                placeholder={t("download_panel.limit_placeholder")}
                min="1"
                max="10000"
                value={formData.limit || ''}
                onChange={(e) => handleInputChange('limit', e.target.value ? parseInt(e.target.value) : undefined)}
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-3">
            <Button
              onClick={handleEstimate}
              disabled={isEstimating || isDownloading}
              variant="outline"
              className="flex items-center space-x-2"
            >
              {isEstimating ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Search className="w-4 h-4" />
              )}
              <span>{t("download_panel.estimate_button")}</span>
            </Button>

            <Button
              onClick={handleDownload}
              disabled={isDownloading || !estimate}
              className="flex items-center space-x-2"
            >
              {isDownloading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Download className="w-4 h-4" />
              )}
              <span>{t("download_panel.download_button")}</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Estimate Results */}
      {estimate && (
        <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
          <CardHeader>
            <div className="flex items-center space-x-2">
              <FileText className="w-5 h-5 text-green-600" />
              <CardTitle className="text-lg">{t("download_panel.estimate_results_title")}</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{estimate.estimated_count.toLocaleString()}</div>
                <div className="text-sm text-gray-600">{t("download_panel.estimated_images")}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{estimate.estimated_size_mb.toFixed(1)} MB</div>
                <div className="text-sm text-gray-600">{t("download_panel.estimated_size")}</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-medium text-purple-600">{estimate.site}</div>
                <div className="text-sm text-gray-600">{t("download_panel.factory")}</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-medium text-orange-600">{estimate.line_id}</div>
                <div className="text-sm text-gray-600">{t("download_panel.production_line")}</div>
              </div>
            </div>
            <Separator className="my-4" />
            <div className="text-sm text-gray-600">
              <div className="flex items-center space-x-2">
                <Calendar className="w-4 h-4" />
                <span>{t("download_panel.time_range", { range: estimate.time_range })}</span>
              </div>
              <div className="flex items-center space-x-2 mt-1">
                <Package className="w-4 h-4" />
                <span>{t("download_panel.part_number_info", { partNumber: estimate.part_number })}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Download Progress */}
      {downloadStatus && (
        <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Clock className="w-5 h-5 text-blue-600" />
                <CardTitle className="text-lg">{t("download_panel.download_progress_title")}</CardTitle>
              </div>
              <Badge variant={
                downloadStatus.status === 'completed' ? 'default' :
                downloadStatus.status === 'failed' ? 'destructive' :
                downloadStatus.status === 'running' ? 'secondary' : 'outline'
              }>
                {downloadStatus.status === 'pending' && t("download_panel.status_waiting")}
                {downloadStatus.status === 'running' && t("download_panel.status_downloading")}
                {downloadStatus.status === 'completed' && t("download_panel.status_completed")}
                {downloadStatus.status === 'failed' && t("download_panel.status_failed")}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>{t("download_panel.progress_label")}</span>
                <span>{downloadStatus.downloaded_count} / {downloadStatus.total_count}</span>
              </div>
              <Progress value={downloadStatus.progress} className="h-2" />
            </div>

            <div className="flex items-center space-x-4 text-sm text-gray-600">
              <div className="flex items-center space-x-1">
                <FileText className="w-4 h-4" />
                <span>{t("download_panel.task_id", { id: downloadStatus.download_id })}</span>
              </div>
              {downloadStatus.status === 'completed' && (
                <div className="flex items-center space-x-1 text-green-600">
                  <CheckCircle className="w-4 h-4" />
                  <span>{t("download_panel.download_complete")}</span>
                </div>
              )}
              {downloadStatus.status === 'failed' && (
                <div className="flex items-center space-x-1 text-red-600">
                  <AlertCircle className="w-4 h-4" />
                  <span>{t("download_panel.download_failed")}</span>
                </div>
              )}
            </div>

            {downloadStatus.error_message && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-md">
                <div className="text-sm text-red-800">{downloadStatus.error_message}</div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}