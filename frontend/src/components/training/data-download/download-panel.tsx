"use client";

import { useState, useEffect } from "react";
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
              toast.success(`下載完成！共下載 ${response.data.downloaded_count} 張圖片`);
            } else {
              toast.error(`下載失敗：${response.data.error_message}`);
            }
          }
        }
      } catch (error) {
        console.error('獲取下載狀態失敗:', error);
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
    if (!formData.site) return "請選擇工廠";
    if (!formData.line_id) return "請輸入產線ID";
    if (!formData.start_date) return "請選擇開始日期";
    if (!formData.end_date) return "請選擇結束日期";
    if (!formData.part_number) return "請輸入料號";

    if (new Date(formData.start_date) > new Date(formData.end_date)) {
      return "開始日期不能晚於結束日期";
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
        toast.success("預估完成");
      } else if (response.error) {
        toast.error(response.error);
      }
    } catch (error) {
      toast.error("預估失敗");
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
        toast.success("下載任務已開始");
      } else if (response.error) {
        toast.error(response.error);
        setIsDownloading(false);
      }
    } catch (error) {
      toast.error("啟動下載失敗");
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
            <CardTitle className="text-2xl">資料下載</CardTitle>
          </div>
          <CardDescription>從資料庫下載訓練用的原始影像資料</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Site and Line Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="site">工廠</Label>
              <Select value={formData.site} onValueChange={(value) => handleInputChange('site', value)}>
                <SelectTrigger>
                  <SelectValue placeholder="選擇工廠" />
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
              <Label htmlFor="line_id">產線ID</Label>
              <Input
                id="line_id"
                placeholder="例如: V31"
                value={formData.line_id}
                onChange={(e) => handleInputChange('line_id', e.target.value)}
              />
            </div>
          </div>

          {/* Date Range */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="start_date">開始日期</Label>
              <Input
                id="start_date"
                type="date"
                value={formData.start_date}
                onChange={(e) => handleInputChange('start_date', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="end_date">結束日期</Label>
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
              <Label htmlFor="part_number">料號</Label>
              <Input
                id="part_number"
                placeholder="輸入料號"
                value={formData.part_number}
                onChange={(e) => handleInputChange('part_number', e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="limit">數量限制 (選填)</Label>
              <Input
                id="limit"
                type="number"
                placeholder="最多下載張數"
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
              <span>預估數量</span>
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
              <span>開始下載</span>
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
              <CardTitle className="text-lg">預估結果</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{estimate.estimated_count.toLocaleString()}</div>
                <div className="text-sm text-gray-600">預估圖片數</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{estimate.estimated_size_mb.toFixed(1)} MB</div>
                <div className="text-sm text-gray-600">預估大小</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-medium text-purple-600">{estimate.site}</div>
                <div className="text-sm text-gray-600">工廠</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-medium text-orange-600">{estimate.line_id}</div>
                <div className="text-sm text-gray-600">產線</div>
              </div>
            </div>
            <Separator className="my-4" />
            <div className="text-sm text-gray-600">
              <div className="flex items-center space-x-2">
                <Calendar className="w-4 h-4" />
                <span>時間範圍: {estimate.time_range}</span>
              </div>
              <div className="flex items-center space-x-2 mt-1">
                <Package className="w-4 h-4" />
                <span>料號: {estimate.part_number}</span>
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
                <CardTitle className="text-lg">下載進度</CardTitle>
              </div>
              <Badge variant={
                downloadStatus.status === 'completed' ? 'default' :
                downloadStatus.status === 'failed' ? 'destructive' :
                downloadStatus.status === 'running' ? 'secondary' : 'outline'
              }>
                {downloadStatus.status === 'pending' && '等待中'}
                {downloadStatus.status === 'running' && '下載中'}
                {downloadStatus.status === 'completed' && '已完成'}
                {downloadStatus.status === 'failed' && '失敗'}
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>下載進度</span>
                <span>{downloadStatus.downloaded_count} / {downloadStatus.total_count}</span>
              </div>
              <Progress value={downloadStatus.progress} className="h-2" />
            </div>

            <div className="flex items-center space-x-4 text-sm text-gray-600">
              <div className="flex items-center space-x-1">
                <FileText className="w-4 h-4" />
                <span>任務ID: {downloadStatus.download_id}</span>
              </div>
              {downloadStatus.status === 'completed' && (
                <div className="flex items-center space-x-1 text-green-600">
                  <CheckCircle className="w-4 h-4" />
                  <span>下載完成</span>
                </div>
              )}
              {downloadStatus.status === 'failed' && (
                <div className="flex items-center space-x-1 text-red-600">
                  <AlertCircle className="w-4 h-4" />
                  <span>下載失敗</span>
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