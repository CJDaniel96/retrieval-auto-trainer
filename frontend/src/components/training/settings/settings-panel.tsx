"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, Settings } from "lucide-react";
import { ApiClient } from "@/lib/api-client";
import { toast } from "sonner";
import { SystemConfigTab } from "./system-config-tab";

export function SettingsPanel() {
  const [loading, setLoading] = useState(false);
  const [configs, setConfigs] = useState<any>({});
  const [isClient, setIsClient] = useState(false);

  // 載入配置
  useEffect(() => {
    setIsClient(true);
    loadConfigs();
  }, []);

  const loadConfigs = async () => {
    try {
      setLoading(true);
      const result = await ApiClient.getCurrentConfig();
      if (result.data) {
        setConfigs(result.data);
      }
    } catch (error) {
      toast.error("載入配置失敗");
    } finally {
      setLoading(false);
    }
  };

  const handleSystemConfigUpdate = async (updates: any) => {
    try {
      setLoading(true);
      const result = await ApiClient.updateSystemConfig({ updates });
      if (result.data) {
        setConfigs({ ...configs, system: result.data });
        toast.success("系統配置更新成功");
      }
    } catch (error) {
      toast.error("系統配置更新失敗");
    } finally {
      setLoading(false);
    }
  };

  if (!isClient || (loading && Object.keys(configs).length === 0)) {
    return (
      <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
        <CardContent className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin mr-2" />
          載入設定中...
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Settings className="w-6 h-6 text-blue-600" />
            <CardTitle className="text-2xl">系統設定</CardTitle>
          </div>
          <CardDescription>管理系統相關配置和偏好設定</CardDescription>
        </CardHeader>
        <CardContent>
          {/* 直接顯示系統配置，不需要分頁 */}
          <div className="space-y-4 mt-6">
            <SystemConfigTab
              config={configs.system || {}}
              onUpdate={handleSystemConfigUpdate}
              loading={loading}
            />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}