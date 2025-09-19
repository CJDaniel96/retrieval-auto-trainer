"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Database } from "lucide-react";

interface DownloadPanelProps {
  // 這個組件將在後續重構中實現完整功能
  // 目前作為佔位符
}

export function DownloadPanel(props: DownloadPanelProps) {
  return (
    <Card className="bg-white/80 backdrop-blur-sm shadow-xl border border-white/20">
      <CardHeader>
        <div className="flex items-center space-x-2">
          <Database className="w-6 h-6 text-blue-600" />
          <CardTitle className="text-2xl">資料下載</CardTitle>
        </div>
        <CardDescription>從資料庫下載訓練用的原始影像資料</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="text-center py-8 text-gray-500">
          資料下載功能正在重構中...
        </div>
      </CardContent>
    </Card>
  );
}