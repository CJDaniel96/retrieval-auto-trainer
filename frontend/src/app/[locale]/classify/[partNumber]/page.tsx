'use client';

import { useTranslations } from 'next-intl';
import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import {
  Target,
  ArrowLeft,
  CheckCircle,
  AlertCircle,
  Loader2,
  ChevronLeft,
  ChevronRight,
  Save
} from 'lucide-react';
import { ApiClient } from '@/lib/api-client';
import { PartImageList, ImageInfo } from '@/lib/types';
import { toast } from 'sonner';

const ITEMS_PER_PAGE = 20;

export default function ClassifyPage() {
  const t = useTranslations();
  const router = useRouter();
  const params = useParams();
  const partNumber = params.partNumber as string;

  const [loading, setLoading] = useState(true);
  const [partImages, setPartImages] = useState<PartImageList | null>(null);
  const [imageClassifications, setImageClassifications] = useState<Record<string, 'OK' | 'NG'>>({});
  const [classifyingImages, setClassifyingImages] = useState(false);

  // 分頁狀態
  const [currentPage, setCurrentPage] = useState(1);
  const totalPages = partImages ? Math.ceil(partImages.images.length / ITEMS_PER_PAGE) : 0;
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = startIndex + ITEMS_PER_PAGE;
  const currentImages = partImages?.images.slice(startIndex, endIndex) || [];

  // 進度統計
  const classifiedCount = Object.keys(imageClassifications).length;
  const totalCount = partImages?.total_images || 0;
  const progressPercentage = totalCount > 0 ? (classifiedCount / totalCount) * 100 : 0;

  useEffect(() => {
    const loadImages = async () => {
      setLoading(true);
      try {
        const result = await ApiClient.listPartImages(partNumber);
        if (result.data) {
          setPartImages(result.data);
        } else {
          toast.error(`載入影像失敗: ${result.error}`);
        }
      } catch (error) {
        toast.error(`載入影像失敗: ${error}`);
      } finally {
        setLoading(false);
      }
    };

    if (partNumber) {
      loadImages();
    }
  }, [partNumber]);

  const handleClassifyImage = (filename: string, classification: 'OK' | 'NG') => {
    setImageClassifications(prev => ({
      ...prev,
      [filename]: classification
    }));
  };

  const handleSubmit = async () => {
    if (classifiedCount === 0) {
      toast.error('請至少分類一張影像');
      return;
    }

    if (classifiedCount < totalCount) {
      const confirmed = window.confirm(
        `您只分類了 ${classifiedCount}/${totalCount} 張影像。未分類的影像將被忽略。確定要繼續嗎？`
      );
      if (!confirmed) return;
    }

    setClassifyingImages(true);
    try {
      const result = await ApiClient.classifyImages(partNumber, {
        part_number: partNumber,
        classifications: imageClassifications
      });

      if (result.data) {
        toast.success(result.data.message);
        router.push('/zh'); // 返回主頁面
      } else {
        toast.error(`分類失敗: ${result.error}`);
      }
    } catch (error) {
      toast.error(`分類失敗: ${error}`);
    } finally {
      setClassifyingImages(false);
    }
  };

  const goToPage = (page: number) => {
    if (page >= 1 && page <= totalPages) {
      setCurrentPage(page);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p className="text-gray-600">載入影像中...</p>
        </div>
      </div>
    );
  }

  if (!partImages) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Alert className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>載入失敗</AlertTitle>
          <AlertDescription>
            無法載入料號 {partNumber} 的影像資料
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* 頁面標題 */}
        <div className="mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <Button
              variant="outline"
              onClick={() => router.push('/zh')}
              className="flex items-center space-x-2"
            >
              <ArrowLeft className="w-4 h-4" />
              <span>返回</span>
            </Button>
            <div className="flex items-center space-x-2">
              <Target className="w-8 h-8 text-purple-600" />
              <h1 className="text-3xl font-bold text-gray-900">
                影像分類 - {partNumber}
              </h1>
            </div>
          </div>

          {/* 進度指示器 */}
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">
                分類進度: {classifiedCount} / {totalCount}
              </span>
              <span className="text-sm text-gray-500">
                {progressPercentage.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progressPercentage}%` }}
              />
            </div>
          </div>
        </div>

        {/* 分頁控制 - 頂部 */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between mb-6 bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => goToPage(currentPage - 1)}
                disabled={currentPage <= 1}
              >
                <ChevronLeft className="w-4 h-4" />
                上一頁
              </Button>
              <span className="text-sm text-gray-600">
                第 {currentPage} 頁，共 {totalPages} 頁
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => goToPage(currentPage + 1)}
                disabled={currentPage >= totalPages}
              >
                下一頁
                <ChevronRight className="w-4 h-4" />
              </Button>
            </div>
            <div className="text-sm text-gray-500">
              顯示 {startIndex + 1}-{Math.min(endIndex, totalCount)} 項，共 {totalCount} 項
            </div>
          </div>
        )}

        {/* 影像網格 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {currentImages.map((image, index) => (
            <Card key={image.filename} className="overflow-hidden">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                      <span className="text-sm font-bold text-blue-600">
                        {startIndex + index + 1}
                      </span>
                    </div>
                    <CardTitle className="text-sm truncate">
                      {image.filename}
                    </CardTitle>
                  </div>
                </div>
                <CardDescription className="text-xs">
                  大小: {(image.size / 1024).toFixed(1)} KB
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                {/* 影像預覽 */}
                <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
                  {image.base64_data ? (
                    <img
                      src={image.base64_data}
                      alt={image.filename}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full bg-gray-200 flex items-center justify-center">
                      <span className="text-gray-500 text-sm">載入中...</span>
                    </div>
                  )}
                </div>

                {/* 分類按鈕 */}
                <div className="flex space-x-2">
                  <Button
                    variant={imageClassifications[image.filename] === 'OK' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => handleClassifyImage(image.filename, 'OK')}
                    className={`flex-1 ${
                      imageClassifications[image.filename] === 'OK'
                        ? 'bg-green-600 hover:bg-green-700'
                        : 'hover:bg-green-50 hover:text-green-600 hover:border-green-600'
                    }`}
                  >
                    OK
                  </Button>
                  <Button
                    variant={imageClassifications[image.filename] === 'NG' ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => handleClassifyImage(image.filename, 'NG')}
                    className={`flex-1 ${
                      imageClassifications[image.filename] === 'NG'
                        ? 'bg-red-600 hover:bg-red-700'
                        : 'hover:bg-red-50 hover:text-red-600 hover:border-red-600'
                    }`}
                  >
                    NG
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* 分頁控制 - 底部 */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center mt-8 space-x-2">
            <Button
              variant="outline"
              onClick={() => goToPage(1)}
              disabled={currentPage <= 1}
            >
              首頁
            </Button>
            <Button
              variant="outline"
              onClick={() => goToPage(currentPage - 1)}
              disabled={currentPage <= 1}
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>

            {/* 頁碼按鈕 */}
            {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
              const pageNum = Math.max(1, Math.min(totalPages - 4, currentPage - 2)) + i;
              if (pageNum > totalPages) return null;

              return (
                <Button
                  key={pageNum}
                  variant={currentPage === pageNum ? 'default' : 'outline'}
                  onClick={() => goToPage(pageNum)}
                  className="w-10"
                >
                  {pageNum}
                </Button>
              );
            })}

            <Button
              variant="outline"
              onClick={() => goToPage(currentPage + 1)}
              disabled={currentPage >= totalPages}
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
            <Button
              variant="outline"
              onClick={() => goToPage(totalPages)}
              disabled={currentPage >= totalPages}
            >
              末頁
            </Button>
          </div>
        )}

        {/* 提交按鈕 */}
        <div className="fixed bottom-6 right-6">
          <Button
            onClick={handleSubmit}
            disabled={classifyingImages || classifiedCount === 0}
            className="bg-purple-600 hover:bg-purple-700 text-white shadow-lg"
            size="lg"
          >
            {classifyingImages ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                分類中...
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                確認分類 ({classifiedCount})
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}