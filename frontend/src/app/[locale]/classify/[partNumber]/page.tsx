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
  Save,
  Trash2
} from 'lucide-react';
import { ApiClient } from '@/lib/api-client';
import { PartImageList, ImageInfo } from '@/lib/types';
import { toast } from 'sonner';

const ITEMS_PER_PAGE = 50;

export default function ClassifyPage() {
  const t = useTranslations();
  const router = useRouter();
  const params = useParams();
  const partNumber = params.partNumber as string;

  const [loading, setLoading] = useState(true);
  const [partImages, setPartImages] = useState<PartImageList | null>(null);
  const [imageClassifications, setImageClassifications] = useState<Record<string, 'OK' | 'NG'>>({});
  const [classifyingImages, setClassifyingImages] = useState(false);
  const [deletedImages, setDeletedImages] = useState<Set<string>>(new Set());
  const [deletingImages, setDeletingImages] = useState<Set<string>>(new Set());

  // 分頁狀態
  const [currentPage, setCurrentPage] = useState(1);
  // 過濾掉已刪除的影像
  const availableImages = partImages?.images.filter(img => !deletedImages.has(img.filename)) || [];
  const totalPages = Math.ceil(availableImages.length / ITEMS_PER_PAGE);
  const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
  const endIndex = Math.min(startIndex + ITEMS_PER_PAGE, availableImages.length);
  // 客戶端分頁: 只顯示當前頁面的影像
  const currentImages = availableImages.slice(startIndex, endIndex);

  // 進度統計
  const classifiedCount = Object.keys(imageClassifications).length;
  const totalCount = availableImages.length;
  const progressPercentage = totalCount > 0 ? (classifiedCount / totalCount) * 100 : 0;

  useEffect(() => {
    const loadImages = async () => {
      setLoading(true);
      try {
        // 載入所有影像資料，但只有第一頁的50張有base64數據
        const result = await ApiClient.listPartImages(partNumber, 1, 10000);
        if (result.data) {
          // 對於沒有base64_data的影像，我們需要逐批次載入
          const imagesWithBase64 = result.data.images.filter(img => img.base64_data);
          const imagesWithoutBase64 = result.data.images.filter(img => !img.base64_data);

          console.log(`載入了 ${imagesWithBase64.length} 張有base64的影像，${imagesWithoutBase64.length} 張需要額外載入`);

          // 先設置已有base64的影像
          setPartImages({
            ...result.data,
            images: result.data.images // 保持所有影像，但某些沒有base64_data
          });

          // 後續可以考慮分批載入其他影像的base64數據
        } else {
          toast.error(`${t('classify.messages.load_failed')}: ${result.error}`);
        }
      } catch (error) {
        toast.error(`${t('classify.messages.load_failed')}: ${error}`);
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

  const handleDeleteImage = async (filename: string) => {
    const confirmed = window.confirm(t('classify.delete.confirm', { filename }));
    if (!confirmed) return;

    setDeletingImages(prev => new Set(prev).add(filename));
    try {
      const result = await ApiClient.deleteImage(partNumber, filename);
      if (result.data) {
        setDeletedImages(prev => new Set(prev).add(filename));
        // 移除已刪除影像的分類記錄
        setImageClassifications(prev => {
          const updated = { ...prev };
          delete updated[filename];
          return updated;
        });
        toast.success(t('classify.delete.success'));
      } else {
        toast.error(`${t('classify.delete.failed')}: ${result.error}`);
      }
    } catch (error) {
      toast.error(`${t('classify.delete.failed')}: ${error}`);
    } finally {
      setDeletingImages(prev => {
        const updated = new Set(prev);
        updated.delete(filename);
        return updated;
      });
    }
  };

  const handleSubmit = async () => {
    if (classifiedCount === 0) {
      toast.error(t('classify.messages.min_classification'));
      return;
    }

    if (classifiedCount < totalCount) {
      const confirmed = window.confirm(
        t('classify.messages.partial_classification', { classified: classifiedCount, total: totalCount })
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
        toast.success(`${result.data.message} - ${t('classify.messages.classification_success')}`);
        // 清空分類狀態，讓用戶可以繼續分類其他圖片
        setImageClassifications({});
      } else {
        toast.error(`${t('classify.messages.classification_failed')}: ${result.error}`);
      }
    } catch (error) {
      toast.error(`${t('classify.messages.classification_failed')}: ${error}`);
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
          <p className="text-gray-600">{t('classify.messages.loading')}</p>
        </div>
      </div>
    );
  }

  if (!partImages) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Alert className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>{t('classify.messages.load_failed')}</AlertTitle>
          <AlertDescription>
            {t('classify.messages.no_images', { partNumber })}
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
              <span>{t('classify.messages.back_to_home')}</span>
            </Button>
            <div className="flex items-center space-x-2">
              <Target className="w-8 h-8 text-purple-600" />
              <h1 className="text-3xl font-bold text-gray-900">
                {t('classify.title')} - {partNumber}
              </h1>
            </div>
          </div>

          {/* 進度指示器 */}
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">
                {t('classify.progress.title')}: {classifiedCount} / {totalCount}
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
          <div className="space-y-4 mb-6">
            <div className="flex items-center justify-between bg-white rounded-lg p-4 shadow-sm">
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => goToPage(currentPage - 1)}
                  disabled={currentPage <= 1}
                >
                  <ChevronLeft className="w-4 h-4" />
                  {t('classify.pagination.previous')}
                </Button>
                <span className="text-sm text-gray-600">
                  {t('classify.pagination.page_info', { current: currentPage, total: totalPages })}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => goToPage(currentPage + 1)}
                  disabled={currentPage >= totalPages}
                >
                  {t('classify.pagination.next')}
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
              <div className="text-sm text-gray-500">
                {t('classify.pagination.item_range', { start: startIndex + 1, end: endIndex, total: totalCount })}
              </div>
            </div>

          </div>
        )}

        {/* 影像網格 */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {currentImages.map((image, index) => (
            <Card key={image.filename} className="overflow-hidden">
              <CardHeader className="pb-2">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center space-x-2 min-w-0 flex-1">
                    <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                      <span className="text-sm font-bold text-blue-600">
                        {startIndex + index + 1}
                      </span>
                    </div>
                    <div className="min-w-0 flex-1">
                      <CardTitle className="text-sm leading-tight break-all" title={image.filename}>
                        {image.filename}
                      </CardTitle>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDeleteImage(image.filename)}
                    disabled={deletingImages.has(image.filename)}
                    className="text-red-500 hover:text-red-700 hover:bg-red-50 h-8 w-8 p-0 flex-shrink-0"
                  >
                    {deletingImages.has(image.filename) ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Trash2 className="w-4 h-4" />
                    )}
                  </Button>
                </div>
                <CardDescription className="text-xs">
                  {t('classify.image_info.size')}: {(image.size / 1024).toFixed(1)} KB
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
                      <div className="text-center p-4">
                        <div className="w-16 h-16 bg-gray-300 rounded-lg mx-auto mb-3 flex items-center justify-center">
                          <svg className="w-8 h-8 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                          </svg>
                        </div>
                        <p className="text-xs text-gray-500 font-medium">{image.filename}</p>
                        <p className="text-xs text-gray-400 mt-1">
                          {(image.size / 1024).toFixed(1)} KB
                        </p>
                      </div>
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
                    {t('classify.classification.ok')}
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
                    {t('classify.classification.ng')}
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
              {t('classify.pagination.first')}
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
              {t('classify.pagination.last')}
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
                {t('classify.classification.classifying')}
              </>
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                {t('classify.classification.confirm_count', { count: classifiedCount })}
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}