'use client';

import { useTranslations } from 'next-intl';
import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { 
  ArrowUp,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  Loader2,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { ApiClient } from '@/lib/api-client';
import { OrientationSample } from '@/lib/types';
import Image from 'next/image';

type OrientationType = 'Up' | 'Down' | 'Left' | 'Right';

export default function OrientationPage() {
  const t = useTranslations();
  const params = useParams();
  const router = useRouter();
  const taskId = params.taskId as string;
  
  const [samples, setSamples] = useState<OrientationSample[]>([]);
  const [orientations, setOrientations] = useState<Record<string, OrientationType>>({});
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchOrientationSamples();
  }, [taskId]);

  const fetchOrientationSamples = async () => {
    try {
      const response = await ApiClient.getOrientationSamples(taskId);
      if (response.data) {
        setSamples(response.data);
      } else if (response.error) {
        setError(response.error);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleOrientationChange = (className: string, orientation: OrientationType) => {
    setOrientations(prev => ({
      ...prev,
      [className]: orientation
    }));
  };

  const handleSubmit = async () => {
    // Check if all orientations are selected
    const missingOrientations = samples.filter(sample => !orientations[sample.class_name]);
    if (missingOrientations.length > 0) {
      setError(t('form.missing_orientation'));
      return;
    }

    setSubmitting(true);
    try {
      const response = await ApiClient.confirmOrientations({
        task_id: taskId,
        orientations
      });
      
      if (response.data) {
        // Redirect back to home page
        router.push('/');
      } else if (response.error) {
        setError(response.error);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const getOrientationIcon = (orientation: OrientationType) => {
    switch (orientation) {
      case 'Up':
        return <ArrowUp className="w-4 h-4" />;
      case 'Down':
        return <ArrowDown className="w-4 h-4" />;
      case 'Left':
        return <ArrowLeft className="w-4 h-4" />;
      case 'Right':
        return <ArrowRight className="w-4 h-4" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="flex items-center space-x-2">
          <Loader2 className="w-8 h-8 animate-spin" />
          <span className="text-lg">{t('common.loading')}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight mb-2">{t('orientation.title')}</h1>
          <p className="text-muted-foreground">{t('orientation.description')}</p>
        </div>

        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>{t('form.error_title')}</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="space-y-8">
          {samples.map((sample) => (
            <Card key={sample.class_name}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>{t('orientation.class_name', { name: sample.class_name })}</span>
                  {orientations[sample.class_name] && (
                    <div className="flex items-center space-x-2 text-green-600">
                      <CheckCircle className="w-5 h-5" />
                      <span>{orientations[sample.class_name]}</span>
                    </div>
                  )}
                </CardTitle>
                <CardDescription>
                  {t('orientation.select_orientation')}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Sample Images */}
                <div className="mb-6">
                  <Label className="text-base font-medium mb-3 block">
                    {t('orientation.sample_images')}
                  </Label>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {sample.sample_images.map((imagePath, index) => (
                      <div key={index} className="border rounded-lg overflow-hidden bg-card shadow-sm hover:shadow-md transition-shadow">
                        <Image
                          src={`http://localhost:8000${imagePath}`}
                          alt={`Sample ${index + 1} for ${sample.class_name}`}
                          width={300}
                          height={200}
                          className="w-full h-48 object-cover"
                        />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Orientation Selection */}
                <div>
                  <Label className="text-base font-medium mb-3 block">
                    {t('orientation.select_orientation')}
                  </Label>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {(['Up', 'Down', 'Left', 'Right'] as OrientationType[]).map((orientation) => (
                      <Button
                        key={orientation}
                        variant={orientations[sample.class_name] === orientation ? "default" : "outline"}
                        className="flex items-center justify-center space-x-2 h-12"
                        onClick={() => handleOrientationChange(sample.class_name, orientation)}
                      >
                        {getOrientationIcon(orientation)}
                        <span>{t(`orientation.${orientation.toLowerCase()}`)}</span>
                      </Button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Submit Button */}
        <div className="mt-8 flex justify-center">
          <Button
            size="lg"
            onClick={handleSubmit}
            disabled={submitting || samples.length === 0}
            className="px-8"
          >
            {submitting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                {t('common.loading')}
              </>
            ) : (
              <>
                <CheckCircle className="w-4 h-4 mr-2" />
                {t('orientation.confirm_all')}
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}