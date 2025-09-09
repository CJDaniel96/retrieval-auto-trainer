'use client';

import { useLocale } from 'next-intl';
import { useRouter, usePathname } from '@/i18n/routing';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Languages } from 'lucide-react';

const languages = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'zh', name: '繁體中文', flag: '🇹🇼' },
  { code: 'zh-CN', name: '简体中文', flag: '🇨🇳' },
  { code: 'vi', name: 'Tiếng Việt', flag: '🇻🇳' },
];

export function LanguageSwitcher() {
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();

  const switchLanguage = (newLocale: string) => {
    router.replace(pathname, { locale: newLocale });
  };

  const currentLanguage = languages.find(lang => lang.code === locale);

  return (
    <div className="flex items-center space-x-2">
      <Languages className="w-4 h-4" />
      <Select value={locale} onValueChange={switchLanguage}>
        <SelectTrigger className="w-[140px] h-9">
          <SelectValue>
            <div className="flex items-center space-x-2">
              <span>{currentLanguage?.flag}</span>
              <span className="hidden sm:inline">{currentLanguage?.name}</span>
            </div>
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {languages.map((language) => (
            <SelectItem key={language.code} value={language.code}>
              <div className="flex items-center space-x-2">
                <span>{language.flag}</span>
                <span>{language.name}</span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}