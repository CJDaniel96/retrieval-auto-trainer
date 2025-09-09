import {defineRouting} from 'next-intl/routing';
import {createNavigation} from 'next-intl/navigation';
 
export const routing = defineRouting({
  // A list of all locales that are supported
  locales: ['en', 'zh', 'zh-CN', 'vi'],
 
  // Used when no locale matches
  defaultLocale: 'zh',

  // The `pathnames` object holds pairs of internal and
  // external paths. Based on the locale, the external
  // paths are rewritten to the shared, internal ones.
  pathnames: {
    // If all locales use the same pathname, a single
    // string can be used for external paths
    '/': '/',
    '/training': {
      en: '/training',
      zh: '/training',
      'zh-CN': '/training',
      vi: '/training'
    },
    '/orientation': {
      en: '/orientation',
      zh: '/orientation',
      'zh-CN': '/orientation',
      vi: '/orientation'
    }
  }
});
 
// Lightweight wrappers around Next.js' navigation APIs
// that will consider the routing configuration
export const {Link, redirect, usePathname, useRouter} = createNavigation(routing);