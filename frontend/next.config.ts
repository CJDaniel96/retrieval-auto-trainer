import type { NextConfig } from "next";
import createNextIntlPlugin from 'next-intl/plugin';

const withNextIntl = createNextIntlPlugin('./src/i18n/request.ts');

const nextConfig: NextConfig = {
  /* config options here */
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/**',
      },
    ],
  },
  // 生產部署時使用靜態導出 - 暫時禁用以解決動態路由問題
  // output: process.env.NODE_ENV === 'production' ? 'export' : undefined,
  trailingSlash: true,
  // 配置 assetPrefix 以便正確處理靜態資源
  assetPrefix: process.env.NODE_ENV === 'production' ? '/static' : undefined,
};

export default withNextIntl(nextConfig);
