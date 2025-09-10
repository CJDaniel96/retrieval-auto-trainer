'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Home, Settings } from 'lucide-react';

const navigationItems = [
  {
    name: '訓練管理',
    href: '/',
    icon: Home
  },
  {
    name: '訓練配置',
    href: '/config',
    icon: Settings
  }
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="flex space-x-2">
      {navigationItems.map((item) => {
        const Icon = item.icon;
        const isActive = pathname === item.href || (item.href !== '/' && pathname.startsWith(item.href));
        
        return (
          <Button
            key={item.href}
            variant={isActive ? 'default' : 'ghost'}
            size="sm"
            asChild
          >
            <Link href={item.href} className="flex items-center space-x-2">
              <Icon className="w-4 h-4" />
              <span>{item.name}</span>
            </Link>
          </Button>
        );
      })}
    </nav>
  );
}