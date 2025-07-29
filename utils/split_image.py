#!/usr/bin/env python3
# sort_images.py
 
import os
import re
import shutil
import argparse
 
def parse_args():
    parser = argparse.ArgumentParser(
        description='將影像依檔名中「@後的元件名稱」分類到對應資料夾'
    )
    parser.add_argument('source_dir',
                        help='來源資料夾路徑')
    parser.add_argument('dest_root',
                        help='分類後的根目錄')
    parser.add_argument('-p', '--pattern',
                        default=r'@(.+?)_\d+_',
                        help='解析元件名稱的正則 (預設: @(.+?)_\\d+_)')
    parser.add_argument('-e', '--extensions',
                        nargs='+',
                        default=['jpg', 'jpeg', 'png'],
                        help='要處理的影像副檔名 (預設: jpg jpeg png)')
    parser.add_argument('-c', '--copy',
                        action='store_true',
                        help='使用複製檔案 (預設為搬移)')
    return parser.parse_args()
 
def main():
    args = parse_args()
    src_dir = args.source_dir
    dest_root = args.dest_root
    pattern = re.compile(args.pattern, re.IGNORECASE)
    exts = tuple(f'.{e.lower()}' for e in args.extensions)
 
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if not fname.lower().endswith(exts):
                continue
 
            m = pattern.search(fname)
            if not m:
                print(f'⚠ 無法解析元件名稱：{fname}')
                continue
 
            comp_name = m.group(1)
            dst_dir = os.path.join(dest_root, comp_name)
            os.makedirs(dst_dir, exist_ok=True)
 
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dst_dir, fname)
 
            if args.copy:
                shutil.copy2(src_path, dst_path)
                action = '複製'
            else:
                shutil.move(src_path, dst_path)
                action = '搬移'
 
            print(f'{action}: {src_path} → {dst_path}')
 
    print('✅ 分類完畢！')
 
if __name__ == '__main__':
    main()