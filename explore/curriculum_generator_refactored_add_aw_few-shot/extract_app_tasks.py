#!/usr/bin/env python3
"""
提取每个app的任务到单独的CSV文件

从生成的任务结果中为每个app创建单独的CSV文件
"""

import pandas as pd
import os
from pathlib import Path
import shutil


def extract_app_tasks():
    """为每个app创建单独的CSV文件"""
    
    # 输入和输出目录
    base_dir = Path("generated_tasks/generated_tasks_refactored_20251103_184949")
    output_dir = base_dir / "all_generated_tasks"
    
    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing tasks from: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # 应用包名到简称的映射
    app_short_names = {
        "code.name.monkey.retromusic": "retromusic",
        "com.android.camera2": "camera2", 
        "com.android.chrome": "chrome",
        "com.android.settings": "settings",
        "com.arduia.expense": "expense",
        "com.dimowner.audiorecorder": "audiorecorder",
        "com.flauschcode.broccoli": "broccoli",
        "com.google.android.contacts": "contacts",
        "com.google.android.deskclock": "deskclock",
        "com.google.android.documentsui": "documentsui",
        "com.simplemobiletools.calendar.pro": "calendar",
        "com.simplemobiletools.draw.pro": "draw",
        "com.simplemobiletools.gallery.pro": "gallery",
        "com.simplemobiletools.smsmessenger": "smsmessenger",
        "de.dennisguse.opentracks": "opentracks",
        "net.cozic.joplin": "joplin",
        "net.gsantner.markor": "markor", 
        "net.osmand": "osmand",
        "org.tasks": "tasks",
        "org.videolan.vlc": "vlc"
    }
    
    processed_apps = []
    failed_apps = []
    
    # 遍历每个app目录
    for app_dir in base_dir.iterdir():
        if app_dir.is_dir() and app_dir.name in app_short_names:
            app_package = app_dir.name
            app_short_name = app_short_names[app_package]
            
            # 检查是否存在generated_tasks.csv文件
            csv_file = app_dir / "generated_tasks.csv"
            if csv_file.exists():
                try:
                    # 读取CSV文件
                    df = pd.read_csv(csv_file)
                    
                    # 创建输出文件名
                    output_filename = f"generated_tasks_refactored_20251103_184949-{app_short_name}.csv"
                    output_path = output_dir / output_filename
                    
                    # 保存到新位置
                    df.to_csv(output_path, index=False)
                    
                    print(f"✅ {app_package} -> {output_filename} (共{len(df)}个任务)")
                    processed_apps.append((app_package, app_short_name, len(df)))
                    
                except Exception as e:
                    print(f"❌ 处理 {app_package} 时出错: {e}")
                    failed_apps.append(app_package)
            else:
                print(f"⚠️  未找到CSV文件: {csv_file}")
                failed_apps.append(app_package)
    
    # 输出统计信息
    print(f"\n{'='*60}")
    print(f"任务提取完成!")
    print(f"{'='*60}")
    print(f"成功处理的应用: {len(processed_apps)}")
    print(f"失败的应用: {len(failed_apps)}")
    
    total_tasks = sum(count for _, _, count in processed_apps)
    print(f"总任务数: {total_tasks}")
    
    print(f"\n📊 详细统计:")
    for app_package, app_short_name, count in sorted(processed_apps, key=lambda x: x[2], reverse=True):
        print(f"  {app_short_name:15} ({app_package:35}): {count:3d} 任务")
    
    if failed_apps:
        print(f"\n❌ 失败的应用:")
        for app in failed_apps:
            print(f"  - {app}")
    
    print(f"\n📁 所有文件已保存到: {output_dir}")
    
    # 列出生成的文件
    print(f"\n📄 生成的文件:")
    for csv_file in sorted(output_dir.glob("*.csv")):
        file_size = csv_file.stat().st_size / 1024  # KB
        print(f"  {csv_file.name} ({file_size:.1f} KB)")
    
    return processed_apps, failed_apps


if __name__ == "__main__":
    extract_app_tasks()
