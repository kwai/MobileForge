#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IRR指标一键运行脚本
"""

import os
import sys
import subprocess

def main():
    print("="*80)
    print("MobileGym-Critic IRR (Information Retention Rate) 指标")
    print("="*80)
    print()
    print("请选择操作:")
    print("1. 检查IRR状态")
    print("2. 功能演示")
    print("3. 添加IRR列到所有agent")
    print("4. 分析失败案例（并行处理）")
    print("5. 查看完整文档")
    print("0. 退出")
    print()
    
    while True:
        choice = input("请输入选择 (0-5): ").strip()
        
        if choice == "0":
            print("退出")
            break
        elif choice == "1":
            print("\n🔍 检查IRR处理状态...")
            subprocess.run([sys.executable, "demo_irr_functionality.py", "--status"])
        elif choice == "2":
            print("\n📋 IRR功能演示...")
            subprocess.run([sys.executable, "demo_irr_functionality.py", "--demo"])
        elif choice == "3":
            print("\n⚙️  添加IRR列到所有agent...")
            confirm = input("确认要为所有agent添加IRR列？(y/N): ").strip().lower()
            if confirm == 'y':
                subprocess.run([sys.executable, "quick_add_irr_columns.py", "--all"])
            else:
                print("已取消")
        elif choice == "4":
            print("\n🚀 并行分析失败案例...")
            print("注意: 这将调用LLM进行详细分析，可能需要较长时间")
            confirm = input("确认要开始分析？(y/N): ").strip().lower()
            if confirm == 'y':
                subprocess.run([sys.executable, "true_parallel_irr_processor.py", "--all"])
            else:
                print("已取消")
        elif choice == "5":
            print("\n📖 查看完整文档...")
            if os.path.exists("IRR_IMPLEMENTATION_README.md"):
                if sys.platform.startswith('linux'):
                    subprocess.run(["less", "IRR_IMPLEMENTATION_README.md"])
                else:
                    subprocess.run(["more", "IRR_IMPLEMENTATION_README.md"])
            else:
                print("文档文件不存在")
        else:
            print("无效选择，请输入0-5")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()
