#!/usr/bin/env python3
"""
快速修复GRPO格式数据的PyArrow兼容性问题

使用方法:
python fix_grpo_format.py --rollout_dir /path/to/rollout --output_dir fixed_data
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加core模块到路径
sys.path.append(str(Path(__file__).parent))

from core import MobileForgeDataProcessor, MobileForgeDataSaver

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="快速修复GRPO格式数据的PyArrow兼容性问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python fix_grpo_format.py --rollout_dir results/session-debug-mobileforge-v25100206
  python fix_grpo_format.py --rollout_dir results/session-debug-mobileforge-v25100206 --max_tasks 5
        """
    )
    
    parser.add_argument("--rollout_dir", required=True, help="rollout结果目录")
    parser.add_argument("--output_dir", default="fixed_grpo_data", help="输出目录")
    parser.add_argument("--max_tasks", type=int, help="最大处理任务数（用于测试）")
    
    args = parser.parse_args()
    
    print(f"输入目录: {args.rollout_dir}")
    print(f"输出目录: {args.output_dir}")
    if args.max_tasks:
        print(f"最大任务数: {args.max_tasks}")
    print("-" * 70)
    
    # 验证输入
    if not os.path.exists(args.rollout_dir):
        print(f"错误: 未找到rollout目录: {args.rollout_dir}")
        return 1
    
    try:
        logger.info("开始修复GRPO格式数据...")
        
        # 1. 处理数据
        processor = MobileForgeDataProcessor(args.rollout_dir, args.output_dir)
        processed_data = processor.process_all_tasks(max_tasks=args.max_tasks)
        
        # 2. 使用修复版本的数据保存器
        data_saver = MobileForgeDataSaver(args.output_dir)
        saved_files = data_saver.save_training_data(processed_data, format_type="grpo")
        data_saver.save_session_summary(processed_data)
        
        # 3. 更新处理结果
        processed_data['session_info'] = data_saver.get_session_info()
        processed_data['saved_files'] = saved_files
        
        # 4. 打印结果
        print("\n" + "=" * 70)
        print("修复完成！")
        print("=" * 70)
        
        stats = processed_data['statistics']
        print(f"总任务数: {stats['total_tasks']}")
        print(f"已处理任务数: {stats['processed_tasks']}")
        print(f"成功轨迹: {stats['successful_trajectories']}")
        print(f"失败轨迹: {stats['failed_trajectories']}")
        print(f"错误轨迹: {stats['error_trajectories']}")
        
        print(f"\n正样本: {len(processed_data['positive_samples'])}")
        print(f"负样本: {len(processed_data['negative_samples'])}")
        print(f"总样本数: {len(processed_data['all_samples'])}")
        
        # 会话信息
        session_info = processed_data['session_info']
        session_dir = session_info.get('session_dir')
        
        print(f"\n✅ 修复后的数据已保存到: {session_dir}")
        print(f"📁 会话时间戳: {session_info.get('timestamp')}")
        
        if 'saved_files' in processed_data:
            saved_files = processed_data['saved_files']
            positive_file = saved_files.get('positive_grpo', '')
            negative_file = saved_files.get('negative_grpo', '')
            
            print(f"\n🚀 可直接用于MobileForge GRPO训练:")
            print(f"正样本文件: {positive_file}")
            print(f"负样本文件: {negative_file}")
            
            print(f"\n训练命令示例:")
            print(f"torchrun --nproc_per_node=4 src/open_r1/grpo_gui.py \\")
            print(f"  --dataset_positive_name {positive_file} \\")
            print(f"  --dataset_negative_name {negative_file} \\")
            print(f"  --model_name_or_path your_model_path \\")
            print(f"  --output_dir ./grpo_output")
        
        print(f"\n🔧 修复说明:")
        print(f"- 已解决PyArrow兼容性问题")
        print(f"- 所有content字段统一为列表格式")
        print(f"- 数据现在可以正常加载到MobileForge训练脚本")
        
        return 0
        
    except Exception as e:
        print(f"错误: 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
