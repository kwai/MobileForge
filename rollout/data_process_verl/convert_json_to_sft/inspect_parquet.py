#!/usr/bin/env python3
"""
检查parquet文件结构的脚本
"""

import pandas as pd
import sys
import os

def inspect_parquet(parquet_path):
    """检查parquet文件的结构和内容"""
    if not os.path.exists(parquet_path):
        print(f"文件不存在: {parquet_path}")
        return

    try:
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)

        print(f"=== Parquet文件信息 ===")
        print(f"文件路径: {parquet_path}")
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(f"数据类型:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")

        print(f"\n=== 前几行数据 ===")
        print(df.head())

        # 如果有图像数据，显示相关信息
        for col in df.columns:
            if 'image' in col.lower() or 'screenshot' in col.lower():
                print(f"\n=== {col} 列信息 ===")
                if df[col].dtype == 'object':
                    sample_val = df[col].iloc[0]
                    if hasattr(sample_val, '__len__'):
                        print(f"  数据类型: {type(sample_val)}")
                        print(f"  数据长度: {len(sample_val)}")
                        if hasattr(sample_val, 'shape'):
                            print(f"  数据形状: {sample_val.shape}")

        return df

    except Exception as e:
        print(f"读取parquet文件时出错: {e}")
        return None

if __name__ == "__main__":
    parquet_path = "./processed_data/source_data/example/train-00000-of-00015.parquet"
    inspect_parquet(parquet_path)