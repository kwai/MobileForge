#!/usr/bin/env python3
"""
简单的parquet到json转换脚本
直接转换为JSONL格式（推荐用于大文件）
"""

import pandas as pd
import json
import base64
import os

def convert_parquet_to_jsonl():
    """转换parquet文件为jsonl格式"""

    parquet_path = "./processed_data/source_data/example/train-00000-of-00015.parquet"
    output_path = "./convert_json_to_sft/train-00000-of-00015.jsonl"

    print(f"开始转换: {parquet_path}")
    print(f"输出到: {output_path}")

    try:
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)
        print(f"成功读取，共 {len(df)} 条记录")

        # 转换并写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    print(f"进度: {idx}/{len(df)}")

                # 处理图像数据
                image_base64 = ""
                if pd.notna(row['image']) and isinstance(row['image'], dict) and 'bytes' in row['image']:
                    image_bytes = row['image']['bytes']
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

                # 构建记录
                record = {
                    "id": idx,
                    "image": image_base64,
                    "thinking": str(row['thinking']) if pd.notna(row['thinking']) else "",
                    "problem": str(row['problem']) if pd.notna(row['problem']) else "",
                    "solution": str(row['solution']) if pd.notna(row['solution']) else ""
                }

                # 写入一行
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"\n转换完成！")
        print(f"输出文件: {output_path}")
        print(f"总记录数: {len(df)}")

        # 检查文件大小
        file_size = os.path.getsize(output_path)
        print(f"输出文件大小: {file_size / (1024*1024):.2f} MB")

    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    convert_parquet_to_jsonl()