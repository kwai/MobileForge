#!/usr/bin/env python3
"""
将parquet文件转换为json格式的脚本
"""

import pandas as pd
import json
import base64
import os
from typing import Dict, Any, List

def convert_image_to_base64(image_data: Dict) -> str:
    """
    将图像字节数据转换为base64编码的字符串

    Args:
        image_data: 包含'bytes'键的字典，值为图像字节数据

    Returns:
        base64编码的图像字符串
    """
    if isinstance(image_data, dict) and 'bytes' in image_data:
        image_bytes = image_data['bytes']
        return base64.b64encode(image_bytes).decode('utf-8')
    else:
        return ""

def convert_parquet_to_json(parquet_path: str, output_path: str = None) -> None:
    """
    将parquet文件转换为json格式

    Args:
        parquet_path: 输入的parquet文件路径
        output_path: 输出的json文件路径，如果为None则自动生成
    """

    if not os.path.exists(parquet_path):
        print(f"错误：文件不存在 {parquet_path}")
        return

    # 如果没有指定输出路径，自动生成
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]
        output_dir = os.path.dirname(parquet_path)
        output_path = os.path.join(output_dir, f"{base_name}.json")

    print(f"开始转换 {parquet_path} -> {output_path}")

    try:
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)
        print(f"成功读取parquet文件，包含 {len(df)} 条记录")

        # 转换数据
        json_data = []

        for idx, row in df.iterrows():
            if idx % 100 == 0:
                print(f"处理进度: {idx}/{len(df)}")

            # 转换图像数据
            image_base64 = ""
            if pd.notna(row['image']):
                image_base64 = convert_image_to_base64(row['image'])

            # 构建json记录
            record = {
                "id": idx,
                "image": image_base64,
                "thinking": str(row['thinking']) if pd.notna(row['thinking']) else "",
                "problem": str(row['problem']) if pd.notna(row['problem']) else "",
                "solution": str(row['solution']) if pd.notna(row['solution']) else ""
            }

            json_data.append(record)

        # 保存为json文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"转换完成！输出文件: {output_path}")
        print(f"总共转换了 {len(json_data)} 条记录")

        # 显示统计信息
        image_count = sum(1 for record in json_data if record['image'])
        print(f"包含图像的记录数: {image_count}")

    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()

def convert_parquet_to_jsonlines(parquet_path: str, output_path: str = None) -> None:
    """
    将parquet文件转换为jsonlines格式（每行一个json对象）

    Args:
        parquet_path: 输入的parquet文件路径
        output_path: 输出的jsonl文件路径，如果为None则自动生成
    """

    if not os.path.exists(parquet_path):
        print(f"错误：文件不存在 {parquet_path}")
        return

    # 如果没有指定输出路径，自动生成
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(parquet_path))[0]
        output_dir = os.path.dirname(parquet_path)
        output_path = os.path.join(output_dir, f"{base_name}.jsonl")

    print(f"开始转换 {parquet_path} -> {output_path} (JSONL格式)")

    try:
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)
        print(f"成功读取parquet文件，包含 {len(df)} 条记录")

        # 转换数据并写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    print(f"处理进度: {idx}/{len(df)}")

                # 转换图像数据
                image_base64 = ""
                if pd.notna(row['image']):
                    image_base64 = convert_image_to_base64(row['image'])

                # 构建json记录
                record = {
                    "id": idx,
                    "image": image_base64,
                    "thinking": str(row['thinking']) if pd.notna(row['thinking']) else "",
                    "problem": str(row['problem']) if pd.notna(row['problem']) else "",
                    "solution": str(row['solution']) if pd.notna(row['solution']) else ""
                }

                # 写入一行json
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"转换完成！输出文件: {output_path}")
        print(f"总共转换了 {len(df)} 条记录")

    except Exception as e:
        print(f"转换过程中出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    parquet_path = "./processed_data/source_data/example/train-00000-of-00015.parquet"

    print("选择输出格式:")
    print("1. JSON格式 (单个数组)")
    print("2. JSONL格式 (每行一个JSON对象)")
    print("3. 两种格式都生成")

    choice = input("请选择 (1/2/3): ").strip()

    if choice == "1":
        convert_parquet_to_json(parquet_path)
    elif choice == "2":
        convert_parquet_to_jsonlines(parquet_path)
    elif choice == "3":
        convert_parquet_to_json(parquet_path)
        convert_parquet_to_jsonlines(parquet_path)
    else:
        print("无效选择，默认生成JSONL格式")
        convert_parquet_to_jsonlines(parquet_path)

if __name__ == "__main__":
    main()