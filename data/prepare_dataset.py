import os
import json
import random
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def process_folder(
    folder_path: str, folder_name: str
) -> Tuple[List[Dict[str, str]], int]:
    """
    处理单个文件夹，读取符合条件的 JSON 文件，返回数据列表及计数。
    """
    partial_dataset = []
    count = 0
    print(f"Processing folder: {folder_name}")
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".json"):
            continue
        json_path = os.path.join(folder_path, file_name)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                count += 1
                index = file_name.rsplit(".", 1)[0]
                ori_image_path = os.path.join(folder_name, f"{index}_ori.png")
                edit_image_path = os.path.join(folder_name, f"{index}_edit.png")
                partial_dataset.append({"ori": ori_image_path, "edit": edit_image_path})
                # 如果你想每个文件夹只取一个json，保持break，不然去掉
                break
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: 无法解析 JSON 文件 {json_path}，错误: {e}")
    return partial_dataset, count


def load_and_save_dataset(
    data_directory: str,
    save_path: str,
    jsonname: str,
    max_workers: int = 100,
) -> None:
    """
    多线程加载数据集并保存为 JSON 文件。

    Args:
        data_directory: 数据根目录，包含子文件夹
        save_path: 结果保存目录
        jsonname: 保存的 JSON 文件名
        max_workers: 线程池最大线程数
    """
    dataset = []
    total_count = 0

    dir_list = [
        d
        for d in os.listdir(data_directory)
        if os.path.isdir(os.path.join(data_directory, d))
    ]

    with tqdm(
        total=len(dir_list), desc="Processing Dataset"
    ) as pbar, ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_folder, os.path.join(data_directory, d), d)
            for d in dir_list
        ]

        for future in futures:
            partial_data, count = future.result()
            dataset.extend(partial_data)
            total_count += count
            pbar.update(1)

    os.makedirs(save_path, exist_ok=True)
    json_path = os.path.join(save_path, jsonname)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"数据加载完成，总数量: {total_count}，结果保存至 {json_path}")


def merge_json_files(json_files: List[str], output_path: str, output_name: str) -> None:
    """
    合并多个 JSON 文件的内容，并保存到一个新的 JSON 文件。

    Args:
        json_files: 待合并的 JSON 文件路径列表
        output_path: 合并后文件保存目录
        output_name: 合并后文件名
    """
    merged_data = []
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: 文件 {file_path} 内容不是列表，跳过。")
        except Exception as e:
            print(f"Error 读取文件 {file_path} 时出错: {e}")

    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, output_name)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)
    print(f"合并完成，保存至 {output_file}，总条目数: {len(merged_data)}")


if __name__ == "__main__":
    random.seed(42)
    data_dir = "/path/to/your/dataset"
    save_dir = "/path/to/your/save_path"
    json_file_name = "your_jsonname.json"
    load_and_save_dataset(data_dir, save_dir, json_file_name)
