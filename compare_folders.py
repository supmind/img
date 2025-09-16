import torch
import pandas as pd
import argparse
import os
import glob
import random
from tqdm import tqdm

# 导入我们自己编写的模块
from feature_extractor import ImageFeatureExtractor

def compare_image_folders(dir1, dir2, weights_path, model_name):
    """
    从两个文件夹中各随机抽取5张图片，计算这10张图片的相似度矩阵，并打印到控制台。
    """
    # 1. 查找并采样图片文件
    supported_formats = ('*.jpg', '*.jpeg', '*.png')

    def get_image_samples(directory, num_samples=5):
        if not os.path.isdir(directory):
            print(f"错误: 文件夹不存在 -> {directory}")
            return None

        all_paths = []
        for fmt in supported_formats:
            all_paths.extend(glob.glob(os.path.join(directory, fmt)))

        if not all_paths:
            print(f"警告: 在文件夹 {directory} 中未找到任何图片。")
            return []

        if len(all_paths) < num_samples:
            print(f"警告: 文件夹 {directory} 中的图片少于{num_samples}张，将使用所有{len(all_paths)}张图片。")
            return all_paths

        return random.sample(all_paths, num_samples)

    sampled_paths1 = get_image_samples(dir1)
    sampled_paths2 = get_image_samples(dir2)

    if sampled_paths1 is None or sampled_paths2 is None:
        return # 提前退出

    combined_paths = sampled_paths1 + sampled_paths2

    if not combined_paths:
        print("错误: 未能从任何一个文件夹中采样到图片。")
        return

    print(f"总共采样了 {len(combined_paths)} 张图片进行比较。")

    # 2. 初始化特征提取器
    print("正在初始化特征提取器...")
    try:
        extractor = ImageFeatureExtractor(
            model_name=model_name,
            weights_path=weights_path
        )
    except Exception as e:
        print(f"错误: 初始化特征提取器失败: {e}")
        return

    # 3. 提取特征向量
    features_list = []
    valid_filenames = []
    progress_bar = tqdm(combined_paths, desc="正在提取特征向量")
    for img_path in progress_bar:
        feature = extractor.extract(img_path)
        if feature is not None:
            features_list.append(feature)
            # 为了简洁，只保留文件名
            valid_filenames.append(os.path.basename(img_path))
        else:
            print(f"警告: 无法为图片 {img_path} 提取特征，已跳过。")

    if not features_list:
        print("错误: 未能成功提取任何图片的特征。")
        return

    # 4. 计算相似度矩阵
    features_tensor = torch.cat(features_list, dim=0).cpu()
    similarity_matrix = torch.matmul(features_tensor, features_tensor.T)

    # 5. 使用pandas格式化并打印结果
    similarity_df = pd.DataFrame(
        similarity_matrix.numpy(),
        index=valid_filenames,
        columns=valid_filenames
    )

    print("\n--- 图片相似度矩阵 ---")
    # 设置pandas的显示选项，以确保矩阵能完整打印
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(similarity_df)
    print("----------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从两个文件夹中各随机抽取5张图片，计算相似度矩阵并打印。")

    parser.add_argument("--dir1", type=str, required=True, help="第一个图片文件夹的路径。")
    parser.add_argument("--dir2", type=str, required=True, help="第二个图片文件夹的路径。")
    parser.add_argument("--weights_path", type=str, required=True, help="微调好的模型权重文件路径 (.pth)。")
    parser.add_argument("--model_name", type=str, default="mobilenetv3_large_100", help="使用的基础模型名称，必须与微调时一致。")

    args = parser.parse_args()

    compare_image_folders(
        dir1=args.dir1,
        dir2=args.dir2,
        weights_path=args.weights_path,
        model_name=args.model_name
    )
