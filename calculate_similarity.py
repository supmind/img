import torch
import pandas as pd
import argparse
import os
import glob
from tqdm import tqdm

# 导入我们自己编写的模块
from feature_extractor import ImageFeatureExtractor

def calculate_similarity_matrix(image_dir, weights_path, model_name, output_path):
    """
    计算一个文件夹下所有图片的特征向量，并生成它们之间的余弦相似度矩阵。

    :param image_dir: 包含图片的文件夹路径。
    :param weights_path: 微调过的模型权重路径 (.pth文件)。
    :param model_name: 使用的基础模型名称。
    :param output_path: 保存相似度矩阵的CSV文件路径。
    """
    # 1. 查找所有支持的图片文件
    supported_formats = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for fmt in supported_formats:
        image_paths.extend(glob.glob(os.path.join(image_dir, fmt)))

    if not image_paths:
        print(f"错误: 在文件夹 {image_dir} 中未找到任何图片文件。")
        return

    print(f"找到了 {len(image_paths)} 张图片。")

    # 2. 初始化特征提取器
    print("正在初始化特征提取器并加载微调过的权重...")
    try:
        extractor = ImageFeatureExtractor(
            model_name=model_name,
            weights_path=weights_path
        )
    except Exception as e:
        print(f"错误: 初始化特征提取器失败。请确保模型名称 '{model_name}' 正确。")
        print(f"原始错误: {e}")
        return

    # 3. 提取所有图片的特征向量
    features_list = []
    valid_filenames = []

    progress_bar = tqdm(image_paths, desc="正在提取特征向量")
    for img_path in progress_bar:
        feature = extractor.extract(img_path)
        if feature is not None:
            features_list.append(feature)
            valid_filenames.append(os.path.basename(img_path))
        else:
            print(f"警告: 无法为图片 {img_path} 提取特征，将跳过此文件。")

    if not features_list:
        print("错误: 未能成功提取任何图片的特征。")
        return

    # 4. 将特征列表堆叠成一个大张量 (N, D)
    # features_list中的每个元素形状是(1, D)，所以我们用torch.cat
    features_tensor = torch.cat(features_list, dim=0)
    print(f"特征矩阵形状: {features_tensor.shape}")

    # 5. 计算余弦相似度矩阵
    # 由于特征向量已经被L2归一化，余弦相似度等于矩阵乘法 (features @ features.T)
    print("正在计算余弦相似度矩阵...")
    # 将张量移动到CPU上进行矩阵乘法和后续的pandas操作
    features_tensor = features_tensor.cpu()
    similarity_matrix = torch.matmul(features_tensor, features_tensor.T)

    # 6. 保存为CSV文件
    print(f"正在将相似度矩阵保存至 {output_path}...")
    # 将张量转换为numpy数组，然后创建pandas DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix.numpy(),
        index=valid_filenames,
        columns=valid_filenames
    )

    try:
        similarity_df.to_csv(output_path)
        print("保存成功！")
    except Exception as e:
        print(f"错误: 保存CSV文件失败。")
        print(f"原始错误: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算一个文件夹下所有图片的特征向量，并生成余弦相似度矩阵。")

    parser.add_argument("--image_dir", type=str, required=True, help="包含待处理图片的文件夹路径。")
    parser.add_argument("--weights_path", type=str, required=True, help="微调好的模型权重文件路径 (.pth)。")
    parser.add_argument("--output_path", type=str, required=True, help="保存相似度矩阵的CSV文件路径。")
    parser.add_argument("--model_name", type=str, default="mobilenetv3_large_100", help="使用的基础模型名称，必须与微调时一致。")

    args = parser.parse_args()

    calculate_similarity_matrix(
        image_dir=args.image_dir,
        weights_path=args.weights_path,
        model_name=args.model_name,
        output_path=args.output_path
    )
