import torch
import numpy as np
from PIL import Image
import os

from feature_extractor import ImageFeatureExtractor

def create_dummy_image(path="dummy_image.jpg"):
    """创建一个简单的虚拟图片用于测试。"""
    # 创建一个 100x100 的随机彩色图片
    array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(array, 'RGB')
    img.save(path)
    print(f"虚拟图片已保存至: {path}")
    return path

def main():
    """主函数，演示特征提取流程。"""
    # 定义虚拟图片路径
    dummy_image_path = "dummy_test_image.jpg"

    try:
        # 1. 创建一个用于测试的虚拟图片
        create_dummy_image(dummy_image_path)

        # 2. 初始化特征提取器
        # 使用默认参数 (mobilenetv3_large_100, 256维)
        print("\n初始化特征提取器...")
        # 为了看到timm的下载进度，我们在这里初始化
        extractor = ImageFeatureExtractor(output_dim=256)

        # 3. 从图片提取特征
        print(f"\n正在从 '{dummy_image_path}' 提取特征...")
        features = extractor.extract(dummy_image_path)

        # 4. 打印结果进行验证
        if features is not None:
            print("\n--- 特征提取成功! ---")
            print(f"特征向量形状: {features.shape}")
            # 由于我们做了L2归一化，其L2范数应该非常接近1.0
            print(f"特征向量L2范数: {torch.linalg.norm(features)}")
            print("特征向量 (前10个元素):")
            print(features[0, :10])
            print("--------------------------")
        else:
            print("\n--- 特征提取失败. ---")

    except Exception as e:
        print(f"\n程序发生错误: {e}")
    finally:
        # 5. 清理创建的虚拟图片
        if os.path.exists(dummy_image_path):
            os.remove(dummy_image_path)
            print(f"\n已清理虚拟图片: {dummy_image_path}")

    print("\n程序执行完毕。")

if __name__ == "__main__":
    main()
