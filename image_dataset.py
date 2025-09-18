import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
import numpy as np

class LabelledImageDataset(Dataset):
    """
    一个为度量学习准备的数据集。
    它接收一个包含图像路径的列表，然后将每张图片视为一个独立的样本，
    并从其父目录名中提取类别标签。
    """
    def __init__(self, image_path_list, transform=None):
        """
        :param image_path_list: 包含所有图像路径的列表。
        :param transform: 应用于图像的torchvision变换。
        """
        self.transform = transform
        self.images, self.labels = self._load_data(image_path_list)

    def _load_data(self, image_path_list):
        """
        从路径列表加载数据，提取唯一的图片和标签。
        """
        if not image_path_list:
            return [], []

        # 获取唯一的图片路径
        image_paths = sorted(list(set(image_path_list)))

        # 从路径中提取标签，并创建从标签名到整数的映射
        # 例如: 'data/cat/img1.png' -> 'cat'
        class_names = [os.path.basename(os.path.dirname(p)) for p in image_paths]

        unique_class_names = sorted(list(set(class_names)))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(unique_class_names)}

        labels = [self.class_to_idx[name] for name in class_names]

        print(f"从路径列表中加载了 {len(image_paths)} 张唯一图片，分属于 {len(unique_class_names)} 个类别。")

        return image_paths, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        返回一个样本，包括经过变换的图像和其对应的整数标签。
        """
        image_path = self.images[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"警告: 找不到文件 {image_path}，将返回一个空张量和-1标签。")
            # 返回一个占位符和无效标签，需要在数据加载循环中处理
            return torch.zeros((3, 224, 224)), -1

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label
