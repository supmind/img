import torch
from torch.utils.data import Dataset
from PIL import Image
import csv

class TripletDataset(Dataset):
    """
    一个自定义的PyTorch数据集，用于加载三元组训练数据。
    它从一个CSV文件中读取三元组的路径，并加载相应的图片。
    """
    def __init__(self, csv_file, transform=None):
        """
        初始化数据集。

        :param csv_file: 包含三元组路径的CSV文件路径。
                         CSV文件应包含'anchor', 'positive', 'negative'三列，并有表头。
        :param transform: 应用于每张图片上的torchvision变换。
        """
        self.triplets = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # 跳过表头
                for row in reader:
                    if len(row) == 3:
                        self.triplets.append(row)
        except FileNotFoundError:
            print(f"错误: CSV文件未找到于 {csv_file}")
            raise

        self.transform = transform
        if not self.triplets:
            print(f"警告: 在 {csv_file} 中没有找到任何三元组数据。")

    def __len__(self):
        """
        返回数据集中的样本总数。
        """
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        根据索引idx，加载并返回一个三元组样本。
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取三元组中三张图片的路径
        anchor_path, positive_path, negative_path = self.triplets[idx]

        try:
            # 加载图片并确保为RGB格式
            anchor_img = Image.open("/content/data/"+anchor_path).convert('RGB')
            positive_img = Image.open("/content/data/"+positive_path).convert('RGB')
            negative_img = Image.open("/content/data/"+negative_path).convert('RGB')
        except FileNotFoundError as e:
            print(f"警告: 无法加载图片 {e}。正在尝试加载下一个样本作为替代。")
            # 处理文件丢失的一个简单方法是加载下一个样本。
            # 这可以避免因单个文件损坏而导致整个训练中断。
            return self.__getitem__((idx + 1) % len(self))

        # 应用图像变换
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img
