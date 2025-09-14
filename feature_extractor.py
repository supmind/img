import torch
import torch.nn as nn
import timm
from PIL import Image

class ImageFeatureExtractor:
    """
    一个用于提取图像特征向量的封装类。

    该类使用预训练的timm模型，移除其分类头，
    并添加一个新的线性层，将特征投影到指定的维度。
    """
    def __init__(self, model_name='mobilenetv3_large_100', output_dim=256):
        """
        初始化特征提取器。

        :param model_name: 要使用的timm模型名称。
        :param output_dim: 期望的最终特征向量维度。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. 加载预训练的timm模型
        # num_classes=0 会移除原始的分类层，使得模型输出特征向量
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval() # 设置为评估模式

        # 4. 创建适用于该模型的图像预处理流程
        # timm.data.create_transform 会自动处理好归一化、缩放等步骤
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        # 2. 动态获取模型输出的特征维度
        # 某些timm模型(如mobilenetv3)的 `num_features` 属性可能不准确。
        # 通过一次虚拟推理来动态获取维度是更稳健的方法。
        print("正在动态确定模型的特征维度...")
        with torch.no_grad():
            # 创建一个符合模型输入的虚拟张量
            dummy_input = torch.randn(1, *data_config['input_size']).to(self.device)
            # 执行一次前向传播
            original_dim = self.model(dummy_input).shape[1]
        print(f"检测到的特征维度: {original_dim}")

        # 3. 创建一个新的线性层，用于将原始特征投影到期望的维度
        self.projection_head = nn.Linear(original_dim, output_dim)
        self.projection_head = self.projection_head.to(self.device)
        self.projection_head.eval() # 同样设置为评估模式

    def extract(self, image_path: str) -> torch.Tensor:
        """
        接收一张图片的路径，提取、投影并归一化其特征向量。

        :param image_path: 图像文件的路径。
        :return: 一个L2归一化后的torch.Tensor，形状为 (1, output_dim)，如果文件未找到则返回None。
        """
        try:
            # 打开图片并确保是RGB格式
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: 图像文件未找到于 {image_path}")
            return None

        # 应用预处理转换，并增加一个batch维度 (C, H, W) -> (1, C, H, W)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 在torch.no_grad()上下文中进行推理，以节省计算资源
        with torch.no_grad():
            # 1. 通过基础模型提取高维特征
            base_features = self.model(tensor)
            # 2. 通过投影层将特征维度调整为目标维度
            projected_features = self.projection_head(base_features)

        # 对特征进行L2归一化，这是向量检索中的常见做法，可以提高相似度比较的稳定性
        normalized_features = torch.nn.functional.normalize(projected_features, p=2, dim=1)

        return normalized_features
