import torch
import torch.nn as nn
import timm
from PIL import Image
import os

class ImageFeatureExtractor:
    """
    一个用于提取图像特征向量的封装类。

    该类使用预训练的timm模型，移除其分类头，
    并添加一个新的线性层，将特征投影到指定的维度。
    """
    def __init__(self, model_name='mobilenetv3_large_100', output_dim=256, weights_path=None):
        """
        初始化特征提取器。

        :param model_name: 要使用的timm模型名称。
        :param output_dim: 期望的最终特征向量维度。
        :param weights_path: (可选) 微调过的投影层权重的路径。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 1. 加载预训练的主干网络
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 2. 创建图像预处理流程
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)

        # 3. 动态获取主干网络输出的特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, *data_config['input_size']).to(self.device)
            original_dim = self.model(dummy_input).shape[1]

        # 4. 创建投影层
        self.projection_head = nn.Linear(original_dim, output_dim)

        # 5. 加载微调过的权重（如果提供了路径）
        if weights_path:
            if os.path.exists(weights_path):
                print(f"正在从 {weights_path} 加载微调过的投影层权重...")
                self.projection_head.load_state_dict(torch.load(weights_path, map_location=self.device))
            else:
                print(f"警告: 权重文件 {weights_path} 未找到。将使用随机初始化的投影层。")
        else:
            print("未提供微调权重，将使用随机初始化的投影层。")

        self.projection_head = self.projection_head.to(self.device)
        self.projection_head.eval()

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
