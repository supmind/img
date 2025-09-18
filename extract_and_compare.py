import torch
import torch.nn as nn
import timm
from PIL import Image
import os
import argparse
from peft import PeftModel
from itertools import combinations
import torch.nn.functional as F

def load_finetuned_model(model_name, output_dim, model_dir):
    """
    加载微调后的模型，包括基础模型、LoRA适配器和投影头。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载不带分类头的timm基础模型
    base_model = timm.create_model(model_name, pretrained=True, num_classes=0)

    # 2. 应用LoRA权重
    try:
        model = PeftModel.from_pretrained(base_model, model_dir)
        model = model.to(device)
        model.eval()
        print(f"成功从 {model_dir} 加载LoRA适配器。")
    except Exception as e:
        print(f"加载LoRA适配器失败: {e}")
        print("将仅使用基础模型。")
        model = base_model.to(device)
        model.eval()

    # 3. 创建并加载投影头的权重
    # 获取基础模型的输出维度
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    with torch.no_grad():
        dummy_input = torch.randn(1, *data_config['input_size']).to(device)
        original_dim = model(dummy_input).shape[1]

    head = nn.Linear(original_dim, output_dim)
    head_weights_path = os.path.join(model_dir, 'head_weights.pth')

    if os.path.exists(head_weights_path):
        head.load_state_dict(torch.load(head_weights_path, map_location=device))
        print(f"成功从 {head_weights_path} 加载投影头权重。")
    else:
        print(f"警告: 在 {head_weights_path} 未找到投影头权重。将使用随机初始化的投影头。")

    head = head.to(device)
    head.eval()

    return model, head, transform, device

def extract_features(image_path, model, head, transform, device):
    """
    为单个图像提取、投影并归一化特征向量。
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误: 图像文件 {image_path} 未找到。")
        return None

    # 预处理图像并增加batch维度
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        base_features = model(tensor)
        projected_features = head(base_features)

    # L2 归一化
    normalized_features = F.normalize(projected_features, p=2, dim=1)
    return normalized_features

def main(args):
    """
    主函数：扫描图像，加载模型，提取特征，计算并打印相似度。
    """
    # 1. 加载模型
    model, head, transform, device = load_finetuned_model(
        args.model_name, args.output_dim, args.model_dir
    )

    # 2. 查找并筛选图像
    image_dir = args.image_dir
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    try:
        image_files = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(supported_formats)
        ]
    except FileNotFoundError:
        print(f"错误: 图像目录 {image_dir} 不存在。")
        return

    if not image_files:
        print(f"在目录 {image_dir} 中没有找到支持的图像文件。")
        return

    # 获取前10张或所有图像（如果总数小于10）
    images_to_process = sorted(image_files)[:10]
    print(f"\n将处理以下 {len(images_to_process)} 张图片:")
    for img_path in images_to_process:
        print(f"- {os.path.basename(img_path)}")

    # 3. 提取所有图像的特征
    feature_vectors = {}
    print("\n--- 开始提取特征 ---")
    for img_path in images_to_process:
        features = extract_features(img_path, model, head, transform, device)
        if features is not None:
            feature_vectors[img_path] = features
    print("--- 特征提取完成 ---\n")

    # 4. 计算并打印余弦相似度
    print("--- 两两图片余弦相似度 ---")
    # 获取所有成功提取特征的图片路径
    valid_image_paths = list(feature_vectors.keys())

    # 使用itertools.combinations来生成所有唯一的对
    for (path1, path2) in combinations(valid_image_paths, 2):
        # 从字典中获取特征向量
        vec1 = feature_vectors[path1]
        vec2 = feature_vectors[path2]

        # 计算余弦相似度
        # 由于向量已经归一化，点积等于余弦相似度
        cosine_similarity = torch.dot(vec1.squeeze(), vec2.squeeze()).item()

        # 打印结果
        filename1 = os.path.basename(path1)
        filename2 = os.path.basename(path2)
        print(f"{filename1} <-> {filename2}: {cosine_similarity:.4f}")

    print("\n任务完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用微调后的模型提取图像特征，并计算它们之间的余弦相似度。")
    parser.add_argument("--image_dir", type=str, required=True, help="包含要处理的图像的目录。")
    parser.add_argument("--model_dir", type=str, default="finetuned_model", help="包含微调模型（LoRA适配器和head_weights.pth）的目录。")
    parser.add_argument("--model_name", type=str, default="mobilenetv4_conv_medium.e500_r224_in1k", help="用于微调的timm基础模型名称。")
    parser.add_argument("--output_dim", type=int, default=256, help="投影头的输出维度。")

    args = parser.parse_args()
    main(args)
