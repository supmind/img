import torch
import torch.nn as nn
import timm
from PIL import Image
import os
import argparse
from peft import PeftModel
import csv
import random
from tqdm import tqdm
from pytorch_metric_learning import metrics

# --- Model Loading Logic (from extract_and_compare.py) ---
def load_finetuned_model(model_name, output_dim, model_dir):
    """
    加载微调后的模型，包括基础模型、LoRA适配器和投影头。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    try:
        base_model = timm.create_model(model_name, pretrained=True, num_classes=0)
    except Exception as e:
        print(f"创建timm模型 '{model_name}' 失败: {e}")
        return None, None, None, None

    try:
        model = PeftModel.from_pretrained(base_model, model_dir)
        print(f"成功从 {model_dir} 加载LoRA适配器。")
    except Exception:
        print(f"在 {model_dir} 中找不到LoRA适配器。将使用基础预训练模型。")
        model = base_model

    model = model.to(device)
    model.eval()

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

# --- Feature Extraction Logic (from extract_and_compare.py) ---
def extract_features(image_path, model, head, transform, device):
    """
    为单个图像提取、投影并归一化特征向量。
    """
    try:
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            base_features = model(tensor)
            projected_features = head(base_features)
        # L2 归一化对于距离计算很重要
        normalized_features = torch.nn.functional.normalize(projected_features, p=2, dim=1)
        return normalized_features
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"处理图片 {os.path.basename(image_path)} 时出错: {e}")
        return None

# --- Main mAP Calculation Logic ---
def calculate_map(args):
    """
    主函数，用于加载模型、数据，提取特征并计算mAP。
    """
    # 1. 加载模型
    print(f"--- 正在加载模型: {args.model_dir} ---")
    model, head, transform, device = load_finetuned_model(
        args.model_name, args.output_dim, args.model_dir
    )
    if model is None:
        return

    # 2. 准备评估数据集
    print(f"\n--- 正在从 {args.csv_path} 准备评估数据集 ---")
    try:
        with open(args.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            if header[0].lower().endswith(('.jpg', '.jpeg', '.png')):
                all_triplets = [header] + [row for row in reader if len(row) == 3]
            else:
                all_triplets = [row for row in reader if len(row) == 3]
    except (FileNotFoundError, StopIteration):
        print(f"错误: 无法读取或解析CSV文件 {args.csv_path}")
        return

    if not all_triplets:
        print("CSV文件中没有找到三元组数据。")
        return

    random.shuffle(all_triplets)

    # 使用与finetune.py相同的逻辑来划分验证集
    if args.val_split > 0 and args.val_split < 1.0:
        split_idx = int(len(all_triplets) * (1 - args.val_split))
        eval_triplets = all_triplets[split_idx:]
    else:
        # 如果不划分，使用所有数据进行评估
        eval_triplets = all_triplets
        print("警告: 未设置val_split，将使用所有数据进行评估，这可能需要较长时间。")

    if not eval_triplets:
        print("未能划分出评估数据集。")
        return

    # 从三元组中获取唯一的图片路径和标签
    eval_image_paths = sorted(list(set([path for triplet in eval_triplets for path in triplet])))
    class_names = [os.path.basename(os.path.dirname(p)) for p in eval_image_paths]
    unique_class_names = sorted(list(set(class_names)))
    class_to_idx = {name: i for i, name in enumerate(unique_class_names)}
    eval_labels = [class_to_idx[name] for name in class_names]

    print(f"已准备好评估集，包含 {len(eval_image_paths)} 张图片，共 {len(unique_class_names)} 个类别。")

    # 3. 提取所有评估图片的特征
    print("\n--- 正在提取特征向量 ---")
    all_embeddings = []
    valid_labels = []

    for i, path in enumerate(tqdm(eval_image_paths, desc="提取特征")):
        embedding = extract_features(path, model, head, transform, device)
        if embedding is not None:
            all_embeddings.append(embedding)
            valid_labels.append(eval_labels[i])

    if not all_embeddings:
        print("未能为任何图片成功提取特征。")
        return

    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.tensor(valid_labels, device=device)

    print(f"成功提取了 {len(labels_tensor)} 个特征向量。")

    # 4. 计算 mAP
    print("\n--- 正在计算 Mean Average Precision (mAP) ---")

    # 在这种场景下，查询集和索引集是相同的
    queries = embeddings_tensor
    query_labels = labels_tensor
    references = embeddings_tensor
    reference_labels = labels_tensor

    # 初始化mAP计算器
    map_calculator = metrics.MeanAveragePrecision(k=None) # k=None 表示考虑所有排名

    # 计算mAP
    mAP = map_calculator.get_accuracy(
        queries=queries,
        references=references,
        query_labels=query_labels,
        reference_labels=reference_labels,
        ref_includes_query=True # 重要：告诉计算器索引集包含了查询本身
    )

    print("\n--- 评估完成 ---")
    print(f"模型 '{os.path.basename(args.model_dir)}' 的 mAP 分数是: {mAP:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用微调后的模型计算mAP分数。")
    parser.add_argument("--model_dir", type=str, required=True, help="包含微调模型（LoRA适配器和head_weights.pth）的目录。")
    parser.add_argument("--csv_path", type=str, required=True, help="定义了图像关系的CSV文件路径。")
    parser.add_argument("--model_name", type=str, default="mobilenetv4_conv_medium.e500_r224_in1k", help="用于微调的timm基础模型名称。")
    parser.add_argument("--output_dim", type=int, default=256, help="投影头的输出维度。")
    parser.add_argument("--val_split", type=float, default=0.1, help="从CSV中用于评估的比例(0.0到1.0之间)。")

    args = parser.parse_args()
    calculate_map(args)
