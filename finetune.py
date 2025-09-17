import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import csv
import random
import os
from peft import LoraConfig, get_peft_model


# 导入我们自己编写的模块
from feature_extractor import ImageFeatureExtractor
from triplet_dataset import TripletDataset

def find_lora_target_modules(model, lora_rank):
    """
    自动查找模型中适合应用LoRA的模块。
    此函数遍历主干模型中的所有模块，并返回一个包含所有
    `nn.Linear` 和兼容的 `nn.Conv2d` 模块名称的列表。
    兼容性意味着对于分组卷积，LoRA的rank必须能被groups数整除。
    """
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            target_modules.append(name)
        elif isinstance(module, nn.Conv2d):
            if lora_rank % module.groups == 0:
                target_modules.append(name)
    print(f"找到 {len(target_modules)} 个可应用LoRA的目标模块。")
    return target_modules

def validate(model, head, val_loader, device):
    """在验证集上评估模型"""
    head.eval()  # 设置为评估模式
    correct = 0
    total = 0

    # 使用L2距离
    pdist = nn.PairwiseDistance(p=2)

    with torch.no_grad():
        # 为验证循环也添加一个进度条
        progress_bar = tqdm(val_loader, desc="[Validating]")
        for anchor_img, positive_img, negative_img in progress_bar:
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            anchor_emb = head(model(anchor_img))
            positive_emb = head(model(positive_img))
            negative_emb = head(model(negative_img))

            # 计算距离
            dist_pos = pdist(anchor_emb, positive_emb)
            dist_neg = pdist(anchor_emb, negative_emb)

            # 如果正样本距离小于负样本距离，则认为是正确的
            correct += (dist_pos < dist_neg).sum().item()
            total += anchor_img.size(0)

            # 在进度条上显示实时准确率
            progress_bar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})

    accuracy = 100 * correct / total
    return accuracy

def finetune(args):
    """主微调函数"""
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载模型和图像变换
    print(f"正在加载预训练模型: {args.model_name}...")
    extractor = ImageFeatureExtractor(model_name=args.model_name, output_dim=256)
    model = extractor.model
    head = extractor.projection_head
    transform = extractor.transform

    model.to(device)
    head.to(device)

    # 应用PEFT (LoRA)配置
    print("正在应用PEFT (LoRA)配置...")
    # 自动查找可应用LoRA的模块
    target_modules = find_lora_target_modules(model, args.lora_r)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    print("PEFT模型创建成功:")
    model.print_trainable_parameters()

    # 确保投影头仍然是可训练的
    for param in head.parameters():
        param.requires_grad = True

    # 3. 准备数据集和数据加载器
    print("正在准备数据集...")

    # 从CSV加载所有数据
    try:
        with open(args.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 兼容没有表头的情况
            try:
                first_row = next(reader)
                all_triplets = [first_row] + [row for row in reader if len(row) == 3]
                # 检查表头是否是路径，如果不是，则认为它不是表头
                if not all_triplets[0][0].lower().endswith(('.jpg', '.jpeg', '.png')):
                     all_triplets = all_triplets[1:] # 真正的表头，跳过
            except StopIteration:
                 all_triplets = []

    except FileNotFoundError:
        print(f"错误: CSV文件未找到于 {args.csv_path}")
        return

    if not all_triplets:
        print(f"错误: 在 {args.csv_path} 中没有找到任何三元组数据。")
        return

    random.shuffle(all_triplets)

    val_loader = None
    if args.val_split > 0 and args.val_split < 1.0:
        split_idx = int(len(all_triplets) * (1 - args.val_split))
        train_triplets = all_triplets[:split_idx]
        val_triplets = all_triplets[split_idx:]

        if not val_triplets:
             print("警告: 验证集为空，请检查val_split值或数据量。将使用所有数据进行训练。")
             train_triplets = all_triplets
        else:
            print(f"数据集划分: {len(train_triplets)} 训练样本, {len(val_triplets)} 验证样本")
            val_dataset = TripletDataset(triplets=val_triplets, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        train_dataset = TripletDataset(triplets=train_triplets, transform=transform)

    else:
        print(f"使用所有 {len(all_triplets)} 样本进行训练，无验证集。")
        train_dataset = TripletDataset(triplets=all_triplets, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 4. 定义损失函数和优化器
    loss_fn = nn.TripletMarginLoss(margin=args.margin)

    # 将投影头和LoRA适配器的可训练参数一起传给优化器
    trainable_params = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.learning_rate)
    print(f"优化器将训练 {len(trainable_params)} 个参数张量。")

    print("--- 开始微调 ---")
    best_val_acc = 0.0

    # 5. 训练循环
    for epoch in range(args.epochs):
        head.train() # 设置为训练模式
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")

        for i, batch in enumerate(progress_bar):
            # 健壮性检查，以防dataloader返回None
            if batch is None: continue
            anchor_img, positive_img, negative_img = batch
            anchor_img, positive_img, negative_img = anchor_img.to(device), positive_img.to(device), negative_img.to(device)

            optimizer.zero_grad()

            anchor_emb = head(model(anchor_img))
            positive_emb = head(model(positive_img))
            negative_emb = head(model(negative_img))

            loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{running_loss / (i + 1):.4f}'})

        avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # 验证步骤
        if val_loader:
            val_acc = validate(model, head, val_loader, device)
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Avg Train Loss: {avg_loss:.4f}, "
                  f"Validation Accuracy: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"✨ 新的最佳验证准确率: {best_val_acc:.2f}%. 正在保存模型至 {args.output_dir}...")
                os.makedirs(args.output_dir, exist_ok=True)
                model.save_pretrained(args.output_dir)
                torch.save(head.state_dict(), os.path.join(args.output_dir, "head_weights.pth"))
        else:
             print(f"Epoch {epoch+1}/{args.epochs} - Avg Train Loss: {avg_loss:.4f}")

    print("--- 微调结束 ---")

    # 如果没有验证集，在训练结束后保存最终模型
    if not val_loader:
        print(f"正在保存最终模型权重至: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        torch.save(head.state_dict(), os.path.join(args.output_dir, "head_weights.pth"))

    print("任务完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微调特征提取模型的投影层，并带有验证功能。")
    parser.add_argument("--model_name", type=str, default="mobilenetv3_large_100", help="要使用的timm模型名称。")
    parser.add_argument("--csv_path", type=str, required=True, help="包含所有三元组数据的CSV文件路径。")
    parser.add_argument("--epochs", type=int, default=5, help="训练的轮次。")
    parser.add_argument("--batch_size", type=int, default=32, help="每个批次的样本数。")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="优化器的学习率。")
    parser.add_argument("--margin", type=float, default=1.0, help="TripletLoss的边际值。")
    parser.add_argument("--output_dir", type=str, default="finetuned_model", help="保存微调后模型权重的目录。")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载器使用的工作进程数。")
    parser.add_argument("--val_split", type=float, default=0.1, help="从CSV中用于验证的比例(0.0到1.0之间)。设为0则禁用验证。")

    # LoRA 相关参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA的秩。")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA的alpha值。")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA的dropout率。")

    args = parser.parse_args()
    if args.val_split < 0 or args.val_split >= 1.0:
        # 如果val_split为0.0，则不进行验证，这是允许的
        if args.val_split != 0.0:
            raise ValueError("val_split 的值必须在 [0.0, 1.0) 范围内。")

    finetune(args)
