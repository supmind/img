import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# 导入我们自己编写的模块
from feature_extractor import ImageFeatureExtractor
from triplet_dataset import TripletDataset

def finetune(args):
    """主微调函数"""
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载模型和图像变换
    # 初始化我们之前创建的特征提取器
    # 注意：这里加载的是预训练模型，还没有经过微调
    print("正在加载预训练模型...")
    extractor = ImageFeatureExtractor(output_dim=256)
    model = extractor.model
    head = extractor.projection_head

    # 将模型和投影层移动到指定设备
    model.to(device)
    head.to(device)

    # 关键步骤：冻结主干网络的权重
    # 我们只训练新添加的投影层
    print("正在冻结主干网络参数...")
    for param in model.parameters():
        param.requires_grad = False

    # 确保投影层的参数是可训练的
    for param in head.parameters():
        param.requires_grad = True

    # 设置模型为训练模式（主要对head有效，比如dropout等层）
    head.train()

    # 3. 准备数据集和数据加载器
    print("正在准备数据集...")
    # 从特征提取器中获取配套的图像变换
    transform = extractor.transform
    train_dataset = TripletDataset(csv_file=args.csv_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 4. 定义损失函数和优化器
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    # 优化器只关注需要训练的投影层的参数
    optimizer = optim.AdamW(head.parameters(), lr=args.learning_rate)

    print("--- 开始微调 ---")
    # 5. 训练循环
    for epoch in range(args.epochs):
        running_loss = 0.0
        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for i, (anchor_img, positive_img, negative_img) in enumerate(progress_bar):
            # 将数据移动到设备上
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            # 分别获取三张图片经过主干网络和投影层的特征向量
            anchor_emb = head(model(anchor_img))
            positive_emb = head(model(positive_img))
            negative_emb = head(model(negative_img))

            # 计算损失
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)

            # 反向传播
            loss.backward()

            # 更新权重
            optimizer.step()

            # 记录并显示损失
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{running_loss / (i + 1):.4f}'})

    print("--- 微调结束 ---")

    # 6. 保存微调后的投影层权重
    print(f"正在保存微调后的投影层权重至: {args.output_weights_path}")
    torch.save(head.state_dict(), args.output_weights_path)
    print("保存成功。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微调特征提取模型的投影层。")
    parser.add_argument("--csv_path", type=str, required=True, help="包含三元组数据的CSV文件路径。")
    parser.add_argument("--epochs", type=int, default=5, help="训练的轮次。")
    parser.add_argument("--batch_size", type=int, default=32, help="每个批次的样本数。")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="优化器的学习率。")
    parser.add_argument("--margin", type=float, default=1.0, help="TripletLoss的边际值。")
    parser.add_argument("--output_weights_path", type=str, default="finetuned_head.pth", help="保存微调后权重的路径。")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载器使用的工作进程数。")

    args = parser.parse_args()
    finetune(args)
