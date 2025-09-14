import os
import random
import argparse
import csv
from tqdm import tqdm

def create_triplets(data_path, output_csv, time_window=1):
    """
    扫描按电影分类的图像目录，并创建用于训练的三元组。

    期望的目录结构:
    - data_path/
      - movie_1/
        - frame_001.jpg
        - frame_002.jpg
        ...
      - movie_2/
        - frame_001.jpg
        ...

    一个三元组由 (anchor, positive, negative) 组成:
    - anchor: 电影中的一张随机图片。
    - positive: 来自同一部电影的、与anchor在时间上相近的图片。
    - negative: 来自不同电影的一张随机图片。
    """
    # 获取所有电影的目录名
    movies = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    if len(movies) < 2:
        raise ValueError("至少需要两个电影目录才能创建三元组。")

    # 读取所有电影及其排序后的图片文件路径
    all_movie_files = {}
    print("正在扫描图片文件...")
    for movie in tqdm(movies, desc="扫描电影目录"):
        movie_path = os.path.join(data_path, movie)
        # 过滤并排序图片文件
        files = sorted([os.path.join(movie_path, f) for f in os.listdir(movie_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if files:
            all_movie_files[movie] = files

    triplets = []
    print("正在生成三元组...")

    movie_list = list(all_movie_files.keys())

    # 遍历所有图片，为每一张图片创建一个三元组
    for movie_name, image_paths in tqdm(all_movie_files.items(), desc="处理电影"):
        num_images = len(image_paths)
        if num_images < 2:
            continue  # 无法在该电影中形成正样本对

        for i in range(num_images):
            anchor_path = image_paths[i]

            # --- 正样本选择 (时间邻近性) ---
            # 在一个时间窗口内选择正样本
            start = max(0, i - time_window)
            end = min(num_images, i + time_window + 1)

            positive_candidates = []
            for j in range(start, end):
                if i == j:  # 图片不能与自身配对
                    continue
                positive_candidates.append(image_paths[j])

            if not positive_candidates:
                # 对于只有一张图片的电影，或者窗口设置问题，跳过
                continue

            positive_path = random.choice(positive_candidates)

            # --- 负样本选择 ---
            # 随机选择一个不同的电影
            negative_movie_name = movie_name
            while negative_movie_name == movie_name:
                negative_movie_name = random.choice(movie_list)

            # 从那个不同的电影中随机选择一张图片
            if all_movie_files[negative_movie_name]:
                negative_path = random.choice(all_movie_files[negative_movie_name])
                triplets.append([anchor_path, positive_path, negative_path])

    print(f"成功生成 {len(triplets)} 个三元组。")
    print(f"正在将三元组保存至 {output_csv}...")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['anchor', 'positive', 'negative'])
        writer.writerows(triplets)

    print("完成。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从电影截图中创建训练三元组。")
    parser.add_argument("--data_path", type=str, required=True, help="包含电影子文件夹的根目录路径。")
    parser.add_argument("--output_csv", type=str, default="triplets.csv", help="输出的CSV文件路径。")
    parser.add_argument("--time_window", type=int, default=2, help="在锚点图片前后选取正样本的时间窗口大小。例如, 2 表示在 [i-2, i-1, i+1, i+2] 中选择。")

    args = parser.parse_args()

    # 检查tqdm是否安装
    try:
        from tqdm import tqdm
    except ImportError:
        print("错误: tqdm 库未安装。请运行 'pip install tqdm' 来安装。")
        exit(1)

    create_triplets(args.data_path, args.output_csv, args.time_window)
