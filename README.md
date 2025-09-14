# 图像特征提取与微调工具

本项目提供了一套完整的工具，用于从图像中提取特征向量，并通过三元组损失（Triplet Loss）对模型进行微调，以适应特定领域的图像相似度任务（例如，电影截图反查）。

## 项目文件结构

- `feature_extractor.py`: 定义了核心的 `ImageFeatureExtractor` 类，负责加载模型和提取特征。
- `prepare_data.py`: 数据准备脚本，用于从您的图像数据中生成训练所需的三元组CSV文件。
- `triplet_dataset.py`: 定义了PyTorch的 `Dataset` 类，用于加载三元组数据。
- `finetune.py`: 核心的微调训练脚本。
- `main.py`: 一个简单的示例，演示如何使用**未经微调**的模型提取特征。

---

## 使用流程

请按照以下步骤来微调模型并使用它。

### 第一步：准备您的图像数据

您需要将您的图像数据按照类别（例如，电影名称）整理到不同的子文件夹中。本工具的根数据目录结构应如下所示：

```
/path/to/your/data/
├── movie_A/
│   ├── 00-00-00.jpg
│   ├── 00-02-00.jpg
│   └── ...
├── movie_B/
│   ├── 00-00-00.jpg
│   ├── 00-02-00.jpg
│   └── ...
└── ...
```

**重要提示**：请确保每个电影文件夹内的图片文件名是**按时间顺序排列**的（例如，使用 `HH-MM-SS.jpg` 或 `frame_0001.jpg` 格式），这样脚本才能正确地找到时间上相邻的图片作为正样本。

### 第二步：生成训练三元组

打开终端，运行 `prepare_data.py` 脚本来扫描您的数据并创建 `triplets.csv` 文件。

```bash
python3 prepare_data.py --data_path /path/to/your/data/
```

- `--data_path`: **必须提供**，指向您在第一步中准备好的数据根目录。
- `--output_csv`: (可选) 输出的CSV文件名，默认为 `triplets.csv`。
- `--time_window`: (可选) 选取正样本的时间窗口大小，默认为2。

运行结束后，您将在当前目录下看到一个 `triplets.csv` 文件。

### 第三步：执行微调训练

现在，使用上一步生成的 `triplets.csv` 文件来运行微调脚本 `finetune.py`。

```bash
python3 finetune.py --csv_path triplets.csv --epochs 5 --batch_size 32 --learning_rate 0.0001
```

- `--csv_path`: **必须提供**，指向 `triplets.csv` 文件。
- `--epochs`: (可选) 训练轮次，默认为5。根据您的数据量和期望的效果，可以适当调整。
- `--batch_size`: (可选) 批处理大小，默认为32。如果您的内存/显存不足，可以减小此值。
- `--learning_rate`: (可选) 学习率，默认为 `1e-4`。
- `--output_weights_path`: (可选) 保存微调后权重的文件名，默认为 `finetuned_head.pth`。

这个过程可能会花费一些时间，具体取决于您的数据量和硬件性能。训练结束后，您将得到一个名为 `finetuned_head.pth` 的权重文件。

### 第四步：使用微调后的模型

现在，您可以像之前一样使用 `ImageFeatureExtractor`，但在初始化时，需要将刚刚生成的权重文件路径传递给它。

您可以修改 `main.py` 或创建一个新的脚本来进行推理。

**示例代码 (`inference_example.py`):**

```python
from feature_extractor import ImageFeatureExtractor
import torch

# 1. 初始化特征提取器，并加载微调过的权重
print("正在初始化微调过的特征提取器...")
extractor = ImageFeatureExtractor(weights_path='finetuned_head.pth')

# 2. 提取一张图片的特征
# 这里的特征向量会比未经微调的模型更懂您的数据
# 请将下面的路径替换为您真实存在的图片路径
image_path = '/path/to/your/data/movie_A/00-00-00.jpg'
print(f"正在从 {image_path} 提取特征...")
features = extractor.extract(image_path)

if features is not None:
    print("特征提取成功!")
    print(f"向量形状: {features.shape}")
else:
    print("特征提取失败。")

```

现在，`extractor` 提取出的特征向量已经针对您的电影截图数据进行过优化，用于相似度查询时，效果会比通用模型更好。

---

## 依赖库

请确保您已安装以下Python库：
- `torch`
- `torchvision`
- `timm`
- `Pillow`
- `tqdm`

可以使用pip进行安装: `pip install torch torchvision timm Pillow tqdm`
(如果您的服务器没有GPU，请参照之前的说明安装PyTorch的CPU版本)
