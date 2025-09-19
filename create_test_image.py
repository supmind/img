import numpy as np
from PIL import Image

def create_dummy_image(path="dummy_test_image.jpg"):
    """创建一个简单的虚拟图片用于测试。"""
    # 创建一个 224x224 的随机彩色图片
    array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(array, 'RGB')
    img.save(path)
    print(f"虚拟测试图片已保存至: {path}")
    return path

if __name__ == "__main__":
    create_dummy_image()
