import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

# --- 配置 ---
DATA_DIR = "data"
CLASSES = ["normal", "viral", "bacterial"]
SETS = {"train": 40, "val": 10}  # 训练集每类40张，验证集每类10张 (足够跑通流程了)
IMG_SIZE = (224, 224) # ResNet 标准输入尺寸

def create_mock_image(class_name):
    """
    创建一个假装是 X 光的图片：
    - Normal: 只有噪点
    - Viral: 噪点 + 模糊的圆形 (模拟病毒性肺炎特征)
    - Bacterial: 噪点 + 明亮的矩形 (模拟细菌性肺炎特征)
    """
    # 1. 创建灰色背景 + 随机噪点
    arr = np.random.randint(50, 150, IMG_SIZE, dtype=np.uint8)
    img = Image.fromarray(arr, mode='L') # 'L' 模式表示黑白灰度图
    draw = ImageDraw.Draw(img)
    
    # 2. 根据类别添加“病灶”
    if class_name == "viral":
        # 病毒性：画一个模糊的圆
        x = random.randint(50, 150)
        y = random.randint(50, 150)
        r = random.randint(20, 40)
        # 画个亮一点的圆
        draw.ellipse([x-r, y-r, x+r, y+r], fill=random.randint(160, 200), outline=None)
        # 稍微模糊一下，模拟“毛玻璃影”
        img = img.filter(ImageFilter.GaussianBlur(radius=3))
        
    elif class_name == "bacterial":
        # 细菌性：画一个明显的矩形
        x = random.randint(50, 150)
        y = random.randint(50, 150)
        w = random.randint(30, 50)
        h = random.randint(30, 50)
        # 画个很亮的矩形
        draw.rectangle([x, y, x+w, y+h], fill=random.randint(200, 255), outline=None)
    
    # Normal 啥也不加，就是纯背景
    
    # 转回 RGB (因为大多数预训练模型需要 3 通道输入)
    return img.convert("RGB")

def main():
    print(f"🏥 开始生成模拟医疗影像数据...")
    
    for set_name, count in SETS.items():
        for class_name in CLASSES:
            # 创建文件夹: data/train/normal, data/val/viral 等
            dir_path = os.path.join(DATA_DIR, set_name, class_name)
            os.makedirs(dir_path, exist_ok=True)
            
            print(f"   正在生成 {set_name}/{class_name} ({count}张)...")
            
            for i in range(count):
                img = create_mock_image(class_name)
                # 保存文件
                file_path = os.path.join(dir_path, f"{class_name}_{i}.jpg")
                img.save(file_path)
                
    print(f"\n✅ 数据生成完毕！存放位置: {os.path.abspath(DATA_DIR)}")
    print("结构如下：")
    print(f"  {DATA_DIR}/train/ (normal, viral, bacterial)")
    print(f"  {DATA_DIR}/val/   (normal, viral, bacterial)")

if __name__ == "__main__":
    main()