import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import time

# 1. 配置参数
DATA_DIR = "data"
BATCH_SIZE = 8       # 一次训练几张图 (根据电脑性能调整)
NUM_EPOCHS = 5       # 训练几轮 (因为是假数据，5轮足够收敛)
LEARNING_RATE = 0.001

# 检测设备 (优先使用 GPU/MPS，没有则用 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 使用设备进行训练: {device}")

def train():
    # 2. 数据增强与加载 (Data Augmentation & Loading)
    # 简历亮点：这里实现了 "Data Augmentation to handle class imbalance"
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(), # 随机翻转 (增强)
            transforms.RandomRotation(10),     # 随机旋转 (增强)
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 标准化
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 读取文件夹中的数据
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes # ['bacterial', 'normal', 'viral']
    
    print(f"📦 类别: {class_names}")
    print(f"📊 训练集数量: {dataset_sizes['train']}, 验证集数量: {dataset_sizes['val']}")

    # 3. 加载预训练模型 ResNet50
    print("🧠 正在加载 ResNet50 预训练模型...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # 4. 修改最后全连接层 (Fine-tuning)
    # ResNet50 原本输出 1000 类，我们要改成 3 类 (Normal, Viral, Bacterial)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 5. 开始训练循环
    print("🔥 开始训练...")
    since = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'-' * 10)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')

        # 每个 Epoch 都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 (只在训练阶段)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    time_elapsed = time.time() - since
    print(f'✅ 训练完成！耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # 6. 保存模型
    torch.save(model.state_dict(), 'medical_resnet.pth')
    print("💾 模型已保存为 medical_resnet.pth")

if __name__ == '__main__':
    train()