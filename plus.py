# main.py 主执行文件
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets as tv_datasets
from torch.utils.data import DataLoader
from shutil import copyfile
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


# ================== 系统配置 ==================
torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================== 数据准备 ==================
def prepare_data():
    """数据分割函数（只需执行一次）"""
    base_dir = "D:/py/xinxilun"
    source_dir = os.path.join(base_dir, "dataset/source")
    target_dir = os.path.join(base_dir, "dataset")

    # 创建目录结构
    for phase in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            os.makedirs(os.path.join(target_dir, phase, cls), exist_ok=True)

    def split_class(class_name):
        src = os.path.join(source_dir, class_name)
        files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)

        n = len(files)
        splits = {
            'train': files[:int(n * 0.7)],
            'val': files[int(n * 0.7):int(n * 0.85)],
            'test': files[int(n * 0.85):]
        }

        for phase, files in splits.items():
            for f in files:
                copyfile(os.path.join(src, f),
                         os.path.join(target_dir, phase, class_name, f))

    split_class('cats')
    split_class('dogs')


# ================== 模型定义 ==================
class AnimalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 输入尺寸：3x128x128
            nn.Conv2d(3, 32, 3, padding=1),  # 32x128x128
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x64x64

            nn.Conv2d(32, 64, 3, padding=1),  # 64x64x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x32x32

            nn.Conv2d(64, 128, 3, padding=1),  # 128x32x32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x16x16

            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.net(x)


# ================== 训练流程 ==================
def train_model(silent=False):
    # 数据预处理
    transform = {
        'train': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    def inverse_normalize(tensor):
        """逆归一化函数"""
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        return tensor * std[:, None, None] + mean[:, None, None]

    # 加载数据集（修正版本）
    data_dir = "D:/py/xinxilun/dataset"
    image_datasets = {
        phase: tv_datasets.ImageFolder(
            root=os.path.join(data_dir, phase),
            transform=transform[phase if phase == 'train' else 'val']
        )
        for phase in ['train', 'val']
    }


    dataloaders = {
        phase: DataLoader(image_datasets[phase],
                         batch_size=32,
                         shuffle=(phase == 'train'),
                         num_workers=0,
                          pin_memory=True if device.type == 'cuda' else False)
        for phase in ['train', 'val']
    }
    # 添加数据集统计
    print("\n=== 数据集统计 ===")
    print(f"训练集: {len(image_datasets['train'])}张")
    print(f"验证集: {len(image_datasets['val'])}张")
    print("==================\n")

    # 显示样本时修改为：
    sample, label = next(iter(dataloaders['train']))
    sample = inverse_normalize(sample)  # 逆归一化
    plt.imshow(sample[0].permute(1, 2, 0).cpu().numpy())
    plt.title("样本标签: 猫" if label[0] == 0 else "样本标签: 狗")
    plt.axis('off')
    plt.show()

    # 初始化模型
    model = AnimalClassifier().to(device)
    # 添加模型参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"=== 模型参数 ===\n总参数量: {total_params / 1e6:.2f}M\n")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    best_acc = 0.0
    history = {'loss': [], 'acc': []}

    for epoch in range(15):
        model.train()
        running_loss = 0.0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(image_datasets['train'])  # 修正变量
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        if not silent:
            print(f'Epoch {epoch + 1}/15 | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
# ================== 预测函数 ==================
def predict(image_path, show=True):
    """增强版预测函数"""
    # 预处理管道（与验证集相同）
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 异常处理机制
    try:
        # 自动处理不同格式图片
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            original_img = img.copy()
    except Exception as e:
        raise RuntimeError(f"图片加载失败: {str(e)}")

    # 预处理并添加批次维度
    input_tensor = transform(img).unsqueeze(0)

    # 设备兼容性加载模型
    model = AnimalClassifier()
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    except FileNotFoundError:
        raise FileNotFoundError("模型文件未找到，请先训练模型")
    model = model.to(device).eval()

    # 执行预测
    with torch.no_grad():
        inputs = input_tensor.to(device)
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        _, pred = torch.max(outputs, 1)

    # 可视化系统
    if show:
        plt.figure(figsize=(12, 5))

        # 原始图片显示（带逆归一化）
        plt.subplot(1, 2, 1)
        show_img = transforms.ToPILImage()(input_tensor.squeeze().cpu())
        plt.imshow(show_img)
        plt.title(f'预处理后图像\n尺寸: {show_img.size}')
        plt.axis('off')

        # 概率分布图
        plt.subplot(1, 2, 2)
        classes = ['猫', '狗']
        colors = ['#FF9999' if i == pred else '#66B2FF' for i in range(2)]
        bars = plt.bar(classes, probs.cpu().numpy(), color=colors)
        plt.ylim(0, 100)
        plt.ylabel('置信度 (%)')
        plt.title('类别概率分布')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.1f}%',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    return classes[pred.item()], probs[pred.item()].item()


# ================== 主执行入口 ==================
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    # 第一步：准备数据
    dataset_root = "D:/py/xinxilun/dataset"
    if not os.path.exists(os.path.join(dataset_root, "train")):
        prepare_data()
        print("数据分割完成！")

    # 第二步：训练模型
    if not os.path.exists('best_model.pth'):
        print("开始模型训练...")
        train_model(silent=False)
    else:
        print("检测到已存在训练好的模型")

    # 第三步：交互式预测
    while True:
        try:
            user_input = input("\n输入图片路径进行预测（或输入'exit'退出）: ").strip()
            if user_input.lower() == 'exit':
                break

            if not os.path.exists(user_input):
                print("错误：文件不存在")
                continue

            pred_class, confidence = predict(user_input)
            print(f"预测结果: {pred_class} | 置信度: {confidence:.1f}%")

        except Exception as e:
            print(f"发生错误: {str(e)}")
            continue