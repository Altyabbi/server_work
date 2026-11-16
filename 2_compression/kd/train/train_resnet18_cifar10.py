import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
from tqdm import tqdm
import os

# ================= 1. 参数配置 =================
# 如果你是训练 Teacher，建议选 ResNet-50
# 如果你是训练 Baseline (待对比的学生)，选 ResNet-18
MODEL_TYPE = "resnet18"  # 可选: "resnet18" 或 "resnet50"
EPOCHS = 60              # 时间紧的话跑 50-60 个 epoch 即可
BATCH_SIZE = 128
LEARNING_RATE = 0.05     # 初始学习率
DEVICE = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
SAVE_PATH = f"./weights/{MODEL_TYPE}_cifar10.pth"

# ================= 2. 数据准备 =================
def get_data_loaders():
    print("==> Preparing data...")
    # CIFAR-10 标准均值和方差
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.2023, 0.1994, 0.2010)

    # 训练集增强：随机裁剪 + 水平翻转
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 测试集仅归一化
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return trainloader, testloader

# ================= 3. 模型构建 (关键步骤) =================
def get_model(model_name="resnet18"):
    print(f"==> Building model: {model_name}...")
    
    # 加载 torchvision 原版模型
    if model_name == "resnet18":
        model = resnet18(weights=None)
    elif model_name == "resnet50":
        model = resnet50(weights=None)
    
    # 【报告重点】修改模型以适应 CIFAR-10 的 32x32 尺寸
    # 原版 ResNet 第一层是 7x7 卷积, stride=2, padding=3 -> 会导致特征图过早变小
    # 修改为 3x3 卷积, stride=1, padding=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 【报告重点】移除第一层的 MaxPool
    # CIFAR-10 图片太小，不需要在此处做池化
    model.maxpool = nn.Identity()
    
    # 修改全连接层，输出 10 类
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model.to(DEVICE)

# ================= 4. 训练与验证流程 =================
def train():
    trainloader, testloader = get_data_loaders()
    model = get_model(MODEL_TYPE)

    criterion = nn.CrossEntropyLoss()
    # 使用 SGD + Momentum + Weight Decay
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    # 余弦退火学习率调度器 (会让后期精度提升更明显)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 进度条
        loop = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix(loss=running_loss/len(trainloader), acc=100.*correct/total)

        scheduler.step()

        # 每个 Epoch 结束后验证一次
        acc = evaluate(model, testloader)
        print(f"Epoch {epoch+1} Results | Train Loss: {running_loss/len(trainloader):.4f} | Test Acc: {acc:.2f}%")

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ Saving best model to {SAVE_PATH} ({best_acc:.2f}%)")

    print(f"Training Finished. Best Accuracy: {best_acc:.2f}%")

def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

if __name__ == "__main__":
    train()