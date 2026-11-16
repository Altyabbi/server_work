import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from tqdm import tqdm
import os

# ================== 1. 参数配置 ==================
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
EPOCHS = 200           # MobileNet 收敛稍慢
BATCH_SIZE = 128
LEARNING_RATE = 0.05   # 初始学习率
WEIGHT_DECAY = 4e-5    # MobileNet 参数少，正则化不宜过大
MOMENTUM = 0.9
SAVE_PATH = "./weights/mobilenetv2_baseline.pth"

# ================== 2. 数据准备 ==================
def get_dataloaders():
    print(">>> 正在准备 CIFAR-100 数据...")
    # CIFAR-100 官方均值和方差
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    
    # 这里的 num_workers 可以根据你的 CPU 核心数调整，通常 2 或 4
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# ================== 3. 模型构建 ==================
def get_mobilenet_baseline():
    print(">>> 构建 MobileNetV2 Baseline (From Scratch)...")
    # 不加载预训练权重，因为这是 Baseline
    model = mobilenet_v2(weights=None)
    
    # 1. 调整第一层卷积以适应 32x32 输入
    # 原版是 stride=2，会导致信息丢失太快。改为 stride=1。
    # MobileNetV2 的第一层结构在 features[0][0]
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. 调整分类器以适应 CIFAR-100 (100类)
    # 原版 classifier[1] 是 Linear(1280, 1000)
    model.classifier[1] = nn.Linear(model.last_channel, 100)
    
    return model.to(DEVICE)



# ================== 4. 训练主循环 ==================
def train():
    trainloader, testloader = get_dataloaders()
    model = get_mobilenet_baseline()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # 使用余弦退火学习率调度器，这是训练轻量化网络的标准配置
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    print(f"开始训练 MobileNetV2 Baseline，设备: {DEVICE}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(trainloader, desc=f"Ep [{epoch+1}/{EPOCHS}]", leave=False)
        
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

        # 验证
        acc = evaluate(model, testloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | LR: {current_lr:.5f} | Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ 保存最佳模型 ({best_acc:.2f}%)")

    print(f"训练结束。Baseline 最佳精度: {best_acc:.2f}%")

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