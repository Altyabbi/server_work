import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm  # 用于显示进度条
import os

# =================配置参数=================
DEVICE = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
EPOCHS = 60           # 只有一天时间，跑60-100个epoch差不多了
BATCH_SIZE = 128      # 显存不够可以调小，比如 64 或 32
LEARNING_RATE = 0.05  # 初始学习率
WEIGHT_DECAY = 5e-4   # 正则化系数
SAVE_PATH = "./weights/teacher_resnet50/resnet50_cifar100.pth"
# ==========================================

def get_data_loaders():
    # CIFAR-100 的标准数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return trainloader, testloader

def get_model():
    # 1. 加载标准 ResNet-50，不使用预训练权重（因为我们要改结构，且类别不同）
    # 如果你想用预训练权重加速收敛，可以用 weights=ResNet50_Weights.DEFAULT
    # 但记得预训练是针对 ImageNet (224x224) 的，效果不一定直接好
    model = resnet50(weights=None) 
    
    # 2. 【关键步骤】修改第一层卷积
    # 原版 ResNet 是为 ImageNet (224x224) 设计的，第一层是 7x7 卷积 stride=2
    # 对于 CIFAR-100 (32x32)，这样做会导致信息丢失太快。
    # 这里改成 3x3 卷积，stride=1，保留更多特征。
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 3. 【关键步骤】移除第一层的 MaxPool
    # 同样是为了防止特征图尺寸变得太小
    model.maxpool = nn.Identity()
    
    # 4. 修改全连接层 (fc) 以匹配 CIFAR-100 的 100 个类别
    model.fc = nn.Linear(model.fc.in_features, 100)
    
    return model.to(DEVICE)

def train():
    trainloader, testloader = get_data_loaders()
    model = get_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    # 使用余弦退火学习率调度器，有助于在后期提升精度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    print(f"Start training ResNet-50 on CIFAR-100 using {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练循环
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

        # 验证循环
        acc = evaluate(model, testloader)
        print(f"Epoch {epoch+1} | Train Loss: {running_loss/len(trainloader):.4f} | Test Acc: {acc:.2f}%")

        # 保存最好的模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved Best Model ({best_acc:.2f}%)")

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