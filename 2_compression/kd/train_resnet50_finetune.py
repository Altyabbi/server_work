import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import os

# ================== 1. 参数配置 ==================
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
LEARNING_RATE = 0.01  # 微调时学习率通常设得小一些
EPOCHS = 100          # 微调 100 轮足以达到高精度
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
SAVE_PATH = "./weights/teacher_finetune_resnet50_cifar100_best.pth" # 教师模型保存路径

# ================== 2. 数据准备 ==================
def get_dataloaders():
    print(">>> 正在准备 CIFAR-100 数据...")
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100 官方均值方差
    
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
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# ================== 3. 模型构建 (Teacher) ==================
def get_teacher_model():
    print(">>> 构建教师模型 (ResNet-50) [ImageNet 预训练 + CIFAR-100 微调]...")
    
    # 【关键】加载 ImageNet 预训练权重
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # 针对 CIFAR-100 调整 ResNet 结构
    # 由于图像分辨率小，需要修改第一层卷积和移除最大池化层，防止信息过早丢失。
    # 尽管改变了预训练模型的conv1结构，但由于后续全连接层会重新训练，且整个网络进行微调，仍能有效利用预训练的特征提取能力。
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # 修改全连接层以匹配 CIFAR-100 的 100 个类别
    model.fc = nn.Linear(model.fc.in_features, 100)
    
    return model.to(DEVICE)

# ================== 4. 训练与评估函数 ==================
def train_teacher():
    trainloader, testloader = get_dataloaders()
    model = get_teacher_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    # MultiStepLR 学习率调度器，在指定里程碑衰减学习率
    milestones = [int(EPOCHS*0.5), int(EPOCHS*0.75)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_acc = 0.0

    print(f"开始训练教师模型，总 Epochs: {EPOCHS}，设备: {DEVICE}")

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
            
            loop.set_postfix(loss=running_loss/len(trainloader), acc=100.*correct/total, lr=optimizer.param_groups[0]['lr'])

        scheduler.step()

        acc = evaluate_model(model, testloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | LR: {current_lr:.5f} | Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"✅ 教师模型已保存！最佳精度: {best_acc:.2f}%")

    print(f"教师模型训练完成。最终最佳精度: {best_acc:.2f}%")

def evaluate_model(model, testloader):
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
    train_teacher()