import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from tqdm import tqdm
import os

# ================= 配置参数 =================
DEVICE = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu')
EPOCHS = 50           # CIFAR-10 比较容易收敛，50个epoch通常能达到90%+
BATCH_SIZE = 128
LEARNING_RATE = 0.05
WEIGHT_DECAY = 5e-4
SAVE_PATH = "./weights/teacher_resnet50/resnet50_cifar10.pth"
# ============================================

def get_data_loaders():
    # CIFAR-10 专用的均值和标准差
    # Mean: (0.4914, 0.4822, 0.4465), Std: (0.2023, 0.1994, 0.2010)
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 改为 CIFAR10 数据集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return trainloader, testloader

def get_model():
    # 加载 ResNet-50 骨架
    model = resnet50(weights=None) 
    
    # 1. 修改第一层卷积：适应 32x32 输入
    # 把 kernel_size 改为 3x3, stride 改为 1, padding 改为 1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. 移除第一层 MaxPool：避免信息过早丢失
    model.maxpool = nn.Identity()
    
    # 3. 修改全连接层：适应 CIFAR-10 的 10 个类别
    # ResNet-50 的 fc 输入是 2048 维
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model.to(DEVICE)

def train():
    trainloader, testloader = get_data_loaders()
    model = get_model()
    
    criterion = nn.CrossEntropyLoss()
    # 优化器设置
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    print(f"Start training ResNet-50 on CIFAR-10 using {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
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

        # 验证
        acc = evaluate(model, testloader)
        print(f"Epoch {epoch+1} | Loss: {running_loss/len(trainloader):.4f} | Test Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            
    print(f"Done! Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {SAVE_PATH}")

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