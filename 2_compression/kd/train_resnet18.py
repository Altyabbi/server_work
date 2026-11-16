import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm
import os

# ================= 1. 配置参数 =================
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
EPOCHS = 200           
BATCH_SIZE = 128
LEARNING_RATE = 0.05
WEIGHT_DECAY = 5e-4
SAVE_PATH = "./weights/resnet18_baseline.pth"



# ================== 2. 数据准备 ==================
def get_data_loaders():
    print("==> Preparing CIFAR-100 data...")
    # CIFAR-100 的官方均值和方差
    norm_mean = (0.5071, 0.4867, 0.4408)
    norm_std = (0.2675, 0.2565, 0.2761)

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


    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return trainloader, testloader


# ================== 3. 模型构建 ==================
def get_model():
    print("==> Building ResNet-18 for CIFAR-100...")
    # 加载 ResNet-18 结构
    model = resnet18(weights=None)
    
    # 1. 修改第一层卷积：适应 32x32 小图
    # 将 7x7 (stride=2) 改为 3x3 (stride=1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. 移除 MaxPool：防止特征图过小
    model.maxpool = nn.Identity()
    
    # 3. 修改全连接层：适应 CIFAR-100 的 100 个类别
    model.fc = nn.Linear(model.fc.in_features, 100)
    
    return model.to(DEVICE)



# ================== 4. 训练主循环 ==================
def train():
    trainloader, testloader = get_data_loaders()
    model = get_model()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
    # 余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0

    print(f"Start training on {DEVICE}...")

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