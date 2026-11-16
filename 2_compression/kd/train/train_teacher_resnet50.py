# train/train_teacher.py
"""
Train Teacher Model: ResNet-50 on CIFAR-100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50

# import sys
# import os

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PROJECT_ROOT)


# =====================
# 基本配置
# =====================
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
print(f'Use {device}')

num_classes = 100
batch_size = 128 # 128
epochs = 200
lr = 1e-4 # 0.1
# save_path = "/mnt/sdc/mxh/zbl/2_compression/kd/weights/resnet50_teacher.pth"

# =====================
# 数据加载
# =====================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

trainset = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
testset = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# =====================
# 模型、损失、优化器
# =====================
model = resnet50(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# =====================
# 训练与测试函数
# =====================
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    return total_loss / len(trainloader), acc


def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


# =====================
# 主训练循环
# =====================
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch()
    test_acc = evaluate()
    scheduler.step()

    print(
        f"[Teacher] Epoch [{epoch+1}/{epochs}] "
        f"Loss: {train_loss:.4f}, "
        f"Train Acc: {train_acc:.2f}%, "
        f"Test Acc: {test_acc:.2f}%"
    )

# =====================
# 保存模型
# =====================
torch.save(model.state_dict(), 
           f'weights/teacher_resnet50/resnet50_epoch{epoch}_acc{train_acc:.2%}.pth')
print("Teacher model saved to:", 
      f'weights/teacher_resnet50/resnet50_epoch{epoch}_acc{train_acc:.2%}.pth')
