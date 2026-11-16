import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
from tqdm import tqdm
import os

# ================== 1. 参数配置 ==================
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 200            # 蒸馏通常需要较长的训练周期
BATCH_SIZE = 128
LEARNING_RATE = 0.05    # 学生模型的初始学习率
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# 蒸馏超参数 (Hinton et al.)
TEMPERATURE = 4.0       # 温度 T
ALPHA = 0.9             # Soft Target 的权重

# 路径设置
TEACHER_PATH = "./weights/teacher_finetune_resnet50_cifar100_best.pth" # /mnt/sdc/mxh/zbl/2_compression/kd/weights/teacher_finetune_resnet50_cifar100_best.pth
os.makedirs("./weights", exist_ok=True)
SAVE_PATH = "./weights/student_resnet18_distilled.pth"  

# ================== 2. 数据准备 ==================
def get_dataloaders():
    print(">>> 正在准备 CIFAR-100 数据...")
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
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return trainloader, testloader

# ================== 3. 模型定义 ==================
def get_teacher():
    # 必须与你训练 Teacher 时的结构完全一致
    model = resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    
    # 加载权重
    if not os.path.exists(TEACHER_PATH):
        raise FileNotFoundError(f"找不到教师权重: {TEACHER_PATH}")
    
    print(f">>> 加载教师模型权重: {TEACHER_PATH}")
    model.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # 极其重要：教师必须处于评估模式
    for param in model.parameters():
        param.requires_grad = False # 冻结参数
    return model

def get_student():
    print(">>> 构建学生模型 ResNet-18...")
    model = resnet18(weights=None) # 从头开始学
    # 针对 CIFAR-100 修改结构
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model.to(DEVICE)

# ================== 4. 蒸馏损失函数 ==================
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # 1. Hard Loss (CE)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 2. Soft Loss (KL Div)
    # 这里的 T**2 是关键，用于保持梯度的量级
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_prob = F.log_softmax(student_logits / T, dim=1)
    
    soft_loss = F.kl_div(student_log_prob, soft_targets, reduction='batchmean') * (T * T)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss

# ================== 5. 主训练循环 ==================
def main():
    trainloader, testloader = get_dataloaders()
    
    teacher_model = get_teacher()
    student_model = get_student()
    
    optimizer = optim.SGD(student_model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0

    print(f"开始蒸馏: R50 -> R18 | Epochs: {EPOCHS} | T={TEMPERATURE} | Alpha={ALPHA}")

    for epoch in range(EPOCHS):
        student_model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(trainloader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # 1. 教师前向推断 (No Grad)
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            # 2. 学生前向推断
            optimizer.zero_grad()
            student_logits = student_model(inputs)
            
            # 3. 计算蒸馏损失
            loss = distillation_loss(student_logits, teacher_logits, targets, TEMPERATURE, ALPHA)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix(acc=100.*correct/total, lr=optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        # 验证
        student_model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = student_model(inputs)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(targets).sum().item()
                test_total += targets.size(0)
        
        acc = 100. * test_correct / test_total
        print(f"Epoch {epoch+1} | Distillation Loss: {train_loss/len(trainloader):.4f} | Student Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(student_model.state_dict(), SAVE_PATH)
            print(f"✅ Best Model Saved: {best_acc:.2f}%")

    print(f"蒸馏完成。ResNet-18 最终最佳精度: {best_acc:.2f}%")

if __name__ == "__main__":
    main()