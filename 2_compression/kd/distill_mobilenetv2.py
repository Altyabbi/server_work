import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, mobilenet_v2
from tqdm import tqdm
import os

# ================== 1. 参数配置 ==================
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
BATCH_SIZE = 128
LEARNING_RATE = 0.05
WEIGHT_DECAY = 4e-5     # MobileNet 参数较少，正则化稍微小一点
MOMENTUM = 0.9

# 蒸馏超参数
TEMPERATURE = 4.0
ALPHA = 0.9

# 路径设置
TEACHER_PATH = "./weights/teacher_finetune_resnet50_cifar100_best.pth"
SAVE_PATH = "./weights/student_mobilenetv2_distilled.pth"
os.makedirs("./weights", exist_ok=True)

# ================== 2. 数据准备 ==================
def get_dataloaders():
    # 与上面代码一致
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
    # 必须与训练时的结构一致
    model = resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    
    if not os.path.exists(TEACHER_PATH):
        raise FileNotFoundError(f"找不到教师权重: {TEACHER_PATH}")
    
    print(f">>> 加载教师模型权重: {TEACHER_PATH}")
    model.load_state_dict(torch.load(TEACHER_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def get_student():
    print(">>> 构建学生模型 MobileNetV2...")
    model = mobilenet_v2(weights=None) # 不使用预训练权重
    
    # 针对 CIFAR-100 修改结构 (防止特征图过早变小)
    # 修改第一层: 32x32 输入 -> stride=1
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 修改分类器: 输出 100 类
    model.classifier[1] = nn.Linear(model.last_channel, 100)
    
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

    print(f"开始蒸馏: R50 -> MobileNetV2 | Epochs: {EPOCHS} | T={TEMPERATURE} | Alpha={ALPHA}")

    for epoch in range(EPOCHS):
        student_model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(trainloader, desc=f"Ep {epoch+1}/{EPOCHS}", leave=False)
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)
            
            optimizer.zero_grad()
            student_logits = student_model(inputs)
            
            loss = distillation_loss(student_logits, teacher_logits, targets, TEMPERATURE, ALPHA)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix(acc=100.*correct/total)
        
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

    print(f"蒸馏完成。MobileNetV2 最终最佳精度: {best_acc:.2f}%")

if __name__ == "__main__":
    main()