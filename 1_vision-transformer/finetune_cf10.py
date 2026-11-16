import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets, models
from tqdm import tqdm
import random, os

# ===================
# 配置
# ===================
seed = 999
dataset_name = 'CIFAR10'  # 可选：'CIFAR10' or 'CIFAR100'
batch_size = 64

lr = 3e-5
num_epochs = 50
freeze_backbone = False  # True时只训练分类头
save_dir = './weights_vit_finetune'

random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(save_dir, exist_ok=True)
print(f'Using device: {device}')

# ===================
# 数据加载与预处理
# ===================
transform_train = T.Compose([
    T.Resize(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])
transform_test = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

if dataset_name == 'CIFAR10':
    train_set = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
elif dataset_name == 'CIFAR100':
    train_set = datasets.CIFAR100(root='data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100(root='data', train=False, download=True, transform=transform_test)
else:
    raise ValueError("dataset_name 必须是 CIFAR10 或 CIFAR100")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
n_classes = len(train_set.classes)

print(f"Dataset: {dataset_name}, Classes: {n_classes}")

# ===================
# 模型加载（ViT-B/16）
# ===================
from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
in_features = model.heads.head.in_features
model.heads.head = nn.Linear(in_features, n_classes)

if freeze_backbone:
    for name, param in model.named_parameters():
        if "heads" not in name:  # 只训练分类头
            param.requires_grad = False

model = model.to(device)

# ===================
# 优化器与损失
# ===================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# ===================
# 训练与验证
# ===================
best_acc = 0.0

for epoch in range(1, num_epochs + 1):
    # ---- Train ----
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}] Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    avg_loss = total_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Epoch [{epoch}/{num_epochs}] Val"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Val Acc: {acc*100:.2f}%")

    # ---- Save best ----
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(save_dir, f"ViT_{dataset_name}_best.pth"))
        print(f"✅ Saved new best model at acc={best_acc*100:.2f}%")

print(f"Training finished! Best Val Acc: {best_acc*100:.2f}%")
