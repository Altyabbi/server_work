import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.vit import VisionTransformer  # 你自己的ViT定义

# ===================
# 超参数与环境设置
# ===================
seed = 999
random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

depth = 6
n_heads = 8
n_classes = 37
n_epoches = 50
batch_size = 32
img_size = 224
patch_size = 16
embed_dim = 256           # ✅ ViT-Tiny
mlp_ratio = 4             # ✅ 标准比例
weight_decay = 0.05
lr = 3e-4

save_dir = 'weights_ox/vit_tiny_pets'
os.makedirs(save_dir, exist_ok=True)

# ===================
# 数据预处理
# ===================
trans = T.Compose([
    T.Resize((img_size, img_size)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])

train_set = OxfordIIITPet(
    root='data',
    split='trainval',
    target_types='category',
    transform=trans,
    download=True
)
val_set = OxfordIIITPet(
    root='data',
    split='test',
    target_types='category',
    transform=trans,
    download=True
)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
print(f"Classes: {len(train_set.classes)} -> {train_set.classes[:5]} ...")
print(f"Example label index: {train_set[0][1]} ({train_set.classes[train_set[0][1]]})")

# ===================
# 模型定义
# ===================
model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    n_classes=n_classes,
    depth=depth,
    n_heads=n_heads,
    embed_dim=embed_dim,
    mlp_ratio=mlp_ratio
).to(device)

print(model.head)

# ===================
# 优化器、损失函数、调度器
# ===================
optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optim, T_max=n_epoches)
criterion = nn.CrossEntropyLoss()

# ===================
# 训练与验证循环
# ===================
for epoch in range(1, n_epoches + 1):

    # ---------- 训练 ----------
    model.train()
    total_loss = 0.0
    for index, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch} [Train]')):
        data, target = data.to(device), target.to(device)

        scores = model(data)
        loss = criterion(scores, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f'[Epoch {epoch}] Train Loss: {avg_loss:.4f}')

    # ---------- 验证 ----------
    model.eval()
    n_correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f'Epoch {epoch} [Val]'):
            data, target = data.to(device), target.to(device)
            scores = model(data)
            pred = scores.argmax(dim=1)
            n_correct += (pred == target).sum().item()
            total += target.size(0)

    acc = n_correct / total
    print(f'[Epoch {epoch}] Validation Accuracy: {acc * 100:.2f}%')

    # ---------- 模型保存 ----------
    torch.save(model.state_dict(),
               os.path.join(save_dir,
                            f'VIT_Tiny_PETS_epoch{epoch}_acc{acc:.2%}.pth'))
