import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms as T
from tqdm import tqdm

from models.vit import VisionTransformer  # 你的ViT定义

# ===================
# 超参数与环境设置
# ===================
seed = 999
depth = 6
n_heads = 8
n_classes = 37     # Oxford-IIIT Pets 有 37 类
n_epoches = 100
batch_size = 64
historical_weights = None

random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ===================
# 数据预处理
# ===================xc
img_size = 224  # ViT 推荐 224x224 输入
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

dataset = OxfordIIITPet(root='data', split='trainval', target_types='category', download=True)

# 查看类别名称
print(dataset.classes)
print(f"类别数：{len(dataset.classes)}")
print(train_set.classes)
print(f"类别数：{len(train_set.classes)}")
print(f"整数：{train_set[0][1]}")




# ===================
# 模型与优化器
# ===================
model = VisionTransformer(img_size=img_size,
                          patch_size=16,     # 通常ViT采用16×16 patch
                          n_classes=n_classes,
                          depth=depth,
                          n_heads=n_heads,
                          embed_dim=256,  #512
                          mlp_ratio=4  # 1
                          ).to(device)
print(model.head)


if historical_weights is not None:
    model.load_state_dict(torch.load(f'weights_ox/2/{historical_weights}'))
    historical_epoch = int(historical_weights.split('_')[4][5:])
else:
    historical_epoch = 0

from torch.optim import AdamW
optim = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epoches)


# optim = Adam(model.parameters(), lr=3e-4, betas=(.5, .999))  #1e-5
criterion = nn.CrossEntropyLoss()

# ===================
# 训练与验证循环
# ===================
for epoch in range(historical_epoch + 1, n_epoches + 1):

    # ---------- 训练 ----------
    model.train()
    total_loss = 0.

    for index, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch} Train')):
        data, target = data.to(device), target.to(device)

        scores = model(data)
        loss = criterion(scores, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'[Epoch {epoch}] Train Loss: {avg_loss:.4f}')

    # ---------- 验证 ----------
    model.eval()
    n_correct = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f'Epoch {epoch} Val'):
            data, target = data.to(device), target.to(device)
            scores = model(data)
            pred = scores.argmax(dim=1)
            n_correct += torch.eq(pred, target).sum().item()

    acc = n_correct / len(val_set)
    print(f'[Epoch {epoch}] Validation Accuracy: {acc * 100:.2f}%')

    # ---------- 模型保存 ----------
    torch.save(model.state_dict(),
               f'weights_ox/2/VIT_PETS_depth{depth}_{n_heads}heads_{n_classes}classes_epoch{epoch}_acc{acc:.2%}.pth')
