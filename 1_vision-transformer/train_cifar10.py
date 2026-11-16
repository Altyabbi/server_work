import csv
import random
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from tqdm import tqdm

from models.vit import VisionTransformer  # 你的ViT定义

# ===================
# 超参数与环境设置
# ===================
seed = 999
depth = 8       #6
n_heads = 8     #8
n_classes = 10
n_epoches = 120
batch_size = 64 #64
show_size = 8
historical_weights = None

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ===================
# 数据预处理
# ===================
trans = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])
untrans = T.Compose([
    T.Normalize([-.485/.229, -.456/.224, -.406/.225],
                [1/.229, 1/.224, 1/.225]),
])

train_set = CIFAR10(root='data', train=True, download=True, transform=trans)
val_set = CIFAR10(root='data', train=False, download=True, transform=trans)

train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size,
                        shuffle=False, num_workers=0)

# ===================
# 模型与优化器
# ===================
model = VisionTransformer(img_size=32,
                          patch_size=2,
                          n_classes=n_classes,
                          depth=depth,
                          n_heads=n_heads,
                          embed_dim=512,
                          mlp_ratio=4
                          ).to(device)

if historical_weights is not None:
    model.load_state_dict(torch.load(f'weights_cifar/3/{historical_weights}'))
    historical_epoch = int(historical_weights.split('_')[4][5:])
else:
    historical_epoch = 0

optim = Adam(model.parameters(), lr=1e-4, betas=(.5, .999))
criterion = nn.CrossEntropyLoss()

# ===================
# 结果保存设置
# ===================
os.makedirs("weights_cifar/3", exist_ok=True)
results_file = "training_results.csv"

# 若文件不存在，则创建并写入表头
if not os.path.exists(results_file):
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc"])


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

    # ---------- 记录结果 ----------
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_loss, acc])

    # ---------- 模型保存 ----------
    torch.save(model.state_dict(),
               f'weights_cifar/3/VIT_depth{depth}_{n_heads}heads_{n_classes}classes_epoch{epoch}_acc{acc:.2%}.pth')
