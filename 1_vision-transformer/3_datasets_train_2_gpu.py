import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms as T, datasets
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
import os, random, csv

# ===================
# 全局配置
# ===================
seed = 999
# batch_size = 64
# batch_size = 32
batch_size = 96


lr = 3e-5
# num_epochs = 30
num_epochs = 15

save_dir = "./weights_vit_2gpu"  # 保存路径
freeze_backbone = False
os.makedirs(save_dir, exist_ok=True)

random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Using {n_gpus} GPUs: {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")

# ===================
# 数据集加载函数
# ===================
# def get_dataset(dataset_name):
#     transform_train = T.Compose([
#         T.Resize(224),
#         T.RandomHorizontalFlip(),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])
#     transform_test = T.Compose([
#         T.Resize(224),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])

#     if dataset_name == "CIFAR10":
#         train_set = datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
#         test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)
#     elif dataset_name == "CIFAR100":
#         train_set = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
#         test_set = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)
#     elif dataset_name == "OxfordIIITPet":
#         from torchvision.datasets import OxfordIIITPet
#         train_set = OxfordIIITPet(root="data", split="trainval", download=True, transform=transform_train, target_types="category") # 🔥 关键：只要分类标签)
#         test_set = OxfordIIITPet(root="data", split="test", download=True, transform=transform_test, target_types="category")
#     else:
#         raise ValueError("未知数据集，请选择 CIFAR10 / CIFAR100 / OxfordIIITPet")

#     return train_set, test_set
def get_dataset(dataset_name):
    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    transform_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    if dataset_name == "CIFAR10":
        train_set = datasets.CIFAR10(root="data", train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root="data", train=False, download=True, transform=transform_test)

    elif dataset_name == "CIFAR100":
        train_set = datasets.CIFAR100(root="data", train=True, download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root="data", train=False, download=True, transform=transform_test)

    elif dataset_name == "OxfordIIITPet":
        from torchvision.datasets import OxfordIIITPet
        train_set = OxfordIIITPet(
            root="data",
            split="trainval",
            target_types="category",
            transform=transform_train,
            download=True
        )
        test_set = OxfordIIITPet(
            root="data",
            split="test",
            target_types="category",
            transform=transform_test,
            download=True
        )

    else:
        raise ValueError("未知数据集，请选择 CIFAR10 / CIFAR100 / OxfordIIITPet")

    # 🔥 核心修复：定义统一的 collate_fn
    def simple_collate(batch):
        imgs, targets = zip(*batch)
        clean_targets = []
        for t in targets:
            # OxfordIIITPet 可能返回 tuple
            if isinstance(t, (tuple, list)):
                t = t[0]
            clean_targets.append(int(t))
        imgs = torch.stack(imgs, dim=0)
        targets = torch.tensor(clean_targets, dtype=torch.long)
        return imgs, targets

    # 🔥 使用该 collate_fn，保证 batch 结构统一
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=simple_collate)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=4, collate_fn=simple_collate)

    return train_set, test_set, train_loader, test_loader


# ===================
# 单数据集训练函数（多GPU）
# ===================
def train_on_dataset(dataset_name):
    print(f"\n{'='*25} Training on {dataset_name} {'='*25}")

    # train_set, test_set = get_dataset(dataset_name)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


    train_set, test_set, train_loader, test_loader = get_dataset(dataset_name)

    n_classes = len(train_set.classes)


    # ---- 模型加载 ----
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, n_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "heads" not in name:
                param.requires_grad = False

    # ---- 使用多GPU并行 ----
    model = nn.DataParallel(model)  # 自动分发到多GPU
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"[{dataset_name}] Epoch {epoch}/{num_epochs} Train"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)

        # ---- 验证 ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"[{dataset_name}] Epoch {epoch}/{num_epochs} Val"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"[{dataset_name}] Epoch {epoch}: Loss={avg_loss:.4f}, Val Acc={acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.module.state_dict(), os.path.join(save_dir, f"ViT_B_16_{dataset_name}_best.pth"))
            print(f"✅ [{dataset_name}] New best model saved (Acc={best_acc*100:.2f}%)")

    return dataset_name, best_acc, best_epoch


# ===================
# 主程序（循环3个数据集）
# ===================
datasets_to_train = ["CIFAR10", "CIFAR100", "OxfordIIITPet"]
# datasets_to_train = ["CIFAR100",]
# datasets_to_train = ["OxfordIIITPet",]




csv_path = os.path.join(save_dir, "vit_b16_2gpu_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Best_Accuracy", "Best_Epoch"])

for ds in datasets_to_train:
    name, best_acc, best_epoch = train_on_dataset(ds)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, f"{best_acc*100:.2f}", best_epoch])

print("\n✅ 全部数据集训练完成，结果已记录至：", csv_path)
