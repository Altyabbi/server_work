import os
import torch
from torchvision import transforms as T
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

# =========================
# 基本参数（保持与训练一致）
# =========================
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# 修改1️⃣：使用与训练一致的权重路径
weights_path = '/mnt/sdc/mxh/zbl/vision-transformer/weights_vit_2gpu/ViT_B_16_CIFAR10_best.pth'
# weights_path = '/mnt/sdc/mxh/zbl/vision-transformer/weights_vit_2gpu/ViT_B_16_CIFAR100_best.pth'

# 修改2️⃣：测试图片所在文件夹
test_dir = '/mnt/sdc/mxh/zbl/vision-transformer/test_img'

# 修改3️⃣：CIFAR-100类别（从官方类名列表）
# classes = [
#     'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
#     'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
#     'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
#     'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
#     'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
#     'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
#     'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
#     'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
#     'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
#     'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
#     'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
#     'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
#     'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
#     'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
#     'worm'
# ]

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# =========================
# 图像预处理
# =========================
transform = T.Compose([
    T.Resize((224, 224)),  # 训练时也是224×224
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# =========================
# 模型加载（修改5️⃣：使用torchvision的vit_b_16）
# =========================
weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)
in_features = model.heads.head.in_features
model.heads.head = torch.nn.Linear(in_features, len(classes))  # 修改分类头

state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print(f'✅ 模型已加载：{weights_path}')
print('=' * 80)

# =========================
# 遍历文件夹推理
# =========================
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)])

if len(image_files) == 0:
    print(f'⚠️ 在目录 {test_dir} 中未找到图片文件！')
    exit()

with torch.no_grad():
    for fname in image_files:
        img_path = os.path.join(test_dir, fname)
        image = Image.open(img_path).convert('RGB')

        input_tensor = transform(image).unsqueeze(0).to(device)

        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        print(f'📷 图片名称: {fname}')
        print(f'   预测类别: {classes[pred_class]}')
        print(f'   置信度: {confidence * 100:.2f}%')
        print('-' * 80)
