import os
import torch
# import pandas as pd
from torchvision import transforms as T
from PIL import Image
from models.vit import VisionTransformer  # ViT定义路径

# =========================
# 基本参数（必须与训练时一致）
# =========================
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

depth = 8   #6
n_heads = 8
n_classes = 10
embed_dim = 512
img_size = 32
patch_size = 2
weights_path = '/mnt/sdc/mxh/zbl/vision-transformer/weights_cifar/2/VIT_depth8_8heads_10classes_epoch111_acc76.74%.pth'
test_dir = '/mnt/sdc/mxh/zbl/vision-transformer/test_img'  # 存放多张图片的文件夹路径

# CIFAR-10 类别名称
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# =========================
# 图像预处理
# =========================
transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])

# =========================
# 加载模型并载入权重
# =========================
model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    n_classes=n_classes,
    depth=depth,
    n_heads=n_heads,
    embed_dim=embed_dim,
    mlp_ratio=1
).to(device)

state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f'✅ 模型已加载：{weights_path}')
print('=' * 80)

# =========================
# 读取所有图片并推理
# =========================
# 支持的图片格式
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# 按名称排序读取
image_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)])

results = []  # 用于保存所有预测结果


if len(image_files) == 0:
    print(f'⚠️ 在目录 {test_dir} 中未找到图片文件！')
    exit()

# 遍历每张图片推理
with torch.no_grad():
    for fname in image_files:
        img_path = os.path.join(test_dir, fname)
        image = Image.open(img_path).convert('RGB')

        # 预处理
        input_tensor = transform(image).unsqueeze(0).to(device)

        # 推理
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        print(f'📷 图片名称: {fname}')
        print(f'   预测类别: {classes[pred_class]}')
        print(f'   置信度: {confidence * 100:.2f}%')
        print('-' * 80)


        # 保存结果到列表
        # results.append({
        #     'filename': fname,
        #     'top1_class': top3_classes[0],
        #     'top1_conf': f'{top3_probs[0]*100:.2f}%',
        #     'top3_classes': ', '.join(top3_classes),
        #     'top3_confs': ', '.join([f'{p*100:.2f}%' for p in top3_probs])
        # })

#         # 可视化预测结果
#         plt.figure()
#         plt.imshow(image)
#         plt.title(f"Pred: {top3_classes[0]} ({top3_probs[0]*100:.1f}%)")
#         plt.axis('off')
#         plt.show()
#         # 若不想弹窗，可改为：
#         # plt.savefig(os.path.join(test_dir, f"vis_{fname}"))
#         # plt.close()

# # =========================
# # 保存推理结果为 CSV
# # =========================
# df = pd.DataFrame(results, columns=['filename', 'top1_class', 'top1_conf', 'top3_classes', 'top3_confs'])
# csv_path = os.path.join(test_dir, 'inference_results.csv')
# df.to_csv(csv_path, index=False, encoding='utf-8-sig')
# print(f'\n💾 推理结果已保存到: {csv_path}')