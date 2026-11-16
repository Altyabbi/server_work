import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18, mobilenet_v2
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from thop import profile  # 用于计算 FLOPs

# ================= 配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
WEIGHTS_DIR = "./weights"

# 数据集准备 (CIFAR-100 Test Set)
def get_test_loader():
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ================= 模型构建器 (结构必须与训练时完全一致) =================
def build_resnet50():
    model = resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def build_resnet18():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

def build_mobilenetv2():
    model = mobilenet_v2(weights=None)
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.last_channel, 100)
    return model

# ================= 评估核心函数 =================
def evaluate_model(model, dataloader, model_name):
    model.to(DEVICE)
    model.eval()
    
    # 1. 计算 FLOPs 和 参数量
    # 创建一个 dummy input 用于 thop 计算
    input_dummy = torch.randn(1, 3, 32, 32).to(DEVICE)
    flops, params = profile(model, inputs=(input_dummy,), verbose=False)
    
    # 2. 推理精度与时间
    correct_1 = 0
    correct_5 = 0
    total = 0
    total_time = 0
    
    print(f"正在评估: {model_name} ...")
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # 计时开始
            start = time.time()
            outputs = model(inputs)
            # 计时结束 (累计 Batch 时间)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            total_time += (end - start)
            
            # Top-1 Accuracy
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct_1 += pred.eq(targets).sum().item()
            
            # Top-5 Accuracy
            _, rank5 = outputs.topk(5, 1, True, True)
            rank5 = rank5.t()
            correct_5 += rank5.eq(targets.view(1, -1).expand_as(rank5)).sum().item()
            
    # 计算指标
    acc1 = 100. * correct_1 / total
    acc5 = 100. * correct_5 / total
    avg_latency = (total_time / total) * 1000  # ms per image
    
    return {
        "name": model_name,
        "acc1": acc1,
        "acc5": acc5,
        "params": params / 1e6,   # 转换为 Million
        "flops": flops / 1e9,     # 转换为 Giga
        "latency": avg_latency
    }

# ================= 可视化函数 =================
def visualize_results(results):
    names = [r['name'] for r in results]
    acc1s = [r['acc1'] for r in results]
    params = [r['params'] for r in results]
    flops = [r['flops'] for r in results]
    
    # 设置图表风格
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 颜色映射
    colors = ['gray', 'skyblue', 'dodgerblue', 'lightgreen', 'forestgreen']
    # 假设顺序是 Teacher, R18_Base, R18_Distill, MV2_Base, MV2_Distill
    
    # 1. 精度对比 (Top-1 Accuracy)
    axes[0].bar(names, acc1s, color=colors)
    axes[0].set_title('Top-1 Accuracy (%) - Higher is Better')
    axes[0].set_ylim(min(acc1s)-5, max(acc1s)+2)
    for i, v in enumerate(acc1s):
        axes[0].text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=15)

    # 2. 参数量对比 (Parameters)
    axes[1].bar(names, params, color=colors)
    axes[1].set_title('Model Size (Params M) - Lower is Better')
    for i, v in enumerate(params):
        axes[1].text(i, v, f"{v:.2f}M", ha='center', va='bottom')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15)

    # 3. 计算量对比 (FLOPs)
    axes[2].bar(names, flops, color=colors)
    axes[2].set_title('Computation (FLOPs G) - Lower is Better')
    for i, v in enumerate(flops):
        axes[2].text(i, v, f"{v:.3f}G", ha='center', va='bottom')
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=15)

    plt.tight_layout()
    plt.savefig('evaluation_comparison.png')
    print(">>> 评估结果图表已保存至 evaluation_comparison.png")
    plt.show()

# ================= 主程序 =================
def main():
    testloader = get_test_loader()
    
    # 定义待评估的模型列表 (名称, 构建函数, 权重路径)
    # 请根据你实际保存的文件名修改 path
    models_config = [
        # Teacher
        ("Teacher(R50)", build_resnet50, "teacher_finetune_resnet50_cifar100_best.pth"),
        
        # ResNet-18 Group
        ("R18_Base", build_resnet18, "resnet18_baseline.pth"),
        ("R18_Distill", build_resnet18, "student_resnet18_distilled.pth"),
        
        # MobileNetV2 Group
        ("MV2_Base", build_mobilenetv2, "mobilenetv2_baseline.pth"),
        ("MV2_Distill", build_mobilenetv2, "student_mobilenetv2_distilled.pth"),
    ]
    
    results = []
    
    print(f"{'Model':<20} | {'Acc@1 (%)':<10} | {'Acc@5 (%)':<10} | {'Params(M)':<10} | {'FLOPs(G)':<10} | {'Latency(ms)':<10}")
    print("-" * 90)
    
    for name, build_func, weight_file in models_config:
        weight_path = os.path.join(WEIGHTS_DIR, weight_file)
        
        # 检查文件是否存在
        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过 {name}: 找不到文件 {weight_path}")
            continue
            
        # 构建并加载模型
        model = build_func()
        try:
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        except RuntimeError as e:
            print(f"❌ 加载 {name} 失败，可能是结构不匹配。\n错误信息: {e}")
            continue
            
        # 评估
        res = evaluate_model(model, testloader, name)
        results.append(res)
        
        # 打印表格行
        print(f"{res['name']:<20} | {res['acc1']:<10.2f} | {res['acc5']:<10.2f} | {res['params']:<10.2f} | {res['flops']:<10.4f} | {res['latency']:<10.4f}")

    # 可视化
    if results:
        visualize_results(results)

if __name__ == "__main__":
    main()