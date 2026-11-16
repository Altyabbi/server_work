import torch

class Config:
    # 数据集路径 (请根据实际情况修改)
    COCO_ROOT = "./data/coco/train2017"
    COCO_ANN = "./data/coco/annotations/instances_train2017.json"
    
    # 强化学习超参数
    LR = 1e-4
    GAMMA = 0.95
    BATCH_SIZE = 32
    MEMORY_SIZE = 10000
    TARGET_UPDATE = 10     # 目标网络更新频率 (Episode)
    
    # Epsilon-Greedy 策略
    EPS_START = 1.0
    EPS_END = 0.1
    EPS_DECAY = 5000       # 衰减步数
    
    # 环境配置
    TARGET_CLASS = 'person'
    MAX_STEPS = 20         # 每个物体的最大寻优步数
    INPUT_SIZE = 224       # 输入给 CNN 的尺寸
    ACTION_DIM = 7         # 0:L, 1:R, 2:U, 3:D, 4:In, 5:Out, 6:Trigger
    MOVE_ALPHA = 0.2       # 动作移动比例
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")