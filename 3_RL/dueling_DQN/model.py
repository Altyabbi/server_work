import torch
import torch.nn as nn
import torchvision.models as models

class DuelingQNet(nn.Module):
    def __init__(self, action_dim):
        super(DuelingQNet, self).__init__()
        
        # 使用预训练的 ResNet18 作为特征提取骨干
        resnet = models.resnet18(pretrained=True)
        # 移除最后的 FC 层，保留特征提取部分 (输出维度 512)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 冻结骨干网络，仅训练决策层 (可选，加速收敛)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Dueling 结构的两个分支
        # 1. Advantage 分支 (每个动作的优势)
        self.advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # 2. Value 分支 (状态的价值)
        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # 提取特征 [batch, 512, 1, 1] -> [batch, 512]
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        
        adv = self.advantage(feat)
        val = self.value(feat)
        
        # Dueling 结合公式: Q = V + (A - mean(A))
        q_val = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_val