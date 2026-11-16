import torch
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    ce = F.cross_entropy(student_logits, labels)
    kd = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    return alpha * ce + (1 - alpha) * kd

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


