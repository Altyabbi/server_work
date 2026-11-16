import torch
import torch.nn as nn
from torchvision.models import resnet50

model = resnet50(num_classes=100)
model = model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
)

criterion = nn.CrossEntropyLoss()
