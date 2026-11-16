import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from model import DuelingQNet

class DuelingDQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.policy_net = DuelingQNet(cfg.ACTION_DIM).to(cfg.DEVICE)
        self.target_net = DuelingQNet(cfg.ACTION_DIM).to(cfg.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.LR)
        self.memory = deque(maxlen=cfg.MEMORY_SIZE)
        self.epsilon = cfg.EPS_START
        self.steps_done = 0

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.cfg.ACTION_DIM)
        
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.cfg.DEVICE)
            return self.policy_net(state).argmax(1).item()

    def update(self):
        if len(self.memory) < self.cfg.BATCH_SIZE:
            return

        batch = random.sample(self.memory, self.cfg.BATCH_SIZE)
        s, a, r, s_next, d = zip(*batch)

        s = torch.stack(s).to(self.cfg.DEVICE)
        a = torch.tensor(a, device=self.cfg.DEVICE).unsqueeze(1)
        r = torch.tensor(r, device=self.cfg.DEVICE)
        s_next = torch.stack(s_next).to(self.cfg.DEVICE)
        d = torch.tensor(d, device=self.cfg.DEVICE).float()

        # Double DQN 核心更新逻辑
        curr_q = self.policy_net(s).gather(1, a).squeeze()
        with torch.no_grad():
            # 用 policy_net 选动作，用 target_net 算值
            next_actions = self.policy_net(s_next).argmax(1).unsqueeze(1)
            next_q = self.target_net(s_next).gather(1, next_actions).squeeze()
            target_q = r + (self.cfg.GAMMA * next_q * (1 - d))

        loss = F.smooth_l1_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 探索率衰减
        self.epsilon = max(self.cfg.EPS_END, self.epsilon - (self.cfg.EPS_START - self.cfg.EPS_END) / self.cfg.EPS_DECAY)