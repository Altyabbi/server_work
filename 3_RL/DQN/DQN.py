import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from collections import deque

# ==========================================
# 0. 全局参数配置 (Global Configuration)
# ==========================================
# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 环境参数
IMG_SIZE = 128
OBS_SIZE = 32         # 输入网络的缩略图尺寸
MAX_STEPS = 40        # 每个Episode最大步数

# RL 超参数
LR = 1e-4
GAMMA = 0.95
MEMORY_SIZE = 5000
BATCH_SIZE = 64
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.996     # 衰减系数
TOTAL_EPISODES = 1500

# 路径配置
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
MODEL_NAME = "dqn_detection_final.pth"

# 实验可复现性设置
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

seed_everything()

# ==========================================
# 1. 环境定义 (Environment)
# ==========================================
class ActiveDetectionEnv(gym.Env):
    def __init__(self):
        super(ActiveDetectionEnv, self).__init__()
        self.action_space = spaces.Discrete(7) # 左,右,上,下,缩小,放大,停止
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, OBS_SIZE, OBS_SIZE), dtype=np.uint8)
        self.reset()

    def _get_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 生成合成数据
        self.img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        self.target_size = random.randint(20, 40)
        self.tx = random.randint(0, IMG_SIZE - self.target_size)
        self.ty = random.randint(0, IMG_SIZE - self.target_size)
        self.target_box = [self.tx, self.ty, self.tx + self.target_size, self.ty + self.target_size]
        self.img[self.ty:self.ty+self.target_size, self.tx:self.tx+self.target_size] = [0, 0, 255]
        
        # 初始窗口为全图
        self.curr_box = [0, 0, IMG_SIZE, IMG_SIZE]
        self.prev_iou = self._get_iou(self.curr_box, self.target_box)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        x1, y1, x2, y2 = map(int, self.curr_box)
        patch = self.img[y1:y2, x1:x2]
        obs = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY), (OBS_SIZE, OBS_SIZE))
        return obs.reshape(1, OBS_SIZE, OBS_SIZE)

    def step(self, action):
        self.steps += 1
        x1, y1, x2, y2 = self.curr_box
        delta = 5
        
        if action == 0: x1, x2 = x1-delta, x2-delta
        elif action == 1: x1, x2 = x1+delta, x2+delta
        elif action == 2: y1, y2 = y1-delta, y2-delta
        elif action == 3: y1, y2 = y1+delta, y2+delta
        elif action == 4: x1, y1, x2, y2 = x1+delta, y1+delta, x2-delta, y2-delta # 缩小
        elif action == 5: x1, y1, x2, y2 = x1-delta, y1-delta, x2+delta, y2+delta # 放大

        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(IMG_SIZE, x2), min(IMG_SIZE, y2)
        if x2 - x1 < 10 or y2 - y1 < 10: x1, y1, x2, y2 = self.curr_box
        self.curr_box = [x1, y1, x2, y2]

        curr_iou = self._get_iou(self.curr_box, self.target_box)
        reward = 1.0 if curr_iou > self.prev_iou else -1.0
        
        done = False
        if action == 6 or self.steps >= MAX_STEPS:
            done = True
            reward = 10.0 if curr_iou > 0.6 else -10.0
        
        self.prev_iou = curr_iou
        return self._get_obs(), reward, done, False, {}

# ==========================================
# 2. 网络结构 (Architecture)
# ==========================================
class DQNet(nn.Module):
    def __init__(self, action_dim):
        super(DQNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Flatten()
        )
        # 计算卷积后的尺寸: 32x32 -> 15x15 -> 7x7
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = x.float() / 255.0
        return self.fc(self.conv(x))

# ==========================================
# 3. 训练函数 (Training)
# ==========================================
def train():
    env = ActiveDetectionEnv()
    q_net = DQNet(env.action_space.n).to(DEVICE)
    target_net = DQNet(env.action_space.n).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPS_START
    
    print(f">>> 开始训练 | 设备: {DEVICE} | 目标: {TOTAL_EPISODES} Episodes")
    
    for episode in range(TOTAL_EPISODES + 1):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Epsilon-Greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    action = q_net(state_t).argmax().item()
            
            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                s, a, r, ns, d = zip(*batch)
                
                s_t = torch.as_tensor(np.array(s), dtype=torch.float32, device=DEVICE)
                a_t = torch.as_tensor(a, device=DEVICE).unsqueeze(1)
                r_t = torch.as_tensor(r, dtype=torch.float32, device=DEVICE)
                ns_t = torch.as_tensor(np.array(ns), dtype=torch.float32, device=DEVICE)
                d_t = torch.as_tensor(d, dtype=torch.float32, device=DEVICE)
                
                curr_q = q_net(s_t).gather(1, a_t).squeeze()
                with torch.no_grad():
                    next_q = target_net(ns_t).max(1)[0]
                    expected_q = r_t + (1 - d_t) * GAMMA * next_q
                
                loss = nn.MSELoss()(curr_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done: break
        
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        if episode % 100 == 0:
            target_net.load_state_dict(q_net.state_dict())
            print(f"Episode: {episode:4d} | Reward: {episode_reward:5.1f} | Epsilon: {epsilon:.3f}")

    # 保存模型
    if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
    save_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
    torch.save(q_net.state_dict(), save_path)
    print(f">>> 训练完成，模型已存至: {save_path}")

# ==========================================
# 4. 测试与可视化 (Testing)
# ==========================================
def test_and_visualize():
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
    
    env = ActiveDetectionEnv()
    model = DQNet(env.action_space.n).to(DEVICE)
    load_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
    
    try:
        # 使用 weights_only=True 且 map_location 确保设备一致
        model.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
        model.eval()
        print(f">>> 成功加载模型: {load_path}")
    except FileNotFoundError:
        print("!!! 未找到模型文件，请先运行训练。")
        return

    state, _ = env.reset()
    frames = []
    
    for i in range(MAX_STEPS):
        # 渲染当前帧
        canvas = env.img.copy()
        x1, y1, x2, y2 = map(int, env.curr_box)
        tx1, ty1, tx2, ty2 = map(int, env.target_box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2) # Agent
        cv2.rectangle(canvas, (tx1, ty1), (tx2, ty2), (255, 0, 0), 1) # GT
        cv2.putText(canvas, f"Step:{i}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        frames.append(canvas)

        # 推理
        state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            action = model(state_t).argmax().item()
        
        state, _, done, _, _ = env.step(action)
        if done: break
    
    # 保存最终结果图
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB))
    plt.title(f"Localization Result (Final Step)")
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, "final_detection.png"), dpi=300)
    
    # 保存过程演变图 (选取4个阶段)
    if len(frames) >= 4:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        indices = [0, len(frames)//3, 2*len(frames)//3, len(frames)-1]
        for idx, step_idx in enumerate(indices):
            axes[idx].imshow(cv2.cvtColor(frames[step_idx], cv2.COLOR_BGR2RGB))
            axes[idx].set_title(f"Step {step_idx}")
            axes[idx].axis('off')
        plt.savefig(os.path.join(RESULTS_DIR, "detection_process.png"), dpi=300)
    
    print(f">>> 可视化图像已保存至: {RESULTS_DIR}/")

# ==========================================
# 5. 评估模块 (Evaluation Module)
# ==========================================
def evaluate(num_test_episodes=100):
    """
    在随机环境下运行多轮测试，统计目标检测的核心评价指标
    """
    env = ActiveDetectionEnv()
    model = DQNet(env.action_space.n).to(DEVICE)
    load_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
    
    try:
        model.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("!!! 评估中止：未找到训练好的模型文件。")
        return

    # 指标记录器
    final_ious = []
    total_steps = []
    success_at_05 = 0
    success_at_07 = 0

    print(f"\n>>> 正在进行评估模块测试 (共 {num_test_episodes} 轮)...")

    for _ in range(num_test_episodes):
        state, _ = env.reset()
        steps = 0
        while True:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = model(state_t).argmax().item()
            
            state, _, done, _, _ = env.step(action)
            steps += 1
            if done:
                # 获取最终 IoU
                final_iou = env._get_iou(env.curr_box, env.target_box)
                final_ious.append(final_iou)
                total_steps.append(steps)
                
                # 统计成功次数
                if final_iou >= 0.5: success_at_05 += 1
                if final_iou >= 0.7: success_at_07 += 1
                break

    # 计算统计结果
    avg_iou = np.mean(final_ious)
    std_iou = np.std(final_ious)
    avg_steps = np.mean(total_steps)
    sr_05 = (success_at_05 / num_test_episodes) * 100
    sr_07 = (success_at_07 / num_test_episodes) * 100


    print("-" * 50)
    print(f"{'Metric':<30} | {'Value':<15}")
    print("-" * 50)
    print(f"{'Average IoU (Final)':<30} | {avg_iou:.4f} ± {std_iou:.3f}")
    print(f"{'Success Rate (IoU > 0.5)':<30} | {sr_05:.2f}%")
    print(f"{'Recall (IoU > 0.7)':<30} | {sr_07:.2f}%")
    print(f"{'Average Steps to Target':<30} | {avg_steps:.2f} steps")
    print("-" * 50)
    
    # 针对强化学习的结论建议
    if sr_05 > 70:
        print("结论: 模型具有较强的目标收敛能力。")
    else:
        print("结论: 模型训练尚不充分或状态空间较复杂，建议增加训练轮数。")
    print("-" * 50)




if __name__ == "__main__":
    train()
    test_and_visualize()
    evaluate(num_test_episodes=800)