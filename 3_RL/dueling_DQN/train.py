from config import Config
from environment import CocoActiveDetEnv
from agent import DuelingDQNAgent
from evaluator import Evaluator

def main():
    cfg = Config()
    env = CocoActiveDetEnv(cfg)
    agent = DuelingDQNAgent(cfg)
    evaluator = Evaluator(env)
    
    print(f"开始训练，目标类别: {cfg.TARGET_CLASS}，设备: {cfg.DEVICE}")

    for episode in range(1, 10001):
        state = env.reset()
        total_reward = 0
        
        for t in range(cfg.MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
            agent.update()
            
            if done: break
            
        # 定期更新目标网络
        if episode % cfg.TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        # 定期打印进度和评估结果
        if episode % 50 == 0:
            avg_iou, sr = evaluator.evaluate(agent)
            print(f"Epi: {episode:4d} | Reward: {total_reward:5.1f} | "
                  f"Eval IoU: {avg_iou:.3f} | Success: {sr*100:4.1f}% | Eps: {agent.epsilon:.2f}")

if __name__ == "__main__":
    main()