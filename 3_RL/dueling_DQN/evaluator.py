import torch

class Evaluator:
    def __init__(self, env):
        self.env = env

    def evaluate(self, agent, num_episodes=20):
        total_iou = 0
        success = 0
        
        # 使用测试集前几个样本
        test_ids = self.env.img_ids[:num_episodes]
        
        for img_id in test_ids:
            state = self.env.reset(img_id=img_id)
            done = False
            while not done:
                action = agent.select_action(state, eval_mode=True)
                state, _, done = self.env.step(action)
            
            iou = self.env._get_iou(self.env.current_box, self.env.gt_box)
            total_iou += iou
            if iou > 0.5: success += 1
            
        return total_iou / num_episodes, success / num_episodes