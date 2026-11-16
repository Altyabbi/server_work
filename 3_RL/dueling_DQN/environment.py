import os
import numpy as np
import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from PIL import Image
from config import Config

class CocoActiveDetEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.coco = COCO(cfg.COCO_ANN)
        cat_ids = self.coco.getCatIds(catNms=[cfg.TARGET_CLASS])
        self.img_ids = self.coco.getImgIds(catIds=cat_ids)
        
        self.transform = T.Compose([
            T.Resize((cfg.INPUT_SIZE, cfg.INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def reset(self, img_id=None):
        if img_id is None:
            img_id = np.random.choice(self.img_ids)
            
        # 获取标注信息
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.coco.getCatIds(catNms=[self.cfg.TARGET_CLASS]))
        anns = self.coco.loadAnns(ann_ids)
        main_ann = max(anns, key=lambda x: x['bbox'][2] * x['bbox'][3])
        bx, by, bw, bh = main_ann['bbox']
        self.gt_box = [bx, by, bx + bw, by + bh] # [x1, y1, x2, y2]
        
        # 加载图像
        img_info = self.coco.loadImgs(int(img_id))[0]
        self.image = Image.open(os.path.join(self.cfg.COCO_ROOT, img_info['file_name'])).convert('RGB')
        self.img_w, self.img_h = self.image.size
        
        # 初始框设定为全图
        self.current_box = [0.0, 0.0, float(self.img_w), float(self.img_h)]
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # 裁剪并缩放
        x1, y1, x2, y2 = map(int, self.current_box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(self.img_w, x2), min(self.img_h, y2)
        if x2 <= x1: x2 = x1 + 10
        if y2 <= y1: y2 = y1 + 10
        roi = self.image.crop((x1, y1, x2, y2))
        return self.transform(roi)

    def _get_iou(self, boxA, boxB):
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        union = (boxA[2]-boxA[0])*(boxA[3]-boxA[1]) + (boxB[2]-boxB[0])*(boxB[3]-boxB[1]) - inter
        return inter / (union + 1e-6)

    def step(self, action):
        old_iou = self._get_iou(self.current_box, self.gt_box)
        x1, y1, x2, y2 = self.current_box
        w, h = x2 - x1, y2 - y1
        alpha = self.cfg.MOVE_ALPHA
        
        done = False
        if action == 0: x1 -= w*alpha; x2 -= w*alpha # L
        elif action == 1: x1 += w*alpha; x2 += w*alpha # R
        elif action == 2: y1 -= h*alpha; y2 -= h*alpha # U
        elif action == 3: y1 += h*alpha; y2 += h*alpha # D
        elif action == 4: x1+=w*alpha; y1+=h*alpha; x2-=w*alpha; y2-=h*alpha # In
        elif action == 5: x1-=w*alpha; y1-=h*alpha; x2+=w*alpha; y2+=h*alpha # Out
        elif action == 6: done = True # Trigger
        
        self.current_box = [np.clip(x1,0,self.img_w), np.clip(y1,0,self.img_h), 
                            np.clip(x2,0,self.img_w), np.clip(y2,0,self.img_h)]
        
        new_iou = self._get_iou(self.current_box, self.gt_box)
        reward = 1 if new_iou > old_iou else -1
        if action == 6: reward = 10 if new_iou > 0.5 else -10
        
        self.steps += 1
        if self.steps >= self.cfg.MAX_STEPS: done = True
        return self._get_obs(), reward, done