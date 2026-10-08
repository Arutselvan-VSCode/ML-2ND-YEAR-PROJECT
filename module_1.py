import os
import json
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torch.utils.data import Dataset

class xBDDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        # Filter strictly for pre_disaster images to act as our base index
        self.pre_images = [f for f in os.listdir(img_dir) if 'pre_disaster.png' in f]
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
        
        # Mapping Kaggle xBD damage strings to integer classes
        self.damage_dict = {
            "un-classified": 1,
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4
        }

    def __len__(self):
        return len(self.pre_images)

    def generate_mask(self, json_path, original_size=(1024, 1024)):
        mask = np.zeros(original_size, dtype=np.uint8)
        if not os.path.exists(json_path):
            return cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        with open(json_path) as f:
            data = json.load(f)

        for feat in data['features']['xy']:
            poly_type = feat['properties'].get('feature_type', 'building')
            if poly_type == 'building':
                damage_type = feat['properties'].get('subtype', 'no-damage')
                damage_val = self.damage_dict.get(damage_type, 1)

                wkt = feat['wkt']
                if wkt.startswith('POLYGON'):
                    coords_str = wkt.replace('POLYGON', '').replace('(', '').replace(')', '').strip()
                    points = []
                    for pt in coords_str.split(','):
                        x, y = map(float, pt.strip().split())
                        points.append([int(x), int(y)])
                    
                    pts = np.array([points], dtype=np.int32)
                    cv2.fillPoly(mask, pts, damage_val)

        mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        return mask_resized

    def __getitem__(self, idx):
        pre_img_name = self.pre_images[idx]
        post_img_name = pre_img_name.replace('_pre_disaster.png', '_post_disaster.png')
        post_json_name = pre_img_name.replace('_pre_disaster.png', '_post_disaster.json')
        
        pre_path = os.path.join(self.img_dir, pre_img_name)
        post_path = os.path.join(self.img_dir, post_img_name)
        json_path = os.path.join(self.label_dir, post_json_name)
        
        pre_img = Image.open(pre_path).convert("RGB")
        post_img = Image.open(post_path).convert("RGB")
        
        pre_tensor = self.transform(pre_img)
        post_tensor = self.transform(post_img)
        
        stacked_tensor = torch.cat((pre_tensor, post_tensor), dim=0)
        
        mask_array = self.generate_mask(json_path, original_size=pre_img.size)
        label_tensor = torch.tensor(mask_array, dtype=torch.long)
        
        return stacked_tensor, label_tensor