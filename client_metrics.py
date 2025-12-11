import torch
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
import math
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class MetricsCalculator:
    def __init__(self, device, ckpt_path=None, server_url="http://localhost:8000") -> None:
        self.device = device
        self.server_url = server_url
        # lpips (keep local)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)

    def _encode_image(self, image):
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Check if it's already PIL
        if not isinstance(image, Image.Image):
             # Try converting assuming it's a tensor or other compatible type
             # For this script, inputs are likely PIL or numpy
             pass
        
        buffered = BytesIO()
        image.save(buffered, format="PNG") 
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def calculate_image_reward(self, image, prompt):
        try:
            img_str = self._encode_image(image)
            resp = requests.post(f"{self.server_url}/score/image_reward", json={"image_base64": img_str, "prompt": prompt})
            resp.raise_for_status()
            return resp.json()['score']
        except Exception as e:
            print(f"Error calculating ImageReward: {e}")
            return 0.0

    def calculate_hpsv21_score(self, image, prompt):
        try:
            img_str = self._encode_image(image)
            resp = requests.post(f"{self.server_url}/score/hpsv2", json={"image_base64": img_str, "prompt": prompt})
            resp.raise_for_status()
            return resp.json()['score']
        except Exception as e:
            print(f"Error calculating HPSv2: {e}")
            return 0.0

    def calculate_aesthetic_score(self, img):
        try:
            img_str = self._encode_image(img)
            resp = requests.post(f"{self.server_url}/score/aesthetic", json={"image_base64": img_str})
            resp.raise_for_status()
            return resp.json()['score']
        except Exception as e:
            print(f"Error calculating Aesthetic Score: {e}")
            return 0.0

    def calculate_clip_similarity(self, img, txt):
        try:
            img_str = self._encode_image(img)
            resp = requests.post(f"{self.server_url}/score/clip", json={"image_base64": img_str, "prompt": txt})
            resp.raise_for_status()
            return resp.json()['score']
        except Exception as e:
            print(f"Error calculating CLIP Score: {e}")
            return 0.0
    
    def calculate_psnr(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255.
        img_gt = np.array(img_gt).astype(np.float32)/255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
        
        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum/difference_size

        if mse < 1.0e-10:
            return 1000
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    
    def calculate_lpips(self, img_gt, img_pred, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255.
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask 
            img_gt = img_gt * mask
            
        img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        score =  self.lpips_metric_calculator(img_pred_tensor*2-1,img_gt_tensor*2-1)
        score = score.cpu().item()
        
        return score
    
    def calculate_mse(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255.
        img_gt = np.array(img_gt).astype(np.float32)/255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
        
        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum()

        mse = difference_square_sum/difference_size

        return mse.item()
