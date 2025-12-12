import torch
import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
import math
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Define base URLs for services
URLS = {
    "clip": "http://localhost:8001/score",
    "aesthetic": "http://localhost:8002/score",
    "pickscore": "http://localhost:8003/score",
    "hpsv2": "http://localhost:8004/score",
    "imagereward": "http://localhost:8005/score",
    "hpsv3": "http://localhost:8006/score",
}

class MetricsCalculator:
    def __init__(self, device, server_urls=URLS) -> None:
        self.device = device
        self.server_urls = server_urls
        # lpips (keep local)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)

    def _encode_image(self, image):
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        # Check if it's already PIL
        if not isinstance(image, Image.Image):
             # Try converting assuming it's a tensor
             pass
        
        buffered = BytesIO()
        image.save(buffered, format="PNG") 
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _call_service(self, metric_name, payload):
        url = self.server_urls.get(metric_name)
        if not url:
            print(f"URL for {metric_name} not configured.")
            return 0.0
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()['score']
        except Exception as e:
            print(f"Error calculating {metric_name}: {e}")
            return 0.0

    def calculate_image_reward(self, image, prompt):
        return self._call_service("imagereward", {"image_base64": self._encode_image(image), "prompt": prompt})

    def calculate_hpsv21_score(self, image, prompt):
        return self._call_service("hpsv2", {"image_base64": self._encode_image(image), "prompt": prompt})

    def calculate_aesthetic_score(self, img):
        return self._call_service("aesthetic", {"image_base64": self._encode_image(img)})

    def calculate_clip_similarity(self, img, txt):
        return self._call_service("clip", {"image_base64": self._encode_image(img), "prompt": txt})
    
    def calculate_pick_score(self, img, txt):
        return self._call_service("pickscore", {"image_base64": self._encode_image(img), "prompt": txt})

    def calculate_hpsv3_score(self, img, txt):
        return self._call_service("hpsv3", {"image_base64": self._encode_image(img), "prompt": txt})
    
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