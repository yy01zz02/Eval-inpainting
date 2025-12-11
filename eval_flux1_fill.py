import torch
import cv2
import json
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError
from urllib.request import urlretrieve 
import open_clip
import hpsv2
import ImageReward as RM
import math
from transformers import AutoProcessor, AutoModel
from diffusers import FluxFillPipeline

def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)

class MetricsCalculator:
    def __init__(self, device,ckpt_path="data/ckpt") -> None:
        self.device=device
        # clip
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        # lpips
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        # aesthetic model
        self.aesthetic_model = torch.nn.Linear(768, 1)
        aesthetic_model_url = (
                    "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
                )
        aesthetic_model_ckpt_path=os.path.join(ckpt_path,"sa_0_4_vit_l_14_linear.pth")
        if not os.path.exists(ckpt_path):
             os.makedirs(ckpt_path)
        if not os.path.exists(aesthetic_model_ckpt_path):
            urlretrieve(aesthetic_model_url, aesthetic_model_ckpt_path)
        self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path))
        self.aesthetic_model.eval()
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        # image reward model
        self.imagereward_model = RM.load("ImageReward-v1.0")
 

    def calculate_image_reward(self,image,prompt):
        reward = self.imagereward_model.score(prompt, [image])
        return reward

    def calculate_hpsv21_score(self,image,prompt):
        result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
        return result.item()

    def calculate_aesthetic_score(self,img):
        image = self.clip_preprocess(img).unsqueeze(0)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.aesthetic_model(image_features)
        return prediction.cpu().item()

    def calculate_clip_similarity(self, img, txt):
        img = np.array(img)
        
        img_tensor=torch.tensor(img).permute(2,0,1).to(self.device)
        
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        
        return score
    
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

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', 
                    type=str, 
                    default="/home/admin/workspace/aop_lab/app_data/.cache/models--black-forest-labs--FLUX.1-Fill-dev/snapshots/6d92c296315be5f54318a6d2659f11465339f934")
parser.add_argument('--image_save_path', 
                    type=str, 
                    default="runs/evaluation_result/BrushBench/flux1_fill/inside")
parser.add_argument('--mapping_file', 
                    type=str, 
                    default="data/BrushBench/mapping_file.json")
parser.add_argument('--base_dir', 
                    type=str, 
                    default="data/BrushBench")
parser.add_argument('--mask_key', 
                    type=str, 
                    default="inpainting_mask")
parser.add_argument('--blended', action='store_true')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Flux.1 Fill model from {args.model_path}")
try:
    pipe = FluxFillPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

with open(args.mapping_file,"r") as f:
    mapping_file=json.load(f)

for key, item in mapping_file.items():
    print(f"generating image {key} ...")
    image_path=item["image"]
    mask=item[args.mask_key]
    caption=item["caption"]
   
    init_image = cv2.imread(os.path.join(args.base_dir,image_path))[:,:,::-1]
    mask_image = rle2mask(mask,(512,512))[:,:,np.newaxis]
    init_image = init_image * (1-mask_image)

    init_image = Image.fromarray(init_image).convert("RGB")
    mask_image = Image.fromarray(mask_image.repeat(3,-1)*255).convert("RGB")

    generator = torch.Generator(device).manual_seed(1234)

    save_path= os.path.join(args.image_save_path,image_path) 
    masked_image_save_path=save_path.replace(".jpg","_masked.jpg")

    if os.path.exists(save_path) and os.path.exists(masked_image_save_path):
        print(f"image {key} exitst! skip...")
        continue

    kwargs = {
        "prompt": caption,
        "image": init_image,
        "mask_image": mask_image,
        "num_inference_steps": 50,
        "generator": generator,
        "height": 512,
        "width": 512,
    }
    image = pipe(**kwargs).images[0]
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    if args.blended:
        mask_np=rle2mask(mask,(512,512))[:,:,np.newaxis]
        image_np=np.array(image)
        init_image_np=cv2.imread(os.path.join(args.base_dir,image_path))[:,:,::-1]

        # blur
        mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
        mask_blurred = mask_blurred[:,:,np.newaxis]
        mask_np = 1-(1-mask_np) * (1-mask_blurred)

        image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
        image_pasted=image_pasted.astype(image_np.dtype)
        image=Image.fromarray(image_pasted)

    image.save(save_path)
    init_image.save(masked_image_save_path)

# evaluation
evaluation_df = pd.DataFrame(columns=['Image ID','Image Reward', 'HPS V2.1', 'Aesthetic Score', 'PSNR', 'LPIPS', 'MSE', 'CLIP Similarity'])

metrics_calculator=MetricsCalculator(device)

for key, item in mapping_file.items():
    print(f"evaluating image {key} ...")
    image_path=item["image"]
    mask=item[args.mask_key]
    prompt=item["caption"]

    src_image_path = os.path.join(args.base_dir, image_path)
    src_image = Image.open(src_image_path).resize((512,512))

    tgt_image_path=os.path.join(args.image_save_path, image_path)
    tgt_image = Image.open(tgt_image_path).resize((512,512))

    evaluation_result=[key]
        
    mask = rle2mask(mask,(512,512))
    mask = 1 - mask[:,:,np.newaxis]

    for metric in evaluation_df.columns.values.tolist()[1:]:
        print(f"evluating metric: {metric}")

        if metric == 'Image Reward':
            metric_result = metrics_calculator.calculate_image_reward(tgt_image,prompt)
            
        if metric == 'HPS V2.1':
            metric_result = metrics_calculator.calculate_hpsv21_score(tgt_image,prompt)
        
        if metric == 'Aesthetic Score':
            metric_result = metrics_calculator.calculate_aesthetic_score(tgt_image)
        
        if metric == 'PSNR':
            metric_result = metrics_calculator.calculate_psnr(src_image, tgt_image, mask)
        
        if metric == 'LPIPS':
            metric_result = metrics_calculator.calculate_lpips(src_image, tgt_image, mask)
        
        if metric == 'MSE':
            metric_result = metrics_calculator.calculate_mse(src_image, tgt_image, mask)
        
        if metric == 'CLIP Similarity':
            metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)

        evaluation_result.append(metric_result)
    
    evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

print("The averaged evaluation result:")
averaged_results=evaluation_df.mean(numeric_only=True)
print(averaged_results)
averaged_results.to_csv(os.path.join(args.image_save_path,"evaluation_result_sum.csv"))
evaluation_df.to_csv(os.path.join(args.image_save_path,"evaluation_result.csv"))

print(f"The generated images and evaluation results is saved in {args.image_save_path}")