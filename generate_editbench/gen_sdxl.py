import torch
import cv2
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
from diffusers import StableDiffusionXLInpaintPipeline

def compute_mask(original_image, masked_image):
    orig = np.array(original_image)
    masked = np.array(masked_image)
    diff = np.abs(orig.astype(int) - masked.astype(int))
    mask = np.any(diff > 10, axis=2).astype(np.uint8) * 255
    return Image.fromarray(mask).convert("RGB")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="/home/admin/workspace/aop_lab/app_data/.cache/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1/snapshots/115134f363124c53c7d878647567d04daf26e41e")
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--dataset_root', type=str, required=True)
parser.add_argument('--output_dir', type=str, default="results/sdxl")
parser.add_argument('--col_image', type=str, default="image_path")
parser.add_argument('--col_masked', type=str, default="masked_image_path")
parser.add_argument('--col_prompt', type=str, default="prompt")

args = parser.parse_args()

device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

print(f"Loading SDXL from {args.model_path}")
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
pipe.to(device)

df = pd.read_csv(args.csv_path)
os.makedirs(args.output_dir, exist_ok=True)

for idx, row in df.iterrows():
    try:
        img_name = row[args.col_image]
        masked_name = row[args.col_masked]
        prompt = row[args.col_prompt]
        
        orig_path = os.path.join(args.dataset_root, img_name)
        masked_path = os.path.join(args.dataset_root, masked_name)
        
        if not os.path.exists(orig_path) or not os.path.exists(masked_path):
            continue
            
        save_name = os.path.basename(img_name)
        save_path = os.path.join(args.output_dir, save_name)
        if os.path.exists(save_path): continue
            
        orig_image = Image.open(orig_path).convert("RGB")
        masked_image_input = Image.open(masked_path).convert("RGB")
        
        if orig_image.size != masked_image_input.size:
            masked_image_input = masked_image_input.resize(orig_image.size)
            
        width, height = orig_image.size
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        if orig_image.size != (width, height):
            orig_image = orig_image.resize((width, height))
            masked_image_input = masked_image_input.resize((width, height))

        mask_image = compute_mask(orig_image, masked_image_input)
        
        print(f"Generating {save_name} ({width}x{height})...")
        
        result = pipe(
            prompt=prompt,
            image=orig_image,
            mask_image=mask_image,
            height=height,
            width=width,
            num_inference_steps=50
        ).images[0]
        
        result.save(save_path)
        
    except Exception as e:
        print(f"Error {idx}: {e}")

print("Done.")
