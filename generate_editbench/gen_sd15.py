import torch
import cv2
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
from diffusers import StableDiffusionInpaintPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="/home/admin/workspace/aop_lab/app_data/.cache/models--stable-diffusion-v1-5--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb")
parser.add_argument('--csv_path', type=str, required=True, help="Path to annotations_*.csv")
parser.add_argument('--image_folder', type=str, required=True, help="Folder containing reference images (e.g. references_generated)")
parser.add_argument('--mask_folder', type=str, required=True, help="Folder containing masks (e.g. masks_generated)")
parser.add_argument('--output_dir', type=str, default="results/sd15")
parser.add_argument('--prompt_col', type=str, default="prompt_mask-rich", help="Column name for prompt (e.g. prompt_mask-rich)")

args = parser.parse_args()

device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# Load Model
print(f"Loading SD 1.5 from {args.model_path}")
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load CSV
df = pd.read_csv(args.csv_path)
print(f"Loaded {len(df)} rows from {args.csv_path}")

os.makedirs(args.output_dir, exist_ok=True)

for idx, row in df.iterrows():
    try:
        aos_id = row['aos']
        prompt = row[args.prompt_col]
        
        # Construct filenames
        img_name = f"{aos_id}.png"
        
        img_path = os.path.join(args.image_folder, img_name)
        mask_path = os.path.join(args.mask_folder, img_name)
        
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"Mask not found: {mask_path}")
            continue
            
        save_path = os.path.join(args.output_dir, img_name)
        if os.path.exists(save_path):
            print(f"Exists: {img_name}")
            continue
            
        # Load and Resize
        # EditBench images are PNG.
        # Force resize to 1024x1024 as requested
        orig_image = Image.open(img_path).convert("RGB").resize((1024, 1024))
        # Mask is binary, convert to L (grayscale)
        mask_image = Image.open(mask_path).convert("L").resize((1024, 1024))
        
        width, height = 1024, 1024
        
        print(f"Generating {img_name} with prompt: '{prompt[:30]}...'")
        
        # Inference
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
        print(f"Error processing row {idx} ({row.get('aos', 'unknown')}): {e}")

print("Done.")
