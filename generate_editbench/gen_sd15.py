import torch
import cv2
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
from diffusers import StableDiffusionInpaintPipeline

def compute_mask(original_image, masked_image):
    # Convert to numpy
    orig = np.array(original_image)
    masked = np.array(masked_image)
    
    # Compute difference
    diff = np.abs(orig.astype(int) - masked.astype(int))
    mask = np.any(diff > 10, axis=2).astype(np.uint8) * 255
    
    return Image.fromarray(mask).convert("RGB")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="/home/admin/workspace/aop_lab/app_data/.cache/models--stable-diffusion-v1-5--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb")
parser.add_argument('--csv_path', type=str, required=True, help="Path to the CSV file")
parser.add_argument('--dataset_root', type=str, required=True, help="Root directory of the dataset")
parser.add_argument('--output_dir', type=str, default="results/sd15")
parser.add_argument('--col_image', type=str, default="image_path")
parser.add_argument('--col_masked', type=str, default="masked_image_path")
parser.add_argument('--col_prompt', type=str, default="prompt")

args = parser.parse_args()

device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

# Load Model
print(f"Loading SD 1.5 from {args.model_path}")
pipe = StableDiffusionInpaintPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
pipe.to(device)

# Load CSV
df = pd.read_csv(args.csv_path)
print(f"Loaded {len(df)} rows from {args.csv_path}")

os.makedirs(args.output_dir, exist_ok=True)

for idx, row in df.iterrows():
    try:
        img_name = row[args.col_image]
        masked_name = row[args.col_masked]
        prompt = row[args.col_prompt]
        
        # Construct full paths
        orig_path = os.path.join(args.dataset_root, img_name)
        masked_path = os.path.join(args.dataset_root, masked_name)
        
        if not os.path.exists(orig_path) or not os.path.exists(masked_path):
            continue
            
        save_name = os.path.basename(img_name)
        save_path = os.path.join(args.output_dir, save_name)
        
        if os.path.exists(save_path):
            print(f"Exists: {save_name}")
            continue
            
        # Load Images and Force Resize to 1024x1024
        orig_image = Image.open(orig_path).convert("RGB").resize((1024, 1024))
        masked_image_input = Image.open(masked_path).convert("RGB").resize((1024, 1024))
        
        width, height = 1024, 1024

        # Compute Mask from resized images
        mask_image = compute_mask(orig_image, masked_image_input)
        
        print(f"Generating {save_name} ({width}x{height})...")
        
        # Inpainting
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
        print(f"Error processing row {idx}: {e}")

print("Done.")