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
    # Assuming the 'removed' parts are different (e.g. black, grey, or noise)
    # compared to the original.
    # If the user provides "original with mask removed", it usually means the hole is black.
    # But checking difference with original is robust.
    diff = np.abs(orig.astype(int) - masked.astype(int))
    mask = np.any(diff > 10, axis=2).astype(np.uint8) * 255
    
    # Post-process mask (fill holes, dilate) if needed, but raw diff is usually fine 
    # if the hole is clean.
    return Image.fromarray(mask).convert("RGB")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="/home/admin/workspace/aop_lab/app_data/.cache/models--stable-diffusion-v1-5--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb")
parser.add_argument('--csv_path', type=str, required=True, help="Path to the CSV file")
parser.add_argument('--dataset_root', type=str, required=True, help="Root directory of the dataset")
parser.add_argument('--output_dir', type=str, default="results/sd15")
# CSV Columns
parser.add_argument('--col_image', type=str, default="image_path", help="Column name for original image filename/path")
parser.add_argument('--col_masked', type=str, default="masked_image_path", help="Column name for masked image filename/path")
parser.add_argument('--col_prompt', type=str, default="prompt", help="Column name for prompt")

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
            print(f"Missing files for {img_name}. Skipping.")
            continue
            
        save_name = os.path.basename(img_name)
        save_path = os.path.join(args.output_dir, save_name)
        
        if os.path.exists(save_path):
            print(f"Exists: {save_name}")
            continue
            
        # Load Images
        orig_image = Image.open(orig_path).convert("RGB")
        masked_image_input = Image.open(masked_path).convert("RGB")
        
        # Ensure sizes match
        if orig_image.size != masked_image_input.size:
            masked_image_input = masked_image_input.resize(orig_image.size)
            
        width, height = orig_image.size
        # Align to 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        if orig_image.size != (width, height):
            orig_image = orig_image.resize((width, height))
            masked_image_input = masked_image_input.resize((width, height))

        # Compute Mask
        mask_image = compute_mask(orig_image, masked_image_input)
        
        print(f"Generating {save_name} ({width}x{height})...")
        
        # Inpainting
        # Note: 'image' in pipe is usually the original image which the pipeline uses reference from.
        # 'mask_image' defines where to paint.
        # SD1.5 Inpaint expects 'image' and 'mask_image'.
        result = pipe(
            prompt=prompt,
            image=orig_image, # Use original as base
            mask_image=mask_image,
            height=height,
            width=width,
            num_inference_steps=50
        ).images[0]
        
        result.save(save_path)
        
    except Exception as e:
        print(f"Error processing row {idx}: {e}")

print("Done.")
