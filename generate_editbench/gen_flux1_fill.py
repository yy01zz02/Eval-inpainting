import torch
import cv2
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
from diffusers import FluxFillPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default="/home/admin/workspace/aop_lab/app_data/.cache/models--black-forest-labs--FLUX.1-Fill-dev/snapshots/6d92c296315be5f54318a6d2659f11465339f934")
parser.add_argument('--csv_path', type=str, required=True, help="Path to annotations_*.csv")
parser.add_argument('--image_folder', type=str, required=True, help="Folder containing reference images")
parser.add_argument('--mask_folder', type=str, required=True, help="Folder containing masks")
parser.add_argument('--output_dir', type=str, default="results/flux1_fill")
parser.add_argument('--prompt_col', type=str, default="prompt_mask-rich")

args = parser.parse_args()

device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

print(f"Loading Flux.1 Fill from {args.model_path}")
pipe = FluxFillPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload(device=device)

df = pd.read_csv(args.csv_path)
os.makedirs(args.output_dir, exist_ok=True)

for idx, row in df.iterrows():
    try:
        aos_id = row['aos']
        prompt = row[args.prompt_col]
        
        img_name = f"{aos_id}.png"
        img_path = os.path.join(args.image_folder, img_name)
        mask_path = os.path.join(args.mask_folder, img_name)
        
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            continue
            
        save_path = os.path.join(args.output_dir, img_name)
        if os.path.exists(save_path): continue
            
        # Resize to 1024x1024
        orig_image = Image.open(img_path).convert("RGB").resize((1024, 1024))
        mask_image = Image.open(mask_path).convert("L").resize((1024, 1024))
        
        width, height = 1024, 1024
        
        print(f"Generating {img_name}...")
        
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
