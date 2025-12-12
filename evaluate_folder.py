import os
import json
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from client_metrics import MetricsCalculator
import torch

def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)

parser = argparse.ArgumentParser()
parser.add_argument('--generated_image_folder', type=str, required=True, help="Path to the folder containing generated images")
parser.add_argument('--base_dir', type=str, default="data/BrushBench", help="Path to the base directory containing ground truth images")
parser.add_argument('--mapping_file', type=str, default="data/BrushBench/mapping_file.json")
parser.add_argument('--mask_key', type=str, default="inpainting_mask")
parser.add_argument('--output_name', type=str, default="evaluation_result")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Evaluation running on {device}")

# Initialize metrics calculator (connects to servers on ports 8001-8006)
metrics_calculator = MetricsCalculator(device)

with open(args.mapping_file, "r") as f:
    mapping_file = json.load(f)

# Extended metrics list
evaluation_df = pd.DataFrame(columns=[
    'Image ID',
    'Image Reward', 'HPS V2.1', 'Aesthetic Score', 'CLIP Score', 'PickScore', 'HPS v3',
    'PSNR', 'LPIPS', 'MSE'
])

print(f"Starting evaluation...")
print(f"Generated Images: {args.generated_image_folder}")
print(f"Ground Truth: {args.base_dir}")

for key, item in mapping_file.items():
    image_path = item["image"]
    mask_rle = item[args.mask_key]
    prompt = item["caption"]

    # Paths
    src_image_path = os.path.join(args.base_dir, image_path)
    tgt_image_path = os.path.join(args.generated_image_folder, image_path)

    # Check if generated image exists
    if not os.path.exists(tgt_image_path):
        print(f"Warning: Generated image not found for {key} at {tgt_image_path}. Skipping.")
        continue

    print(f"Evaluating {key}...")

    # Load Images (force 512x512 for standard evaluation)
    src_image = Image.open(src_image_path).convert("RGB").resize((512, 512))
    tgt_image = Image.open(tgt_image_path).convert("RGB").resize((512, 512))

    # Prepare Mask
    mask = rle2mask(mask_rle, (512, 512))
    mask = 1 - mask[:, :, np.newaxis]
    
    evaluation_result = [key]

    # Calculate Metrics
    # 1. Image Reward
    evaluation_result.append(metrics_calculator.calculate_image_reward(tgt_image, prompt))
    
    # 2. HPS v2.1
    evaluation_result.append(metrics_calculator.calculate_hpsv21_score(tgt_image, prompt))
    
    # 3. Aesthetic Score
    evaluation_result.append(metrics_calculator.calculate_aesthetic_score(tgt_image))

    # 4. CLIP Score
    evaluation_result.append(metrics_calculator.calculate_clip_similarity(tgt_image, prompt))

    # 5. PickScore
    evaluation_result.append(metrics_calculator.calculate_pick_score(tgt_image, prompt))

    # 6. HPS v3
    evaluation_result.append(metrics_calculator.calculate_hpsv3_score(tgt_image, prompt))
    
    # 7. PSNR
    evaluation_result.append(metrics_calculator.calculate_psnr(src_image, tgt_image, mask))
    
    # 8. LPIPS
    evaluation_result.append(metrics_calculator.calculate_lpips(src_image, tgt_image, mask))
    
    # 9. MSE
    evaluation_result.append(metrics_calculator.calculate_mse(src_image, tgt_image, mask))

    evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

# Save Results
print("The averaged evaluation result:")
averaged_results = evaluation_df.mean(numeric_only=True)
print(averaged_results)

save_dir = args.generated_image_folder
averaged_results.to_csv(os.path.join(save_dir, f"{args.output_name}_sum.csv"))
evaluation_df.to_csv(os.path.join(save_dir, f"{args.output_name}.csv"))

print(f"Evaluation results saved in {save_dir}")