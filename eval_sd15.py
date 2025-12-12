import torch
import cv2
import json
import os
import numpy as np
from PIL import Image
import argparse
from diffusers import StableDiffusionInpaintPipeline

def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', 
                    type=str, 
                    default="/home/admin/workspace/aop_lab/app_data/.cache/models--stable-diffusion-v1-5--stable-diffusion-inpainting/snapshots/8a4288a76071f7280aedbdb3253bdb9e9d5d84bb")
parser.add_argument('--image_save_path', 
                    type=str, 
                    default="runs/evaluation_result/BrushBench/sd15_inpainting/inside")
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

print(f"Loading SD 1.5 Inpainting model from {args.model_path}")
try:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16
    )
    pipe.to(device)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

with open(args.mapping_file,"r") as f:
    mapping_file=json.load(f)

print(f"Generating images to {args.image_save_path}...")
for key, item in mapping_file.items():
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
        print(f"image {key} exists! skip...")
        continue
    
    print(f"Generating {key}...")
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

print("Generation complete.")
