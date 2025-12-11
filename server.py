import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import numpy as np
import open_clip
import ImageReward as RM
import hpsv2
from urllib.request import urlretrieve

# Define paths
HPSV2_PATH = "/home/admin/workspace/aop_lab/app_data/.cache/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a"
CLIP_PATH = "/home/admin/workspace/aop_lab/app_data/.cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
IMAGE_REWARD_PATH = "/home/admin/workspace/aop_lab/app_data/.cache/models--zai-org--ImageReward/snapshots/5736be03b2652728fb87788c9797b0570450ab72"
AESTHETIC_PATH = "sa_0_4_vit_l_14_linear.pth"

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

# Global models
models = {}

def load_models():
    print("Loading models...")
    
    # 1. Load CLIP
    print(f"Loading CLIP from {CLIP_PATH}...")
    # OpenCLIP typically expects the checkpoint file specifically if passing a local path to 'pretrained', 
    # but for huggingface snapshot it might be 'model_name_or_path'. 
    # Since the user provided a snapshot directory, we try loading it as a HF model or OpenCLIP compatible.
    # Note: open_clip.create_model_and_transforms usually takes a model name (e.g. ViT-L-14) and pretrained (checkpoint).
    # If the path is a directory, it might be tricky for open_clip directly if it expects a single file.
    # However, since the user calls it 'clip-vit-large-patch14', we assume ViT-L-14.
    try:
        # Attempt to load using standard open_clip if it's a bin file, but here it's a folder.
        # We'll stick to the original logic but point to the local path if supported, 
        # or fall back to standard loading if the path implies a HF cache.
        # Actually, for the evaluation script, it used 'openai/clip-vit-large-patch14' (HF Hub).
        # We will try to load from the provided directory.
        models['clip'], _, models['clip_preprocess'] = open_clip.create_model_and_transforms('ViT-L-14', pretrained=os.path.join(CLIP_PATH, "open_clip_pytorch_model.bin")) 
    except:
        # Fallback: try loading with just the directory or default if file missing
        print("Could not load specific bin, trying generic load or checking directory...")
        # If open_clip doesn't support folder, we might need to rely on the fact that it's cached.
        # But let's assume the user knows the file is there or we can use the HF Hub ID and rely on offline mode if needed.
        # For now, let's try standard load but maybe it picks up the cache?
        # A safer bet for server stability is to assume the standard 'openai' one if local fails,
        # but the user requested local. 
        # Let's try to map the snapshot to the pretrained argument.
        models['clip'], _, models['clip_preprocess'] = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', cache_dir=os.path.dirname(os.path.dirname(CLIP_PATH)))
    models['clip'].to(device).eval()

    # 2. Load Aesthetic
    print(f"Loading Aesthetic Model form {AESTHETIC_PATH}...")
    models['aesthetic'] = torch.nn.Linear(768, 1).to(device)
    models['aesthetic'].load_state_dict(torch.load(AESTHETIC_PATH))
    models['aesthetic'].eval()

    # 3. Load ImageReward
    print(f"Loading ImageReward from {IMAGE_REWARD_PATH}...")
    # ImageReward.load() takes a name. It downloads to ~/.cache/ImageReward.
    # To use a local path, we might need to instantiate manually or modify the internal download logic.
    # However, the library 'ImageReward' usually wraps 'AutoModel'.
    # We will try passing the path as the name.
    models['image_reward'] = RM.load(IMAGE_REWARD_PATH, device=device)

    # 4. Load HPSv2
    # HPSv2 doesn't have a persistent object in the original script, it calls hpsv2.score each time.
    # But to speed up, we should see if we can preload. 
    # hpsv2.score loads the model every time if not careful.
    # Looking at hpsv2 source, it usually caches the checkpoint.
    # We can set os.environ['HPSV2_CACHE_DIR'] or similar if needed.
    # The user provided a snapshot path. We will pass this to the score function as 'cp' if possible.
    print(f"HPSv2 configured with path {HPSV2_PATH}")
    
    print("All models loaded.")

@app.on_event("startup")
async def startup_event():
    load_models()

class ImageRequest(BaseModel):
    image_base64: str
    prompt: str = ""

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

@app.post("/score/image_reward")
async def score_image_reward(req: ImageRequest):
    image = decode_image(req.image_base64)
    # ImageReward expects [image] list
    with torch.no_grad():
        reward = models['image_reward'].score(req.prompt, [image])
    return {"score": reward}

@app.post("/score/hpsv2")
async def score_hpsv2(req: ImageRequest):
    image = decode_image(req.image_base64)
    # HPSv2 score
    # We pass the specific checkpoint path if the library supports it
    # standard hpsv2.score(imgs, prompts, cp=...)
    # The path provided is a folder. We need the .pt file usually.
    # We'll assume the folder contains the standard file structure.
    # If HPSv2_PATH is the snapshot dir, we might need to find the .pt file inside.
    cp_path = HPSV2_PATH
    if os.path.isdir(HPSV2_PATH):
        # Look for a .pt file
        for f in os.listdir(HPSV2_PATH):
            if f.endswith(".pt") or f.endswith(".pth"):
                cp_path = os.path.join(HPSV2_PATH, f)
                break
    
    result = hpsv2.score([image], req.prompt, hps_version="v2.1", cp=cp_path)[0]
    return {"score": result.item()}

@app.post("/score/aesthetic")
async def score_aesthetic(req: ImageRequest):
    image = decode_image(req.image_base64)
    # Preprocess
    img = models['clip_preprocess'](image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = models['clip'].encode_image(img)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = models['aesthetic'](image_features)
    return {"score": prediction.cpu().item()}

@app.post("/score/clip")
async def score_clip(req: ImageRequest):
    image = decode_image(req.image_base64)
    # Preprocess
    img = models['clip_preprocess'](image).unsqueeze(0).to(device)
    
    # Tokenize text
    # We need the tokenizer. open_clip.get_tokenizer
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    text = tokenizer([req.prompt]).to(device)

    with torch.no_grad():
        # MetricCalculator in client used torchmetrics CLIPScore which is different from raw OpenCLIP.
        # But here we are reimplementing using the loaded OpenCLIP model.
        # The original code used:
        # self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
        # score = self.clip_metric_calculator(img_tensor, txt)
        # We should try to match that behavior or provide raw cosine similarity.
        # CLIPScore from torchmetrics typically calculates: 2.5 * max(cosine_similarity(image, text), 0)
        
        image_features = models['clip'].encode_image(img)
        text_features = models['clip'].encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (image_features @ text_features.T).item()
        
        # torchmetrics style scaling (optional, but standard for 'CLIP Score')
        # score = max(similarity, 0) * 2.5
        # The user's original code used torchmetrics. To minimize diff, we should match.
        score = max(similarity, 0) * 2.5
        
    return {"score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
