import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
import open_clip
import glob

SNAPSHOT_PATH = "/home/admin/workspace/aop_lab/app_data/.cache/models--xswu--HPSv2/snapshots/697403c78157020a1ae59d23f111aa58ced35b0a"
PORT = 8004

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
preprocess = None
tokenizer = None

class ImageRequest(BaseModel):
    image_base64: str
    prompt: str

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

@app.on_event("startup")
def load_model():
    global model, preprocess, tokenizer
    print(f"[HPSv2] Loading from {SNAPSHOT_PATH}")
    
    # Find .pt file
    pt_files = glob.glob(os.path.join(SNAPSHOT_PATH, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt file found in {SNAPSHOT_PATH}")
    checkpoint_path = pt_files[0]
    print(f"[HPSv2] Checkpoint: {checkpoint_path}")

    try:
        # Create model structure (ViT-H-14)
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14',
            pretrained=None,
            precision='amp',
            device=device,
            force_quick_gelu=False,
            pretrained_image=False
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-H-14')
        print("[HPSv2] Loaded successfully.")
    except Exception as e:
        print(f"[HPSv2] Failed to load: {e}")
        raise e

@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text_tensor = tokenizer([req.prompt]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tensor)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            hps_score = image_features @ text_features.T
            
        return {"score": hps_score.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
