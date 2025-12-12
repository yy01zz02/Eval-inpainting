import os
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
from transformers import CLIPProcessor, CLIPModel

# Config
CLIP_PATH = "/home/admin/workspace/aop_lab/app_data/.cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
AESTHETIC_PATH = "sa_0_4_vit_l_14_linear.pth"
PORT = 8002

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = None
clip_processor = None
aesthetic_model = None

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1)
        )

    def forward(self, x):
        return self.layers(x)

class ImageRequest(BaseModel):
    image_base64: str

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

@app.on_event("startup")
def load_models():
    global clip_model, clip_processor, aesthetic_model
    print(f"[Aesthetic] Loading CLIP from {CLIP_PATH}")
    clip_model = CLIPModel.from_pretrained(CLIP_PATH, local_files_only=True).to(device).eval()
    clip_processor = CLIPProcessor.from_pretrained(CLIP_PATH, local_files_only=True)
    
    print(f"[Aesthetic] Loading linear weights from {AESTHETIC_PATH}")
    aesthetic_model = AestheticPredictor(768).to(device).eval()
    
    try:
        state_dict = torch.load(AESTHETIC_PATH, map_location=device)
        # 尝试直接加载
        try:
            aesthetic_model.load_state_dict(state_dict)
        except:
            # 适配常见的简单保存格式
            if 'weight' in state_dict and 'bias' in state_dict:
                aesthetic_model.layers[0].weight.data = state_dict['weight']
                aesthetic_model.layers[0].bias.data = state_dict['bias']
            else:
                # 尝试其他可能的 key 映射
                new_state_dict = {}
                for k, v in state_dict.items():
                    if "weight" in k: new_state_dict["layers.0.weight"] = v
                    if "bias" in k: new_state_dict["layers.0.bias"] = v
                aesthetic_model.load_state_dict(new_state_dict)
        print("[Aesthetic] Loaded successfully.")
    except Exception as e:
        print(f"[Aesthetic] Failed to load weights: {e}")
        raise e

@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        # CLIP features (text is ignored for aesthetic model usually, just image features)
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            vision_outputs = clip_model.get_image_features(**inputs)
            # Normalize
            vision_embeds = vision_outputs / vision_outputs.norm(dim=-1, keepdim=True)
            # Linear probe
            prediction = aesthetic_model(vision_embeds)
            
        return {"score": prediction.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
