import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
from transformers import AutoProcessor, AutoModel

MODEL_PATH = "/home/admin/workspace/aop_lab/app_data/.cache/models--yuvalkirstain--PickScore_v1/snapshots/a4e4367c6dfa7288a00c550414478f865b875800"
PORT = 8003

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
processor = None

class ImageRequest(BaseModel):
    image_base64: str
    prompt: str

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

@app.on_event("startup")
def load_model():
    global model, processor
    print(f"[PickScore] Loading from {MODEL_PATH}")
    try:
        processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).eval().to(device)
        print("[PickScore] Loaded successfully.")
    except Exception as e:
        print(f"[PickScore] Failed to load: {e}")
        raise e

@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        inputs = processor(
            images=image,
            text=req.prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            image_embs = model.get_image_features(**inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
            
            text_embs = model.get_text_features(**inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
            
            # PickScore calc
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)
            
        return {"score": scores.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
