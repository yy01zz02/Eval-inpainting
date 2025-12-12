import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
from transformers import CLIPProcessor, CLIPModel

MODEL_PATH = "/home/admin/workspace/aop_lab/app_data/.cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"
PORT = 8001

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
processor = None

class ImageRequest(BaseModel):
    image_base64: str
    prompt: str = ""

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

@app.on_event("startup")
def load_model():
    global model, processor
    print(f"[CLIP] Loading from {MODEL_PATH}")
    try:
        model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device).eval()
        processor = CLIPProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        print("[CLIP] Loaded successfully.")
    except Exception as e:
        print(f"[CLIP] Failed to load: {e}")
        raise e

@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        inputs = processor(
            text=[req.prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Normalize
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            
            # Cosine similarity
            score = (text_embeds @ image_embeds.T).item()
            
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
