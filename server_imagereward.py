import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
import ImageReward as RM

MODEL_PATH = "/home/admin/workspace/aop_lab/app_data/.cache/models--zai-org--ImageReward/snapshots/5736be03b2652728fb87788c9797b0570450ab72"
PORT = 8005

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None

class ImageRequest(BaseModel):
    image_base64: str
    prompt: str

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

@app.on_event("startup")
def load_model():
    global model
    print(f"[ImageReward] Loading from {MODEL_PATH}")
    try:
        # Attempt direct load
        model = RM.load(MODEL_PATH, device=device)
        print("[ImageReward] Loaded via RM.load")
    except Exception as e:
        print(f"[ImageReward] RM.load failed ({e}), trying manual load...")
        try:
            from ImageReward.ImageReward import ImageReward
            import glob
            
            # Find .pt
            state_dict_path = os.path.join(MODEL_PATH, "ImageReward.pt")
            if not os.path.exists(state_dict_path):
                pts = glob.glob(os.path.join(MODEL_PATH, "*.pt"))
                if pts: state_dict_path = pts[0]
            
            print(f"Loading weights from {state_dict_path}")
            model = ImageReward(device=device)
            state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model.to(device)
            print("[ImageReward] Loaded manually.")
        except Exception as e2:
            print(f"[ImageReward] Manual load failed: {e2}")
            raise e2

@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        # ImageReward.score accepts prompt and list of images
        with torch.no_grad():
            score = model.score(req.prompt, [image])
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
