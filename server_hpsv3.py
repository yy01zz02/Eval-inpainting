import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import base64
import io
import uvicorn
from hpsv3 import HPSv3RewardInferencer

# Default ModelScope path usually
MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/MizzenAI/HPSv3")
PORT = 8006

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
inferencer = None

class ImageRequest(BaseModel):
    image_base64: str
    prompt: str

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

@app.on_event("startup")
def load_model():
    global inferencer
    print(f"[HPSv3] Loading from {MODEL_PATH}")
    
    # Check if path exists, otherwise try standard load which might download
    path_to_use = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    
    try:
        if path_to_use:
            inferencer = HPSv3RewardInferencer(model_name_or_path=path_to_use, device=device)
        else:
            print("[HPSv3] Local path not found, letting HPSv3 library handle loading (might download)...")
            inferencer = HPSv3RewardInferencer(device=device)
        print("[HPSv3] Loaded successfully.")
    except Exception as e:
        print(f"[HPSv3] Failed to load: {e}")
        # Fallback attempt
        try:
             inferencer = HPSv3RewardInferencer(device=device)
        except Exception as e2:
             print(f"[HPSv3] Retry failed: {e2}")
             raise e2

@app.post("/score")
async def score(req: ImageRequest):
    try:
        image = decode_image(req.image_base64)
        # HPSv3 expects file paths mostly, but let's check if it accepts PIL
        # Looking at md: "image_paths list".
        # We need to save to temp file because the library might expect paths.
        # Or check if it supports PIL.
        # If not, we save temp.
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp, format="JPEG")
            tmp_path = tmp.name
        
        try:
            # result is list of scores
            result = inferencer.reward(prompts=[req.prompt], image_paths=[tmp_path])
            score = result[0].item()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
