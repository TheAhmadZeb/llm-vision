"""
Florence-2 + EasyOCR Vision API Server
Florence-2-base (230M params, ~770MB) for captioning + object detection,
EasyOCR for dedicated text extraction. OpenAI-compatible endpoint.
"""
import argparse, base64, io, re, time, json, os
from typing import Optional, List
import concurrent.futures
from collections import Counter

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import easyocr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np

app = FastAPI(title="Vision API (Florence-2-base + EasyOCR)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Globals ───────────────────────────────────────────────────────────
florence_model = None
florence_processor = None
easyocr_reader = None
device = "cpu"
dtype = torch.float32

# ─── Request/Response Models ───────────────────────────────────────────
class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None

class ChatMessage(BaseModel):
    role: str
    content: List[MessageContent]

class ChatRequest(BaseModel):
    model: str = "florence-2"
    messages: List[ChatMessage]
    max_tokens: int = 1024
    temperature: Optional[float] = None

class ChatChoice(BaseModel):
    index: int = 0
    message: dict

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]

# ─── Florence-2 Helpers ───────────────────────────────────────────────
def run_florence_task(task: str, image) -> dict:
    inputs = florence_processor(text=task, images=image, return_tensors="pt").to(device, dtype)
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return florence_processor.post_process_generation(
        generated_text, task=task, image_size=(image.width, image.height)
    )

def format_florence_caption(parsed: dict) -> str:
    text = parsed.get("<DETAILED_CAPTION>", "") or parsed.get("<CAPTION>", "")
    return text if text else "I couldn't make out what's in this image."

def format_florence_od(parsed: dict) -> str:
    result = parsed.get("<OD>", {})
    labels = result.get("labels", [])
    if not labels:
        return ""
    counts = Counter(labels)
    lines = [f"**Detected {len(labels)} objects ({len(counts)} unique types):**"]
    for label, count in counts.most_common():
        lines.append(f"  • {label}: {count}")
    return "\n".join(lines)

# ─── EasyOCR Helper ───────────────────────────────────────────────────
def run_easyocr(image) -> str:
    img_array = np.array(image)
    results = easyocr_reader.readtext(img_array, detail=1, paragraph=False)
    if not results:
        return ""
    min_confidence = 0.5
    min_text_length = 2
    clean = [(text, conf) for (_, text, conf) in results
             if conf >= min_confidence and len(text.strip()) >= min_text_length]
    if not clean:
        return ""
    lines = ["**Text found in image:**"]
    for text, conf in clean:
        lines.append(f"  • {text}")
    return "\n".join(lines)

# ─── Combined Pipeline ────────────────────────────────────────────────
def process_image(image) -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fut_caption = executor.submit(run_florence_task, "<DETAILED_CAPTION>", image)
        fut_od = executor.submit(run_florence_task, "<OD>", image)
        fut_ocr = executor.submit(run_easyocr, image)
        
        florence_caption_result = fut_caption.result()
        florence_od_result = fut_od.result()
        ocr_text = fut_ocr.result()
    
    caption_text = format_florence_caption(florence_caption_result)
    od_text = format_florence_od(florence_od_result)
    
    parts = [caption_text]
    if od_text:
        parts.append(od_text)
    if ocr_text:
        parts.append(ocr_text)
    
    combined = "\n\n".join(parts)
    tasks_used = "<DETAILED_CAPTION>+<OD>+<OCR>"
    
    return {
        "text": combined,
        "tasks": tasks_used,
        "ocr_detected": bool(ocr_text),
        "objects_detected": len(florence_od_result.get("<OD>", {}).get("labels", [])) > 0
    }

# ─── API Endpoint ─────────────────────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    global florence_model, florence_processor, easyocr_reader
    
    user_msg = None
    for msg in reversed(req.messages):
        if msg.role == "user":
            user_msg = msg
            break
    
    if not user_msg:
        raise HTTPException(400, "No user message found")
    
    image = None
    prompt = ""
    
    for part in user_msg.content:
        if part.type == "image_url" and part.image_url:
            url_or_data = part.image_url.get("url", "")
            if url_or_data.startswith("data:image"):
                header, _, b64data = url_or_data.partition(",")
                img_bytes = base64.b64decode(b64data)
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            elif url_or_data.startswith("http"):
                raise HTTPException(400, "URL-based images not supported, use base64")
            else:
                raise HTTPException(400, "Unsupported image format")
        elif part.type == "text":
            prompt = part.text or ""
    
    if image is None:
        raise HTTPException(400, "No image found in request")
    
    try:
        t0 = time.time()
        result = process_image(image)
        elapsed = time.time() - t0
        footer = f"\n\n_({elapsed:.1f}s, {result['tasks']})_"
        response_text = result["text"] + footer
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    
    return ChatResponse(
        id=f"vision-{int(time.time())}",
        created=int(time.time()),
        model=req.model,
        choices=[ChatChoice(message={"role": "assistant", "content": response_text})]
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "florence_loaded": florence_model is not None,
        "easyocr_loaded": easyocr_reader is not None,
        "device": device
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "florence-2", "object": "model", "created": int(time.time()), "owned_by": "microsoft"}]
    }

# ─── Startup ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision API Server (Florence-2-base + EasyOCR)")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--model-size", choices=["base", "large"], default="base",
                        help="Florence-2 model size (base=230M, large=770M)")
    args = parser.parse_args()
    
    model_id = f"microsoft/Florence-2-{args.model_size}"
    
    print(f"Loading Florence-2 model ({model_id})...")
    t0 = time.time()
    florence_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    florence_model.eval()
    print(f"Florence-2 loaded in {time.time()-t0:.1f}s on {device}")
    
    print("Loading EasyOCR...")
    t0 = time.time()
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
    print(f"EasyOCR loaded in {time.time()-t0:.1f}s")
    
    print(f"Models loaded. Starting server on port {args.port}...")
    print("Memory: Florence-2-base (~770MB) + EasyOCR (~500MB) permanently resident")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
