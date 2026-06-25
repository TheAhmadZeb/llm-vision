# Jeeves Vision API

A local, CPU‑first vision API server that combines **Florence-2** (image captioning + object detection) with **EasyOCR** (text extraction) behind an OpenAI‑compatible endpoint.

## Features

- **OpenAI‑compatible** — drop‑in replacement for GPT‑4 Vision in any app (LobeChat, Open WebUI, custom tools)
- **Triple pipeline** — every image gets:
  - **Detailed caption** (Florence‑2 `<DETAILED_CAPTION>` — describes scene, objects, people, actions)
  - **Object detection** (Florence‑2 `<OD>` — counts labelled objects per class)
  - **OCR** (EasyOCR — extracts printed text with confidence filtering)
- **Concurrent** — all three tasks run in parallel via `ThreadPoolExecutor`, returning in ~30‑40s on CPU
- **CPU‑native** — no GPU required. Runs on any machine with 4‑core CPU and ~4GB free RAM

## Quick Start

```bash
# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run (downloads models on first launch — ~770MB Florence-2 + ~500MB EasyOCR)
python server.py --port 8010

# Or as a systemd service (see systemd/jeeves-vision.service)
```

## API Usage

### `POST /v1/chat/completions`

```bash
curl -X POST http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "florence-2",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What do you see?"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
      }
    ]
  }'
```

### `GET /health`

```json
{"status": "ok", "florence_loaded": true, "easyocr_loaded": true, "device": "cpu"}
```

### `GET /v1/models`

Returns the available model (`florence-2`).

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│  Client     │────▶│  FastAPI Server  │────▶│  ThreadPoolExecutor │
│  (OpenAI    │     │  (port 8010)     │     │  ├── Florence Caption│
│   SDK)      │     │                  │     │  ├── Florence OD     │
└─────────────┘     └─────────────────┘     │  └── EasyOCR         │
                                            └──────────────────────┘
```

## Model Sizes

| Model | Parameters | RAM | Speed (CPU, 4‑core) |
|---|---|---|---|
| Florence-2-base | 230M | ~770MB | ~30s per image |
| Florence-2-large | 770M | ~2.8GB | ~60s per image |

Pass `--model-size large` for higher accuracy at the cost of speed.

## Why This Exists

Most vision APIs require a GPU. This server was built for a home server running on CPU — a practical solution for adding vision capabilities to a private AI stack without cloud costs or GPU hardware.
