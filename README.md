# ModeX – AI Visual Search (CLIP + FAISS)

ModeX is the core backend service for an AI-driven e-commerce **visual search** experience. It uses **OpenAI CLIP** to generate image embeddings and **FAISS** to run fast similarity search over a catalog of product images.

## Project Overview

- **Goal**: Given a query image (uploaded by a client), return the most visually similar products.
- **Approach**: Encode images with **CLIP ViT-B/32** into 512‑dimensional vectors, then retrieve nearest neighbors using **FAISS**.
- **API**: A **FastAPI** service exposes an image search endpoint (`POST /search`).

## System Architecture

### Indexing (offline, one-time / periodic)

1. Place product images under the `images/` folder (any nested structure is supported).
2. Run `build_index.py` to:
   - Recursively scan `images/` for `.jpg/.jpeg/.png`
   - Generate CLIP embeddings (with background removal via `rembg`)
   - Build a FAISS index (`IndexFlatIP`) for fast retrieval
   - Save:
     - `index.faiss`: the FAISS vector index
     - `metadata.json`: per-vector product metadata including `image_path`, `product_name`, `store_name`, and `price`

### Query-time search (online, per request)

1. The FastAPI service loads:
   - CLIP model once at startup
   - `index.faiss` + `metadata.json` from the project directory
2. For an uploaded query image:
   - Generate its CLIP embedding (512D)
   - Search the FAISS index for nearest neighbors (inner product)
   - Filter results using a configurable similarity `threshold`
   - Return a JSON response with product metadata and similarity scores

> Note: `IndexFlatIP` uses inner product. If embeddings are L2-normalized, inner product corresponds to cosine similarity. The query embedding is normalized in `model_utils.py`.

## Tech Stack

- **API**: FastAPI
- **ML**: PyTorch
- **Embedding model**: OpenAI CLIP (`ViT-B/32`)
- **Vector search**: FAISS (`IndexFlatIP`)
- **Image processing**: Pillow, `rembg` (background removal)

## Repository Layout (core)

- `main.py`: FastAPI app (loads model, loads index + metadata, exposes `/search`)
- `model_utils.py`: CLIP loading, embedding extraction, similarity search helpers
- `build_index.py`: builds `index.faiss` + `metadata.json` from `images/`
- `images/`: product images (not included by default unless you add them)

## How to Run

### 1) Create a virtual environment

Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

If you already have a `requirements.txt`, install from it:

```bash
pip install -r requirements.txt
```

If you don’t yet have one, you’ll need (at minimum) packages for:
FastAPI + Uvicorn, PyTorch, CLIP, FAISS, Pillow, NumPy, and `rembg`.

### 3) Build the FAISS index (required before searching)

Ensure you have an `images/` directory with product images, then:

```bash
python build_index.py
```

This produces `index.faiss` and `metadata.json` in the project root.

### 4) Start the API server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:

- `GET /health`: component status (model/index/metadata)
- `POST /search` (or `POST /api/search`): upload an image to search

### 5) Example request (curl)

```bash
curl -X POST "http://127.0.0.1:8000/search?threshold=0.85&top_k=5" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "image=@path\to\query.jpg"
```

## Notes & Gotchas

- **Index required**: If `index.faiss` is missing, the API returns a `503` asking you to run `build_index.py`.
- **Paths in metadata**: `build_index.py` stores `image_path` values as **relative paths** (intended to work well with typical media serving setups).
- **GPU optional**: The model runs on CUDA if available; otherwise CPU is used.

## License

Add your preferred license (e.g., MIT) before publishing if this will be open source.

