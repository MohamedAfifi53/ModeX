"""
Model utilities extracted from mode_x_final.ipynb
Handles image embedding extraction and search functionality
"""
import torch
import clip
import faiss
import numpy as np
from PIL import Image
from rembg import remove
import json
import os

# Initialize device and model (global, loaded once)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
preprocess = None

# ==========================================
# Store Name Mapping (Strict - Only Known Brands)
# ==========================================
STORE_MAPPING = {
    'bershka': 'Bershka',
    'zara': 'Zara',
    'hm': 'H&M',
    'h&m': 'H&M',
    'h_m': 'H&M',
    'handm': 'H&M',
}

# Fallback store name when no known brand is found
DEFAULT_STORE_NAME = "Mode X Collection"

# Similarity threshold (70% for balanced precision/recall)
HIGH_PRECISION_THRESHOLD = 0.70
NO_MATCH_MESSAGE = "No matching products found, please try a different photo"

# Available stores for random assignment
AVAILABLE_STORES = ['H&M', 'ZARA', 'BERSHKA']


def get_store_name_from_path(image_path: str) -> str:
    """
    Identify store name based on folder path (STRICT MAPPING).
    
    Priority:
    1. Check for 'zara' in path (case-insensitive) → Zara
    2. Check for 'bershka' in path (case-insensitive) → Bershka
    3. Check for 'hm' or 'h&m' in path (case-insensitive) → H&M
    4. If no known brand found → 'Mode X Collection'
    
    NOTE: Does NOT use folder name as fallback.
    Only returns known brands or 'Mode X Collection'.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Store name: 'Zara', 'Bershka', 'H&M', or 'Mode X Collection'
    """
    path_lower = image_path.lower()
    
    # Priority check for known store names (case-insensitive)
    # Check Zara first
    if 'zara' in path_lower:
        return 'Zara'
    
    # Check Bershka
    if 'bershka' in path_lower:
        return 'Bershka'
    
    # Check H&M variants
    if 'h&m' in path_lower or 'hm' in path_lower or 'h_m' in path_lower or 'handm' in path_lower:
        return 'H&M'
    
    # No known brand found - use default fallback (NOT folder name)
    return DEFAULT_STORE_NAME


def load_model():
    """Load CLIP model - called once at startup"""
    global model, preprocess
    if model is None:
        print("⏳ Loading CLIP model with background removal...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        print("✅ Model loaded successfully!")
    return model, preprocess


def get_image_embedding(image_path_or_pil):
    """
    Extract embedding from image (matches notebook Cell 4)
    
    Args:
        image_path_or_pil: Either a file path (str) or PIL Image object
        
    Returns:
        numpy array of normalized embedding (512 dimensions)
    """
    global model, preprocess
    
    # Load model if not already loaded
    if model is None or preprocess is None:
        load_model()
    
    # Handle both file path and PIL Image
    if isinstance(image_path_or_pil, str):
        image = Image.open(image_path_or_pil).convert("RGB")
    else:
        image = image_path_or_pil.convert("RGB")
    
    # Remove background (like notebook)
    image = remove(image)
    
    # Preprocess and encode
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    
    return emb.cpu().numpy()


def search_image(query_embedding, index, metadata, threshold=0.70, top_k=4):
    """
    Search for similar images with optimized quality filtering.
    
    Args:
        query_embedding: numpy array of query image embedding
        index: FAISS index
        metadata: dict mapping index_id to product info
        threshold: minimum similarity threshold (0-1), default 0.70 (70%) for precision/recall balance
        top_k: number of results to return, default 4 for best quality matches
        
    Returns:
        dict with 'match' (bool) and 'results' (list sorted by similarity descending)
    """
    query_emb = query_embedding.astype("float32")
    
    # Search for more candidates to allow for filtering and sorting
    search_k = min(top_k * 5, index.ntotal)  # Search 5x to have enough candidates
    scores, indices = index.search(query_emb, k=search_k)
    
    print(f"\n🔍 SEARCH DEBUG: Threshold={threshold*100:.1f}%, Searching top {search_k} candidates...")
    
    results = []
    seen_products = set()  # Avoid duplicates from augmented images
    
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx not in metadata:
            continue
        
        # Calculate similarity percentage
        similarity_percent = float(score) * 100
        
        # Get product info for logging
        item = metadata.get(int(idx), {})
        product_id = item.get("product_id", "unknown")
        
        # Print each candidate's score for monitoring
        status = "✅ PASS" if similarity_percent >= (threshold * 100) else "❌ FAIL"
        print(f"   [{i+1}] Score: {similarity_percent:.2f}% | Product: {product_id} | {status}")
        
        # Filter by threshold (70% minimum)
        if similarity_percent < (threshold * 100):
            continue
        
        # Skip duplicates (same product, different augmentation)
        if product_id in seen_products:
            continue
        seen_products.add(product_id)
        
        # Get store name from path using mapping
        image_path = item.get("image_path", "")
        store_name = get_store_name_from_path(image_path)
        
        # Override with metadata store if it's a known store
        metadata_store = item.get("store", "")
        if metadata_store.lower() in [k for k in STORE_MAPPING.keys()]:
            store_name = STORE_MAPPING.get(metadata_store.lower(), store_name)
        
        results.append({
            "similarity": similarity_percent,
            "product_name": item.get("product_name", ""),
            "store": store_name,
            "store_name": store_name,  # Explicit store_name field for mobile app
            "price": item.get("price", 0),
            "image_path": image_path,
            "product_id": product_id
        })
    
    # Sort results by similarity score (highest first)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Limit to top 4 results only
    results = results[:4]
    
    print(f"   📊 Final: {len(results)} results after filtering & sorting (top 4 max)")
    
    if not results:
        return {
            "match": False, 
            "message": NO_MATCH_MESSAGE
        }
    
    return {"match": True, "results": results}
