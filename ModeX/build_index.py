"""
Build FAISS index from images - matches mode_x_final.ipynb approach
Recursively scans all subdirectories in /images folder
Includes store name mapping for Bershka, Zara, H&M
"""
import os
import faiss
import numpy as np
import json
import random
from pathlib import Path
from tqdm import tqdm
from model_utils import get_image_embedding, load_model

# Use the script directory as the project root to keep paths portable
PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================
# Configuration
# ==========================================
MAX_IMAGES = None  # Set to None to process ALL images, or a number to limit (e.g., 100 for testing)
# Main images folder (relative to this script, not the current working directory)
IMAGES_DIR = PROJECT_ROOT / "images"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}  # Valid image extensions

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
    path_lower = str(image_path).lower()
    
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

# ==========================================
# Step 1: Load CLIP Model
# ==========================================
print("=" * 60)
print("🚀 Mode X Index Builder")
print("=" * 60)
load_model()

# ==========================================
# Step 2: Recursively find ALL images in subdirectories
# ==========================================
print(f"\n📂 Scanning '{IMAGES_DIR}' recursively for images...")

def find_all_images(base_dir: Path) -> list:
    """
    Recursively find all valid image files in the directory.
    Uses Path.rglob for recursive globbing.
    """
    all_images = []
    
    for ext in VALID_EXTENSIONS:
        # Use rglob to recursively find all files with this extension
        # Also check uppercase extensions
        all_images.extend(base_dir.rglob(f"*{ext}"))
        all_images.extend(base_dir.rglob(f"*{ext.upper()}"))
    
    # Remove duplicates (in case of case-insensitive filesystem)
    unique_images = list(set(all_images))
    
    # Sort for consistent ordering
    unique_images.sort()
    
    return unique_images

# Find all images
all_image_paths = find_all_images(IMAGES_DIR)

# ==========================================
# Step 3: Print total count before processing
# ==========================================
total_found = len(all_image_paths)
print(f"\n✅ Found {total_found:,} images in '{IMAGES_DIR}' and subdirectories")

if total_found == 0:
    print("❌ No images found! Please check the images directory path.")
    exit(1)

# Apply limit if set
if MAX_IMAGES:
    images_to_process = all_image_paths[:MAX_IMAGES]
    print(f"⚠️  LIMITED to {MAX_IMAGES} images for testing")
else:
    images_to_process = all_image_paths
    print(f"📊 Processing ALL {total_found:,} images")

total_to_process = len(images_to_process)

# ==========================================
# Step 4: Extract embeddings with progress bar
# ==========================================
print(f"\n🔄 Extracting embeddings from {total_to_process:,} images...")
print("   (This may take a while for large datasets)")

metadata = {}
embeddings = []
failed_images = []

# Use tqdm for progress bar with total count
for index_id, img_path in enumerate(tqdm(images_to_process, 
                                          total=total_to_process,
                                          desc="📸 Processing",
                                          unit="img",
                                          ncols=100)):
    try:
        # Extract product info from path
        # Path structure: images/product_folder/image_variant.jpg
        # e.g., images/jacket_1/original.jpg → product_id = "jacket_1"
        
        # Get the parent folder name as product_id
        product_folder = img_path.parent.name
        
        # If the parent is 'images' itself, use the filename as product_id
        if product_folder == IMAGES_DIR.name or product_folder == "":
            product_name = img_path.stem  # filename without extension
        else:
            product_name = product_folder
        
        # Get store name using mapping function
        # This checks for bershka, zara, hm/h&m in the path
        store_name = get_store_name_from_path(str(img_path))
        
        # Generate random price (consistent per product)
        # Use hash of product_name for consistent pricing
        price_seed = hash(product_name) % 1000
        random.seed(price_seed)
        price = random.randint(200, 1500)
        
        # Extract embedding using CLIP with background removal
        emb = get_image_embedding(str(img_path))
        embeddings.append(emb)
        
        # Store metadata with store_name for mobile app
        # CRITICAL: Always store RELATIVE paths (not absolute) for Django MEDIA_URL compatibility
        # Convert to forward slashes for cross-platform compatibility
        if img_path.is_absolute():
            # Prefer paths relative to the project root for consistent portability.
            try:
                relative_path = str(img_path.relative_to(PROJECT_ROOT))
            except ValueError:
                # Fallback: keep as-is if the image is outside the project root.
                relative_path = str(img_path)
        else:
            relative_path = str(img_path)
        relative_path = relative_path.replace("\\", "/")  # Normalize to forward slashes
        
        metadata[index_id] = {
            "product_id": product_name,
            "product_name": product_name.replace("_", " "),
            "store": store_name,
            "store_name": store_name,  # Explicit field for mobile app
            "price": price,
            "image_path": relative_path  # Always relative, e.g., "images/jacket_1/original.jpg"
        }
        
    except Exception as e:
        failed_images.append((str(img_path), str(e)))
        continue

# Reset random seed
random.seed()

# ==========================================
# Step 5: Report processing results
# ==========================================
print(f"\n📊 Processing Results:")
print(f"   ✅ Successfully indexed: {len(embeddings):,} images")
if failed_images:
    print(f"   ❌ Failed: {len(failed_images)} images")
    if len(failed_images) <= 10:
        for path, error in failed_images:
            print(f"      - {path}: {error}")
    else:
        print(f"      (Showing first 10 failures)")
        for path, error in failed_images[:10]:
            print(f"      - {path}: {error}")

if len(embeddings) == 0:
    print("\n❌ No embeddings extracted! Cannot build index.")
    exit(1)

# ==========================================
# Step 6: Build FAISS Index
# ==========================================
print(f"\n🔧 Building FAISS index...")

dimension = 512  # CLIP ViT-B/32 embedding dimension
index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity

# Stack all embeddings into a single array
embeddings_array = np.vstack(embeddings).astype("float32")
index.add(embeddings_array)

print(f"   ✅ FAISS index created with {index.ntotal:,} vectors")

# ==========================================
# Step 7: Save index and metadata
# ==========================================
print(f"\n💾 Saving files...")

# Save FAISS index
faiss.write_index(index, str(PROJECT_ROOT / "index.faiss"))
print(f"   ✅ Saved 'index.faiss' ({index.ntotal:,} vectors)")

# Save metadata as JSON
with open(PROJECT_ROOT / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"   ✅ Saved 'metadata.json' ({len(metadata):,} entries)")

# ==========================================
# Final Summary
# ==========================================
print("\n" + "=" * 60)
print("🎉 INDEX BUILD COMPLETE!")
print("=" * 60)
print(f"   📂 Source directory: {IMAGES_DIR}")
print(f"   🖼️  Images found: {total_found:,}")
print(f"   ✅ Images indexed: {len(embeddings):,}")
print(f"   📦 Index file: index.faiss")
print(f"   📋 Metadata file: metadata.json")
print("=" * 60)
