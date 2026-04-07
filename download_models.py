import os
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from huggingface_hub import hf_hub_download
import requests

# Directories
MODEL_CACHE_DIR = "/model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# 1. Download BigLust Safetensors (API Token Included)
BIGLUST_DOWNLOAD_URL = "https://civitai.com/api/download/models/1081768?type=Model&format=SafeTensor&size=full&fp=fp16&token=85a9d6503e3bd953780fa2250a93452a" 
BIGLUST_PATH = os.path.join(MODEL_CACHE_DIR, "biglust.safetensors")

print("Downloading BigLust model...")
# Add headers to prevent Civitai from rejecting a raw python request
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Stream the download to avoid loading the massive file into RAM all at once
response = requests.get(BIGLUST_DOWNLOAD_URL, headers=headers, stream=True)

if response.status_code == 200:
    with open(BIGLUST_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("BigLust model downloaded successfully.")
else:
    print(f"Failed to download model. Status code: {response.status_code}")
    print(f"Response text: {response.text}")

# 2. Pre-cache standard components (Safety checker, feature extractor, etc.)
# We load a dummy base model just to cache the standard SD1.5 components locally
print("Caching standard SD1.5 components...")
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe.save_pretrained(os.path.join(MODEL_CACHE_DIR, "sd-base-components"))

# 3. Cache IP-Adapter weights
print("Caching IP-Adapter weights...")
hf_hub_download(repo_id="h94/IP-Adapter", filename="models/ip-adapter_sd15.bin", cache_dir=MODEL_CACHE_DIR)
hf_hub_download(repo_id="h94/IP-Adapter", filename="models/image_encoder/pytorch_model.bin", cache_dir=MODEL_CACHE_DIR)

print("Download phase complete.")