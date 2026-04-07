import runpod
import torch
import base64
import io
import os
import requests
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from huggingface_hub import hf_hub_download

# Point HuggingFace cache and custom models to the RunPod Network Volume
os.environ["HF_HOME"] = "/runpod-volume/huggingface"
MODEL_CACHE_DIR = "/runpod-volume/models"
BIGLUST_PATH = os.path.join(MODEL_CACHE_DIR, "biglust.safetensors")

pipe = None

def setup_models():
    """Downloads models to the Network Volume if they don't already exist."""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    
    # 1. Download BigLust if missing
    if not os.path.exists(BIGLUST_PATH):
        print("BigLust not found. Downloading to Network Volume (This only happens on the very first boot)...")
        BIGLUST_DOWNLOAD_URL = "https://civitai.com/api/download/models/1081768?type=Model&format=SafeTensor&size=full&fp=fp16&token=85a9d6503e3bd953780fa2250a93452a"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(BIGLUST_DOWNLOAD_URL, headers=headers, stream=True)
        if response.status_code == 200:
            with open(BIGLUST_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("BigLust model downloaded securely to volume.")
        else:
            print(f"Failed to download BigLust. Status code: {response.status_code}")

    # 2. Cache IP-Adapter weights
    print("Verifying IP-Adapter weights on volume...")
    hf_hub_download(repo_id="h94/IP-Adapter", filename="models/ip-adapter_sd15.bin")
    hf_hub_download(repo_id="h94/IP-Adapter", filename="models/image_encoder/pytorch_model.bin")

def load_pipeline():
    global pipe
    setup_models()
    
    print("Loading pipeline into VRAM...")
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        BIGLUST_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None
    )
    
    pipe.load_ip_adapter(
        "h94/IP-Adapter", 
        subfolder="models", 
        weight_name="ip-adapter_sd15.bin"
    )
    
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")
    print("Pipeline ready.")

def decode_base64_image(image_string):
    if "," in image_string:
        image_string = image_string.split(",")[1]
    image_bytes = base64.b64decode(image_string)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def encode_image_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handler(job):
    global pipe
    job_input = job.get("input", {})

    prompt = job_input.get("prompt", "masterpiece, best quality")
    negative_prompt = job_input.get("negative_prompt", "lowres, bad anatomy, worst quality")
    strength = float(job_input.get("strength", 0.6))
    guidance_scale = float(job_input.get("guidance_scale", 7.5))
    num_inference_steps = int(job_input.get("steps", 30))
    ip_adapter_scale = float(job_input.get("ip_adapter_scale", 0.5))
    
    init_image_b64 = job_input.get("init_image")
    ip_adapter_image_b64 = job_input.get("ip_adapter_image")

    if not init_image_b64 or not ip_adapter_image_b64:
        return {"error": "Both init_image and ip_adapter_image must be provided as base64 strings."}

    try:
        init_image = decode_base64_image(init_image_b64).resize((512, 512))
        ip_image = decode_base64_image(ip_adapter_image_b64).resize((512, 512))
    except Exception as e:
        return {"error": f"Failed to decode images: {str(e)}"}

    pipe.set_ip_adapter_scale(ip_adapter_scale)

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            ip_adapter_image=ip_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]

        return {
            "status": "success",
            "image": encode_image_base64(result)
        }
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

if __name__ == "__main__":
    load_pipeline()
    runpod.serverless.start({"handler": handler})
