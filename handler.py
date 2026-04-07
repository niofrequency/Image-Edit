import runpod
import torch
import base64
import io
import os
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

# Global variables to hold the pipeline in memory across invocations
pipe = None
MODEL_CACHE_DIR = "/model_cache"
BIGLUST_PATH = os.path.join(MODEL_CACHE_DIR, "biglust.safetensors")

def load_pipeline():
    global pipe
    print("Loading pipeline into VRAM...")
    
    # Load the custom safetensors model
    pipe = StableDiffusionImg2ImgPipeline.from_single_file(
        BIGLUST_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None
    )
    
    # Load IP-Adapter
    # Assuming standard SD1.5 IP-Adapter setup
    pipe.load_ip_adapter(
        "h94/IP-Adapter", 
        subfolder="models", 
        weight_name="ip-adapter_sd15.bin",
        cache_dir=MODEL_CACHE_DIR
    )
    
    # Optimize for inference speed
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")
    print("Pipeline ready.")

def decode_base64_image(image_string):
    if "," in image_string:
        image_string = image_string.split(",")[1]
    image_bytes = base64.b64decode(image_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

def encode_image_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def handler(job):
    """
    RunPod Serverless Handler.
    Expects job_input payload:
    {
        "prompt": "text prompt",
        "negative_prompt": "bad quality...",
        "init_image": "base64_string",
        "ip_adapter_image": "base64_string",
        "strength": 0.7,
        "guidance_scale": 7.5,
        "steps": 30
    }
    """
    global pipe
    job_input = job.get("input", {})

    # Extract parameters with defaults
    prompt = job_input.get("prompt", "masterpiece, best quality")
    negative_prompt = job_input.get("negative_prompt", "lowres, bad anatomy, worst quality")
    strength = float(job_input.get("strength", 0.6))
    guidance_scale = float(job_input.get("guidance_scale", 7.5))
    num_inference_steps = int(job_input.get("steps", 30))
    ip_adapter_scale = float(job_input.get("ip_adapter_scale", 0.5))
    
    # Extract Images
    init_image_b64 = job_input.get("init_image")
    ip_adapter_image_b64 = job_input.get("ip_adapter_image")

    if not init_image_b64 or not ip_adapter_image_b64:
        return {"error": "Both init_image and ip_adapter_image must be provided as base64 strings."}

    try:
        init_image = decode_base64_image(init_image_b64)
        ip_image = decode_base64_image(ip_adapter_image_b64)
        
        # Resize images to standard dimensions for SD1.5 (e.g., 512x512) if needed
        init_image = init_image.resize((512, 512))
        ip_image = ip_image.resize((512, 512))

    except Exception as e:
        return {"error": f"Failed to decode images: {str(e)}"}

    # Set IP-Adapter scale
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    # Run Inference
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

        # Convert result to base64 to send back to your frontend/SaaS
        result_b64 = encode_image_base64(result)
        
        return {
            "status": "success",
            "image": result_b64
        }

    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

# Initialize the model on container startup
if __name__ == "__main__":
    load_pipeline()
    runpod.serverless.start({"handler": handler})