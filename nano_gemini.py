import torch
import requests
import base64
import numpy as np
import json
import concurrent.futures
from io import BytesIO
from PIL import Image

# --- Helper Functions ---

def tensor2pil(image_tensor):
    """Convert ComfyUI tensor (B=1, H, W, C) to PIL Image (RGB)"""
    if image_tensor is None or image_tensor.shape[0] == 0:
        return None
    i = 255. * image_tensor[0].cpu().numpy()  # (H, W, C)
    image = np.clip(i, 0, 255).astype(np.uint8)
    
    c = image.shape[-1]
    if c == 1:
        image = np.repeat(image, 3, axis=-1)
    elif c == 3:
        pass
    elif c == 4:
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported channels: {c}. Expected 1, 3, or 4.")
    
    return Image.fromarray(image, mode='RGB')

def pil2tensor(pil_image):
    """Convert PIL Image (RGB) back to ComfyUI tensor (B=1, H, W, C)"""
    if pil_image is None:
        return None
    arr = np.array(pil_image).astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]
    return torch.from_numpy(arr)

class NanoBEditGemini:
    """
    ComfyUI Node for Google Gemini Image Editing.
    Supports Gemini 3 Pro (Nano Banana Pro) and Gemini 2.5 Flash (Nano Banana).
    Directly hits the Google Generative Language API.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Accepts a batch of images (1 to 16)
                "prompt": ("STRING", {"default": "Edit the image according to this prompt.", "multiline": True}),
                "model": ([
                    "gemini-3-pro-image-preview", # Nano Banana Pro
                    "gemini-2.5-flash-image"      # Nano Banana
                ], {"default": "gemini-3-pro-image-preview"}),
                "gemini_api_key": ("STRING", {"default": "", "multiline": False}),
                "aspect_ratio": ([
                    "1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "5:4", "4:5", "21:9"
                ], {"default": "1:1"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "safety_filter": (["block_none", "block_few", "block_some", "block_most"], {"default": "block_none"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_images",)
    FUNCTION = "process"
    CATEGORY = "NanoGemini"
    OUTPUT_NODE = True

    def process(self, images, prompt, model, gemini_api_key, aspect_ratio="1:1", num_images=1, safety_filter="block_none"):
        if not gemini_api_key or gemini_api_key.strip() == "":
            raise ValueError("Please provide a valid Google Gemini API Key.")

        # 1. Prepare Input Images
        # 'images' is a tensor of shape [B, H, W, C]. We convert each to base64.
        input_parts = []
        
        # Add the text prompt first
        input_parts.append({"text": prompt})

        # Process input image batch
        batch_size = images.shape[0]
        if batch_size > 16:
             print(f"Warning: Gemini API usually supports up to 16 input images. Truncating {batch_size} to 16.")
             images = images[:16]

        for i in range(images.shape[0]):
            img_tensor = images[i:i+1] # Slice to keep dims for helper
            pil_img = tensor2pil(img_tensor)
            if pil_img:
                buffer = BytesIO()
                pil_img.save(buffer, format="PNG") # PNG is robust
                b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                input_parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64_data
                    }
                })

        # 2. Configure API Payload
        # Mapping safety settings
        safety_map = {
            "block_none": "BLOCK_NONE",
            "block_few": "BLOCK_ONLY_HIGH",
            "block_some": "BLOCK_MEDIUM_AND_ABOVE",
            "block_most": "BLOCK_LOW_AND_ABOVE"
        }
        
        # We need to set safety settings for all categories to ensure the filter is applied
        safety_categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT"
        ]
        safety_settings = [
            {"category": cat, "threshold": safety_map[safety_filter]} for cat in safety_categories
        ]

        payload = {
            "contents": [{"parts": input_parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE"], # Force image output
                "aspectRatio": aspect_ratio,
                "candidateCount": 1, # Gemini often limits this to 1 per request for images
            },
            "safetySettings": safety_settings
        }

        # 3. Define the Request Function
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_api_key}"
        headers = {"Content-Type": "application/json"}

        def send_request(_):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Gemini API Request Failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                     print(f"Error details: {e.response.text}")
                return None

        # 4. Execute Parallel Requests (if num_images > 1)
        # We use a ThreadPool to simulate 'num_images' since we can't always get >1 candidate per call reliably
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_images) as executor:
            # Launch 'num_images' tasks. The argument (i) is just a placeholder.
            futures = [executor.submit(send_request, i) for i in range(num_images)]
            
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if data:
                    results.append(data)

        if not results:
             raise ValueError("All API requests failed. Check your API key and console logs.")

        # 5. Process Responses and Convert to Tensors
        output_tensors = []

        for res in results:
            # Google API structure: candidates -> content -> parts -> inline_data
            try:
                candidates = res.get("candidates", [])
                if not candidates:
                    # Check for prompt feedback block
                    if "promptFeedback" in res:
                        print(f"Prompt Feedback: {res['promptFeedback']}")
                    continue

                for candidate in candidates:
                    parts = candidate.get("content", {}).get("parts", [])
                    for part in parts:
                        inline_data = part.get("inline_data", {})
                        if inline_data:
                            b64_img = inline_data.get("data")
                            if b64_img:
                                img_data = base64.b64decode(b64_img)
                                pil_out = Image.open(BytesIO(img_data))
                                tensor_out = pil2tensor(pil_out)
                                output_tensors.append(tensor_out)
            except Exception as e:
                print(f"Error parsing response: {e}")
                continue

        if not output_tensors:
            raise ValueError("API returned valid response but no images were found. Possible safety refusal.")

        # 6. Stack and Return
        # Concatenate all resulting images into a single batch
        result_batch = torch.cat(output_tensors, dim=0)
        return (result_batch,)