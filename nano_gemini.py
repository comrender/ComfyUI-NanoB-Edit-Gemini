import os
import torch
import requests
import base64
import numpy as np
import json
import copy
import concurrent.futures
from io import BytesIO
from PIL import Image

# --- Helper Functions ---

def tensor2pil(image_tensor):
    """Convert ComfyUI tensor (B=1, H, W, C) to PIL Image (RGB)"""
    if image_tensor is None or image_tensor.shape[0] == 0:
        return None
    # Take the first image in the batch if a batch is passed to a single socket
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
                "prompt": ("STRING", {"default": "Edit the image according to this prompt.", "multiline": True}),
                "model": ([
                    "gemini-3-pro-image-preview", # Nano Banana Pro
                    "gemini-2.5-flash-image"      # Nano Banana
                ], {"default": "gemini-3-pro-image-preview"}),
                "gemini_api_key": ("STRING", {"default": "", "multiline": False}),
                "aspect_ratio": ([
                    "1:1", "16:9", "9:16", "4:3", "3:4", "2:3", "3:2", "5:4", "4:5", "21:9"
                ], {"default": "1:1"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "safety_filter": (["block_none", "block_few", "block_some", "block_most"], {"default": "block_none"}),
                "debug_payload": ("BOOLEAN", {"default": False, "label_on": "Enabled", "label_off": "Disabled"}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),
                "image6_14": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_images",)
    FUNCTION = "process"
    CATEGORY = "NanoGemini"
    OUTPUT_NODE = True

    def process(self, prompt, model, gemini_api_key, aspect_ratio="1:1", resolution="1K", num_images=1, safety_filter="block_none", debug_payload=False,
                image1=None, image2=None, image3=None, image4=None, image5=None, image6_14=None):
        
        # Check environment variable if UI field is empty
        if not gemini_api_key or gemini_api_key.strip() == "":
            gemini_api_key = os.environ.get("GEMINI_API_KEY")

        if not gemini_api_key or gemini_api_key.strip() == "":
            raise ValueError("Please provide a valid Google Gemini API Key via the UI or the GEMINI_API_KEY environment variable.")

        # 1. Prepare Input Images
        input_tensors = []
        
        # Add individual optional images
        for img in [image1, image2, image3, image4, image5]:
            if img is not None:
                input_tensors.append(img) # These are usually [1, H, W, C]
        
        # Add batch images from image6_14
        if image6_14 is not None:
            # image6_14 is [B, H, W, C]
            batch_size = image6_14.shape[0]
            for i in range(batch_size):
                input_tensors.append(image6_14[i:i+1]) # Keep dims as [1, H, W, C]

        # Enforce limits
        total_inputs = len(input_tensors)
        if total_inputs == 0:
             raise ValueError("At least one image input is required.")
        
        # Enforce strict 14 limit as requested
        if total_inputs > 14:
            print(f"NanoGemini Warning: {total_inputs} images provided. Truncating to 14 (Gemini 3 Pro limit).")
            input_tensors = input_tensors[:14]

        input_parts = []
        
        # Add the text prompt first
        input_parts.append({"text": prompt})

        # Process input images
        for i, img_tensor in enumerate(input_tensors):
            pil_img = tensor2pil(img_tensor)
            if pil_img:
                buffer = BytesIO()
                pil_img.save(buffer, format="PNG") # PNG is robust
                b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # FIX: Use camelCase keys for REST API (inlineData, mimeType)
                input_parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": b64_data
                    }
                })

        # 2. Configure API Payload
        safety_map = {
            "block_none": "BLOCK_NONE",
            "block_few": "BLOCK_ONLY_HIGH",
            "block_some": "BLOCK_MEDIUM_AND_ABOVE",
            "block_most": "BLOCK_LOW_AND_ABOVE"
        }
        
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
            # Explicitly wrapping in role="user" is safer for newer Pro models
            "contents": [{
                "role": "user",
                "parts": input_parts
            }],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"], 
                "candidateCount": 1, 
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": resolution
                }
            },
            "safetySettings": safety_settings
        }

        # --- DEBUG OUTPUT ---
        if debug_payload:
            # Create a safe copy for printing (don't print 20MB of base64 text)
            debug_print = copy.deepcopy(payload)
            total_size_estimate = 0
            
            # Truncate base64 strings in the copy
            for part in debug_print["contents"][0]["parts"]:
                if "inlineData" in part:
                    data_len = len(part["inlineData"]["data"])
                    total_size_estimate += data_len
                    part["inlineData"]["data"] = f"<Base64 Image Data: {data_len} chars>"
            
            print("--- NANO GEMINI DEBUG PAYLOAD START ---")
            print(json.dumps(debug_print, indent=2))
            print(f"Approximate Payload Size (Images Only): {total_size_estimate / (1024*1024):.2f} MB")
            if total_size_estimate > 20 * 1024 * 1024:
                print("WARNING: Payload exceeds 20MB limit! API will likely fail.")
            print("--- NANO GEMINI DEBUG PAYLOAD END ---")

        # 3. Define the Request Function
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_api_key}"
        headers = {"Content-Type": "application/json"}

        def send_request(idx):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print(f"Gemini API Request {idx+1} Failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                     print(f"Error details: {e.response.text}")
                return None

        # 4. Execute Parallel Requests (if num_images > 1)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_images) as executor:
            futures = [executor.submit(send_request, i) for i in range(num_images)]
            
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if data:
                    results.append(data)

        if not results:
             raise ValueError("All API requests failed. Check your API key and console logs.")

        # 5. Process Responses and Convert to Tensors
        output_tensors = []
        collected_errors = []

        for i, res in enumerate(results):
            try:
                candidates = res.get("candidates", [])
                
                if "promptFeedback" in res:
                    pf = res["promptFeedback"]
                    if pf.get("blockReason"):
                         print(f"Request {i+1} Blocked: {pf['blockReason']}")

                if not candidates:
                    collected_errors.append(f"Req {i+1}: No candidates returned.")
                    continue

                for candidate in candidates:
                    finish_reason = candidate.get("finishReason", "UNKNOWN")
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    
                    image_found_in_candidate = False
                    text_response = ""

                    for part in parts:
                        if "text" in part:
                            text_response += part["text"]
                        
                        # FIX: Use camelCase 'inlineData' for REST API response parsing
                        inline_data = part.get("inlineData", {})
                        if inline_data:
                            b64_img = inline_data.get("data")
                            if b64_img:
                                try:
                                    img_data = base64.b64decode(b64_img)
                                    pil_out = Image.open(BytesIO(img_data))
                                    tensor_out = pil2tensor(pil_out)
                                    output_tensors.append(tensor_out)
                                    image_found_in_candidate = True
                                except Exception as e:
                                    print(f"Error decoding image in Req {i+1}: {e}")

                    if not image_found_in_candidate:
                        msg = f"Req {i+1}: FinishReason={finish_reason}"
                        if text_response:
                            msg += f", Text='{text_response[:200]}...'"
                        collected_errors.append(msg)
                        
                        print(f"--- Full JSON Response for Req {i+1} (No Image Found) ---")
                        print(json.dumps(res, indent=2))
                        print("---------------------------------------------------------")

            except Exception as e:
                print(f"Error parsing response {i+1}: {e}")
                continue

        if not output_tensors:
            error_summary = "; ".join(collected_errors)
            raise ValueError(f"No images found in API responses. Details: {error_summary}")

        # 6. Stack and Return
        result_batch = torch.cat(output_tensors, dim=0)
        return (result_batch,)
