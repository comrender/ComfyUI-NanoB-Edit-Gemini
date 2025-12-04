from .nano_gemini import NanoBEditGemini

NODE_CLASS_MAPPINGS = {
    "NanoBEditGemini": NanoBEditGemini,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBEditGemini": "NanoB Edit Gemini"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
