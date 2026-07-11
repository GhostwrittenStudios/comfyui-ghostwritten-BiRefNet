"""
Ghostwritten Studios — BiRefNet Background Removal for ComfyUI
==============================================================

Self-contained custom node. It owns its own copy of the inference logic so a
ComfyUI update (or a third-party node breaking) can never silently break the
app that depends on it. The only external moving part is the BiRefNet model
weights, pulled once from Hugging Face via ``transformers`` and then cached
locally by the standard HF cache.

Nodes
-----
* **Ghostwritten BiRefNet Loader** — loads/caches a BiRefNet model, outputs a
  ``BIREFNET_MODEL`` handle.
* **Ghostwritten Remove Background** — runs BiRefNet on an IMAGE and returns an
  RGBA IMAGE (transparent background) plus the raw MASK.

The loader is optional: *Remove Background* also accepts a ``model_name`` and
will lazily load + cache the model itself, so a workflow can be a single node.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageSegmentation

# ── Model registry ────────────────────────────────────────────────────────────
# Friendly name -> Hugging Face repo id. All loaded with trust_remote_code.
BIREFNET_MODELS = {
    "BiRefNet (general)":      "ZhengPeng7/BiRefNet",
    "BiRefNet_lite (faster)":  "ZhengPeng7/BiRefNet_lite",
    "BiRefNet-portrait":       "ZhengPeng7/BiRefNet-portrait",
    "BiRefNet HR (1536)":      "ZhengPeng7/BiRefNet_HR",
}

# Native input resolution per model. BiRefNet is trained at 1024; the HR
# variant at 1536.
MODEL_INPUT_SIZE = {
    "ZhengPeng7/BiRefNet_HR": 1536,
}
DEFAULT_INPUT_SIZE = 1024

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Global cache so repeated API calls reuse the loaded weights.
# key: (repo_id, device_str, dtype_str) -> torch.nn.Module
_MODEL_CACHE = {}


def _pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_birefnet(repo_id, device, use_fp16):
    dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32
    key = (repo_id, str(device), str(dtype))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached, dtype

    print(f"[Ghostwritten BiRefNet] Loading {repo_id} ({dtype}) on {device} ...")
    model = AutoModelForImageSegmentation.from_pretrained(
        repo_id, trust_remote_code=True
    )
    model.eval().to(device=device, dtype=dtype)
    # BiRefNet benefits from TF32 matmuls on Ampere+.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    _MODEL_CACHE[key] = model
    print(f"[Ghostwritten BiRefNet] {repo_id} ready.")
    return model, dtype


class GhostwrittenBiRefNetLoader:
    """Loads a BiRefNet model and outputs a reusable handle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list(BIREFNET_MODELS.keys()),),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("BIREFNET_MODEL",)
    RETURN_NAMES = ("birefnet_model",)
    FUNCTION = "load"
    CATEGORY = "Ghostwritten Studios"

    def load(self, model_name, precision):
        repo_id = BIREFNET_MODELS[model_name]
        device = _pick_device()
        model, dtype = _load_birefnet(repo_id, device, precision == "fp16")
        size = MODEL_INPUT_SIZE.get(repo_id, DEFAULT_INPUT_SIZE)
        handle = {"model": model, "device": device, "dtype": dtype,
                  "input_size": size, "repo_id": repo_id}
        return (handle,)


class GhostwrittenRemoveBackground:
    """Runs BiRefNet and returns a transparent (RGBA) image plus the mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "birefnet_model": ("BIREFNET_MODEL",),
                "model_name": (list(BIREFNET_MODELS.keys()),),
                "precision": (["fp16", "fp32"], {"default": "fp16"}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -32, "max": 32, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba_image", "mask")
    FUNCTION = "remove"
    CATEGORY = "Ghostwritten Studios"

    def _resolve_model(self, birefnet_model, model_name, precision):
        if birefnet_model is not None:
            return birefnet_model
        repo_id = BIREFNET_MODELS[model_name]
        device = _pick_device()
        model, dtype = _load_birefnet(repo_id, device, precision == "fp16")
        size = MODEL_INPUT_SIZE.get(repo_id, DEFAULT_INPUT_SIZE)
        return {"model": model, "device": device, "dtype": dtype,
                "input_size": size, "repo_id": repo_id}

    @torch.inference_mode()
    def remove(self, image, birefnet_model=None, model_name="BiRefNet (general)",
               precision="fp16", mask_blur=0, mask_offset=0, invert_mask=False):
        handle = self._resolve_model(birefnet_model, model_name, precision)
        model, device, dtype = handle["model"], handle["device"], handle["dtype"]
        size = handle["input_size"]

        mean = torch.tensor(_IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  device=device, dtype=dtype).view(1, 3, 1, 1)

        # ComfyUI IMAGE: [B, H, W, C] float 0..1
        batch = image.shape[0]
        rgba_out, mask_out = [], []

        for b in range(batch):
            img = image[b]                       # [H, W, 3]
            H, W = img.shape[0], img.shape[1]
            x = img.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,H,W]
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
            x = (x - mean) / std

            preds = model(x)
            pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
            pred = pred.sigmoid().float()        # [1,1,size,size]

            # Resize mask back to original resolution.
            pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
            m = pred[0, 0].clamp(0, 1)           # [H, W]

            m = self._postprocess_mask(m, mask_blur, mask_offset, invert_mask)

            rgb = img.to("cpu", torch.float32)   # [H, W, 3]
            alpha = m.unsqueeze(-1).cpu()        # [H, W, 1]
            rgba = torch.cat([rgb, alpha], dim=-1)  # [H, W, 4]

            rgba_out.append(rgba)
            mask_out.append(m.cpu())

        rgba_image = torch.stack(rgba_out, dim=0)   # [B, H, W, 4]
        mask = torch.stack(mask_out, dim=0)         # [B, H, W]
        return (rgba_image, mask)

    @staticmethod
    def _postprocess_mask(m, blur, offset, invert):
        # m: [H, W] float on cpu/gpu, 0..1
        if offset != 0:
            # Grow (+) or shrink (-) the mask via max/min pooling.
            k = abs(offset) * 2 + 1
            x = m.unsqueeze(0).unsqueeze(0)
            if offset > 0:
                x = F.max_pool2d(x, kernel_size=k, stride=1, padding=offset)
            else:
                x = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=-offset)
            m = x[0, 0]
        if blur > 0:
            k = blur * 2 + 1
            x = m.unsqueeze(0).unsqueeze(0)
            x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=blur)
            m = x[0, 0].clamp(0, 1)
        if invert:
            m = 1.0 - m
        return m


NODE_CLASS_MAPPINGS = {
    "GhostwrittenBiRefNetLoader": GhostwrittenBiRefNetLoader,
    "GhostwrittenRemoveBackground": GhostwrittenRemoveBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GhostwrittenBiRefNetLoader": "Ghostwritten BiRefNet Loader",
    "GhostwrittenRemoveBackground": "Ghostwritten Remove Background",
}
