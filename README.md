# Ghostwritten Studios — BiRefNet Background Removal

A **self-contained** ComfyUI custom node for BiRefNet background removal. It does
not depend on any other third-party ComfyUI extension, so a ComfyUI update can't
silently break the app that relies on it. The only external piece is the model
weights, pulled once from Hugging Face and cached locally.

## Nodes

| Node | Category | Purpose |
|------|----------|---------|
| **Ghostwritten Remove Background** | Ghostwritten Studios | IMAGE → RGBA IMAGE + MASK. Loads the model itself if no loader is connected. |
| **Ghostwritten BiRefNet Loader** | Ghostwritten Studios | (Optional) Load a model once and reuse the handle across multiple nodes. |

## Outputs

- **rgba_image** (`IMAGE`) — 4-channel RGBA. Feed straight into a `Save Image`
  node to get a transparent PNG (ComfyUI's SaveImage writes the alpha channel).
- **mask** (`MASK`) — raw foreground matte, 0–1.

## Options (Remove Background)

- **model_name** — `BiRefNet (general)`, `BiRefNet_lite (faster)`,
  `BiRefNet-portrait`, `BiRefNet HR (1536)`.
- **precision** — `fp16` (default, CUDA) or `fp32`.
- **mask_blur** — feather the matte edge (px).
- **mask_offset** — grow (+) / shrink (−) the matte (px).
- **invert_mask** — keep the background instead of the subject.

## Model weights

Downloaded automatically on first use via `transformers` +
`trust_remote_code=True`, cached under `~/.cache/huggingface`. Repos:

- `ZhengPeng7/BiRefNet` · `ZhengPeng7/BiRefNet_lite`
- `ZhengPeng7/BiRefNet-portrait` · `ZhengPeng7/BiRefNet_HR`

Requires `timm`, `einops`, `kornia` (already present in this ComfyUI install).

## Minimal API workflow (single node)

```json
{
  "10": { "class_type": "LoadImage", "inputs": { "image": "input.png" } },
  "11": { "class_type": "GhostwrittenRemoveBackground",
          "inputs": { "image": ["10", 0], "model_name": "BiRefNet (general)",
                      "precision": "fp16", "mask_blur": 0, "mask_offset": 0,
                      "invert_mask": false } },
  "12": { "class_type": "SaveImage",
          "inputs": { "filename_prefix": "GhostwrittenStudios_cutout",
                      "images": ["11", 0] } }
}
```

Restart ComfyUI after installing so the nodes register.
