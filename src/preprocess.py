import numpy as np
from PIL import Image
import torch


def preprocess_canvas_image(image_data, target_size=28, padding_ratio=0.18, threshold=20):
    """
    Standardized, robust preprocessing pipeline for drawing canvas input:
    
    1. Determines canvas format and detects background vs stroke polarity.
    2. Identifies bounding box of drawn strokes.
    3. Crops content and adds aspect-ratio preserving padding (centering).
    4. Resizes to 28x28 with bilinear resampling.
    5. Formats as Quick, Draw! contract: black background (0), light strokes (1..255).
    6. Normalizes to float32 in [0, 1] tensor of shape (1, 1, 28, 28).
    
    Returns:
        tensor: torch.FloatTensor (1, 1, 28, 28) or None if canvas is empty
        meta: dict containing intermediate images, bbox, and diagnostics
    """
    if image_data is None:
        return None, {"error": "No image data provided", "is_empty": True}
        
    arr = np.array(image_data)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        return None, {"error": f"Invalid canvas array shape: {arr.shape}", "is_empty": True}
        
    h, w, c = arr.shape
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    
    # Standard luminosity formula
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    if c == 4:
        alpha = arr[:, :, 3].astype(np.float32) / 255.0
    else:
        alpha = np.ones((h, w), dtype=np.float32)
        
    # Detect background polarity from corners
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    corner_mean = np.mean(corners)
    
    if corner_mean > 127.0:
        # Light background (e.g. white canvas #FFFFFF with black ink)
        # Ink intensity = 255 - gray
        stroke = (255.0 - gray) * alpha
    else:
        # Dark background (e.g. black canvas with white ink)
        stroke = gray * alpha
        
    # Clip to uint8 range
    stroke = np.clip(stroke, 0, 255).astype(np.uint8)
    
    # Identify drawn strokes above threshold
    nonzero_mask = stroke > threshold
    total_stroke_pixels = int(np.sum(nonzero_mask))
    
    # Handle empty or trivial canvas (< 12 stroke pixels)
    if total_stroke_pixels < 12:
        return None, {
            "error": "Canvas is blank or drawing is too faint. Please draw a clearer object.",
            "is_empty": True,
            "stroke_pixels": total_stroke_pixels
        }
        
    # Find bounding box
    rows = np.any(nonzero_mask, axis=1)
    cols = np.any(nonzero_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    crop_h = rmax - rmin + 1
    crop_w = cmax - cmin + 1
    
    cropped_stroke = stroke[rmin:rmax+1, cmin:cmax+1]
    
    # Aspect-ratio preserving padding to make square
    max_dim = max(crop_h, crop_w)
    # Add border margin proportional to size
    margin = int(max_dim * padding_ratio)
    padded_dim = max_dim + 2 * margin
    
    square_canvas = np.zeros((padded_dim, padded_dim), dtype=np.uint8)
    
    # Center the cropped stroke
    y_offset = margin + (max_dim - crop_h) // 2
    x_offset = margin + (max_dim - crop_w) // 2
    square_canvas[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped_stroke
    
    # Resize to 28x28 using PIL bilinear filter
    pil_square = Image.fromarray(square_canvas, mode='L')
    pil_28 = pil_square.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)
    processed_28 = np.array(pil_28, dtype=np.uint8)
    
    # QuickDraw tensor format: (1, 1, 28, 28), float32 in [0, 1]
    tensor = torch.from_numpy(processed_28).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    meta = {
        "is_empty": False,
        "stroke_pixels": total_stroke_pixels,
        "bbox": (int(cmin), int(rmin), int(crop_w), int(crop_h)),
        "crop_shape": (crop_h, crop_w),
        "padded_dim": padded_dim,
        "cropped_image": cropped_stroke,
        "square_image": square_canvas,
        "processed_28x28": processed_28
    }
    
    return tensor, meta


