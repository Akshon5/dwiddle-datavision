import os
import json
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import torch

from src.model import QuickDrawCNN
from src.preprocess import preprocess_canvas_image

MODELS_DIR = 'models'
MODEL_CONFIGS = {
    20: {
        'model_path': os.path.join(MODELS_DIR, 'expanded_cnn_20.pt'),
        'label_path': os.path.join(MODELS_DIR, 'label_mapping_20.json'),
        'fallback_label_path': os.path.join('data/expanded_20', 'label_mapping.json')
    },
    8: {
        'model_path': os.path.join(MODELS_DIR, 'baseline_cnn.pt'),
        'label_path': os.path.join(MODELS_DIR, 'label_mapping.json'),
        'fallback_label_path': os.path.join('data', 'split_metadata.json')
    }
}

def get_default_model_version() -> int:
    env_ver = os.environ.get("DRAW_MODEL_VERSION", "")
    return int(env_ver) if env_ver in ("8", "20") else 8


class QuickDrawPredictor:
    """
    Inference wrapper for QuickDrawCNN.
    Supports both the expanded 20-class model and the original 8-class baseline.
    Strictly pairs matching checkpoints with corresponding label mappings to avoid mismatch.
    """
    def __init__(
        self,
        model_version: Optional[int] = None,
        model_path: Optional[str] = None,
        label_path: Optional[str] = None
    ):
        if model_version is None and model_path is None:
            model_version = get_default_model_version()
            
        self.model_version = model_version
        
        # Determine paths based on version or explicit parameters
        if model_path is not None and label_path is not None:
            self.model_path = model_path
            self.label_path = label_path
        elif model_version in MODEL_CONFIGS:
            cfg = MODEL_CONFIGS[model_version]
            self.model_path = cfg['model_path']
            if os.path.exists(cfg['label_path']):
                self.label_path = cfg['label_path']
            elif 'fallback_label_path' in cfg and os.path.exists(cfg['fallback_label_path']):
                self.label_path = cfg['fallback_label_path']
            else:
                self.label_path = cfg['label_path']
        else:
            raise ValueError(f"Unsupported model_version {model_version}. Must be 20 or 8.")
            
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Label mapping not found: {self.label_path}")
            
        with open(self.label_path, 'r') as f:
            self.label_mapping = json.load(f)
            
        self.rev_mapping = {v: k for k, v in self.label_mapping.items()}
        self.class_names = [self.rev_mapping[i] for i in range(len(self.rev_mapping))]
        self.num_classes = len(self.class_names)
        
        # Load model to CPU in evaluation mode
        checkpoint = torch.load(self.model_path, map_location='cpu')
        self.model = QuickDrawCNN(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def predict_canvas(self, canvas_image_data: Any) -> Tuple[Optional[str], float, List[Tuple[str, float]], Dict[str, Any]]:
        """
        Processes canvas array through shared preprocessing pipeline and returns:
        - top_class: predicted class string (or None if empty/invalid)
        - top_confidence: float in [0.0, 1.0]
        - top3: list of (class_name, confidence)
        - meta: preprocessing diagnostics (is_empty, bbox, stroke_pixels, processed_28x28)
        """
        tensor, meta = preprocess_canvas_image(canvas_image_data)
        
        if tensor is None or meta.get('is_empty', False):
            return None, 0.0, [], meta
            
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            
        top3_indices = np.argsort(probs)[::-1][:3]
        top_idx = top3_indices[0]
        top_class = self.class_names[top_idx]
        top_confidence = float(probs[top_idx])
        
        top3 = [(self.class_names[idx], float(probs[idx])) for idx in top3_indices]
        return top_class, top_confidence, top3, meta
