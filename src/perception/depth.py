
import numpy as np


class DepthAnythingMetric:
    # Depth Anything v2

    def __init__(self, size="Small", device=None, scale=1.0):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        name = f"depth-anything/Depth-Anything-V2-Metric-Indoor-{size}-hf"
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(self.device).eval()
        self.scale = scale   # fitted from your tape-measure table (Step 3)

    def __call__(self, frame_rgb: np.ndarray) -> np.ndarray:
        # frame_rgb
        torch = self.torch
        inputs = self.proc(images=frame_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self.model(**inputs).predicted_depth            # (1, h, w)
        pred = torch.nn.functional.interpolate(
            pred[:, None], size=frame_rgb.shape[:2], mode="bilinear", align_corners=False)
        return (pred[0, 0].cpu().numpy() * self.scale).astype(np.float32)
