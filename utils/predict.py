"""Inference wrapper used by the Streamlit app.

Quick test from the repo root (never from inside utils/):
    python -m utils.predict path\\to\\photo.jpg
"""
import json
import sys

import torch

from utils import config
from utils.preprocess import bgr_to_model_input
from utils.train import build_model


class SkinClassifier:
    def __init__(self):
        ckpt = torch.load(config.MODEL_PATH, map_location="cpu")
        self.class_names = ckpt["classes"]
        self.img_size = tuple(ckpt["img_size"])

        self.model = build_model(len(self.class_names), pretrained=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()  # inference mode: no dropout, fixed batch-norm

        try:
            self.threshold = json.loads(config.THRESHOLD_PATH.read_text())["threshold"]
        except FileNotFoundError:
            self.threshold = getattr(config, "DEFAULT_CONF_THRESHOLD", 0.60)

    def predict(self, bgr, top_k=3):
        """bgr: an OpenCV image (numpy array). Returns a dict the app can display."""
        x = bgr_to_model_input(bgr, self.img_size)  # tensor, shape (1, 3, H, W)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].tolist()

        order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
        top = [(self.class_names[i], float(probs[i])) for i in order[:top_k]]
        return {
            "label": top[0][0],
            "confidence": top[0][1],
            "top": top,
            "all_scores": {n: float(p) for n, p in zip(self.class_names, probs)},
            "low_confidence": top[0][1] < self.threshold,
            "threshold": self.threshold,
        }


if __name__ == "__main__":
    import cv2

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m utils.predict path\\to\\photo.jpg")
    image = cv2.imread(sys.argv[1])
    if image is None:
        raise SystemExit(f"could not read an image from {sys.argv[1]}")

    result = SkinClassifier().predict(image)
    print(f"Top guess: {result['label']} ({result['confidence']:.1%})")
    print(f"Inconclusive: {result['low_confidence']} (threshold {result['threshold']:.2f})")
    print("All scores:")
    for name, score in sorted(result["all_scores"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:20s} {score:.1%}")