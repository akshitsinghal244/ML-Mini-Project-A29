"""Predict the bird species in an image with the trained weights.

Usage:
    python src/predict.py path/to/photo.jpg --weights bird_model.pth --topk 3
"""
import argparse

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device)  # dict: state_dict, arch, names, use_bbox
    model = models.get_model(ckpt["arch"], weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ckpt["names"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["names"]


def predict(image_path, model, names, device, topk=3):
    x = EVAL_TF(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1)[0].cpu()
    conf, idx = probs.topk(topk)
    return [(names[i], float(c)) for i, c in zip(idx, conf)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default="bird_model.pth")
    ap.add_argument("--topk", type=int, default=3)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, names = load_model(a.weights, device)
    for name, c in predict(a.image, model, names, device, a.topk):
        print(f"{c:6.1%}  {name}")
