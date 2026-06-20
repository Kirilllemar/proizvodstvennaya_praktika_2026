import json
import time

import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)

from app.config import CONFUSION_PATH, EXPERIMENTS_PATH, FOOD_CLASSES_RU

_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_model_cache = {}
_imagenet_labels = None
_food_indices = None


def _load_imagenet_labels():
    global _imagenet_labels
    if _imagenet_labels is None:
        weights = EfficientNet_B0_Weights.DEFAULT
        _imagenet_labels = weights.meta["categories"]
    return _imagenet_labels


def _get_food_indices():
    global _food_indices
    if _food_indices is None:
        labels = _load_imagenet_labels()
        indices = {}
        for idx, name in enumerate(labels):
            key = name.lower()
            if key in FOOD_CLASSES_RU:
                indices[idx] = FOOD_CLASSES_RU[key]
        _food_indices = indices
    return _food_indices


def _build_model(model_id):
    builders = {
        "resnet18": lambda: models.resnet18(weights=ResNet18_Weights.DEFAULT),
        "resnet50": lambda: models.resnet50(weights=ResNet50_Weights.DEFAULT),
        "mobilenet_v3_small": lambda: models.mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        ),
        "efficientnet_b0": lambda: models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        ),
        "densenet121": lambda: models.densenet121(weights=DenseNet121_Weights.DEFAULT),
    }
    model = builders[model_id]()
    model.eval()
    return model


def get_model(model_id):
    if model_id not in _model_cache:
        _model_cache[model_id] = _build_model(model_id)
    return _model_cache[model_id]


def predict(image_path, model_id="efficientnet_b0", top_k=3):
    model = get_model(model_id)
    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    inference_ms = (time.perf_counter() - start) * 1000

    food_indices = _get_food_indices()
    food_probs = []
    for idx, ru_name in food_indices.items():
        food_probs.append((ru_name, float(probs[idx].item())))

    food_probs.sort(key=lambda x: x[1], reverse=True)

    if not food_probs or food_probs[0][1] < 0.01:
        labels = _load_imagenet_labels()
        top_vals, top_idxs = torch.topk(probs, top_k)
        results = []
        for prob, idx in zip(top_vals, top_idxs):
            en = labels[idx].lower()
            ru = FOOD_CLASSES_RU.get(en, labels[idx])
            results.append({"class": ru, "confidence": round(float(prob), 4)})
    else:
        results = [
            {"class": name, "confidence": round(conf, 4)}
            for name, conf in food_probs[:top_k]
        ]

    return results, round(inference_ms, 1)


def compare_all_models(image_path):
    experiments = load_experiments()
    comparisons = []
    for arch in experiments["architectures"]:
        top3, ms = predict(image_path, arch["id"], top_k=3)
        comparisons.append(
            {
                "model_id": arch["id"],
                "model_name": arch["name"],
                "top3": top3,
                "inference_ms": ms,
            }
        )
    return comparisons


def load_experiments():
    with open(EXPERIMENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_confusion_matrix():
    with open(CONFUSION_PATH, encoding="utf-8") as f:
        return json.load(f)
