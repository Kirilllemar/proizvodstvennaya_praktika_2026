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

from app.config import (
    CONFUSION_PATH,
    DISH_GROUPS,
    EXPERIMENTS_PATH,
    GROUPED_IMAGENET_INDICES,
    IMAGENET_FOOD_RU,
    MODEL_NAMES_RU,
    NON_FOOD_IMAGENET,
)

_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_model_cache = {}
_food_indices = None


def _get_food_indices():
    global _food_indices
    if _food_indices is None:
        _food_indices = {
            idx: ru_name
            for idx, ru_name in IMAGENET_FOOD_RU.items()
            if idx not in NON_FOOD_IMAGENET
        }
    return _food_indices


def get_model_name(model_id):
    experiments = load_experiments()
    for arch in experiments["architectures"]:
        if arch["id"] == model_id:
            return arch["name"]
    return MODEL_NAMES_RU.get(model_id, model_id)


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


def _infer_probs(model, image):
    tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        return torch.nn.functional.softmax(outputs[0], dim=0)


def _food_top_k(probs, top_k=3):
    scores = {}

    for dish_name, indices in DISH_GROUPS.items():
        score = sum(float(probs[idx]) for idx in indices if idx < len(probs))
        if score > 0:
            scores[dish_name] = score

    for idx, ru_name in _get_food_indices().items():
        if idx in GROUPED_IMAGENET_INDICES:
            continue
        prob = float(probs[idx])
        if prob > 0:
            scores[ru_name] = scores.get(ru_name, 0) + prob

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if not ranked:
        return [{"class": "блюдо не распознано", "confidence": 0.0}]

    return [
        {"class": name, "confidence": round(conf, 4)}
        for name, conf in ranked[:top_k]
    ]


def predict(image_path, model_id="efficientnet_b0", top_k=3):
    model = get_model(model_id)
    image = Image.open(image_path).convert("RGB")

    start = time.perf_counter()
    probs = _infer_probs(model, image)
    inference_ms = (time.perf_counter() - start) * 1000

    results = _food_top_k(probs, top_k)
    return results, round(inference_ms, 1)


def predict_from_image(image, model_id="efficientnet_b0", top_k=1):
    model = get_model(model_id)
    start = time.perf_counter()
    probs = _infer_probs(model, image)
    inference_ms = (time.perf_counter() - start) * 1000
    return _food_top_k(probs, top_k), round(inference_ms, 1)


def predict_multi_dishes(
    image_path,
    model_id="efficientnet_b0",
    grid=3,
    min_confidence=0.001,
    max_dishes=8,
):
    """Приблизительный поиск нескольких блюд: сканирование областей 3×3."""
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    cell_w = max(width // grid, 1)
    cell_h = max(height // grid, 1)

    aggregated = {}
    total_ms = 0.0

    for row in range(grid):
        for col in range(grid):
            left = col * cell_w
            top = row * cell_h
            right = width if col == grid - 1 else left + cell_w
            bottom = height if row == grid - 1 else top + cell_h
            crop = image.crop((left, top, right, bottom))
            top1, ms = predict_from_image(crop, model_id, top_k=1)
            total_ms += ms
            if not top1:
                continue
            pred = top1[0]
            if pred["confidence"] < min_confidence:
                continue
            name = pred["class"]
            if name == "блюдо не распознано":
                continue
            region = f"область {row + 1}×{col + 1}"
            if name in aggregated:
                aggregated[name]["confidence"] = max(
                    aggregated[name]["confidence"], pred["confidence"]
                )
                aggregated[name]["regions"].append(region)
            else:
                aggregated[name] = {
                    "class": name,
                    "confidence": pred["confidence"],
                    "regions": [region],
                }

    dishes = sorted(aggregated.values(), key=lambda x: x["confidence"], reverse=True)
    dishes = dishes[:max_dishes]
    for dish in dishes:
        dish["confidence"] = round(dish["confidence"], 4)
        dish["regions"] = ", ".join(dish["regions"])

    return dishes, round(total_ms, 1)


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
