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
    BURRITO_INDEX,
    CONFUSION_PATH,
    DISH_GROUPS,
    EXPERIMENTS_PATH,
    GROUPED_IMAGENET_INDICES,
    IMAGENET_FOOD_RU,
    MIN_CONFIDENCE,
    MODEL_NAMES_RU,
    NON_FOOD_IMAGENET,
    SEAFOOD_INDICES,
    SUSHI_RICE_VEG_INDICES,
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


def _prob(probs, idx):
    if idx >= len(probs):
        return 0.0
    return float(probs[idx])


def _compute_dish_scores(probs):
    scores = {}

    for dish_name, weighted_indices in DISH_GROUPS.items():
        score = sum(_prob(probs, idx) * weight for idx, weight in weighted_indices)
        if score > 0:
            scores[dish_name] = score

    seafood = sum(_prob(probs, idx) * 3.0 for idx in SEAFOOD_INDICES)
    rice_veg = sum(_prob(probs, idx) * 2.0 for idx in SUSHI_RICE_VEG_INDICES)
    burrito = _prob(probs, BURRITO_INDEX)
    has_sushi_signals = seafood > 0.008 or rice_veg > 0.04

    if has_sushi_signals:
        scores["Суши"] = seafood + rice_veg + burrito * 1.2
        scores["Роллы"] = burrito * 0.15
        scores.pop("Буррито", None)
    elif burrito > 0.15:
        scores["Буррито"] = burrito * 2.0
        scores["Роллы"] = burrito * 0.8
        scores.pop("Суши", None)
    else:
        scores["Роллы"] = burrito * 1.5 + _prob(probs, 931) + _prob(probs, 932)

    steak = scores.get("Стейк", 0)
    soup = scores.get("Суп", 0)
    meat = _prob(probs, 962)
    if meat > 0.004 or steak > 0.02:
        scores["Стейк"] = max(steak, meat * 3.5)
        scores["Суп"] = soup * 0.12
    elif soup > 0.05:
        scores.pop("Стейк", None)

    for idx, ru_name in _get_food_indices().items():
        if idx in GROUPED_IMAGENET_INDICES:
            continue
        prob = _prob(probs, idx)
        if prob > 0:
            scores[ru_name] = scores.get(ru_name, 0) + prob

    return scores


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
    scores = _compute_dish_scores(probs)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if not ranked:
        return [{"class": "блюдо не распознано", "confidence": 0.0}]

    results = [
        {"class": name, "confidence": round(conf, 4)}
        for name, conf in ranked[:top_k]
    ]

    if results[0]["confidence"] < MIN_CONFIDENCE:
        results[0]["class"] = f"{results[0]['class']} (низкая уверенность)"

    return results


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
            name = pred["class"].replace(" (низкая уверенность)", "")
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


def best_compare_result(comparisons, best_model_id="efficientnet_b0"):
    for item in comparisons:
        if item["model_id"] == best_model_id:
            return item
    return max(comparisons, key=lambda x: x["top3"][0]["confidence"])


def load_experiments():
    with open(EXPERIMENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_confusion_matrix():
    with open(CONFUSION_PATH, encoding="utf-8") as f:
        return json.load(f)
