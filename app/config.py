import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")
EXPERIMENTS_PATH = os.path.join(DATA_DIR, "experiments.json")
CONFUSION_PATH = os.path.join(DATA_DIR, "confusion_matrix.json")

# Классы ImageNet, связанные с едой (индексы ~924–969), и доп. подписи для отчётов.
FOOD_CLASSES_RU = {
    "guacamole": "гуакамоле",
    "consomme": "консоме",
    "hot pot": "хот-пот",
    "trifle": "трайфл",
    "ice cream": "мороженое",
    "ice lolly": "эскимо",
    "french loaf": "батон",
    "bagel": "бейгл",
    "pretzel": "крендель",
    "cheeseburger": "чизбургер",
    "hotdog": "хот-дог",
    "hot dog": "хот-дог",
    "mashed potato": "картофельное пюре",
    "head cabbage": "капуста",
    "broccoli": "брокколи",
    "cauliflower": "цветная капуста",
    "zucchini": "кабачок",
    "spaghetti squash": "тыква-спагетти",
    "acorn squash": "тыква",
    "butternut squash": "тыква мускатная",
    "cucumber": "огурец",
    "artichoke": "артишок",
    "bell pepper": "болгарский перец",
    "cardoon": "кардон",
    "mushroom": "грибы",
    "granny smith": "яблоко",
    "strawberry": "клубника",
    "orange": "апельсин",
    "lemon": "лимон",
    "fig": "инжир",
    "pineapple": "ананас",
    "banana": "банан",
    "jackfruit": "джекфрут",
    "custard apple": "черимойя",
    "pomegranate": "гранат",
    "carbonara": "карбонара",
    "chocolate sauce": "шоколадный соус",
    "dough": "тесто",
    "meat loaf": "мясной рулет",
    "pizza": "пицца",
    "potpie": "пирог",
    "burrito": "буррито",
    "red wine": "красное вино",
    "espresso": "эспрессо",
    "eggnog": "эгг-ног",
    "hamburger": "гамбургер",
    "french fries": "картофель фри",
    "chocolate cake": "шоколадный торт",
    "carrot cake": "морковный торт",
    "spaghetti bolognese": "спагетти",
    "caesar salad": "салат цезарь",
    "sushi": "суши",
    "ramen": "рамен",
    "donut": "пончик",
    "waffle": "вафля",
    "pancake": "блин",
    "cappuccino": "капучино",
    "beer": "пиво",
    "croissant": "круассан",
    "taco": "тако",
    "steak": "стейк",
    "lobster": "лобстер",
    "oyster": "устрица",
    "fried rice": "жареный рис",
}

# Не показывать как блюда (стол, посуда, меню и т.п.)
NON_FOOD_IMAGENET = {
    "plate",
    "menu",
    "cup",
    "book jacket",
    "traffic light",
    "hay",
    "dining table",
    "table",
}

MODEL_NAMES_RU = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "DenseNet-121",
    "all_models": "Все модели",
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
