import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")
EXPERIMENTS_PATH = os.path.join(DATA_DIR, "experiments.json")
CONFUSION_PATH = os.path.join(DATA_DIR, "confusion_matrix.json")

FOOD_CLASSES_RU = {
    "pizza": "пицца",
    "cheeseburger": "бургер",
    "hot dog": "хот-дог",
    "mashed potato": "пюре",
    "french fries": "картофель фри",
    "ice cream": "мороженое",
    "chocolate cake": "шоколадный торт",
    "carrot cake": "морковный торт",
    "spaghetti bolognese": "спагетти",
    "carbonara": "карбонара",
    "caesar salad": "салат цезарь",
    "sushi": "суши",
    "ramen": "рамен",
    "guacamole": "гуакамоле",
    "hamburger": "гамбургер",
    "bagel": "бейгл",
    "donut": "пончик",
    "waffle": "вафля",
    "pancake": "блин",
    "espresso": "эспрессо",
    "cappuccino": "капучино",
    "red wine": "красное вино",
    "beer": "пиво",
    "croissant": "круассан",
    "burrito": "буррито",
    "taco": "тако",
    "steak": "стейк",
    "lobster": "лобстер",
    "oyster": "устрица",
    "fried rice": "жареный рис",
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}
