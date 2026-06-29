import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")
EXPERIMENTS_PATH = os.path.join(DATA_DIR, "experiments.json")
CONFUSION_PATH = os.path.join(DATA_DIR, "confusion_matrix.json")

FOOD_CLASSES_CATALOG = [
    {"index": 924, "en": "guacamole", "ru": "Гуакамоле", "desc": "Соус из авокадо"},
    {"index": 925, "en": "consomme", "ru": "Мясной бульон", "desc": "В группе «Суп»"},
    {"index": 926, "en": "hot pot", "ru": "Мясной суп с овощами", "desc": "В группе «Суп»"},
    {"index": 927, "en": "trifle", "ru": "Слоёный десерт", "desc": "В группе «Торт»"},
    {"index": 928, "en": "ice cream", "ru": "Мороженое", "desc": "Мороженое в чаше или рожке"},
    {"index": 929, "en": "ice lolly", "ru": "Мороженое на палочке", "desc": "Эскимо"},
    {"index": 930, "en": "French loaf", "ru": "Батон", "desc": "Белый хлеб"},
    {"index": 931, "en": "bagel", "ru": "Бейгл", "desc": "Круглая булочка"},
    {"index": 932, "en": "pretzel", "ru": "Крендель", "desc": "Солёная выпечка"},
    {"index": 933, "en": "cheeseburger", "ru": "Чизбургер", "desc": "Бургер с сыром"},
    {"index": 934, "en": "hotdog", "ru": "Хот-дог", "desc": "Сосиска в булке"},
    {"index": 935, "en": "mashed potato", "ru": "Картофельное пюре", "desc": "Признак риса для суши"},
    {"index": 936, "en": "head cabbage", "ru": "Капуста", "desc": "Белокочанная капуста"},
    {"index": 937, "en": "broccoli", "ru": "Брокколи", "desc": "Зелёные овощи"},
    {"index": 938, "en": "cauliflower", "ru": "Цветная капуста", "desc": "Цветная капуста"},
    {"index": 939, "en": "zucchini", "ru": "Кабачок", "desc": "Кабачок"},
    {"index": 940, "en": "spaghetti squash", "ru": "Тыква", "desc": "Овощная тыква"},
    {"index": 941, "en": "acorn squash", "ru": "Тыква", "desc": "Жёлтая тыква"},
    {"index": 942, "en": "butternut squash", "ru": "Тыква мускатная", "desc": "Мускатная тыква"},
    {"index": 943, "en": "cucumber", "ru": "Огурец", "desc": "Огурец"},
    {"index": 944, "en": "artichoke", "ru": "Артишок", "desc": "Артишок"},
    {"index": 945, "en": "bell pepper", "ru": "Болгарский перец", "desc": "Сладкий перец"},
    {"index": 946, "en": "cardoon", "ru": "Кардон", "desc": "Стеблевой овощ"},
    {"index": 947, "en": "mushroom", "ru": "Грибы", "desc": "В группе «Грибы»"},
    {"index": 948, "en": "Granny Smith", "ru": "Яблоко", "desc": "Яблоко"},
    {"index": 949, "en": "strawberry", "ru": "Клубника", "desc": "Клубника"},
    {"index": 950, "en": "orange", "ru": "Апельсин", "desc": "Апельсин"},
    {"index": 951, "en": "lemon", "ru": "Лимон", "desc": "Лимон"},
    {"index": 952, "en": "fig", "ru": "Инжир", "desc": "Инжир"},
    {"index": 953, "en": "pineapple", "ru": "Ананас", "desc": "Ананас"},
    {"index": 954, "en": "banana", "ru": "Банан", "desc": "Банан"},
    {"index": 955, "en": "jackfruit", "ru": "Джекфрут", "desc": "Джекфрут"},
    {"index": 956, "en": "custard apple", "ru": "Черимойя", "desc": "Фрукт черимойя"},
    {"index": 957, "en": "pomegranate", "ru": "Гранат", "desc": "Гранат"},
    {"index": 959, "en": "carbonara", "ru": "Спагетти карбонара", "desc": "Паста карбонара"},
    {"index": 960, "en": "chocolate sauce", "ru": "Шоколадный соус", "desc": "В группе «Торт»"},
    {"index": 961, "en": "dough", "ru": "Тесто", "desc": "В группе «Торт»"},
    {"index": 962, "en": "meat loaf", "ru": "Стейк", "desc": "Жареное мясо (класс meat loaf)"},
    {"index": 963, "en": "pizza", "ru": "Пицца", "desc": "Пицца"},
    {"index": 964, "en": "potpie", "ru": "Пирог", "desc": "Пирог с начинкой"},
    {"index": 965, "en": "burrito", "ru": "Буррито", "desc": "Лепёшка с начинкой; для суши — в группе «Суши»"},
    {"index": 966, "en": "red wine", "ru": "Красное вино", "desc": "Красное вино"},
    {"index": 967, "en": "espresso", "ru": "Эспрессо", "desc": "Эспрессо"},
    {"index": 969, "en": "eggnog", "ru": "Яичный коктейль", "desc": "Яичный коктейль"},
]

IMAGENET_FOOD_RU = {item["index"]: item["ru"] for item in FOOD_CLASSES_CATALOG}

# Веса > 1 используются только для ранжирования; уверенность в UI — всегда 0..100%.
DISH_GROUPS = {
    "Грибы": [(947, 1.0), (992, 1.2), (997, 1.2), (991, 0.8), (993, 0.8)],
    "Торт": [(927, 2.5), (415, 2.0), (960, 1.5), (961, 0.6), (928, 0.3)],
    "Суп": [(925, 1.5), (926, 1.5)],
    "Пицца": [(963, 1.0)],
    "Бургер": [(933, 1.0)],
    "Стейк": [(962, 3.0)],
    "Паста": [(959, 1.0)],
    "Пирог": [(964, 1.0)],
    "Мороженое": [(928, 1.0), (929, 1.0)],
    "Хот-дог": [(934, 1.0)],
    "Кофе": [(967, 1.0)],
}

SEAFOOD_INDICES = [390, 122, 118, 119, 120, 121, 123, 124, 125, 393, 396]
SUSHI_RICE_VEG_INDICES = [935, 943, 937, 950, 953]  # рис, огурец, зелень, лосось/икра
BURRITO_INDEX = 965

DISH_GROUPS_CATALOG = [
    {"ru": "Суши", "desc": "Морепродукты + рис/огурец; burrito только при признаках суши"},
    {"ru": "Роллы", "desc": "Burrito без морепродуктов — японские роллы"},
    {"ru": "Буррито", "desc": "Лаваш с начинкой (класс burrito)"},
    {"ru": "Грибы", "desc": "mushroom, agaric, bolete"},
    {"ru": "Торт", "desc": "trifle, bakery, шоколад"},
    {"ru": "Суп", "desc": "consomme, hot pot (если нет стейка на тарелке)"},
    {"ru": "Стейк", "desc": "meat loaf — приоритет над супом для жареного мяса"},
    {"ru": "Пицца", "desc": "pizza"},
    {"ru": "Бургер", "desc": "cheeseburger"},
    {"ru": "Паста", "desc": "carbonara"},
    {"ru": "Пирог", "desc": "potpie"},
    {"ru": "Мороженое", "desc": "ice cream"},
    {"ru": "Хот-дог", "desc": "hotdog"},
    {"ru": "Кофе", "desc": "espresso"},
]

GROUPED_IMAGENET_INDICES = {idx for pairs in DISH_GROUPS.values() for idx, _ in pairs}
GROUPED_IMAGENET_INDICES.update(SEAFOOD_INDICES + SUSHI_RICE_VEG_INDICES + [BURRITO_INDEX])

NON_FOOD_IMAGENET = {920, 921, 922, 923, 958, 968, 970, 971}

FOOD_CLASSES_RU = {item["en"]: item["ru"] for item in FOOD_CLASSES_CATALOG}

MODEL_NAMES_RU = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "densenet121": "DenseNet-121",
    "all_models": "Все модели",
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "gif"}

MIN_CONFIDENCE = 0.12
