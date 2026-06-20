import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
DB_PATH = os.path.join(BASE_DIR, "data", "history.db")
EXPERIMENTS_PATH = os.path.join(DATA_DIR, "experiments.json")
CONFUSION_PATH = os.path.join(DATA_DIR, "confusion_matrix.json")

# Каталог пищевых классов ImageNet: индекс → русское название + пояснение.
# Названия подобраны для понятного отображения в интерфейсе (без лишней транслитерации).
FOOD_CLASSES_CATALOG = [
    {"index": 924, "en": "guacamole", "ru": "Гуакамоле", "desc": "Соус из авокадо"},
    {"index": 925, "en": "consomme", "ru": "Мясной бульон", "desc": "Прозрачный консоме"},
    {"index": 926, "en": "hot pot", "ru": "Мясной суп с овощами", "desc": "Тушёное мясо с овощами в бульоне"},
    {"index": 927, "en": "trifle", "ru": "Слоёный десерт", "desc": "В распознавании объединяется в группу «Торт»"},
    {"index": 928, "en": "ice cream", "ru": "Мороженое", "desc": "Мороженое в чаше или рожке"},
    {"index": 929, "en": "ice lolly", "ru": "Мороженое на палочке", "desc": "Эскимо, фруктовый лед"},
    {"index": 930, "en": "French loaf", "ru": "Батон", "desc": "Белый хлеб"},
    {"index": 931, "en": "bagel", "ru": "Бейгл", "desc": "Круглая булочка с отверстием"},
    {"index": 932, "en": "pretzel", "ru": "Крендель", "desc": "Солёная выпечка"},
    {"index": 933, "en": "cheeseburger", "ru": "Чизбургер", "desc": "Бургер с сыром"},
    {"index": 934, "en": "hotdog", "ru": "Хот-дог", "desc": "Сосиска в булке"},
    {"index": 935, "en": "mashed potato", "ru": "Картофельное пюре", "desc": "Пюре из картофеля"},
    {"index": 936, "en": "head cabbage", "ru": "Капуста", "desc": "Белокочанная капуста"},
    {"index": 937, "en": "broccoli", "ru": "Брокколи", "desc": "Капуста брокколи"},
    {"index": 938, "en": "cauliflower", "ru": "Цветная капуста", "desc": "Капуста цветная"},
    {"index": 939, "en": "zucchini", "ru": "Кабачок", "desc": "Кабачок, цукини"},
    {"index": 940, "en": "spaghetti squash", "ru": "Тыква", "desc": "Овощная тыква (не паста)"},
    {"index": 941, "en": "acorn squash", "ru": "Тыква", "desc": "Жёлтая тыква"},
    {"index": 942, "en": "butternut squash", "ru": "Тыква мускатная", "desc": "Мускатная тыква"},
    {"index": 943, "en": "cucumber", "ru": "Огурец", "desc": "Свежий огурец"},
    {"index": 944, "en": "artichoke", "ru": "Артишок", "desc": "Артишок"},
    {"index": 945, "en": "bell pepper", "ru": "Болгарский перец", "desc": "Сладкий перец"},
    {"index": 946, "en": "cardoon", "ru": "Кардон", "desc": "Стеблевой овощ"},
    {"index": 947, "en": "mushroom", "ru": "Грибы", "desc": "Объединяется с agaric, bolete и др."},
    {"index": 948, "en": "Granny Smith", "ru": "Яблоко", "desc": "Зелёное яблоко"},
    {"index": 949, "en": "strawberry", "ru": "Клубника", "desc": "Свежая клубника"},
    {"index": 950, "en": "orange", "ru": "Апельсин", "desc": "Апельсин"},
    {"index": 951, "en": "lemon", "ru": "Лимон", "desc": "Лимон"},
    {"index": 952, "en": "fig", "ru": "Инжир", "desc": "Инжир"},
    {"index": 953, "en": "pineapple", "ru": "Ананас", "desc": "Ананас"},
    {"index": 954, "en": "banana", "ru": "Банан", "desc": "Банан"},
    {"index": 955, "en": "jackfruit", "ru": "Джекфрут", "desc": "Джекфрут"},
    {"index": 956, "en": "custard apple", "ru": "Черимойя", "desc": "Фрукт черимойя"},
    {"index": 957, "en": "pomegranate", "ru": "Гранат", "desc": "Гранат"},
    {"index": 959, "en": "carbonara", "ru": "Спагетти карбонара", "desc": "Паста с беконом и сыром"},
    {"index": 960, "en": "chocolate sauce", "ru": "Шоколадный соус", "desc": "Шоколадная глазурь или соус"},
    {"index": 961, "en": "dough", "ru": "Тесто", "desc": "Сырое или готовое тесто"},
    {
        "index": 962,
        "en": "meat loaf",
        "ru": "Стейк",
        "desc": "Жареное мясо; в ImageNet нет отдельного класса «steak»",
    },
    {"index": 963, "en": "pizza", "ru": "Пицца", "desc": "Пицца"},
    {"index": 964, "en": "potpie", "ru": "Пирог", "desc": "Пирог с начинкой"},
    {"index": 965, "en": "burrito", "ru": "Буррито", "desc": "Мексиканская лепёшка с начинкой"},
    {"index": 966, "en": "red wine", "ru": "Красное вино", "desc": "Красное вино"},
    {"index": 967, "en": "espresso", "ru": "Эспрессо", "desc": "Кофе эспрессо"},
    {"index": 969, "en": "eggnog", "ru": "Яичный коктейль", "desc": "Молочный напиток с яйцом и специями"},
]

IMAGENET_FOOD_RU = {item["index"]: item["ru"] for item in FOOD_CLASSES_CATALOG}

# Группы блюд: суммируем вероятности нескольких классов ImageNet.
# Нужно, потому что в ImageNet нет «суши», «роллов», «торта» как отдельных классов.
DISH_GROUPS = {
    "Грибы": [947, 992, 997, 991, 993],
    "Торт": [927, 415, 960, 961],
    "Суши": [390, 122, 118, 119, 120, 121, 123, 124, 125, 393, 396, 935, 943],
    "Роллы": [965, 931, 932],
    "Пицца": [963],
    "Бургер": [933],
    "Стейк": [962],
    "Паста": [959],
    "Пирог": [964],
    "Мороженое": [928, 929],
    "Хот-дог": [934],
    "Кофе": [967],
}

DISH_GROUPS_CATALOG = [
    {"ru": "Грибы", "desc": "mushroom, agaric, bolete и др. виды грибов в ImageNet"},
    {"ru": "Торт", "desc": "trifle, bakery, шоколад — ближайшие классы для торта и выпечки"},
    {"ru": "Суши", "desc": "морепродукты, рыба, рис (mashed potato), огурец — типичные признаки суши"},
    {"ru": "Роллы", "desc": "burrito, hotdog, bagel — цилиндрические блюда, похожие на роллы"},
    {"ru": "Пицца", "desc": "класс pizza"},
    {"ru": "Бургер", "desc": "класс cheeseburger"},
    {"ru": "Стейк", "desc": "класс meat loaf (жареное мясо в ImageNet)"},
    {"ru": "Паста", "desc": "класс carbonara"},
    {"ru": "Пирог", "desc": "класс potpie"},
    {"ru": "Мороженое", "desc": "ice cream, ice lolly"},
    {"ru": "Кофе", "desc": "класс espresso"},
]

GROUPED_IMAGENET_INDICES = {idx for indices in DISH_GROUPS.values() for idx in indices}

# Индексы ImageNet, которые не являются едой (посуда, меню, стол и т.п.)
NON_FOOD_IMAGENET = {
    920,  # traffic light
    921,  # book jacket
    922,  # menu
    923,  # plate
    958,  # hay
    968,  # cup
    970,  # alp
    971,  # bubble
}

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
