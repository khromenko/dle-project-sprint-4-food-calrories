import re

# парсинг списка в виде строки (так хранится в csv) в реальный список
INGR_RE = re.compile(r"""['"]([^'"]*)['"]""")
EXCLUDE = {'plate only', 'deprecated'}

def parse_str_to_list(raw_str_ingr: str) -> list:
    """
    Парсинг строки со списом ингредиентов в список
    """
    ingrs = INGR_RE.findall(raw_str_ingr)
    return ingrs

def preprocess_ingredients(raw_ingr_list_str: str) -> list:
    """
    Препроцессинг строки `raw_ingr_list_str` из csv -> сырой список названий ингредиентов.
    Убирает ненужные символы '(', ')', '.', ',') и выкидывает пустые/'plate only' токены.
    """

    ingrs = parse_str_to_list(raw_ingr_list_str)
    cleaned = []
    for name in ingrs:
        ingr = clean_ingr_name(name)
        if ingr and ingr not in EXCLUDE:
            cleaned.append(ingr)
    return cleaned

def clean_ingr_name(ingr: str):
    """
    Убирает лишние символы ('(', ')', '.', ',') и схлопывает пробелы
    """

    ingr_clean = ingr.replace('(', ' ').replace(')', ' ') 
    ingr_clean = re.sub(r'[.,;]', ' ', ingr_clean)
    ingr_clean = re.sub(r'\s+', ' ', ingr_clean).strip().lower()
    
    return ingr_clean