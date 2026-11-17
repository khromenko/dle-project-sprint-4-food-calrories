import numpy as np
import re
from typing import Dict
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pathlib as path
from PIL import Image
import albumentations as A
import timm
import transformers
import scipy.sparse as sparse

# locals
import scripts.models as models

# парсинг списка в виде строки (так хранится в csv) в реальный список
_INGR_RE = re.compile(r"""['"]([^'"]*)['"]""")

# Эксперимент 9 - подсказка о типе блюда по ключевым ингредиентам.
# также как префикс с количеством блюд (ing_count), 
# который помогает DistilBERT'у выявлять калорийные блюда
_HINT_RULES = [
    ('high-water',  ('water', 'juice', 'soup', 'broth', 'melon', 'watermelon',
                     'cucumber', 'lettuce', 'celery', 'ice', 'milk')),
    ('high-fat',    ('oil', 'butter', 'cream', 'cheese', 'bacon', 'mayonnaise',
                     'fried', 'almond', 'nut', 'peanut', 'avocado')),
    ('high-carb',   ('rice', 'pasta', 'bread', 'potato', 'flour', 'sugar',
                     'honey', 'tortilla', 'noodle')),
    ('protein',     ('chicken', 'beef', 'pork', 'fish', 'salmon', 'tuna',
                     'egg', 'turkey', 'shrimp')),
]

def _preprocess_ingredients(ingr_name: str) -> list:
    """
    Препроцессинг строки `ingr_name` из csv -> список названий ингредиентов.

    Убирает мусорные символы ('(', ')', '.', ',') и выкидывает пустые/'plate only' токены.
    """
    names = _INGR_RE.findall(ingr_name)
    cleaned = []
    for n in names:
        n = n.replace('(', ' ').replace(')', ' ')
        n = re.sub(r'[.,;]', ' ', n)
        n = re.sub(r'\s+', ' ', n).strip().lower()
        if n and n != 'plate only':
            cleaned.append(n)
    return cleaned

def _dish_hint(ingredients: list) -> str:
    """Определяет грубый тип блюда по списку ингредиентов (по приоритету правил)."""
    joined = ' '.join(ingredients).lower()
    for hint, keywords in _HINT_RULES:
        if any(kw in joined for kw in keywords):
            return hint
    return 'mixed'

def make_ingr2cal_matrix(ingr_cal_mapping_df: pd.DataFrame):
    """
    Создает квадратную диагональную csr-матрицу калорийности ингредиентов:
    - строка и столбцы - соответсвует train-ингредиентам
    - значения - справочная калорийность/100г ингредиента

    Матрица используется для вычисления суммарного вектора калорийности блюда по его ингредиентам

    Returns:
        (ingr2kal_matrix, ingr_encoder) (tuple): 
            - ingr2kal_matrix: csr-матрица калорийности ингредиентов
            - ingr_encoder: кодировщик ингредиентов для поиска ингредиента по индексу в матрице
    """
    # read cal data for ingredients from provided mapping
    ingr_cal_mapping = ingr_cal_mapping_df[['food_ingr', 'energy_kcal']].to_numpy()
    
    print('Ingredients kal mapping:', ingr_cal_mapping.shape)
    print('top 10 records:', ingr_cal_mapping[0:10])
    print('------')

    # create encoder for ingredients to make positional index search
    ingr_encoder = {row[0]: i  for i, row in enumerate(ingr_cal_mapping)}

    # make csr-matrix - square matrix with diagonal items
    kal_data = ingr_cal_mapping[:,1]    
    ingr2kal_matrix = sparse.csr_matrix(
        (kal_data, (list(ingr_encoder.values()), list(ingr_encoder.values()))),
        dtype=int
    )

    return ingr2kal_matrix, ingr_encoder
        
def make_dish2cal_reference_vector(dish_ingredients: str|list, ingr2kal_matrix: sparse.csr_matrix, ingr_encoder: dict):
    """
    Создает (разреженный) вектор калорийности блюда размером (1, N_ingr):
    - N_ingr - количество ингредиентов (фиксируется train-датасетом)
    - значения - справочная калорийность/100г ингредиента в соответсвующей позиции матрицы калорийностей

    Фактически выполняется суммирование строк-ингредиентов из `ingr2kal_matrix` по их индексам, определяемым `ingr_encoder`

    Returns:
        kal_vector (np.ndarray): 
            суммарный вектор калорийностей по ингредиентам блюда
    """
    if type(dish_ingredients) == str:
        dish_ingredients = _preprocess_ingredients(dish_ingredients)

    result_matrix = np.zeros(len(ingr_encoder))
    ingr_ids = [ingr_encoder[ingr] for ingr in dish_ingredients]
        
    kal_vector = ingr2kal_matrix[ingr_ids].sum(axis=0)
    return kal_vector



class MealDataset(Dataset):
    def __init__(self, model_config: models.MealModelConfig, split='train', ingr2kal_ref_matrix: sparse.csr_matrix = None, ingr_encoder: dict = None): # type: ignore
        super().__init__()
        
        try:
            df = pd.read_csv(path.Path(model_config.DATASET_PATH))
            # убираем блюда с нулевой калорийностью - это просто пустая тарелка да еще и с ошибкой в массе в датасете
            # для расчета логарифмического масштаба 0 - это запрещенное значение, и в целом для модели это не нужно.
            df = df[(df['split'] == split) & (df['total_calories'] > 0)]

            # заглушаем текстовую составляющую для проверки только визуальной модели
            if model_config.STUB_TEXT:
                df['ingr_name'] = "['stub_text']"

        except Exception as e:
            print(f'Error reading dataset from: {model_config.DATASET_PATH}, exc = {e}')
            raise RuntimeError(e)

        if split == 'train': 
            try:
                # Эксперимент 2.1 - матрица калорийности и кодировщик ингридиентов
                ref_kal_mapping_df = pd.read_csv(model_config.DATASET_EXT_KAL_MAPPING)
            except Exception as e:
                print(f'Error reading dataset from: {model_config.DATASET_EXT_KAL_MAPPING}, exc = {e}')
                raise RuntimeError(e)
            
        self.df = df
        self.config = model_config

        if split == 'train': 
            # Эксперимент 2.1 - матрица калорийности и кодировщик ингридиентов
            self.ingr2kal_ref_matrix, self.ingr_encoder = make_ingr2cal_matrix(ref_kal_mapping_df)

            # проверить что значения распределений не отличаются от заданных в конфиге
            self.verify_mean_std(df, ref_kal_mapping_df)
        else:
            # на test передать готовые объекты после train
            self.ingr2kal_ref_matrix, self.ingr_encoder = ingr2kal_ref_matrix, ingr_encoder

        # images
        self.img_path = path.Path(model_config.DATASET_IMAGES_PATH)
        img_model_config = timm.get_pretrained_cfg(model_name=model_config.IMG_MODEL_NAME); assert img_model_config is not None
        img_size = img_model_config.input_size[1] # CxHxW
        
        # базовые трансформы 
        base_transforms_train = [
            A.SmallestMaxSize(max_size=img_size),
            A.RandomCrop(height=img_size, width=img_size),
        ]
        base_transforms_val = [
            A.SmallestMaxSize(max_size=img_size),
            A.CenterCrop(height=img_size, width=img_size),
        ]
        
        # аугментации для train - расширяем датасет
        aug_transforms = [
            A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), translate_percent=(-0.1, 0.1),
                        shear=(-10, 10), fill=0, p=0.5),
            A.CoarseDropout(num_holes_range=(2, 8),
                            hole_height_range=(int(0.07 * img_size), int(0.15 * img_size)),
                            hole_width_range=(int(0.1 * img_size), int(0.15 * img_size)),
                            fill=0, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        ]
        
        post_transforms = [
            A.Normalize(mean=img_model_config.mean, std=img_model_config.std),
            A.ToTensorV2()
        ]

        if split == 'train':
            self.img_transforms = A.Compose(base_transforms_train + aug_transforms + post_transforms)
        else:
            self.img_transforms = A.Compose(base_transforms_val + post_transforms) 

    def verify_mean_std(self, df, ref_kal_mappping):
        """
        Проверка, что значения распределений не отличаются от заданных в конфиге
        """
        kal = df["total_calories"]
        assert np.isclose(self.config.TARGET_MEAN, kal.mean(), atol=.5), f'expected TARGET_MEAN = {self.config.TARGET_MEAN}, real data = {kal.mean():.2f}'
        assert np.isclose(self.config.TARGET_STD, kal.std(), atol=0.5) , f'expected TARGET_STD  = {self.config.TARGET_STD}, real data = {kal.std():.2f}'

        mass = np.log(df['total_mass'])
        assert np.isclose(self.config.MASS_LOG_MEAN, mass.mean(), atol=.05), f'expected MASS_LOG_MEAN = {self.config.MASS_LOG_MEAN}, real = {mass.mean():.4f}'
        assert np.isclose(self.config.MASS_LOG_STD, mass.std(), atol=.05), f'expected MASS_LOG_STD = {self.config.MASS_LOG_STD}), real = {mass.std():.4f}'
        
        k_100 = np.log(100 * df['total_calories'] / df['total_mass'])
        assert np.isclose(self.config.K_100_LOG_MEAN, k_100.mean(), atol=.005), f'expected K_100_LOG_MEAN  = {self.config.K_100_LOG_MEAN}, real = {k_100.mean():.4f}'
        assert np.isclose(self.config.K_100_LOG_STD, k_100.std(), atol=.005), f'expected K_100_LOG_STD  = {self.config.K_100_LOG_STD}, real = {k_100.std():.4f}'

        ref_k100 = ref_kal_mappping['energy_kcal']
        assert np.isclose(self.config.K_100_REF_MEAN, ref_k100.mean(), atol=.5), f'expected K_100_REF_MEAN  = {self.config.K_100_REF_MEAN}, real = {ref_k100.mean():.4f}'
        assert np.isclose(self.config.K_100_REF_STD, ref_k100.std(), atol=.5), f'expected K_100_REF_STD  = {self.config.K_100_REF_STD}, real = {ref_k100.std():.4f}'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        Преобразует картинку с помощью набора transformers согласно конфигурации IMAGE_MODEL.
        Текст должен быть токенизирован в рамках DataLoader `collate_fn`.
        
        Returns:

            dict
                - text: list[str] - список "слов" - названий ингредиентов
                - image: torch.Tensor - трансформированное изображение
                - target: float - целевая переменная (total_calories) - нормализованное значание на MEAN/STD       

                - kal_ref_vector - вектор справочной калорийности/100г по ингредиентам блюда

                - target_mass - масса блюда (как дополнительный таргет) - лог масштаб и нормализация на MEAN/STD       
                - target_kal_100 - калорий на 100г (как дополнительный таргет) - лог масштаб и нормализация на MEAN/STD
        """

        item = self.df.iloc[index]
        dish_id = item['dish_id']

        # парсим текст со списком в реальный список
        text = _preprocess_ingredients(item['ingr_name'])
        
        # # Эксперимент 2.1 - формируем вектор калорийности ингредиентов блюда по справочнику
        kal_ref_vector = make_dish2cal_reference_vector(text, self.ingr2kal_ref_matrix, self.ingr_encoder)

        # Эксперимент 9: добавляем текстовые подсказки о БЖУ:
        # - количество ингредиентов 
        # - о типе блюда (жиры, углеводы и проч.)
        hint_count = item['ing_count']
        hint_bju = _dish_hint(text)
        hinted_text = f"ingredients: {hint_count}. type: {hint_bju}. " + ", ".join(text)

        # target - norm
        target = item['total_calories']        
        target_norm = (target - self.config.TARGET_MEAN)/self.config.TARGET_STD

        # масса - log + norm
        mass = item['total_mass']
        mass_log = np.log(mass)
        mass_lognorm = (mass_log - self.config.MASS_LOG_MEAN) / self.config.MASS_LOG_STD

        # kal_100 - log + norm
        kal_100 = target / mass * 100
        kal_100_log = np.log(kal_100)
        kal_100_lognorm = (kal_100_log - self.config.K_100_LOG_MEAN) / self.config.K_100_LOG_STD

        # image + transform
        original_image = Image.open(self.img_path / dish_id / 'rgb.png')
        image = self.img_transforms(image=np.array(original_image))["image"]

        return {
            'text': hinted_text,
            'image': image,
            'kal_ref_vector': kal_ref_vector,
            'target': target_norm,
            'target_mass': mass_lognorm,
            'target_kal_100': kal_100_lognorm
        }
    
def collate_fn(batch, tokenizer: transformers.BertTokenizer):
    """
    Собирает батч из токенизированных текстов и картинок

    Returns:

        dict:
            - input_ids - токенизированный текст, выровненный padding-ом до максимальной длины в батче,
            - attention_mask - маска токенизированного текста,
            - images - изображения (передаются как есть, уже трансформированы)
            
            - kal_ref_vector - вектор справочной калорийности/100г по ингредиентам блюда
            
            - target - целевые значения калорийности блюда
            
            - target_mass - масса блюда (как дополнительный таргет)
            - target_kal_100 - калорий на 100г (как дополнительный таргет)    
    """

    texts = [item['text'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    kal_ref_vector = torch.stack([item['kal_ref_vector'] for item in batch])

    targets = torch.FloatTensor([item['target'] for item in batch])
    target_mass = torch.FloatTensor([item['target_mass'] for item in batch])
    target_kal_100 = torch.FloatTensor([item['target_kal_100'] for item in batch])

    tokenized = tokenizer(texts, return_tensors='pt', padding='longest')

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'images': images,
        'kal_ref_vector': kal_ref_vector,
        
        'target': targets,
        'target_mass': target_mass,
        'target_kal_100': target_kal_100
    }
