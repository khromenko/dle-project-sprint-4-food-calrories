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

# locals
import scripts.utils as utils

_INGR_RE = re.compile(r"""['"]([^'"]*)['"]""")

class MealDataset(Dataset):
    def __init__(self, model_config: utils.MealModelConfig, split='train'):
        super().__init__()
        
        try:
            df = pd.read_csv(path.Path(model_config.DATASET_PATH))
            # убираем блюда с нулевой калорийностью - это просто пустая тарелка да еще и с ошибкой в массе в датасете
            # для расчета логарифмического масштаба 0 - это запрещенное значение, и в целом для модели это не нужно.
            df = df[(df['split'] == split) & (df['total_calories'] > 0)]
        except Exception as e:
            print(f'Error reading dataset from: {model_config.DATASET_PATH}, exc = {e}')
            raise RuntimeError(e)
                
        self.df = df
        self.config = model_config
        
        # проверить что значения не отличаются
        if split == 'train':
            print(f'expected TARGET_MEAN = {model_config.TARGET_MEAN}, real data =  {df["total_calories"].mean():.2f}')
            print(f'expected TARGET_STD  = {model_config.TARGET_STD}, real data = {df["total_calories"].std():.2f}')           

            _m = np.log(df['total_mass'])
            print(f'expected MASS_LOG  = ({model_config.MASS_LOG_MEAN}, {model_config.MASS_LOG_STD}), real = ({_m.mean():.4f}, {_m.std():.4f})')
            _k = df.apply(lambda r: r['total_calories']/r['total_mass']*100, axis=1)
            _k = np.log(_k[_k>0])
            print(f'expected K100_LOG  = ({model_config.K_100_LOG_MEAN}, {model_config.K_100_LOG_STD}), real = ({_k.mean():.4f}, {_k.std():.4f})')
       
        self.img_path = path.Path(model_config.IMAGES_PATH)

        img_model_config = timm.get_pretrained_cfg(model_name=model_config.IMG_MODEL_NAME); assert img_model_config is not None

        img_size = img_model_config.input_size[1] # CxHxW
        self.img_transforms = A.Compose([
            A.SmallestMaxSize(max_size=img_size),
            A.RandomCrop(height=img_size, width=img_size),

            # дополнительные аугментации для Эксперимента 6
            # A.Affine(scale=(0.8, 1.2),
            #             rotate=(-15, 15),
            #             translate_percent=(-0.1, 0.1),
            #             shear=(-10, 10),
            #             fill=0,
            #             p=0.5),
            # A.CoarseDropout(
            #     num_holes_range=(2, 8),
            #     hole_height_range=(int(0.07 * img_size),
            #                         int(0.15 * img_size)),
            #     hole_width_range=(int(0.1 * img_size),
            #                         int(0.15 * img_size)),
            #     fill=0,
            #     p=0.5),
            # A.ColorJitter(brightness=0.2,
            #                 contrast=0.2,
            #                 saturation=0.2,
            #                 hue=0.1,
            #                 p=0.5),

            A.Normalize(mean=img_model_config.mean, std=img_model_config.std),
            A.ToTensorV2()
        ])        

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

                - target_mass - масса блюда (как дополнительный таргет) - лог масштаб и нормализация на MEAN/STD       
                - target_kal_100 - калорий на 100г (как дополнительный таргет) - лог масштаб и нормализация на MEAN/STD
        """

        item = self.df.iloc[index]
        dish_id = item['dish_id']

        # парсим текст со списком в реальный список
        text = item['ingr_name']
        text = _INGR_RE.findall(text)
        
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
            'text': ", ".join(text),
            'image': image,
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
            - target - целевые значения калорийности блюда
            
            - target_mass - масса блюда (как дополнительный таргет)
            - target_kal_100 - калорий на 100г (как дополнительный таргет)

    
    """

    texts = [item['text'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    targets = torch.FloatTensor([item['target'] for item in batch])
    target_mass = torch.FloatTensor([item['target_mass'] for item in batch])
    target_kal_100 = torch.FloatTensor([item['target_kal_100'] for item in batch])

    tokenized = tokenizer(texts, return_tensors='pt', padding='longest')

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'images': images,
        'target': targets,
        'target_mass': target_mass,
        'target_kal_100': target_kal_100
    }
