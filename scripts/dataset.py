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
            df = df[df['split'] == split]
        except Exception as e:
            print(f'Error reading dataset from: {model_config.DATASET_PATH}, exc = {e}')
            raise RuntimeError(e)
                
        self.df = df
        self.target_mean = model_config.TARGET_MEAN
        self.target_std = model_config.TARGET_STD

        # проверить что значения не отличаются
        if split == 'train':
            print(f'expected TARGET_MEAN = {model_config.TARGET_MEAN}, real data =  {df["total_calories"].mean():.2f}')
            print(f'expected TARGET_STD  = {model_config.TARGET_STD}, real data = {df["total_calories"].std():.2f}')           

        
        self.img_path = path.Path(model_config.IMAGES_PATH)

        img_model_config = timm.get_pretrained_cfg(model_name=model_config.IMG_MODEL_NAME); assert img_model_config is not None

        img_size = img_model_config.input_size[1] # CxHxW
        self.img_transforms = A.Compose([
            A.SmallestMaxSize(max_size=img_size),
            A.RandomCrop(height=img_size, width=img_size),
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
                - target: float - целевая переменная (total_calories)            
        """

        item = self.df.iloc[index]

        dish_id = item['dish_id']
        
        text = item['ingr_name']
        text = _INGR_RE.findall(text)
        
        target = item['total_calories']
        # масштабируем
        target_norm = (target - self.target_mean)/self.target_std

        original_image = Image.open(self.img_path / dish_id / 'rgb.png')
        image = self.img_transforms(image=np.array(original_image))["image"]
        
        return {
            'text': ", ".join(text),
            'image': image,
            'target': target_norm
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
    
    """

    texts = [item['text'] for item in batch]
    images = torch.stack([item['image'] for item in batch])
    targets = torch.FloatTensor([item['target'] for item in batch])

    tokenized = tokenizer(texts, return_tensors='pt', padding='longest')

    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'images': images,
        'target': targets      
    }
