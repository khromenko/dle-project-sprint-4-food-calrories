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
import torch.nn.utils.rnn as rnn

# locals
import scripts.v2_attention.config as config
import scripts.text_utils as text_utils
        
class MealDataset(Dataset):

    """
    Датасет для новой модели (v2) внимания признаков на изображение
      - собирает заранее подготовленные текстовые эмбеддинги, референсную калорийность ингредиентов 
      - готовит и аугментирует изображения
      - дополняет признаки массы
      - выводит таргет
    """

    def __init__(self, model_config: config.MealModelConfig, split='train'):
        super().__init__()

        self.config = model_config        
        try:
            df = pd.read_csv(path.Path(model_config.DATASET_PATH))

            # убираем блюда с нулевой калорийностью - это просто пустая тарелка да еще и с ошибкой в массе в датасете
            # для расчета логарифмического масштаба 0 - это запрещенное значение, и в целом для модели это не нужно.
            df = df[(df['split'] == split) & (df['total_calories'] > 0)]

            self.df = df
        except Exception as e:
            print(f'Error reading dataset from: {model_config.DATASET_PATH}, exc = {e}')
            raise e            
    
        try:
            # внешний справочник калорийности блюд для референсного значения
            self.ref_kal_mapping_df = pd.read_csv(model_config.DATASET_EXT_KAL_MAPPING)            

        except Exception as e:
            print(f'Error reading dataset from: {model_config.DATASET_EXT_KAL_MAPPING}, exc = {e}')
            raise e
            
        if split == 'train': 
            # проверить что значения распределений не отличаются от заданных в конфиге
            self.verify_mean_std(self.df, self.ref_kal_mapping_df)            

        # загрузка подготовленных эмбеддингов ингредиентов
        try:
            text_embs_loaded = torch.load(f'{self.config.TEXT_EMBED_PATH}/{self.config.TEXT_EMBED_FILE}')
            self.ingr_encoder = text_embs_loaded['ingr_encoder']
            self.all_ingr_embs = text_embs_loaded['text_embs']            
            
            assert len(self.ingr_encoder) == len(self.all_ingr_embs) and len(self.all_ingr_embs) > 0, 'Ingr embeddings size wrong'
        except Exception as e:
            print(f'Error reading ingr embeddings from: {self.config.TEXT_EMBED_PATH}, exc = {e}')
            raise e

        # images
        self.img_path = path.Path(model_config.DATASET_IMAGES_PATH)
        img_model_config = timm.get_pretrained_cfg(model_name=model_config.IMG_MODEL_NAME); assert img_model_config is not None
        img_size = img_model_config.input_size[1] # CxHxW
        
        # базовые трансформации изображения
        # приведение изображения к требованиям модели W=img_size, H=img_size
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
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(-0.1, 0.1),
                        shear=(-10, 10), fill=0, p=0.5),
            A.CoarseDropout(num_holes_range=(2, 8),
                            hole_height_range=(int(0.07 * img_size), int(0.15 * img_size)),
                            hole_width_range=(int(0.1 * img_size), int(0.15 * img_size)),
                            fill=0, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
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

        ref_k100 = ref_kal_mappping['energy_kcal']
        assert np.isclose(self.config.K_100_REF_MEAN, ref_k100.mean(), atol=.5), f'expected K_100_REF_MEAN  = {self.config.K_100_REF_MEAN}, real = {ref_k100.mean():.4f}'
        assert np.isclose(self.config.K_100_REF_STD, ref_k100.std(), atol=.5), f'expected K_100_REF_STD  = {self.config.K_100_REF_STD}, real = {ref_k100.std():.4f}'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        - текстовые эмбеддинги загружаются из предварительно подготовленного хранилища в файле `config.TEXT_EMBED_PATH`
        - картинки преобразуется с помощью набора трансформаций (размер приводится к конфигурации IMAGE_MODEL)
        
        Returns:
            dict
                - ingr_embs: torch.Tensor - тензор эмбеддингов ингредиентов [S, emb_dim]
                - ingr_ids: torch.Tensor - id ингредиентов в словаре эмбеддингов
                - ingr_ref_k100: torch.FloatTensor - тензор справочной калорийности/100г ингредиентов [S, 1], нормализованное значание на MEAN/STD
                - ingr_mask: torch.Tensor(bool) - маска ингредиентов - фиксирует сколько реальных ингредиентов в тензоре, дополненном до максимальной длины батча
                - image: torch.Tensor - трансформированное изображение
                - mass: torch.Tensor - масса блюда, г
                - target: torch.Tensor - целевая переменная (total_calories), кал
        """

        item = self.df.iloc[index]
        dish_id = item['dish_id']

        # сырой список ингредиентов
        raw_ingr_str = item['ingr_name']
        raw_ingr_list = text_utils.parse_str_to_list(raw_ingr_str)

        # чистый (отфильтрованный) список игредиентов
        ingr_list = text_utils.preprocess_ingredients(raw_ingr_str)

        # список эмбеддингов ингредиентов
        ingr_ids = [self.ingr_encoder[ingr] for ingr in ingr_list]
        ingr_embs = self.all_ingr_embs[ingr_ids]
        
        # список справочных калорийностей ингредиентов - составлен маппингом с сырыми названиями ингредиентов
        ref_k100 = []
        for raw_ingr in raw_ingr_list:
            ingr_k100 = self.ref_kal_mapping_df.query('food_ingr == @raw_ingr')['energy_kcal']
            assert len(ingr_k100) == 1, f'Ref kal product not found for {raw_ingr}'

            ref_k_100_raw = ingr_k100.iloc[0]
            ref_k_100_norm = (ref_k_100_raw - self.config.K_100_REF_MEAN) / self.config.K_100_REF_STD
            ref_k100.append(ref_k_100_norm)  

        # target
        target = item['total_calories']        
        
        # масса 
        mass = item['total_mass']
        
        # image + transform
        original_image = Image.open(self.img_path / dish_id / 'rgb.png')
        image = self.img_transforms(image=np.array(original_image))["image"]

        return {
            'ingr_embs': ingr_embs,
            'ingr_ids': torch.LongTensor(ingr_ids),
            'ingr_ref_k100': torch.FloatTensor(ref_k100),
            'ingr_mask': torch.ones(len(ingr_list), dtype=torch.bool),
            'image': image,            
            'mass': torch.FloatTensor([mass]),
            'target': torch.FloatTensor([target]),
        }
    
def collate_fn(batch):
    """
    Собирает батч из эмбеддингов ингредиентов, калорийности, массы, картинок и целевой калорийности и калорийности/100г блюда
    Выравнивает размерность длины ингредиентов S_i до S_max в батче.
    Маска `mask` задает сколько реальных ингредиентов.

    Returns:
        dict:
            - ingr_embs: torch.Tensor - эмбеддинги ингредиентов, выровненный padding-ом до максимальной длины в батче,
            - ingr_ids: torch.Tensor - id ингредиентов в словаре эмбеддингов
            - ingr_embs_attention_mask - маскм эмбеддингов ингредиентов,
            - ref_k100: torch.FloatTensor - справочная калорийности/100г ингредиентов, нормализованное
            - image: torch.Tensor - трансформированные изображения
            - mass: torch.Tensor - масса блюда, г 
            - target: torch.Tensor - целевая переменная (total_calories), кал
    """

    ingr_embs = rnn.pad_sequence([item['ingr_embs'] for item in batch], batch_first=True, padding_value=0)
    ingr_ids = rnn.pad_sequence([item['ingr_ids'] for item in batch], batch_first=True, padding_value=0)
    ingr_ref_k100 = rnn.pad_sequence([item['ingr_ref_k100'] for item in batch], batch_first=True, padding_value=0)
    ingr_mask = rnn.pad_sequence([item['ingr_mask'] for item in batch], batch_first=True, padding_value=0)
    
    image = torch.stack([item['image'] for item in batch])
    mass = torch.stack([item['mass'] for item in batch])

    target = torch.stack([item['target'] for item in batch])
    
    return {
        'ingr_embs': ingr_embs,
        'ingr_ids': ingr_ids,
        'ingr_ref_k100': ingr_ref_k100,
        'ingr_mask': ingr_mask,
        'images': image,
        'mass': mass,
        'target': target
    }
