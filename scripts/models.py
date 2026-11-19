import matplotlib.pyplot as plt
import os, re, random
import pathlib as path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DistilBertModel
import timm
from timm.models import MobileNetV3
from functools import partial
from tqdm import tqdm
from time import time, perf_counter

# locals
import scripts.utils as utils
import scripts.config as config

class MealModelNN(nn.Module):
    """
    Мультимодальная модель нейросети для распознавания калорийность блюд по текстовому составу ингредиентов и фото блюда.
    Использует дополнительно внешний справочник калорийности продуктов как дополнительные признаки (`USAD FNDDS`).
    """
    def __init__(self, config: config.MealModelConfig):
        super().__init__()

        self.log = partial(utils.log_fn, config=config)
        self.config = config

        # специализированные модели для модальностей текста и изображений
        self.text_model: DistilBertModel = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model: MobileNetV3 = timm.create_model(config.IMG_MODEL_NAME, pretrained=True, num_classes=0)

        # проекционный слой для выравнивания размерностей признаков разных модальностей       
        self.text_proj = nn.Linear(in_features=self.text_model.config.hidden_size, out_features=config.COMMON_FEATURES_DIM)
        self.img_proj = nn.Linear(in_features=self.config.IMG_MODEL_FEATURE_DIM, out_features=config.COMMON_FEATURES_DIM) # type: ignore

        # Эксп.2.1 - добавление проекции вектора справочной калорийности
        self.ref_k100_proj = nn.Linear(in_features=self.config.INGR_DIM, out_features=config.COMMON_FEATURES_DIM) # type: ignore

        # слой нормализации для проекций признаков + dropout
        self.norm_text = nn.LayerNorm(config.COMMON_FEATURES_DIM)
        self.norm_img = nn.LayerNorm(config.COMMON_FEATURES_DIM)
        self.norm_ref_k100 = nn.LayerNorm(config.COMMON_FEATURES_DIM)

        self.dropout = nn.Dropout(config.DROPOUT_FEATURES)

        # объединение признаков разных модальностей 
        # вар. 1 - модель перекрестного-внимания 
        # Эксперимент 0 - не сработало - на одном текстовом токене [CLS] - Softmax внутри Attention дает 1 и текстовые признаки вообще не учитываются
        # self.attention = nn.MultiheadAttention(embed_dim=config.COMMON_FEATURES_DIM, num_heads=4, batch_first=True)
        
        # TODO - стоит пересмотреть механизм внимания, чтобы опрашивать картику по каждому токену-ингредиенту а не по сводному токену [CLS] как было изначально
        # Exp 2.2 - новый механизм внимания
        # self.attention = nn.MultiheadAttention(embed_dim=config.COMMON_FEATURES_DIM, num_heads=4, batch_first=True)

        # вар. 2 - слой конкатенации
        # Эксперимент 1
        self.fusion = nn.Sequential(
            nn.Linear(3 * config.COMMON_FEATURES_DIM, config.COMMON_FEATURES_DIM), # x3 - добавлен эмбеддинг справочной калорийности
            nn.LayerNorm(config.COMMON_FEATURES_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_FUSION)
        )
      
        # Эксперимент 4 - разделяем таргет на две головы
        # Эксперимент 9 - разделяем таргет на 3 головы головы - 1 основная (прямой таргет как в Эксп.4) + 2 вспомогательные
        # регрессионный выход 1 - k_100
        self.regression_head_k_100 = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=config.REGRESSION_HEAD_DIM)
        # регрессионный выход 2 - mass
        self.regression_head_mass = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=config.REGRESSION_HEAD_DIM)    
        # Эксперимент 9: прямой выход (голова) калорийности:
        #   - учится по target как в Эксп.4 до разделения на 2 головы - устраняет накопление/рассинхронизацию ошибок двух голов
        #   - выход через произведение k100 * mass - оставлен как дополнительный  
        self.regression_head_kal = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=config.REGRESSION_HEAD_DIM) 

        # Инициализируем только новые слои, не трогая backbone-модели
        self._init_custom_weights()

    def _init_custom_weights(self):
        # 1. Проекции модальностей и блок слияния (Xavier Uniform хорошо подходит под LayerNorm + GELU)
        custom_modules = [
            self.text_proj, 
            self.img_proj, 
            self.ref_k100_proj, 
            self.fusion
        ]
        for module in custom_modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # 2. Регрессионные головы: делаем начальные веса очень маленькими.
        # Так как таргеты нормализованы (mean=0), старт с предсказаний около нуля
        # предотвратит скачки значений после операции torch.exp() на первых батчах.
        heads = [
            self.regression_head_k_100,
            self.regression_head_mass,
            self.regression_head_kal
        ]
        for head in heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor, ref_k100: torch.Tensor): # type: ignore
        """
        Выполняет расчет слоев модели

        Returns:
            (kal_ensemble_norm, kal_direct_norm, kal_product_norm, kal_100_lognorm, mass_lognorm) (tuple):
                - kal_ensemble_norm  - ensemble = KAL_DIRECT_WEIGHT*direct + (1-KAL_DIRECT_WEIGHT)*product, нормализованное значение на TARGET_MEAN/STD (используется для метрики)
                - kal_direct_norm    - предсказание калорийности прямым head'ом, нормализованное
                - kal_product_norm   - предсказание калорийности через k100*mass/100, нормализованное
                - kal_100_lognorm    - предсказанный результат калорийности/100г, лог масштаб и нормализация на MEAN/STD
                - mass_lognorm       - предсказанный результат массы блюда, лог масштаб и нормализация на MEAN/STD
        """
        #self.log(f'forward() input_ids = {input_ids.shape}, images = {images.shape}')

        # 1 - text features
        if self.config.STUB_TEXT:
            # stub: constant [CLS] features, batch-dependent, no text model forward
            hidden_size = self.text_model.config.hidden_size
            text_features = torch.ones(
                input_ids.shape[0], hidden_size,
                device=input_ids.device, dtype=self.text_proj.weight.dtype
            )
        else:           
            text_out = self.text_model(input_ids, attention_mask).last_hidden_state      
            text_features = text_out[:, 0, :] # берем состояние 1-ого [CLS] токена          

        # 2 - image features
        img_out = self.img_model(images)
        img_features = img_out

        # 3 - features projection
        text_proj = self.text_proj(text_features)
        img_proj = self.img_proj(img_features)
        # Exp.2.1 - reference kal features
        ref_k100_proj = self.ref_k100_proj(ref_k100)
                
        # 4 - normalization + dropout
        text_proj = self.dropout(self.norm_text(text_proj))
        img_proj = self.dropout(self.norm_img(img_proj))
        ref_k100_proj = self.dropout(self.norm_ref_k100(ref_k100_proj))
       
        # 5 - fusion
        if self.config.STUB_TEXT:
            # stub test with images twice to keep settings unchanged
            fusion_in = torch.concat([img_proj, img_proj], dim=1)
        else:
            fusion_in = torch.concat([text_proj, img_proj, ref_k100_proj], dim=1)

        fusion_out = self.fusion(fusion_in)

        # 6.1 - regression head - kal_100
        k100_lognorm = self.regression_head_k_100(fusion_out).squeeze(-1) # [B, 1] → [B]
        # 6.2 - regression head - mass
        mass_lognorm = self.regression_head_mass(fusion_out).squeeze(-1) 
        # 6.3 - regression head - калорийность напрямую (в нормализованном пространстве target)
        kal_direct_norm = self.regression_head_kal(fusion_out).squeeze(-1)
        
        # 7 - reconstruct to real units (from log -> norm)
        pred_mass = torch.exp(mass_lognorm * self.config.MASS_LOG_STD + self.config.MASS_LOG_MEAN)
        pred_k100 = torch.exp(k100_lognorm * self.config.K_100_LOG_STD + self.config.K_100_LOG_MEAN)
        pred_kal = pred_k100 * pred_mass / 100.0

        kal_product_norm = (pred_kal - self.config.TARGET_MEAN)/self.config.TARGET_STD

        # 8 - Эксперимент 9: ensemble прямой + разделенный (через произведение/product) путь
        kal_ensemble_norm = (
            self.config.KAL_DIRECT_WEIGHT * kal_direct_norm + (1.0 - self.config.KAL_DIRECT_WEIGHT) * kal_product_norm
        )

        return kal_ensemble_norm, kal_direct_norm, kal_product_norm, k100_lognorm, mass_lognorm        

class MealMode_WithAttention_NN(nn.Module):
    """
    Мультимодальная модель нейросети для распознавания калорийность блюд по текстовому составу ингредиентов и фото блюда.
    Используется кросс-модальное внимание текстовых токенов-ингредиентов и пространственных эмбеддингов картики
    """
    def __init__(self, config: config.MealModelConfig):
        super().__init__()

        self.log = partial(utils.log_fn, config=config)
        self.config = config

        # специализированные модели для модальностей текста и изображений
        self.text_model: DistilBertModel = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model: MobileNetV3 = timm.create_model(config.IMG_MODEL_NAME, pretrained=True, num_classes=0)

        # проекционный слой для выравнивания размерностей признаков разных модальностей       
        self.text_proj = nn.Linear(in_features=self.text_model.config.hidden_size, out_features=config.COMMON_FEATURES_DIM)
        self.img_proj = nn.Linear(in_features=self.config.IMG_MODEL_FEATURE_DIM, out_features=config.COMMON_FEATURES_DIM) # type: ignore

        # Эксп.2.1 - добавление проекции вектора справочной калорийности
        self.ref_k100_proj = nn.Linear(in_features=self.config.INGR_DIM, out_features=config.COMMON_FEATURES_DIM) # type: ignore

        # слой нормализации для проекций признаков + dropout
        self.norm_text = nn.LayerNorm(config.COMMON_FEATURES_DIM)
        self.norm_img = nn.LayerNorm(config.COMMON_FEATURES_DIM)
        self.norm_ref_k100 = nn.LayerNorm(config.COMMON_FEATURES_DIM)

        self.dropout = nn.Dropout(config.DROPOUT_FEATURES)

        # объединение признаков разных модальностей 
        # вар. 1 - модель перекрестного-внимания 
        # Эксперимент 0 - не сработало - на одном текстовом токене [CLS] - Softmax внутри Attention дает 1 и текстовые признаки вообще не учитываются
        # self.attention = nn.MultiheadAttention(embed_dim=config.COMMON_FEATURES_DIM, num_heads=4, batch_first=True)
        
        # TODO - стоит пересмотреть механизм внимания, чтобы опрашивать картику по каждому токену-ингредиенту а не по сводному токену [CLS] как было изначально
        # Exp 2.2 - новый механизм внимания
        # self.attention = nn.MultiheadAttention(embed_dim=config.COMMON_FEATURES_DIM, num_heads=4, batch_first=True)

        # вар. 2 - слой конкатенации
        # Эксперимент 1
        self.fusion = nn.Sequential(
            nn.Linear(3 * config.COMMON_FEATURES_DIM, config.COMMON_FEATURES_DIM), # x3 - добавлен эмбеддинг справочной калорийности
            nn.LayerNorm(config.COMMON_FEATURES_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_FUSION)
        )
      
        # Эксперимент 4 - разделяем таргет на две головы
        # Эксперимент 9 - разделяем таргет на 3 головы головы - 1 основная (прямой таргет как в Эксп.4) + 2 вспомогательные
        # регрессионный выход 1 - k_100
        self.regression_head_k_100 = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=config.REGRESSION_HEAD_DIM)
        # регрессионный выход 2 - mass
        self.regression_head_mass = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=config.REGRESSION_HEAD_DIM)    
        # Эксперимент 9: прямой выход (голова) калорийности:
        #   - учится по target как в Эксп.4 до разделения на 2 головы - устраняет накопление/рассинхронизацию ошибок двух голов
        #   - выход через произведение k100 * mass - оставлен как дополнительный  
        self.regression_head_kal = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=config.REGRESSION_HEAD_DIM) 

        # Инициализируем только новые слои, не трогая backbone-модели
        self._init_custom_weights()

    def _init_custom_weights(self):
        # 1. Проекции модальностей и блок слияния (Xavier Uniform хорошо подходит под LayerNorm + GELU)
        custom_modules = [
            self.text_proj, 
            self.img_proj, 
            self.ref_k100_proj, 
            self.fusion
        ]
        for module in custom_modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # 2. Регрессионные головы: делаем начальные веса очень маленькими.
        # Так как таргеты нормализованы (mean=0), старт с предсказаний около нуля
        # предотвратит скачки значений после операции torch.exp() на первых батчах.
        heads = [
            self.regression_head_k_100,
            self.regression_head_mass,
            self.regression_head_kal
        ]
        for head in heads:
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor, ref_k100: torch.Tensor): # type: ignore
        """
        Выполняет расчет слоев модели

        Returns:
            (kal_ensemble_norm, kal_direct_norm, kal_product_norm, kal_100_lognorm, mass_lognorm) (tuple):
                - kal_ensemble_norm  - ensemble = KAL_DIRECT_WEIGHT*direct + (1-KAL_DIRECT_WEIGHT)*product, нормализованное значение на TARGET_MEAN/STD (используется для метрики)
                - kal_direct_norm    - предсказание калорийности прямым head'ом, нормализованное
                - kal_product_norm   - предсказание калорийности через k100*mass/100, нормализованное
                - kal_100_lognorm    - предсказанный результат калорийности/100г, лог масштаб и нормализация на MEAN/STD
                - mass_lognorm       - предсказанный результат массы блюда, лог масштаб и нормализация на MEAN/STD
        """
        #self.log(f'forward() input_ids = {input_ids.shape}, images = {images.shape}')

        # 1 - text features
        if self.config.STUB_TEXT:
            # stub: constant [CLS] features, batch-dependent, no text model forward
            hidden_size = self.text_model.config.hidden_size
            text_features = torch.ones(
                input_ids.shape[0], hidden_size,
                device=input_ids.device, dtype=self.text_proj.weight.dtype
            )
        else:           
            text_out = self.text_model(input_ids, attention_mask).last_hidden_state      
            text_features = text_out[:, 0, :] # берем состояние 1-ого [CLS] токена          

        # 2 - image features
        img_out = self.img_model(images)
        img_features = img_out

        # 3 - features projection
        text_proj = self.text_proj(text_features)
        img_proj = self.img_proj(img_features)
        # Exp.2.1 - reference kal features
        ref_k100_proj = self.ref_k100_proj(ref_k100)
                
        # 4 - normalization + dropout
        text_proj = self.dropout(self.norm_text(text_proj))
        img_proj = self.dropout(self.norm_img(img_proj))
        ref_k100_proj = self.dropout(self.norm_ref_k100(ref_k100_proj))
       
        # 5 - fusion
        if self.config.STUB_TEXT:
            # stub test with images twice to keep settings unchanged
            fusion_in = torch.concat([img_proj, img_proj], dim=1)
        else:
            fusion_in = torch.concat([text_proj, img_proj, ref_k100_proj], dim=1)

        fusion_out = self.fusion(fusion_in)

        # 6.1 - regression head - kal_100
        k100_lognorm = self.regression_head_k_100(fusion_out).squeeze(-1) # [B, 1] → [B]
        # 6.2 - regression head - mass
        mass_lognorm = self.regression_head_mass(fusion_out).squeeze(-1) 
        # 6.3 - regression head - калорийность напрямую (в нормализованном пространстве target)
        kal_direct_norm = self.regression_head_kal(fusion_out).squeeze(-1)
        
        # 7 - reconstruct to real units (from log -> norm)
        pred_mass = torch.exp(mass_lognorm * self.config.MASS_LOG_STD + self.config.MASS_LOG_MEAN)
        pred_k100 = torch.exp(k100_lognorm * self.config.K_100_LOG_STD + self.config.K_100_LOG_MEAN)
        pred_kal = pred_k100 * pred_mass / 100.0

        kal_product_norm = (pred_kal - self.config.TARGET_MEAN)/self.config.TARGET_STD

        # 8 - Эксперимент 9: ensemble прямой + разделенный (через произведение/product) путь
        kal_ensemble_norm = (
            self.config.KAL_DIRECT_WEIGHT * kal_direct_norm + (1.0 - self.config.KAL_DIRECT_WEIGHT) * kal_product_norm
        )

        return kal_ensemble_norm, kal_direct_norm, kal_product_norm, k100_lognorm, mass_lognorm        

