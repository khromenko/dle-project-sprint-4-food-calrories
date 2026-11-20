import matplotlib.pyplot as plt
import os, re, random
import pathlib as path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DistilBertModel, AutoConfig, DistilBertConfig
import timm
from timm.models import MobileNetV3
from functools import partial
from tqdm import tqdm
from time import time, perf_counter

# locals
import scripts.v2_attention.utils as utils
import scripts.v2_attention.config as config

class MealMode_WithAttention_NN(nn.Module):
    """
    Мультимодальная модель нейросети для распознавания калорийность блюд по текстовому составу ингредиентов и фото блюда.
    Используется кросс-модальное внимание текстовых эмбеддингов-ингредиентов и пространственных эмбеддингов картики.
    Также как отдельные признаки ингредиентов к эмбеддингам добавлена справочная калорийность блюда.
    Общая масса блюда используется как входной признак вместе с эмбеддингами ингредиентов
    Модель предсказывает:
    1) долю каждого ингредиента в блюде
    2) калорийность/100г (`k_100`) ингредиентов как корректировку справочной калорийности

    """
    def __init__(self, config: config.MealModelConfig, freeze_backbone=True):
        super().__init__()

        self.log = partial(utils.log_fn, config=config, name='model')
        self.config = config

        # 1 - backbone модель для выведения признаков изображений (model.features() вместо model())
        text_model_config: DistilBertConfig = AutoConfig.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model: MobileNetV3 = timm.create_model(config.IMG_MODEL_NAME, pretrained=True, num_classes=0)
        if freeze_backbone:
            for param in self.img_model.parameters():
                param.requires_grad = False

        # 2- проекционный слой для выравнивания размерностей признаков разных модальностей       
        # добавляем нелинейность для возможности извлекать сложные зависимости

        # эмбдеддинги ингредиентов + масса блюда
        ingr_ref_feat_dim = 1
        text_feat_dim = text_model_config.dim + ingr_ref_feat_dim

        # вход + нелинейность + нормализация + dropout
        self.ingr_proj = nn.Sequential( 
            nn.Linear(in_features=text_feat_dim, out_features=config.COMMON_FEATURES_DIM),
            nn.GELU(),
            nn.LayerNorm(config.COMMON_FEATURES_DIM),
            nn.Dropout(config.DROPOUT_FEATURES)            
        )
        self.mass_proj = nn.Sequential( 
                    nn.Linear(in_features=1, out_features=config.COMMON_FEATURES_DIM),
                    nn.GELU(),
                    nn.LayerNorm(config.COMMON_FEATURES_DIM),
                    nn.Dropout(config.DROPOUT_FEATURES)            
                )
        self.img_proj = nn.Sequential( 
            nn.Conv2d(in_channels=self.config.IMG_MODEL_FEATURE_DIM, out_channels=config.COMMON_FEATURES_DIM, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(config.COMMON_FEATURES_DIM),
            nn.Dropout2d(config.DROPOUT_FEATURES)            
        )

        # 3 - объединение признаков разных модальностей - кросс-внимание
        # Q - объединенные эмбеддинги ингредиентов и масса, K,V - эмбеддинги изображения
        num_heads = 4 
        self.attention = nn.MultiheadAttention(embed_dim=config.COMMON_FEATURES_DIM, num_heads=num_heads, batch_first=True, dropout=config.DROPOUT_FUSION)

        # 4 - нормализация входа в ffn
        self.attention_norm = nn.LayerNorm(config.COMMON_FEATURES_DIM)

        # 5 - ffn слой для обработки результатов внимания + передеча входа внимания как residual connection - расширение и сужение с нелинейностью
        self.ffn = nn.Sequential(
            nn.Linear(self.config.COMMON_FEATURES_DIM, out_features=config.COMMON_FEATURES_DIM * 2),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_FUSION),
            nn.Linear(self.config.COMMON_FEATURES_DIM * 2, out_features=config.COMMON_FEATURES_DIM)
        )
        
        # 6 - для нормализации выхода ffn + residual connection
        self.ffn_norm = nn.LayerNorm(config.COMMON_FEATURES_DIM)

        # 7 - дообучение и корректировка калорийности/100г
        # глобальная поправка справочной калорийности - обучаемый словарь
        self.k_100_vocab_delta = nn.Embedding(num_embeddings=config.INGR_DIM, embedding_dim=1, padding_idx=0)
                
        # локальная поправка по извлеченным признакам изображения
        self.k_100_vocab_img_delta = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=1)
        
        # 7 - голова для результатов предсказания доли ингредиента и калорийности/100г
        # z - доли ингредиентов
        self.regression_head_z = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=config.REGRESSION_HEAD_DIM)
        
        # Инициализируем новые слои
        self._init_custom_weights()

    def _init_custom_weights(self):
        # 1. Проекции модальностей и блок внимания (Xavier Uniform хорошо подходит под LayerNorm + GELU)
        custom_modules = [
            self.ingr_proj, 
            self.mass_proj, 
            self.img_proj, 
            self.attention,
            self.ffn,
            self.regression_head_z      
        ]
        for module in custom_modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # дельта для справочной калорийности
        nn.init.zeros_(self.k_100_vocab_delta.weight) 
        nn.init.zeros_(self.k_100_vocab_img_delta.weight) 
        nn.init.zeros_(self.k_100_vocab_img_delta.bias)        

    def forward(self, 
                ingr_embs: torch.Tensor,        # [B, S_max, emd_dim] 
                ingr_ids: torch.Tensor,         # [B, S_max]
                ingr_ref_k100: torch.Tensor,    # [B, S_max] 
                ingr_mask: torch.Tensor,        # [B, S_max]
                images: torch.Tensor,           # [B, C, H, W]
                mass: torch.Tensor):            # [B, S_max]
        """
        Выполняет расчет слоев модели
    
        Returns:
            tuple
                - total_kal      - целевая калорийность блюда (таргет), в калориях (используется для метрики)
                - total_k_100    - калорийности/100г блюда, в калориях
                - mass_i         - масса ингредиентов, г
                - (k_100_delta_global, k_100_delta_img) - поправки для справочной калорийности ингредиентов, в калориях
        """
        self.log(f'forward() ingr_embs = {ingr_embs.shape}, ingr_ids = {ingr_ids.shape}, ingr_ref_k100 = {ingr_ref_k100.shape}, ingr_mask = {ingr_mask.shape}, images = {images.shape}, mass: {mass.shape}')

        # 1 - объединяем признаки ингредиентов
        ingr_features = torch.cat([ingr_embs, ingr_ref_k100.unsqueeze(-1)], dim=-1) # [B, S_max, emd_dim] + [B, S_max, 1] = [B, S_max, emd_dim + 1]
        self.log(f'ingr_features = {ingr_features.shape}')

        # 2 - формируем признаки изображений (до GAP-слоя)
        img_features = self.img_model.forward_features(images) # [B, img_dim, H', W'], H', W' - размер финальной карты признаков
        self.log(f'img_features = {img_features.shape}')

        # 3 - проецируем признаки в единое пространство (feat_dim)
        ingr_proj = self.ingr_proj(ingr_features)           # [B, S_max, feat_dim]
        self.log(f'ingr_proj = {ingr_proj.shape}')
        
        img_proj = self.img_proj(img_features)              # [B, feat_dim, H', W']
        self.log(f'img_proj = {img_proj.shape}')
        
        # расширяем карту признаков (токенов) до плоского вида H'x W' - "попиксельно"/патчи
        img_tokens = img_proj.flatten(2)      # [B, img_dim, H' x W']
        # переворачиваем, чтобы каждый "пиксель" был представлен img_dim каналами
        img_tokens = img_tokens.transpose(1, 2) # [B, H' x W', img_dim]
        self.log(f'img_tokens tranpose = {img_tokens.shape}')

        # приводим массу к лог масштабу для стабилизации
        mass_log = torch.log1p(mass)
        mass_proj = self.mass_proj(mass_log).unsqueeze(dim=1)   # [B, feat_dim] -> [B, 1, feat_dim]
        self.log(f'mass_proj = {mass_proj.shape}')

        # 4 - attention
        # query - добавляем массу блюда к эмбеддингам ингредиентов, не меняя размерность длины последовательности S
        queries = ingr_proj + mass_proj # [B, S_max, feat_dim]
        self.log(f'query = {queries.shape}')

        attention_out, _ = self.attention(query=ingr_proj, key=img_tokens, value=img_tokens, need_weights=False) # [B, S_max, feat_dim] 
        self.log(f'attention_out = {attention_out.shape}')
        
        # 5 - ffn - обработка результатов внимания
        # передаем query признаков ингредиентов как residual connection 
        queries = attention_out + queries
        # нормализация перед входом в ffn
        queries = self.attention_norm(queries)
        ffn_out = self.ffn(queries) # [B, S_max, feat_dim]
        self.log(f'ffn_out = {ffn_out.shape}')

        # итоговые признаки - еще раз дополняем входом / residual connection - и нормализуем
        final_features = ffn_out + queries 
        final_features = self.ffn_norm(final_features) # [B, S_max, feat_dim]
        self.log(f'final_features = {final_features.shape}')

        # 6 - вычисление результатов 
        # логиты долей ингредиентов
        z_logits = self.regression_head_z(final_features).squeeze(-1)   # [B, S_max, 1] -> [B, S_max]
        self.log(f'z_logits = {z_logits.shape}')

        # убираем padded токены - заменяем paddede на -inf, 
        # т.к. на 0 нельзя заменять при использовании softmax,
        # e^0 = 1 -> приведет к увеличению знаменателя и уменьшении доли всех ингредиентов
        z_logits = z_logits.masked_fill(~ingr_mask, -float('inf')) # [B, S_max]

        # вычисляем доли через softmax
        z = nn.functional.softmax(z_logits, dim=-1)  # [B, S_max]
        self.log(f'z = {z.shape}')
        
        # 7 - обучение поправок для справочной калорийности        
        k_100_delta_1 = self.k_100_vocab_delta(ingr_ids).squeeze(-1)         # [B, S_max, 1] -> [B, S_max]
        self.log(f'k_100_delta_1 = {k_100_delta_1.shape}')

        # ограничиваем потолок поправки до 250 кал (защита от ошибок), 
        # и растягиваем зону плавного градиента внутри tanh (x/250), 
        # чтобы более плавно проходило обучение
        k_100_delta_1 = 250.0 * torch.tanh(k_100_delta_1 / 250)              # [B, S_max]
        
        k_100_delta_2 = self.k_100_vocab_img_delta(final_features).squeeze(-1)      # [B, S_max, 1] -> [B, S_max]
        self.log(f'k_100_delta_2 = {k_100_delta_2.shape}')
        
        # ограничиваем потолок поправки до 150 кал (защита от ошибок), порог поменьше
        k_100_delta_2 = 150.0 * torch.tanh(k_100_delta_2 / 150)              # [B, S_max]
        
        # восстанавливаем реальные величины
        ingr_ref_k_100_real = ingr_ref_k100 * self.config.K_100_REF_STD + self.config.K_100_REF_MEAN
        
        # защищаем от отрицательных значений через relu
        k_100_pred = nn.functional.relu(ingr_ref_k_100_real + k_100_delta_1 + k_100_delta_2)      # [B, S_max]
        
        # обнуляем padded позиции
        k_100_pred = k_100_pred * ingr_mask.float()   # [B, S_max]
        self.log(f'k_100_pred = {k_100_pred.shape}')

        # 9 - итоговое предсказание
        # доли массы ингредиентов
        mass_i = mass * z # [B, S_max]
        self.log(f'mass_i = {mass_i.shape}')

        # итоговая калорийность/100г блюда по долям ингредиентов
        total_k_100_pred = torch.sum(k_100_pred * z, dim=-1)           # [B]
        self.log(f'total_k_100_pred = {total_k_100_pred.shape}')

        # итоговая общая калорийность 
        total_kal_pred = torch.sum(k_100_pred * mass_i / 100, dim=-1)    # [B]
        self.log(f'total_kal_pred = {total_kal_pred.shape}')

        return total_kal_pred.unsqueeze(-1), total_k_100_pred.unsqueeze(-1), mass_i, (k_100_delta_1, k_100_delta_2)

def unfreeze_model_params(model: MealMode_WithAttention_NN, config: config.MealModelConfig):
    """
    Размораживает слои модели для обучения согласно их названияем из конфга
    """
    # freeze all text and img model parameters unless they have config to unfreeze
    for m, unfreeze, model_name in zip([model.img_model], [config.IMG_MODEL_UNFREEZE], [config.IMG_MODEL_NAME]):
        
        for pname, p in m.named_parameters():
            p.requires_grad_(False)            

        # unfreeze by config
        if len(unfreeze) > 0:
            pattern = unfreeze.split("|")

            for pname, p in m.named_parameters():
                if any([pname.startswith(p) for p in pattern]):
                    p.requires_grad_(True)
                    utils.log_fn(f'{model_name} - parameter id unfreezed: {pname}', config=config)

            # verification
            did_unfreeze = False
            for pname, p in m.named_parameters():
                if p.requires_grad:
                    did_unfreeze = True
            assert did_unfreeze, 'No one parameter did unfreeze for config: ' + unfreeze