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


def set_random(seed=43):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


class MealModelConfig:

    SEED = 43
    DATASET_PATH_ORIGINGAL = 'data/meal_combined.csv'
    DATASET_PATH_NO_OUTLIERS = 'data/meal_combined_cleaned.csv'
    DATASET_PATH = DATASET_PATH_NO_OUTLIERS
    IMAGES_PATH = 'data/images'

    # параметры средних и отклонений train датасета
    #TARGET_MEAN = 255
    #TARGET_STD = 221
    # удаление выброса - Эксперимент 6 (не было исправлено) + очищенный датасет
    # Эксперимент 12 - подтвердил 
    # + Эксперимент 13 фиксируем как base с конфигами Эксперимента 9
    TARGET_MEAN = 254
    TARGET_STD = 209

    # для 2-х задачной модели - масса и кал/100г
    # активируем удаление выброса - эксперимент 6.
    # MASS_LOG_MEAN = 5.0932
    # MASS_LOG_STD  = 0.8334
    # K_100_LOG_MEAN = 4.5942
    # K_100_LOG_STD  = 0.7317

    # удаление выброса - Эксперимент 6 + очищенный датасет
    MASS_LOG_MEAN = 5.0921
    MASS_LOG_STD  = 0.8317
    K_100_LOG_MEAN = 4.5941
    K_100_LOG_STD  = 0.7318
    ###

    # models config
    TEXT_MODEL_NAME = 'distilbert-base-uncased'
    TEXT_MODEL_UNFREEZE = "transformer.layer.5" # выбрать последние 1-2 слоя при необходимости
    #TEXT_MODEL_UNFREEZE = "" # to stub text

    IMG_MODEL_NAME = 'mobilenetv4_conv_aa_large' #.e230_r448_in12k_ft_in1k'
    IMG_MODEL_UNFREEZE = "blocks.4|conv_head|norm_head" # выбрать последние 1-2 слоя при необходимости

    IMG_MODEL_FEATURE_DIM = 1280 # img_model.num_features lies telling it is 960
    COMMON_FEATURES_DIM = 256
    REGRESSION_HEAD_DIM = 1

    # Эксперименит 9 - усиленная регуляризация (было 0.2/0.2) — борьба с переобучением после эп. 21
    DROPOUT_FEATURES = 0.3
    DROPOUT_FUSION = 0.3

    # train config
    N_BATCH = 32
    N_EPOCHS = 30 # 10 for CPU (2.5 hours)
    PATIENCE = 6 # ранняя остановка обучения, если метрики не растут
    N_ITER = -1 #32 # для отладки, -1 без ограничений
    
    # optimizer
    LR_TEXT_MODEL = 1e-5
    LR_IMG_MODEL = 1e-5
    LR_PROJ = 1e-4
    LR_FUSION = 1e-4
    LR_REGRESSION_HEAD = 1e-4

    # Эксперимент 9: усиливаем L2-регуляризация (weight decay) - по умолчанию 0.01 — добавляется во все группы AdamW
    WEIGHT_DECAY = 1e-4

    # Эксперимент 8 - больше веса ошибки 2-ой головы - по массе (0.3 -> 0.5)
    # Эксперимент 10 - меньше веса ошибки 2-ой головы - по массе (0.5 -> 0.4)
    #LOSS_MASS_WEIGHT = 0.4
    # Эксперимент 13 -> возврат к Эксп. 9 - 0.4 -> 0.5
    LOSS_MASS_WEIGHT = 0.5

    # Эксперимент 9: вес ошибки k_100 головы (для 3-головой схемы: direct + mass + k100)
    LOSS_K_100_WEIGHT = 0.3
    
    # Эксперимент 9: доля прямого head в финальном предсказании калорийности (ensemble)
    # kal = KAL_DIRECT_WEIGHT * direct + (1 - KAL_DIRECT_WEIGHT) * (k100 * mass / 100)
    # Эксперимент 10 - увеличиваем долю прямой головы до 0.5 -> 0.65
    # KAL_DIRECT_WEIGHT = 0.65
    # Эксперимент 13 -> возврат к Эксп. 9 - 0.65 -> 0.5
    KAL_DIRECT_WEIGHT = 0.5

    # Эксперимент 12 - снижаем с 1.0 до 0.5 для kal_direct - чтобы большие ошибки меньше оттягивали на себя (ушли из MSE зоны)
    HUBER_DELTA_KAL_DIRECT = 0.5 # 0.5 * TARGET_STD ≈ 110 kcal — порог MSE/L1 для калорий
    # Эксперимент 13 - возврат к 1 как в Эксперимент 9
    HUBER_DELTA_KAL_DIRECT = 1.0


    TARGET_METRIC = 50

    MODEL_SAVE_PATH = 'data/model'
    MODEL_SAVE_NAME = 'best_model.pth'
    MODEL_LOG_SAVE_PATH = 'data/model'

    # для тестирования отдельной визуальной модели - выключение текстовой модели (заглушка)
    STUB_TEXT = False

    # отладочная печать
    DEBUG = False

def log_fn(debug_msg: str, config):
    if config and config.DEBUG:
        print(debug_msg)

class MealModelNN(nn.Module):
    """
    Мультимодальная модель нейросети для распознавания калорийность блюд по текстовому составу ингредиентов и фото блюда    
    """

    def __init__(self, config: MealModelConfig):
        super().__init__()

        self.log = partial(log_fn, config=config)
        self.config = config

        # специализированные модели для модальностей текста и изображений
        self.text_model: DistilBertModel = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.img_model: MobileNetV3 = timm.create_model(config.IMG_MODEL_NAME, pretrained=True, num_classes=0)

        # проекционный слой для выравнивания размерностей признаков разных модальностей       
        self.text_proj = nn.Linear(in_features=self.text_model.config.hidden_size, out_features=config.COMMON_FEATURES_DIM)
        self.img_proj = nn.Linear(in_features=self.config.IMG_MODEL_FEATURE_DIM, out_features=config.COMMON_FEATURES_DIM) # type: ignore

        # слой нормализации для проекций признаков + dropout
        self.norm_layer = nn.LayerNorm(config.COMMON_FEATURES_DIM)
        self.dropout = nn.Dropout(config.DROPOUT_FEATURES)

        # объединение признаков разных модальностей 
        # вар. 1 - модель перекрестного-внимания 
        # Эксперимент 0 - не сработало - на одном текстовом токене [CLS] - Softmax внутри Attention дает 1 и текстовые признаки вообще не учитываются
        # self.attention = nn.MultiheadAttention(embed_dim=config.COMMON_FEATURES_DIM, num_heads=4, batch_first=True)
        
        # вар. 2 - слой конкатенации
        # Эксперимент 1
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.COMMON_FEATURES_DIM, config.COMMON_FEATURES_DIM),
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

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor): # type: ignore
        """
        Выполняет расчет слоев модели

        Returns:
            tuple(kal_ensemble_norm, kal_direct_norm, kal_product_norm, kal_100_lognorm, mass_lognorm)
                - kal_ensemble_norm  - ensemble = KAL_DIRECT_WEIGHT*direct + (1-KAL_DIRECT_WEIGHT)*product,
                                       нормализованное значение на TARGET_MEAN/STD (используется для метрики)
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
        
        # 4 - normalization + dropout
        text_proj = self.dropout(self.norm_layer(text_proj))
        img_proj = self.dropout(self.norm_layer(img_proj))
        
        # 5 - fusion
        if self.config.STUB_TEXT:
            # stub test with images twice to keep settings unchanged
            fusion_in = torch.concat([img_proj, img_proj], dim=1)
        else:
            fusion_in = torch.concat([text_proj, img_proj], dim=1)

        fusion_out = self.fusion(fusion_in)

        # 6.1 - regression head - kal_100
        kal_100_lognorm = self.regression_head_k_100(fusion_out).squeeze(-1) # [B, 1] → [B]
        # 6.2 - regression head - mass
        mass_lognorm = self.regression_head_mass(fusion_out).squeeze(-1) 
        # 6.3 - regression head - калорийность напрямую (в нормализованном пространстве target)
        kal_direct_norm = self.regression_head_kal(fusion_out).squeeze(-1)

        # 7 - reconstruct to real units (from log -> norm)
        pred_mass_log = mass_lognorm * self.config.MASS_LOG_STD + self.config.MASS_LOG_MEAN
        pred_mass = torch.exp(pred_mass_log)      
        pred_kal_100_log = kal_100_lognorm * self.config.K_100_LOG_STD + self.config.K_100_LOG_MEAN
        pred_kal_100 = torch.exp(pred_kal_100_log)   
        pred_kal = pred_kal_100 * pred_mass / 100.0

        kal_product_norm = (pred_kal - self.config.TARGET_MEAN)/self.config.TARGET_STD

        # 8 - Эксперимент 9: ensemble прямой + разделенный (через произведение/product) путь
        kal_ensemble_norm = (
            self.config.KAL_DIRECT_WEIGHT * kal_direct_norm + (1.0 - self.config.KAL_DIRECT_WEIGHT) * kal_product_norm
        )

        return kal_ensemble_norm, kal_direct_norm, kal_product_norm, kal_100_lognorm, mass_lognorm        

def unfreeze_model_params(model: MealModelNN, config: MealModelConfig):
    """
    Размораживает слои модели для обучения согласно их названияем из конфга
    """

    # freeze all text and img model parameters unless they have config to unfreeze
    for m, unfreeze, model_name in zip([model.text_model, model.img_model], [config.TEXT_MODEL_UNFREEZE, config.IMG_MODEL_UNFREEZE], [config.TEXT_MODEL_NAME, config.IMG_MODEL_NAME]):
        
        for pname, p in m.named_parameters():
            p.requires_grad_(False)            

        # unfreeze by config
        if len(unfreeze) > 0:
            pattern = unfreeze.split("|")

            for pname, p in m.named_parameters():
                if any([pname.startswith(p) for p in pattern]):
                    p.requires_grad_(True)
                    log_fn(f'{model_name} - parameter id unfreezed: {pname}', config=config)

            # verification
            did_unfreeze = False
            for pname, p in m.named_parameters():
                if p.requires_grad:
                    did_unfreeze = True
            assert did_unfreeze, 'No one parameter did unfreeze for config: ' + unfreeze

def optimizer_params_sanity_check(model, optimizer):
    """
    Проверка - все параметры модели с requires_grad должны быть добавлены в оптимизаторе
    """

    optimizer_params = {id(p) for g in optimizer.param_groups for p in g['params']}
    missing = [
        n for n, p in model.named_parameters()
        if p.requires_grad and id(p) not in optimizer_params
    ]
    assert not missing, f"Параметры без оптимизатора: {missing}"

def use_device():
    """
    Определяет возможность использовать GPU или CPU устройство
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

def mae_metric(pred: torch.Tensor, real: torch.Tensor):
    """
    Функция ошибки MAE
    """
    return torch.abs(pred-real).mean()

def huber_metric(pred: torch.Tensor, real: torch.Tensor, delta=1.0):
    """
    Функция ошибки Huber - сочетание MSE (для ошибки < delta) и MAE (для ошибки > delta)
    """
    return nn.functional.huber_loss(pred, real, reduction='mean', delta=delta)

def train(model: MealModelNN, config: MealModelConfig, optimizer: torch.optim.AdamW, dataloader: DataLoader, scheduler = None):
    log = partial(log_fn, config=config)
    log('   --- train()')
    
    # перенос на устройство
    device = use_device()
    model.to(device)
    model.train()

    i = 0
    running_loss = 0
    running_mae = 0

    running_kal_100_mae = 0
    running_mass_mae = 0
    running_direct_mae = 0

    start = perf_counter()
    for data in tqdm(dataloader, 'training'):
        i += 1

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        images = data['images'].to(device)
        target = data['target'].to(device)
        target_kal_100 = data['target_kal_100'].to(device)
        target_mass = data['target_mass'].to(device)
        
        optimizer.zero_grad()

        # forward возвращает 5 значений: ensemble, direct, product, k100, mass
        result_kal, result_kal_direct, result_kal_product, result_k_100, result_mass = model(input_ids, attention_mask, images)
                
        # loss составной: прямой kal (вес 1.0) + mass + k100
        # Эксперимент 12 - delta = 0.5 (default = 1)
        loss_huber_kal_direct = huber_metric(result_kal_direct, target, delta=config.HUBER_DELTA_KAL_DIRECT)
        loss_huber_kal_100 = huber_metric(result_k_100, target_kal_100)
        loss_huber_mass = huber_metric(result_mass, target_mass)

        loss = (loss_huber_kal_direct
            + config.LOSS_MASS_WEIGHT * loss_huber_mass
            + config.LOSS_K_100_WEIGHT * loss_huber_kal_100
        )
        loss.backward()
        
        # clip grad norm
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=2.0
        )

        optimizer.step()

        # warm-up learning_rate
        if scheduler is not None:
            scheduler.step()

        # метрики MAE
        with torch.no_grad():            
            mass_real = torch.exp(target_mass * model.config.MASS_LOG_STD + model.config.MASS_LOG_MEAN)
            mass_pred = torch.exp(result_mass * model.config.MASS_LOG_STD + model.config.MASS_LOG_MEAN)
            mass_mae = mae_metric(mass_pred, mass_real)

            kal_100_real = torch.exp(target_kal_100 * model.config.K_100_LOG_STD + model.config.K_100_LOG_MEAN)
            kal_100_pred = torch.exp(result_k_100 * model.config.K_100_LOG_STD + model.config.K_100_LOG_MEAN)
            kal_100_mae = mae_metric(kal_100_pred, kal_100_real)

            kal_pred = result_kal * model.config.TARGET_STD + model.config.TARGET_MEAN
            kal_real = target * model.config.TARGET_STD + model.config.TARGET_MEAN
            kal_mae = mae_metric(kal_pred, kal_real)

            # Эксперимент 9: раздельные MAE по частям (для диагностики)
            kal_direct_pred = result_kal_direct * model.config.TARGET_STD + model.config.TARGET_MEAN
            kal_direct_mae = mae_metric(kal_direct_pred, kal_real)
            kal_product_pred = result_kal_product * model.config.TARGET_STD + model.config.TARGET_MEAN
            kal_product_mae = mae_metric(kal_product_pred, kal_real)

        running_loss += loss.item()
        running_mae += kal_mae.item()

        running_kal_100_mae += kal_100_mae.item()
        running_mass_mae += mass_mae.item()
        running_direct_mae += kal_direct_mae.item()

        log(f'loss = {loss.item():.4f}, \
            loss_huber_kal_direct = {loss_huber_kal_direct.item():.4f}, \
            loss_huber_kal_100 = {loss_huber_kal_100.item():.4f}, \
            loss_huber_mass = {loss_huber_mass.item():.4f}')

        log(f'mae = {kal_mae:.4f}, \
              kal_100_mae = {kal_100_mae:.4f},\
              mass_mae = {mass_mae:.4f}')

        if i % 50 == 0:
            log(f'running_loss = {running_loss/i:.4f}, \
                running_mae = {running_mae/i:.4f}, \
                running_kal_100_mae = {running_kal_100_mae/i:.4f},\
                running_mass_mae = {running_mass_mae/i:.4f}')
        
        if i == config.N_ITER:
            print(f'debug early stop - i = {i}')
            break

    train_huber_loss = running_loss / i
    train_mae_loss = running_mae / i
    train_metric_kal_100_mae = running_kal_100_mae / i
    train_metric_mass_mae = running_mass_mae / i
    train_metric_direct_mae = running_direct_mae / i

    return train_huber_loss, train_mae_loss, train_metric_kal_100_mae, train_metric_mass_mae, train_metric_direct_mae, perf_counter() - start

def validate(model: MealModelNN, config: MealModelConfig, dataloader: DataLoader):
    log = partial(log_fn, config=config)
    log('   -- validate()')

    device = use_device()

    start = perf_counter()
    model.eval()
    i = 0
    running_huber = 0
    running_mae = 0
    running_kal_100_mae = 0
    running_mass_mae = 0
    running_direct_mae = 0
    for data in tqdm(dataloader, 'validation'):
        i += 1

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        images = data['images'].to(device)
        target = data['target'].to(device)
        target_kal_100 = data['target_kal_100'].to(device)
        target_mass = data['target_mass'].to(device)

        with torch.no_grad():        
            # forward возвращает 5 значений: ensemble, direct, product, k100, mass
            result_kal, result_kal_direct, result_kal_product, result_k_100, result_mass = model(input_ids, attention_mask, images)

            # loss huber (direct + mass + k100)
            # Эксперимент 12 - delta = 0.5 (default = 1)
            loss_huber_kal_direct = huber_metric(result_kal_direct, target, delta=config.HUBER_DELTA_KAL_DIRECT)
            loss_huber_kal_100 = huber_metric(result_k_100, target_kal_100)
            loss_huber_mass = huber_metric(result_mass, target_mass)
            loss_huber = (
                loss_huber_kal_direct
                + config.LOSS_MASS_WEIGHT * loss_huber_mass
                + config.LOSS_K_100_WEIGHT * loss_huber_kal_100
            )

            # mae
            mass_real = torch.exp(target_mass * model.config.MASS_LOG_STD + model.config.MASS_LOG_MEAN)
            mass_pred = torch.exp(result_mass * model.config.MASS_LOG_STD + model.config.MASS_LOG_MEAN)
            mass_mae = mae_metric(mass_pred, mass_real)

            kal_100_real = torch.exp(target_kal_100 * model.config.K_100_LOG_STD + model.config.K_100_LOG_MEAN)
            kal_100_pred = torch.exp(result_k_100 * model.config.K_100_LOG_STD + model.config.K_100_LOG_MEAN)
            kal_100_mae = mae_metric(kal_100_pred, kal_100_real)

            kal_pred = result_kal * model.config.TARGET_STD + model.config.TARGET_MEAN
            kal_real = target * model.config.TARGET_STD + model.config.TARGET_MEAN
            kal_mae = mae_metric(kal_pred, kal_real)

            # Эксперимент 9: раздельные MAE по каналам (diagnostics)
            kal_direct_pred = result_kal_direct * model.config.TARGET_STD + model.config.TARGET_MEAN
            kal_direct_mae = mae_metric(kal_direct_pred, kal_real)
            kal_product_pred = result_kal_product * model.config.TARGET_STD + model.config.TARGET_MEAN
            kal_product_mae = mae_metric(kal_product_pred, kal_real)

            log(f'result_kal = {result_kal}')
            log(f'target = {target}')
            
            log(f'result_k_100 = {result_k_100}')
            log(f'target_kal_100 = {target_kal_100}')
            
            log(f'result_mass = {result_mass}')
            log(f'target_mass = {target_mass}')
            
            log(f'huber loss = {loss_huber.item():.4f}, \
                loss_huber_kal_direct: {loss_huber_kal_direct.item():.4f}, \
                loss_huber_kal_100: {loss_huber_kal_100.item():.4f}, \
                loss_huber_mass = {loss_huber_mass.item():.4f}')
            
            log(f'mae metric = {kal_mae.item():.4f}, \
                kal_direct_mae = {kal_direct_mae.item():.4f}, \
                kal_product_mae = {kal_product_mae.item():.4f}, \
                kal_100_mae = {kal_100_mae.item():.4f},\
                mass_mae = {mass_mae.item():.4f}')
            
            running_huber += loss_huber.item()
            running_mae += kal_mae.item()
            running_kal_100_mae += kal_100_mae.item()
            running_mass_mae += mass_mae.item()
            running_direct_mae += kal_direct_mae.item()

        if i == config.N_ITER:
            log(f'debug early stop - i = {i}')
            break

    val_loss_huber = running_huber/i
    val_metric_mae = running_mae/i
    val_metric_kal_100_mae = running_kal_100_mae / i
    val_metric_mass_mae = running_mass_mae / i
    val_metric_direct_mae = running_direct_mae / i

    return val_loss_huber, val_metric_mae, val_metric_kal_100_mae, val_metric_mass_mae, val_metric_direct_mae, perf_counter() - start

def save_model(model: MealModelNN, config: MealModelConfig):
    """
    Сохранение весов модели (checkpoint)
    """
    save_path = path.Path(config.MODEL_SAVE_PATH)
    save_path.mkdir(exist_ok=True)
    file_path = save_path / config.MODEL_SAVE_NAME
    
    print(f'model saving to - [{file_path}]')
    torch.save(model.state_dict(), file_path)
def save_checkpoint(model: MealModelNN, config: MealModelConfig, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR, current_epoch, best_epoch, best_metric):
    """
    Сохранение checkpoint обучения
    """

    save_path = path.Path(config.MODEL_SAVE_PATH)
    save_path.mkdir(exist_ok=True)
    file_path = save_path / config.MODEL_SAVE_NAME
    
    print(f'checkpoint saving to - [{file_path}]')
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': current_epoch,
        'best_epoch': best_epoch,
        'best_metric': best_metric
    }
    torch.save(checkpoint, file_path)    

def restore_checkpoint(model: MealModelNN, config: MealModelConfig, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR):
    model_path = path.Path(config.MODEL_SAVE_PATH) / config.MODEL_SAVE_NAME
    ckpt = torch.load(model_path, weights_only=False)

    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch   = ckpt.get('epoch', 0) + 1
    best_metric   = ckpt.get('best_metric', 1000)
    best_epoch    = ckpt.get('best_epoch', -1)

    return start_epoch, best_epoch, best_metric


def plof_train_log(train_loss_history, val_loss_history, train_mae_history, val_mae_history, train_time_history, val_time_history):
    fix, axs = plt.subplots(1,3, figsize=(12,5))
    ax=axs[0]
    ax.plot(train_loss_history)
    ax.plot(val_loss_history)
    ax.grid();ax.set_xlabel('epoch');ax.set_ylabel('Huber loss');ax.legend(['train_loss', 'val_loss'])
    ax.set_title('Loss')

    ax=axs[1]
    ax.plot(train_mae_history)
    ax.plot(val_mae_history)
    ax.grid();ax.set_xlabel('epoch');ax.set_ylabel('MAE');ax.legend(['train_mae', 'val_mae'])
    ax.set_title('MAE')

    ax=axs[2]
    ax.plot(train_time_history)
    ax.plot(val_time_history)
    ax.grid();ax.set_xlabel('epoch');ax.set_ylabel('time, s');l=ax.legend(['train_time', 'val_time'])
    ax.set_title('Time')

    plt.tight_layout()