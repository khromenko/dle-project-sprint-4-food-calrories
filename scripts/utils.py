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
    DATASET_PATH = 'data/meal_combined.csv'
    IMAGES_PATH = 'data/images'

    TARGET_MEAN = 255
    TARGET_STD = 221

    # models config
    TEXT_MODEL_NAME = 'distilbert-base-uncased'
    TEXT_MODEL_UNFREEZE = "transformer.layer.5" # выбрать последние 1-2 слоя при необходимости

    IMG_MODEL_NAME = 'mobilenetv4_conv_aa_large' #.e230_r448_in12k_ft_in1k'
    IMG_MODEL_UNFREEZE = "blocks.4|conv_head|norm_head" # выбрать последние 1-2 слоя при необходимости

    IMG_MODEL_FEATURE_DIM = 1280 # img_model.num_features lies telling it is 960
    COMMON_FEATURES_DIM = 256
    REGRESSION_HEAD_DIM = 1
    DROPOUT_FEATURES = 0.2
    DROPOUT_FUSION = 0.2

    # train config
    N_BATCH = 32
    N_EPOCHS = 20 # 10 for CPU (2.5 hours)
    PATIENCE = 4 # ранняя остановка обучения, если метрики не растут
    N_ITER = -1 #32 # для отладки, -1 без ограничений
    
    # optimizer
    LR_TEXT_MODEL = 1e-5
    LR_IMG_MODEL = 1e-5
    LR_PROJ = 1e-4
    LR_FUSION = 1e-4
    LR_REGRESSION_HEAD = 1e-4

    TARGET_METRIC = 50

    MODEL_SAVE_PATH = 'data/model'
    MODEL_SAVE_NAME = 'best_model.pth'
    MODEL_LOG_SAVE_PATH = 'data/model'

    DEBUG = True # False


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

        # нормализация выходов признаков
        # self.text_norm = nn.LayerNorm(config.COMMON_FEATURES_DIM)
        # self.img_norm  = nn.LayerNorm(config.COMMON_FEATURES_DIM)
                
        # проекционный слой для выравнивания размерностей признаков разных модальностей       
        self.text_proj = nn.Linear(in_features=self.text_model.config.hidden_size, out_features=config.COMMON_FEATURES_DIM)
        self.img_proj = nn.Linear(in_features=config.IMG_MODEL_FEATURE_DIM, out_features=config.COMMON_FEATURES_DIM) # type: ignore

        # слой нормализации для проекций признаков + dropout
        self.norm_layer = nn.LayerNorm(config.COMMON_FEATURES_DIM)
        self.dropout = nn.Dropout(config.DROPOUT_FEATURES)

        # объединение признаков разных модальностей 
        # вар. 1 - модель перекрестного-внимания - не сработало - на одном текстовом токене [CLS] - Softmax внутри Attention дает 1 и текстовые признаки вообще не учитываются
        # self.attention = nn.MultiheadAttention(embed_dim=config.COMMON_FEATURES_DIM, num_heads=4, batch_first=True)
        
        # вар. 2 - слой конкатенации
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.COMMON_FEATURES_DIM, config.COMMON_FEATURES_DIM),
            nn.LayerNorm(config.COMMON_FEATURES_DIM),
            nn.GELU(),
            nn.Dropout(config.DROPOUT_FUSION)
        )
        
        # финальный выход - регрессионный полносвязный слой
        self.regression_head = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=config.REGRESSION_HEAD_DIM) 


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor):
        #self.log(f'forward() input_ids = {input_ids.shape}, images = {images.shape}')

        # 1 - text features
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
        # 5.1 - attention
        # attention_out, _ = self.attention(query=text_proj, key=img_proj, value=img_proj)
        
        # 5.2 - fusion
        fusion_out = self.fusion(torch.concat([text_proj, img_proj], dim=1))
        
        # 6 - regression head
        regression_out = self.regression_head(fusion_out).squeeze(-1) # [B, 1] → [B]
        
        return regression_out

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
    running_huber = 0
    running_mae = 0

    start = perf_counter()
    for data in tqdm(dataloader, 'training'):
        i += 1

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        images = data['images'].to(device)
        target = data['target'].to(device)
        
        target_mean = config.TARGET_MEAN
        target_std = config.TARGET_STD

        optimizer.zero_grad()
        
        result = model(input_ids, attention_mask, images)
        
        #log(f'result = {result}')
        #log(f'target = {target}')

        loss_huber = huber_metric(result, target)
        loss_huber.backward()
        
        # clip grad norm
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()

        # warm-up learning_rate
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            result_real = result * target_std + target_mean
            target_real = target * target_std + target_mean
            mae = mae_metric(result_real, target_real)

        #log(f'loss_huber = {loss_huber.item():.2f}, metric mae = {mae.item():.2f}')

        running_huber += loss_huber.item()
        running_mae += mae.item()

        if i % 50 == 0:
            log(f'running_huber = {running_huber/i:.4f}, running_mae = {running_mae/i:.4f}')
        
        if i == config.N_ITER:
            print(f'debug early stop - i = {i}')
            break

    train_huber_loss = running_huber / i
    train_mae_loss = running_mae / i
    return train_huber_loss, train_mae_loss, perf_counter() - start

def validate(model: MealModelNN, config: MealModelConfig, dataloader: DataLoader):
    log = partial(log_fn, config=config)
    log('   -- validate()')

    device = use_device()

    start = perf_counter()
    model.eval()
    i = 0
    running_huber = 0
    running_mae = 0
    for data in tqdm(dataloader, 'validation'):
        i += 1

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        images = data['images'].to(device)
        target = data['target'].to(device)

        target_mean = config.TARGET_MEAN
        target_std = config.TARGET_STD

        with torch.no_grad():        
            result = model(input_ids, attention_mask, images)

            # huber
            loss_huber = huber_metric(result, target)

            # mae
            result_real = result * target_std + target_mean
            target_real = target * target_std + target_mean
            mae = mae_metric(result_real, target_real)

            #log(f'result = {result}')
            #log(f'target = {target}')
            #log(f'huber loss = {loss_huber.item():.4f}, mae metric = {mae.item():.4f}')
            
            running_huber += loss_huber.item()
            running_mae += mae.item()

        if i == config.N_ITER:
            log(f'debug early stop - i = {i}')
            break

    val_loss_huber = running_huber/i
    val_metric_mea = running_mae/i
    return val_loss_huber, val_metric_mea, perf_counter() - start

def save_model(model: MealModelNN, config: MealModelConfig):
    """
    Сохранение весов модели (checkpoint)
    """
    save_path = path.Path(config.MODEL_SAVE_PATH)
    save_path.mkdir(exist_ok=True)
    file_path = save_path / config.MODEL_SAVE_NAME
    
    print(f'model saved to - [{file_path}]')
    torch.save(model.state_dict(), file_path)