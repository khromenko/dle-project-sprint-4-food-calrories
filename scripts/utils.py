import os, re, random
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

    # models config
    TEXT_MODEL_NAME = 'distilbert-base-uncased'
    TEXT_MODEL_UNFREEZE = "" # выбрать последние 1-2 слоя при необходимости

    IMG_MODEL_NAME = 'mobilenetv4_conv_aa_large' #.e230_r448_in12k_ft_in1k'
    IMG_MODEL_UNFREEZE = "" # выбрать последние 1-2 слоя при необходимости

    IMG_MODEL_FEATURE_DIM = 1280 # img_model.num_features lies telling it is 960
    COMMON_FEATURES_DIM = 256
    REGRESSION_HEAD_DIM = 1

    # train config
    N_BATCH = 8
    N_EPOCHS = 10 # 30
    PATIENCE = 3 # ранняя остановка обучения, если метрики не растут
    N_ITER = 10 #32 # для отладки, -1 без ограничений
    
    # optimizer
    LR_TEXT_MODEL = 3e-5
    LR_IMG_MODEL = 1e-4
    LR_PROJ = 1e-4
    LR_ATTENTION = 1e-4
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
        # self.drop = nn.Dropout(0.1)
        
        # проекционный слой для выравнивания размерностей признаков разных модальностей
        # надо ли свести сначала в общую размерность, какое значение выбрать - config.COMMON_FEATURES_DIM?
        
        self.text_proj = nn.Linear(in_features=self.text_model.config.hidden_size, out_features=config.COMMON_FEATURES_DIM)
        self.img_proj = nn.Linear(in_features=config.IMG_MODEL_FEATURE_DIM, out_features=config.COMMON_FEATURES_DIM) # type: ignore

        # объединение признаков разных модальностей 
        # вар. 1 - модель перекрестного-внимания
        # # не сработало - на одном текстовом токене [CLS] - Softmax внутри Attention дает 1 и текстовые признаки вообще не учитываются
        # self.attention = nn.MultiheadAttention(embed_dim=config.COMMON_FEATURES_DIM, num_heads=4, batch_first=True)
        
        # вар. 2 - слой конкатенации
        self.fusion = nn.Sequential(
            nn.Linear(2 * config.COMMON_FEATURES_DIM, config.COMMON_FEATURES_DIM),
            nn.LayerNorm(config.COMMON_FEATURES_DIM),
            nn.GELU(),
            #nn.Dropout(0.1)
        )
        
        # финальный выход - регрессионный полносвязный слой
        self.regression_head = nn.Linear(in_features=config.COMMON_FEATURES_DIM, out_features=config.REGRESSION_HEAD_DIM) 


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor):
        #self.log(f'forward() input_ids = {input_ids.shape}, images = {images.shape}')

        start = perf_counter()

        # 1 - text features
        text_out = self.text_model(input_ids, attention_mask).last_hidden_state
        #self.log(f'  text_out = {text_out.shape} ({(perf_counter()-start):.2f} sec)')
        
        start = perf_counter()

        text_features = text_out[:, 0, :] # берем состояние 1-ого [CLS] токена
        #self.log(f'  text_features = {text_features.shape}')
        self.log(f'  text_features = {text_features}')

        # 2 - image features
        img_out = self.img_model(images)
        img_features = img_out
        #self.log(f'  img_features = {img_features.shape} ({(perf_counter()-start):.2f} sec)')
        self.log(f'  img_features = {img_features}')

        self.log(f'text feat norm = {text_features.norm(dim=1).mean().item():.2f}, img feat norm = {img_features.norm(dim=1).mean().item():.2f}')

        start = perf_counter()

        # 3 - text features projection
        text_proj = self.text_proj(text_features)
        #text_proj = self.text_norm(text_proj)
        #text_proj = self.drop(text_proj)
        #self.log(f'  text_proj = {text_proj.shape} ({(perf_counter()-start):.2f} sec)')

        start = perf_counter()

        # 4 - image features projection
        img_proj = self.img_proj(img_features)
        #img_proj = self.img_norm(img_proj)
        #img_proj  = self.drop(img_proj)
        #self.log(f'  img_proj = {img_proj.shape} ({(perf_counter()-start):.2f} sec)')

        start = perf_counter()

        # 5 - fusion
        # 5.1 - attention
        # attention_out, _ = self.attention(query=text_proj, key=img_proj, value=img_proj)
        #self.log(f'  attention_out = {attention_out.shape} ({(perf_counter()-start):.2f} sec)')
        #self.log(f'  attention_out = {attention_out}')

        # 5.2 - fusion
        fusion_out = self.fusion(torch.concat([text_proj, img_proj], dim=1))
        self.log(f'  fusion_out = {fusion_out.shape} ({(perf_counter()-start):.2f} sec)')
        self.log(f'  fusion_out = {fusion_out}')

        start = perf_counter()

        # 6 - regression head
        regression_out = self.regression_head(fusion_out)    
        #self.log(f'  regression_out = {regression_out.shape} ({(perf_counter()-start):.2f} sec)')
        self.log(f'  regression_out = {regression_out}')

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

def mae_loss(pred: torch.Tensor, real: torch.Tensor):
    """
    Функция ошибки MAE
    """
    return torch.abs(pred-real).mean()


def train(model: MealModelNN, config: MealModelConfig, optimizer: torch.optim.AdamW, dataloader: DataLoader):
    log = partial(log_fn, config=config)
    log('   --- train()')
    
    # перенос на устройство
    device = use_device()
    model.to(device)

    model.train()
    i = 0
    running_loss = 0

    start = perf_counter()
    for data in tqdm(dataloader, 'training'):
        i += 1

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        images = data['images'].to(device)
        target = data['target'].to(device)

        optimizer.zero_grad()
        
        result = model(input_ids, attention_mask, images)
        
        log(f'result = {result.squeeze()}')
        log(f'target = {target}')

        loss = mae_loss(result, target)
        log(f'loss = {loss}')

        loss.backward()

        # clip grad norm
        # torch.nn.utils.clip_grad_norm_(
        #     [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        # )

        optimizer.step()

        running_loss += loss.item()

        if i % 20 == 0:
            log(f'running_loss = {running_loss/i:.4f}')
        
        if i == config.N_ITER:
            print(f'debug early stop - i = {i}')
            break

    train_loss = running_loss/i
    return train_loss, perf_counter() - start

def validate(model: MealModelNN, config: MealModelConfig, dataloader: DataLoader):
    log = partial(log_fn, config=config)
    log('   -- validate()')

    device = use_device()

    start = perf_counter()
    model.eval()
    i = 0
    running_loss = 0
    for data in tqdm(dataloader, 'validation'):
        i += 1

        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        images = data['images'].to(device)
        target = data['target'].to(device)

        with torch.no_grad():        
            result = model(input_ids, attention_mask, images)

            # проверка - насколько большие значения - возможно из-за этого взрывные MAE
            tf = model.text_model(input_ids, attention_mask).last_hidden_state[:,0]
            imf = model.img_model(images)

            log(f'check text feat norm = {tf.norm(dim=1).mean().item():.2f}, img feat norm = {imf.norm(dim=1).mean().item():.2f}')

            tp = model.text_proj(tf)      
            ip = model.img_proj(imf)
            log(f'text_proj per-sample norms = {tp.norm(dim=1)}')  
            log(f'img_proj  per-sample norms = {ip.norm(dim=1)}')

            loss = mae_loss(result, target)

            log(f'result = {result.squeeze()}')
            log(f'target = {target}')
            log(f'loss = {loss}')
            
        running_loss += loss.item()

        if i == config.N_ITER:
            log(f'debug early stop - i = {i}')
            break

    val_loss = running_loss/i
    return val_loss, perf_counter() - start