import matplotlib.pyplot as plt
import os, re, random
import pandas as pd
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

# local
import scripts.v2_attention.config as config
import scripts.text_utils as text_utils

def set_random(seed=43):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def log_fn(debug_msg: str, config: config.MealModelConfig, name=None):
    if config and config.DEBUG:
        if name and name in config.LOG_NOT_SHOW:
            pass
        else:
            print(debug_msg)

def read_preprocessed_meal_dataset():
    """
    Возвращает рабочий подготовленный датасет с блюдами
    """
    meal_df = pd.read_csv(config.MealModelConfig.DATASET_PATH)

    # распарсим ингредиенты из текста обратно в список
    if meal_df['ingr_name'].dtype == 'object':
        meal_df['ingr_name'] = meal_df['ingr_name'].apply(text_utils.parse_str_to_list)

    return meal_df

def get_optimizer(model:nn.Module, backbone_lr=1e-4, head_lr=1e-3, k_100_ref_lr=5e-2, weight_decay=1e-2):
    """
    Настройка оптимизатора, три группы параметров с разными learning rate и weight_decay:
    - `backbone_params` - отдельный шаг (минимальный) для backbone моделей, чтобы не разрушить предобученное состояние
    - `head_params` - параметры для слияния признаков и предсказания - основная обучаемая группа - шаг более грубый
    - `k_100_ref_lr` - обучаемые слои для отклонения справочной калорийности:
                       по ним используется явная регуляризация на уровне loss 
                       (которая избирательна по ингредиентам из-за примерения маски)
    - `no_decay_params` - параметры без регуляризации (которая стремится занулить веса)
                        - защита нормализации и смещения от искусственного искажения
    """

    backbone_params = []
    k_100_delta_params = []
    head_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue        
        if "img_model" in name:
            backbone_params.append(param)
        elif "norm" in name or "bias" in name :
            no_decay_params.append(param)
        elif "k_100_vocab" in name:
            k_100_delta_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': head_params, 'lr': head_lr, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'lr': head_lr, 'weight_decay': 0.0},
        {"params": k_100_delta_params, "lr": k_100_ref_lr, "weight_decay": 0.0}
    ])

    return optimizer

def optimizer_params_sanity_check(model: nn.Module, optimizer: torch.optim.AdamW):
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
    Функция ошибки Huber - сочетание MSE (для ошибки < delta) и MAE x delta (для ошибки > delta) 
    """
    return nn.functional.huber_loss(pred, real, reduction='mean', delta=delta)

def smooth_l1_metric(pred: torch.Tensor, real: torch.Tensor, delta=1.0):
    """
    Функция ошибки smooth_l1 - сочетание MSE (для ошибки < delta) и MAE (для ошибки > delta)
    Не делает умножение на delta в зоне MAE, поэтому loss не такой большой
    """

    return nn.functional.smooth_l1_loss(pred, real, beta=delta)


def train(model:nn.Module, config: config.MealModelConfig, optimizer: torch.optim.AdamW, dataloader: DataLoader, scheduler = None):
    """
    Выполнение цикла обучения

    Returns:
        tuple:
            - train_loss_huber - значение loss-функции
            - train_mae - значение метрики MAE для целевой переменной (калорийность блюда) 
            - train_kal_100_mae - значение метрики MAE для вспомогательного предсказания - калорийность/100г
            - ingr_mass - вектор предсказний массы ингредиентов
            - train_k_100_delta_global - вектор глобальных корректировок справочника калорийности продуктов
            - train_k_100_delta_img - вектор дополнительных корректировок справочника калорийности продуктов на основе признаков изображения
            - train_time - время цикла (сек)
    """
    
    log = partial(log_fn, config=config, name='train')
    log('   --- train()')
    
    # перенос на устройство
    device = use_device()
    model.to(device)
    model.train()

    i = 0
    running_loss = 0
    running_mae = 0
    running_kal_100_mae = 0

    running_k_100_delta_global = 0
    running_k_100_delta_img = 0    

    start = perf_counter()
    for data in tqdm(dataloader, 'training'):
        i += 1

        ingr_embs = data['ingr_embs'].to(device)
        ingr_ids = data['ingr_ids'].to(device)
        ingr_mask = data['ingr_mask'].to(device)
        images = data['images'].to(device)
        ingr_ref_k100 = data['ingr_ref_k100'].to(device)
        mass = data['mass'].to(device)        
        target_kal = data['target'].to(device)        
        
        target_kal_100 = 100 * target_kal / mass

        log(f'mass = {mass.shape} - {mass}')
        log(f'target_kal = {target_kal.shape} - {target_kal}')
        log(f'target_kal_100 = {target_kal_100.shape} - {target_kal_100}')
                
        optimizer.zero_grad()

        total_kal, total_k_100, ingr_mass, (k_100_delta_global, k_100_delta_img) = model(ingr_embs, ingr_ids, ingr_ref_k100, ingr_mask, images, mass)
        log(f'total_kal = {total_kal.shape} - {total_kal}')
        log(f'total_k_100 = {total_k_100.shape} - {total_k_100}')
        log(f'ingr_mass = {ingr_mass.shape} - {ingr_mass}')
        log(f'k_100_delta_global = {k_100_delta_global.shape} - {k_100_delta_global}')
        log(f'k_100_delta_img = {k_100_delta_img.shape} - {k_100_delta_img}')
        
        loss_huber_kal = smooth_l1_metric(total_kal, target_kal, delta=config.HUBER_DELTA_KAL)
        loss_huber_k_100 = smooth_l1_metric(total_k_100, target_kal_100, delta=config.HUBER_DELTA_KAL_100)

        loss = (loss_huber_kal + config.LOSS_K_100_WEIGHT * loss_huber_k_100)
        
        # L2-регуляризация поправок к справочнику 
        # удерживает модель от переобучения на редких продуктах 
        # стимулирует не корректировать значения справочника, а уточнять пропорции блюда
        loss_reg = 0.002 * (k_100_delta_global[ingr_mask] ** 2).mean() + (k_100_delta_img[ingr_mask] ** 2).mean()
        
        loss = loss + loss_reg

        loss.backward()
        
        # clip grad norm (рекомендуется при обучении Attention)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        optimizer.step()

        # warm-up learning_rate
        if scheduler is not None:
            scheduler.step()

        # метрики MAE
        with torch.no_grad():            
            kal_100_mae = mae_metric(total_k_100, target_kal_100)
            kal_mae = mae_metric(total_kal, target_kal)

        running_loss += loss.item()
        running_mae += kal_mae.item()
        running_kal_100_mae += kal_100_mae.item()

        running_k_100_delta_global += torch.mean(k_100_delta_global).item()
        running_k_100_delta_img += torch.mean(k_100_delta_img).item()
        
        log(f'loss = {loss.item():.4f}, ' 
            f'loss_huber_kal = {loss_huber_kal.item():.4f}, '
            f'loss_huber_k_100 = {loss_huber_k_100.item():.4f}, '
            f'loss_reg = {loss_reg.item():.4f}')

        log(f'mae = {kal_mae.item():.4f}, kal_100_mae = {kal_100_mae.item():.4f}')

        if i % 50 == 0:
            log(f'running_loss = {running_loss/i:.4f}, '
                f'running_mae = {running_mae/i:.4f}, '
                f'running_kal_100_mae = {running_kal_100_mae/i:.4f}, '
                f'running_k_100_delta_global = {running_k_100_delta_global/i:.4f}, '
                f'running_k_100_delta_img = {running_k_100_delta_img/i:.4f}')
        
        if i == config.N_ITER:
            print(f'debug early stop - i = {i}')
            break

    train_loss_huber = running_loss / i
    train_mae = running_mae / i
    train_kal_100_mae = running_kal_100_mae / i
    train_k_100_delta_global = running_k_100_delta_global / i
    train_k_100_delta_img = running_k_100_delta_img /i

    return train_loss_huber, train_mae, train_kal_100_mae, ingr_mass, train_k_100_delta_global, train_k_100_delta_img, perf_counter() - start

def validate(model: nn.Module, config: config.MealModelConfig, dataloader: DataLoader):
    """
    Выполнение цикла валидации

    Returns:
        tuple:
            - val_loss_huber - значение loss-функции
            - val_mae - значение метрики MAE для целевой переменной (калорийность блюда) 
            - val_kal_100_mae - значение метрики MAE для вспомогательного предсказания - калорийность/100г
            - ingr_mass - вектор предсказний массы ингредиентов
            - val_k_100_delta_global - вектор глобальных корректировок справочника калорийности продуктов
            - val_k_100_delta_img - вектор дополнительных корректировок справочника калорийности продуктов на основе признаков изображения
            - val_time - время цикла (сек)
    """
    log = partial(log_fn, config=config, name='validate')
    log('   -- validate()')

    device = use_device()

    start = perf_counter()
    model.eval()
    i = 0
    
    running_loss = 0
    running_mae = 0
    running_kal_100_mae = 0

    running_k_100_delta_global = 0
    running_k_100_delta_img = 0    

    for data in tqdm(dataloader, 'validation'):
        i += 1

        ingr_embs = data['ingr_embs'].to(device)
        ingr_ids = data['ingr_ids'].to(device)
        ingr_mask = data['ingr_mask'].to(device)
        images = data['images'].to(device)
        ingr_ref_k100 = data['ingr_ref_k100'].to(device)
        mass = data['mass'].to(device)        
        target_kal = data['target'].to(device)       

        with torch.no_grad():                    
            total_kal, total_k_100, ingr_mass, (k_100_delta_global, k_100_delta_img) = model(ingr_embs, ingr_ids, ingr_ref_k100, ingr_mask, images, mass)
            target_kal_100 = 100 * target_kal / mass

            loss_huber_kal = smooth_l1_metric(total_kal, target_kal, delta=config.HUBER_DELTA_KAL)
            loss_huber_k_100 = smooth_l1_metric(total_k_100, target_kal_100, delta=config.HUBER_DELTA_KAL_100)

            loss = (loss_huber_kal + config.LOSS_K_100_WEIGHT * loss_huber_k_100)
            loss_reg = 0.002 * (k_100_delta_global[ingr_mask] ** 2).mean() + (k_100_delta_img[ingr_mask] ** 2).mean()
            loss = loss + loss_reg
            
            kal_100_mae = mae_metric(total_k_100, target_kal_100)
            kal_mae = mae_metric(total_kal, target_kal)

            running_loss += loss.item()
            running_mae += kal_mae.item()
            running_kal_100_mae += kal_100_mae.item()
            
            running_k_100_delta_global += torch.mean(torch.abs(k_100_delta_global)).item()
            running_k_100_delta_img += torch.mean(torch.abs(k_100_delta_img)).item()
        
            log(f'loss = {loss.item():.4f}, ' 
                f'loss_huber_kal = {loss_huber_kal.item():.4f}, '
                f'loss_huber_k_100 = {loss_huber_k_100.item():.4f}, '
                f'loss_reg = {loss_reg.item():.4f}')

            log(f'mae = {kal_mae.item():.4f}, kal_100_mae = {kal_100_mae.item():.4f}')

            if i % 50 == 0:
                log(f'running_loss = {running_loss/i:.4f}, '
                    f'running_mae = {running_mae/i:.4f}, '
                    f'running_kal_100_mae = {running_kal_100_mae/i:.4f}, '
                    f'running_k_100_delta_global = {running_k_100_delta_global/i:.4f}, '
                    f'running_k_100_delta_img = {running_k_100_delta_img/i:.4f}')

        if i == config.N_ITER:
            log(f'debug early stop - i = {i}')
            break

    val_loss_huber = running_loss/i
    val_mae = running_mae/i
    val_kal_100_mae = running_kal_100_mae / i
    val_k_100_delta_global = running_k_100_delta_global / i
    val_k_100_delta_img = running_k_100_delta_img /i

    return val_loss_huber, val_mae, val_kal_100_mae, ingr_mass, val_k_100_delta_global, val_k_100_delta_img, perf_counter() - start


def save_model(model: nn.Module, config: config.MealModelConfig):
    """
    Сохранение весов модели (checkpoint)
    """
    save_path = path.Path(config.MODEL_SAVE_PATH)
    save_path.mkdir(exist_ok=True)
    file_path = save_path / config.MODEL_SAVE_NAME
    
    print(f'model saving to - [{file_path}]')
    torch.save(model.state_dict(), file_path)
def save_checkpoint(model: nn.Module, config: config.MealModelConfig, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.SequentialLR, current_epoch, best_epoch, best_metric):
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

def restore_checkpoint(model: nn.Module, config: config.MealModelConfig, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.SequentialLR):
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
    fig, axs = plt.subplots(1,3, figsize=(12,5))
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