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

# local
import scripts.models as models

def set_random(seed=43):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def log_fn(debug_msg: str, config: models.MealModelConfig):
    if config and config.DEBUG:
        print(debug_msg)

def unfreeze_model_params(model: models.MealModelNN, config: models.MealModelConfig):
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

def optimizer_params_sanity_check(model: models.MealModelNN, optimizer: torch.optim.AdamW):
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

def train(model: models.MealModelNN, config: models.MealModelConfig, optimizer: torch.optim.AdamW, dataloader: DataLoader, scheduler = None):
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
        ref_k100 = data['ref_k100'].to(device)
        
        target = data['target'].to(device)
        target_kal_100 = data['target_kal_100'].to(device)
        target_mass = data['target_mass'].to(device)
        
        optimizer.zero_grad()

        # forward возвращает 5 значений: ensemble, direct, product, k100, mass
        result_kal, result_kal_direct, result_kal_product, result_k_100, result_mass = model(input_ids, attention_mask, images, ref_k100)
                
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

def validate(model: models.MealModelNN, config: models.MealModelConfig, dataloader: DataLoader):
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
        ref_k100 = data['ref_k100'].to(device)
        target = data['target'].to(device)
        target_kal_100 = data['target_kal_100'].to(device)
        target_mass = data['target_mass'].to(device)

        with torch.no_grad():        
            # forward возвращает 5 значений: ensemble, direct, product, k100, mass
            result_kal, result_kal_direct, result_kal_product, result_k_100, result_mass = model(input_ids, attention_mask, images, ref_k100)

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

def save_model(model: models.MealModelNN, config: models.MealModelConfig):
    """
    Сохранение весов модели (checkpoint)
    """
    save_path = path.Path(config.MODEL_SAVE_PATH)
    save_path.mkdir(exist_ok=True)
    file_path = save_path / config.MODEL_SAVE_NAME
    
    print(f'model saving to - [{file_path}]')
    torch.save(model.state_dict(), file_path)
def save_checkpoint(model: models.MealModelNN, config: models.MealModelConfig, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR, current_epoch, best_epoch, best_metric):
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

def restore_checkpoint(model: models.MealModelNN, config: models.MealModelConfig, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR):
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