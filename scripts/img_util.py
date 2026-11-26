import numpy as np
import pandas as pd
import torch
from PIL import Image
import albumentations as A
import timm
import matplotlib.pyplot as plt
import pathlib as path

import scripts.v2_attention.config as config
import scripts.v2_attention.dataset as dataset

def plot_dish_image(dish: pd.Series, title_prefix = ''):
    plot_image(dish_id=dish['dish_id'], total_calories=dish['total_calories'], total_mass=dish['total_mass'], ingredients=dish['ingr_name'], title_prefix=title_prefix)

def plot_image(dish_id = '', total_calories='', total_mass='', ingredients:list = None, title_prefix=''):
    """
    Вывод изображения блюда и его характеристи
    """
    
    img_path = path.Path(config.MealModelConfig.DATASET_IMAGES_PATH)
    image = Image.open(path.Path(img_path) / dish_id / 'rgb.png')
    
    plt.imshow(image)

    title_prefix = f'{title_prefix} \n {dish_id}' if title_prefix else dish_id
    plt.title(f"{title_prefix} \n cal: {total_calories:.2f} kcal, mass: {total_mass:.2f} g")
    plt.xlabel(" \n ".join(ingredients))
    
def plot_image_with_transforms(dish_id = 'dish_1560356627'):
    """
    Вывод изображения и его трансформированной версии при аугментации в датасете.
    """

    model_config = config.MealModelConfig()
    img_path = path.Path(model_config.DATASET_IMAGES_PATH)
        
    # вытаскиваем трансформации из датасета
    train_ds = dataset.MealDataset(model_config, split='train')
    img_transforms = train_ds.img_transforms

    # достаем параметры нормализации, чтобы восстановить изображение в реальные значения
    # проверим, что предпоследний трансформ это нормализация
    norm_transform: A.Normalize = img_transforms[-2] # type: ignore
    assert str(type(norm_transform)).find('Normalize') != -1, 'Трансформация [Normalize] на предпоследнем месте не найдена'
    
    # среднее и отклонение для восстановления изображения
    img_mean, img_std = norm_transform.mean, norm_transform.std
    
    print('img_transforms: ', img_transforms)
    original_image = Image.open(path.Path(img_path) / dish_id / 'rgb.png')
    print('original image shape:', original_image.size)

    image_transform = img_transforms(image=np.array(original_image))["image"]
    print('image_transform shape:', image_transform.shape)

    # денормализация в исходные значения
    image_transform = (image_transform * torch.Tensor(img_std).view(3,1,1)) + torch.Tensor(img_mean).view(3,1,1)

    plt.subplot(1,2,1)
    plt.imshow(original_image)
    plt.title('Original image')
    
    plt.subplot(1,2,2)
    plt.imshow(image_transform.permute(1,2,0))
    plt.title('Transformed image')

    plt.tight_layout()
    
