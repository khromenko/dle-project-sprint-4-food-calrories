import numpy as np
import torch
from PIL import Image
import albumentations as A
import timm
import scripts.config as config
import matplotlib.pyplot as plt
import pathlib as path

"""
Вывод изображения и его трансформированной версии.
Трансформации скопированы из датасета
[TODO] - использовать датасет для предоставления изображения напрямую
"""

def plot_image_with_transforms(dish_id = 'dish_1560356627'):
    model_config = config.MealModelConfig()
    img_path = path.Path(model_config.DATASET_IMAGES_PATH)
    img_model_config = timm.get_pretrained_cfg(model_name=model_config.IMG_MODEL_NAME); assert img_model_config is not None
    img_size = img_model_config.input_size[1] # CxHxW

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
        A.Affine(scale=(0.95, 1.05), rotate=(-15, 15), translate_percent=(-0.1, 0.1),
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
    img_transforms = A.Compose(base_transforms_train + aug_transforms + post_transforms)

    print('img_transforms: ', img_transforms)

    original_image = Image.open(path.Path(img_path) / dish_id / 'rgb.png')

    image = img_transforms(image=np.array(original_image))["image"]

    print('image:', image.shape)

    image_restore = (image * torch.Tensor(img_model_config.std).view(3,1,1)) + torch.Tensor(img_model_config.mean).view(3,1,1)

    plt.subplot(1,2,1)
    plt.imshow(original_image)
    plt.title('Original image')
    
    plt.subplot(1,2,2)
    plt.imshow(image_restore.permute(1,2,0))
    plt.title('Transformed image')

    plt.tight_layout()
    
