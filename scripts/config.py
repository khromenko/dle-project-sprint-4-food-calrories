class MealModelConfig:

    SEED = 43
    DATASET_PATH_ORIGINGAL = 'data/meal_combined.csv'
    DATASET_PATH_NO_OUTLIERS = 'data/meal_combined_cleaned.csv'
    DATASET_PATH = DATASET_PATH_NO_OUTLIERS
    DATASET_IMAGES_PATH = 'data/images'
    
    # внешний справочник калорийности продуктов
    DATASET_EXT_KAL_MAPPING = 'data/nutrients/processed/ingr_nutr_mapping_result.csv'

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

    # Эксперимент 2.1 - данные справочника калорийностей
    K_100_REF_MEAN = 168.79
    K_100_REF_STD = 177.34

    # models config
    TEXT_MODEL_NAME = 'distilbert-base-uncased'
    TEXT_MODEL_UNFREEZE = "transformer.layer.5" # выбрать последние 1-2 слоя при необходимости
    #TEXT_MODEL_UNFREEZE = "" # to stub text

    IMG_MODEL_NAME = 'mobilenetv4_conv_aa_large' #.e230_r448_in12k_ft_in1k'
    IMG_MODEL_UNFREEZE = "blocks.4|conv_head|norm_head" # выбрать последние 1-2 слоя при необходимости

    IMG_MODEL_FEATURE_DIM = 1280 # img_model.num_features lies telling it is 960
    COMMON_FEATURES_DIM = 256
    INGR_DIM = 197 # количество ингредиентов (известных на train)
    REGRESSION_HEAD_DIM = 1

    # Эксперименит 9 - усиленная регуляризация (было 0.2/0.2) — борьба с переобучением после эп. 21
    DROPOUT_FEATURES = 0.3
    DROPOUT_FUSION = 0.3

    # train config
    N_BATCH = 32 #32
    N_EPOCHS = 10 #30 # 10 for CPU (2.5 hours)
    PATIENCE = 6 # ранняя остановка обучения, если метрики не растут
    N_ITER = 30 #-1 #32 # для отладки, -1 без ограничений
    
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
    #HUBER_DELTA_KAL_DIRECT = 0.5 # 0.5 * TARGET_STD ≈ 110 kcal — порог MSE/L1 для калорий
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