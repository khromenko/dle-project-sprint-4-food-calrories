class MealModelConfig:

    SEED = 43
    DATASET_PATH_ORIGINGAL = 'data/meal_combined.csv'
    DATASET_PATH_NO_OUTLIERS = 'data/meal_combined_cleaned.csv'
    DATASET_PATH = DATASET_PATH_NO_OUTLIERS
    DATASET_IMAGES_PATH = 'data/images'
    
    # подготовленные текстовые эмбеддинги ингредиентов и кодирощик name-id
    TEXT_EMBED_PATH = 'data/preprocess'
    TEXT_EMBED_FILE = 'text_emb.pt'

    # внешний справочник калорийности продуктов
    DATASET_EXT_KAL_MAPPING = 'data/nutrients/processed/ingr_nutr_mapping_result.csv'

    # параметры средних и отклонений train датасета
    K_100_REF_MEAN = 168.79
    K_100_REF_STD = 177.34

    # models config
    TEXT_MODEL_NAME = 'distilbert-base-uncased'
    
    IMG_MODEL_NAME = 'mobilenetv4_conv_aa_large' #.e230_r448_in12k_ft_in1k'
    IMG_MODEL_UNFREEZE = "blocks.4|conv_head|norm_head" #

    IMG_MODEL_FEATURE_DIM = 960 # 1280 # img_model.num_features lies telling it is 960
    COMMON_FEATURES_DIM = 256
    INGR_DIM = 198 # 197 количество используемых ингредиентов + 1 <PAD>
    REGRESSION_HEAD_DIM = 1

    # Эксперименит 9 - усиленная регуляризация (было 0.2/0.2) — борьба с переобучением после эп. 21
    DROPOUT_FEATURES = 0.3
    DROPOUT_FUSION = 0.3

    # train config
    N_BATCH = 32
    N_EPOCHS = 30 #30 # 10 for CPU (2.5 hours)
    PATIENCE = 6 # ранняя остановка обучения, если метрики не растут
    N_ITER = -1 #32 # для отладки, -1 без ограничений
    
    # optimizer
    LR_IMG_MODEL = 1e-4
    LR_HEAD = 1e-3    
    LR_K_100_REF = 1e-1 # больший шаг, так как значения в реальных величинах
    
    WEIGHT_DECAY = 1e-4

    # loss
    HUBER_DELTA_KAL = 20 # порог для расчета MAE/MSE
    HUBER_DELTA_KAL_100 = 10 #
    # 2ая голова - k_100
    LOSS_K_100_WEIGHT = 0.5

    # target
    TARGET_METRIC = 20 #50

    MODEL_SAVE_PATH = 'data/model'
    MODEL_SAVE_NAME = 'best_model.pth'
    MODEL_LOG_SAVE_PATH = 'data/model'

    # отладочная печать
    DEBUG = True
    LOG_NOT_SHOW=['model', 'train', 'validate']