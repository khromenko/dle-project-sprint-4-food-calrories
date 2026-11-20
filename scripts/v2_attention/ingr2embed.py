import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertTokenizerFast, DistilBertModel
import torch, torch.nn as nn

# locals
import scripts.v2_attention.config as config
import scripts.v2_attention.text_utils as text_utils

def make_ingr_embeddings(tokenizer: DistilBertTokenizer, text_model: DistilBertModel, dataset_path: str):
    """
    Формирование текстовых эмбеддингов для названий ингредиентов.
        - используется предобученная backbone модель
        - подготовка эмбеддингов ингредиентов
        - сохранение в файл для быстрого извлечения в датасете

        - для формирования контекста фраза дополняется пищевым контекстом (на случай слов омонимов из других областей)
            - "food ingredient: <название ингредиента>"
        - на выходе `text_model` применяется mean pooling по всем токенам фразы
            - результат - единый вектор-эмбеддинг на каждый ингредиент [1, text_dim]
            - при этом удаляются спец-токены CLS, SEP, чтобы они не размывали значения эмбеддингов

        - результат сохраняется в файл `data/preprocess/text_emb.pt`
            - `ingr_encoder` - ингредиентов - `name -> id`, где id используется для индексирования в словаре и справочнике калорий
            - `text_embs` - тензор ембеддингов, i-ая позиция = id в encoder
    """

    df = pd.read_csv(dataset_path)

    # список ингредиентов
    ingrs = np.unique(np.concat([text_utils.preprocess_ingredients(ingr_list) for ingr_list in df['ingr_name'].unique()]))
    print('ingrs:', ingrs.shape, ingrs[0:5])

    # encoder ингредиентов для позиционного индексирования
    # бронируем нулевой id для спец токена - паддинга для выравнивания    
    ingr_encoder = {'<PAD>': 0 }
    ingr_encoder.update({str(ingr): (i+1)  for i, ingr in enumerate(ingrs)})
    
    print('ingr_encoder: len =', len(ingr_encoder), [item for i, item in enumerate(ingr_encoder.items()) if i < 5])

    text_embs_list = [torch.zeros(size=[text_model.config.hidden_size], dtype=torch.float32)]
    for ingr in ingrs:
        text = f"food ingredient: {ingr}"
        token_out = tokenizer(text, return_tensors='pt')
        model_out = text_model(**token_out).last_hidden_state.squeeze(0)
        text_features = model_out[1:-1, :] # берем состояние всех токенов кроме 1-ого [CLS] и последнего [SEP] токена
        # mean pooling
        pooled_emb = torch.mean(text_features, dim=0)

        # нормирование чтобы значения были в нормальном масштабе перед слиянием с калорийностью и применением во внинимании
        pooled_emb = torch.nn.functional.normalize(pooled_emb, p=2, dim=0)

        text_embs_list.append(pooled_emb)

    text_embs_tenzor = torch.stack(text_embs_list)
    
    save_path = config.MealModelConfig.TEXT_EMBED_PATH
    print('text_embs:', text_embs_tenzor.shape, 'save to file: ', save_path)
    
    torch.save(
        {
            'text_embs': torch.Tensor(text_embs_tenzor),
            'ingr_encoder': ingr_encoder
        },
        save_path
    )

def _test(tokenizer: DistilBertTokenizer, text_model: DistilBertModel, dataset_path: str):
    """
    Тест для проверки/отладки.
    Выводит обработку эмбедднга для 3-х первых ингредиентов
    """
    
    df = pd.read_csv(dataset_path) 

    # список ингредиентов
    ingrs = np.unique(np.concat([text_utils.preprocess_ingredients(ingr_list) for ingr_list in df['ingr_name'].unique()]))
    print('ingrs:', ingrs.shape, ingrs[0:5])

    # encoder ингредиентов для позиционного индексирования
    ingr_encoder = {'<PAD>': 0 }
    ingr_encoder.update({str(ingr): (i+1)  for i, ingr in enumerate(ingrs)})
    print('ingr_encoder:', [item for i, item in enumerate(ingr_encoder.items()) if i < 5])

    text_embs_list = []
    for ingr in ingrs[0:3]:
        text = f"food ingredient: {ingr}"
        print('text:', text)

        token_out = tokenizer(text, return_tensors='pt')
        print('token_out:', token_out)
        print('decode:', tokenizer.decode(token_out['input_ids'][0])) # type: ignore

        model_out = text_model(**token_out).last_hidden_state.squeeze(0) #  [1, seq_len, emd_dim] - убираем размерность батча 
        print('model_out:', model_out.shape)

        text_features = model_out[1:-1, :] # берем состояние всех токенов кроме 1-ого [CLS] и последнего [SEP] токена
        print('text_features:', text_features.shape)

        text_emb = torch.mean(text_features, dim=0)        
        print('text_emb:', text_emb.shape)

        text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=0)

        text_embs_list.append(text_emb)

    text_embs_tenzor = torch.stack(text_embs_list)

    text_emp_path = 'data/preprocess/text_emb_test.pt'
    print('text_embs:', text_embs_tenzor.shape, 'save to file: ', text_emp_path)

    torch.save(
        {
            'text_embs': torch.Tensor(text_embs_tenzor),
            'ingr_encoder': ingr_encoder
        },
        text_emp_path
    )

    text_embs_loaded = torch.load(text_emp_path)
    print('text_embs_loaded:', text_embs_loaded.keys())

# ---------

# проверка
if __name__ == "__main__":
    cfg = config.MealModelConfig
    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
    text_model = AutoModel.from_pretrained(cfg.TEXT_MODEL_NAME)

    _test(tokenizer, text_model, cfg.DATASET_PATH)
    

# test() out:
# ingrs: (197,) ['almonds' 'apple' 'artichokes' 'arugula' 'asparagus']
# ingr_encoder: [('almonds', 0), ('apple', 1), ('artichokes', 2), ('arugula', 3), ('asparagus', 4)]
# text: food ingredient: almonds
# token_out: {'input_ids': tensor([[  101,  2833, 21774,  1024, 26011,  2015,   102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
# decode: [CLS] food ingredient : almonds [SEP]
# model_out: torch.Size([7, 768])
# text_features: torch.Size([5, 768])
# text_emb: torch.Size([768])
# text: food ingredient: apple
# token_out: {'input_ids': tensor([[  101,  2833, 21774,  1024,  6207,   102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}
# decode: [CLS] food ingredient : apple [SEP]
# model_out: torch.Size([6, 768])
# text_features: torch.Size([4, 768])
# text_emb: torch.Size([768])
# text: food ingredient: artichokes
# token_out: {'input_ids': tensor([[  101,  2833, 21774,  1024,  2396,  7033, 11045,  2015,   102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}
# decode: [CLS] food ingredient : artichokes [SEP]
# model_out: torch.Size([9, 768])
# text_features: torch.Size([7, 768])
# text_emb: torch.Size([768])
# text_embs: torch.Size([3, 768])
# text_embs_loaded: dict_keys(['text_embs', 'ingr_encoder'])