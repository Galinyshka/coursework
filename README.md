```data_collect.ipynb``` - Сбор данных и разделение на тренировочную и тестовую выборку \
```tokinizer.py``` - токенайзер для первого подхода \
```tokinizer_kz.py``` - токенайзер для второго подхода \
```train_ru_tat.py``` - обучение с руского на татарский для первого подхода \
```train_tat_ru.py``` - обучение с татарского на русский для первого подхода \
```train_ru_tat_kz.py``` - обучение с руского на татарский для второго подхода


# How to use models


## For models that use tatar language as new language
```python
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
MODEL = "Galinyshka/mBART50_rus-tat_v1"

def fix_tokenizer(tokenizer):
    """ Add a new language token to the tokenizer vocabulary (this should be done each time after its initialization) """
    old_len = len(tokenizer) - int('tt_XX' in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id['tt_XX'] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = 'tt_XX'
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if 'tt_XX' not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append('tt_XX')

def translate(text, src='ru_RU', trg='tt_XX', max_length=256, num_beams=5, repetition_penalty=5.0, **kwargs):
    tokenizer.src_lang = src
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded.to(model.device),
        forced_bos_token_id=tokenizer.lang_code_to_id[trg], 
        max_length=max_length, 
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        early_stopping=True,
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

model = MBartForConditionalGeneration.from_pretrained(MODEL)
tokenizer = MBart50Tokenizer.from_pretrained(MODEL)

fix_tokenizer(tokenizer)

translation = translate("Привет, мир!")
print(translation)
```

## For models that use tatar language as modification of kazakh language do not apply `fix_tokenizer` function, and use kk_KZ as target language
```python
MODEL = "Galinyshka/mBART50_rus-tatkz_v1"
model = MBartForConditionalGeneration.from_pretrained(MODEL)
tokenizer = MBart50Tokenizer.from_pretrained(MODEL)
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

translation = translate("Привет, мир!", src='ru_RU', trg='kk_KZ')
print(translation)
```