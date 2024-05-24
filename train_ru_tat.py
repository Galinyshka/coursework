import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import random

import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import gc 
from transformers.optimization import Adafactor 
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler

from sacrebleu import CHRF, BLEU

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def translate(text, src='ru_RU', trg='tt_XX', max_length=200, num_beams=5, repetition_penalty=5.0, **kwargs):
    tokenizer.src_lang = src
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded.to(model.device),
        forced_bos_token_id=tokenizer.lang_code_to_id[trg], 
        max_length=max_length, 
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
        # early_stopping=True,
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def evaluate_bleu_chrf(model, tokenizer, data, src='ru_RU', trg='tt_XX'):
    '''Функция для рассчета метрик'''
    model.eval()
    test_src = data['Russian'].to_list()
    test_tgt = data['Tatar'].to_list()
    test_pred = [translate(x, src=src, trg=trg).replace('tt_XX', '').strip() for x in tqdm(test_src)]
    with open('test_pred.txt', 'w') as f:
        for item in test_pred:
            f.write("%s\n" % item)
    test_bleu = BLEU()
    test_chrf = CHRF(word_order=2)
    test_data['pred'] = test_pred
    chrf = test_chrf.corpus_score(test_pred, [[x for x in  test_tgt]]).score
    bleu = test_bleu.corpus_score(test_pred, [[x for x in  test_tgt]]).score
    model.train()
    return bleu, chrf

cur_path = os.path.dirname(os.path.realpath(__file__))
model = MBartForConditionalGeneration.from_pretrained(os.path.join(cur_path, 'tokenizers/new_tokenizer'))#"/home/ryi-v/work/bibble_scrap/runs_ru_tat/004"))
tokenizer = MBart50Tokenizer.from_pretrained(os.path.join(cur_path, 'tokenizers/new_tokenizer'))#"/home/ryi-v/work/bibble_scrap/runs_ru_tat/004"))
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

fix_tokenizer(tokenizer)
CURRENT_RUN = len(os.listdir(os.path.join(cur_path, 'runs_ru_tat')))
save_path = os.path.join(cur_path, 'runs_ru_tat', str(CURRENT_RUN).zfill(3))
os.mkdir(save_path)
model.cuda()

for p in model.parameters():
    # замораживаем параметры
    p.requires_grad = False
for p in model.model.shared.parameters():
    # делаем обучаемыми только эмбединги
    p.requires_grad = True

# оптимизатор градиентного спуска     
optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad], 
    scale_parameter=False, 
    relative_step=False, 
    lr=5e-5,
    clip_threshold=1.0
)


batch_size = 4  
report_steps = 500 
epochs = 1 # warmup эпоха
losses = []

tokenizer.src_lang = "ru_RU"
tokenizer.tgt_lang = "tt_XX"
all_data = pd.read_csv(os.path.join(cur_path, 'data/train_data.csv'))
all_pairs = list(zip(all_data['Tatar'], all_data['Russian']))
test_data = pd.read_csv(os.path.join(cur_path, 'data/test_data.csv'))
cleanup()
model.train()

for epoch in range(epochs):
    print('EPOCH', epoch)
    random.shuffle(all_pairs)
    for i in trange(0, int(len(all_pairs) / batch_size)):
        batch = all_pairs[i * batch_size: (i + 1) * batch_size]
        x = tokenizer([p[1] for p in batch], return_tensors='pt', padding=True, truncation=True, max_length=256).to(model.device)
        with tokenizer.as_target_tokenizer():
            y = tokenizer([p[0] for p in batch], return_tensors='pt', padding=True, truncation=True, max_length=256).to(model.device)
        y.input_ids[y.input_ids == 0] = -100
        # вычисляем loss
        try:
            loss = model(
                input_ids=x.input_ids,
                attention_mask=x.attention_mask,
                labels=y.input_ids,
                decoder_attention_mask=y.attention_mask,
                return_dict=True
            ).loss
            # делаем шаг градиентного спуска
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except RuntimeError as e:
            print('error')
            loss = None
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            continue

        # печатаем скользящее среднее значение функции потерь
        losses.append(loss.item())
        if i % report_steps == 0:
            print('step', i, 'loss', np.mean(losses[-report_steps:]))
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# строим и сохраняем график loss
pd.Series(losses).ewm(100).mean().plot()
plt.yscale('log')
plt.savefig(os.path.join(save_path, 'loss0.png'))
plt.clf()



loss = None
optimizer.zero_grad(set_to_none=True)
optimizer = None
cleanup()

for p in model.parameters():
    # размораживаем все параметры
    p.requires_grad = True
     

optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad], 
    scale_parameter=False, 
    relative_step=False, 
    lr=1e-5,
    clip_threshold=1.0
)

batch_size = 4
report_steps = 500 
epochs = 3
losses = []

cleanup()
model.train()
for epoch in range(epochs):
    print('EPOCH', epoch)
    random.shuffle(all_pairs)
    for i in trange(0, int(len(all_pairs) / batch_size)):
        batch = all_pairs[i * batch_size: (i + 1) * batch_size]
        x = tokenizer([p[1] for p in batch], return_tensors='pt', padding=True, truncation=True, max_length=256).to(model.device)
        with tokenizer.as_target_tokenizer():
            y = tokenizer([p[0] for p in batch], return_tensors='pt', padding=True, truncation=True, max_length=256).to(model.device)
        y.input_ids[y.input_ids == 0] = -100
        # вычисляем функцию потерь
        try:
            loss = model(
                input_ids=x.input_ids,
                attention_mask=x.attention_mask,
                labels=y.input_ids,
                decoder_attention_mask=y.attention_mask,
                return_dict=True
            ).loss
            # делаем шаг градиентного спуска
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except RuntimeError as e:
            print('error')
            loss = None
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            continue

        # печатаем скользящее среднее значение функции потерь
        losses.append(loss.item())
        if i % report_steps == 0:
            print('step', i, 'loss', np.mean(losses[-report_steps:]))
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    bleu, chrf  = evaluate_bleu_chrf(model, tokenizer, test_data)
    print(f'After {epoch+1} epoch BLEU: {bleu:2.2f}, CHRF: {chrf:2.2f}')
    
# строим и сохраняем график loss    
pd.Series(losses).ewm(1000).mean().plot()
plt.yscale('log')
plt.savefig(os.path.join(save_path, 'loss1.png'))
plt.close()


