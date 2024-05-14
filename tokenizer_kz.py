import os 
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import json
import torch
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from collections import Counter, defaultdict
from tqdm.auto import tqdm, trange
import random
import re
from copy import deepcopy
from heapdict import heapdict
import numpy as np 

from transformers.models.mbart50.tokenization_mbart50 import FAIRSEQ_LANGUAGE_CODES

import sentencepiece_model_pb2 as model

def get_words(text, tokenizer, verbose=False):
    toks = tokenizer.tokenize(text)
    words = []
    word = []
    prev_punct = False
    for tok in toks:
        is_punct = tok.lstrip(SPACE) and all(c in PUNCT for c in tok.lstrip(SPACE))
        if tok.startswith(SPACE) or prev_punct != is_punct:
            if word:
                words.append(word)
            word = []
        word.append(tok)
        prev_punct = is_punct
    if word:
        words.append(word)
    if verbose:
        print(words)
    res = words
    # assert tokenizer.decode([tok for t in res for tok in t]) == text
    return res
def get_ru_toks(myv_tok):
    ru_toks = []
    ru_weights = []
    for t, w in token_to_others[myv_tok].items():
        ru_toks.append(t)
        ru_weights.append(w**2 / token_priors[t])
    ru_weights = np.array(ru_weights)
    ru_weights = ru_weights / (sum(ru_weights) + 1e-4)
    return ru_weights, ru_toks

def replace_pair(old_tokens, pair, new_token):
    result = []
    prev = old_tokens[0]
    for tok in old_tokens[1:]:
        if (prev, tok) == pair:
            result.append(new_token)
            prev = None
        else:
            if prev is not None:
                result.append(prev)
            prev = tok
    if prev is not None:
        result.append(prev)
    return result

tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
#get file path
cur_path = os.path.dirname(os.path.realpath(__file__))
corpus = pd.read_csv(os.path.join(cur_path, 'data/all_data.csv'))
corpus = corpus.dropna(ignore_index=True)
all_sentences = corpus['Tatar']
char_count = Counter()
for text in tqdm(all_sentences):
    char_count.update(text)
PUNCT = '.,-—:)(»«!?–/;„"“…*́№Ёҥ[]”^%+І=і•_􏰀²|}{#‘■>⁠’á<°\§\''
SPACE = '▁'

word_count = Counter()
word2toks = {}
for text in tqdm(all_sentences):
    for word_toks in get_words(text, tokenizer):
        word = ''.join(word_toks)
        word_count[word] += 1
        word2toks[word] = word_toks
print(len(word_count))
word_count2 = deepcopy(word_count)
word2toks2 = deepcopy(word2toks)
word_count = deepcopy(word_count2)
word2toks = deepcopy(word2toks2)

pairs_count = Counter()
pair2word = defaultdict(set)
for w, c in tqdm(word_count.items(), total=len(word_count)):
    enc = word2toks[w]
    for pair in zip(enc[:-1], enc[1:]):
        pairs_count[pair] += c
        pair2word[pair].add(w)
        
hd = heapdict()
for w, c in pairs_count.items():
    hd[w] = -c
    
steps = 100000
min_count = 15  
extra_vocab = []
extra_counts = []
extra_pairs = []

for _ in trange(steps):
    pair, c = hd.peekitem()
    c = -c
    
    if c < min_count:
        break
    new_token = ''.join(pair) 
    extra_vocab.append(pair)
    extra_counts.append(c)
    extra_pairs.append(pair)
    
    delta = Counter()
    for word in list(pair2word[pair]):
        # calculate old and new ways to tokenize the word
        old_toks = word2toks[word]
        # new_toks = " ".join(old_toks).replace(' '.join(pair), new_token).split(" ")
        new_toks = replace_pair(old_toks, pair, new_token)
        word2toks[word] = new_toks
        wc = word_count[word]
        # update the index concerning the tokens of the word
        for old_pair in zip(old_toks[:-1], old_toks[1:]):
            #pairs_count[old_pair] -= wc
            delta[old_pair] -= wc
            if word in pair2word[old_pair]:
                pair2word[old_pair].remove(word)
        for new_pair in zip(new_toks[:-1], new_toks[1:]):
            # pairs_count[new_pair] += wc
            delta[new_pair] += wc
            pair2word[new_pair].add(word)
    # update the heap
    for a_pair, a_delta in delta.items():
        if a_delta == 0:
            continue
        if a_pair not in hd:
            hd[a_pair] = 0
        hd[a_pair] -= a_delta
tokenizer.save_pretrained(os.path.join(cur_path,'tokenizers/old_tokenizer'))
m = model.ModelProto()
m.ParseFromString(open(os.path.join(cur_path,"tokenizers/old_tokenizer/sentencepiece.bpe.model"), "rb").read())
scores = [p.score for p in m.pieces]
min_score = min(scores)
epsilon = 1e-4

for i, pair in enumerate(extra_vocab):
    new_token = model.ModelProto().SentencePiece()
    new_token.piece = ''.join(pair)
    new_token.score = min_score - epsilon * (i+1)
    m.pieces.append(new_token)
    
with open(os.path.join(cur_path,"tokenizers/old_tokenizer/sentencepiece.bpe.model"), 'wb') as f:
    f.write(m.SerializeToString())


new_tokenizer = MBart50Tokenizer.from_pretrained(
    os.path.join(cur_path,"tokenizers/old_tokenizer/sentencepiece.bpe.model"), 
    additional_special_tokens = tokenizer.additional_special_tokens
)

print(f'New tokenizer has {new_tokenizer.vocab_size} tokens')
 
print(new_tokenizer.vocab_size)
print(new_tokenizer.vocab_size)
print(new_tokenizer.additional_special_tokens_ids)

token_priors = Counter()
token_to_others = defaultdict(Counter)

all_pairs = zip(corpus['Tatar'], corpus['Russian'])
for myv, ru in tqdm(all_pairs):
    myv_toks = new_tokenizer.convert_tokens_to_ids(new_tokenizer.tokenize(myv))
    ru_toks = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ru))
    token_priors.update(ru_toks)
    for myv_tok in myv_toks:
        token_to_others[myv_tok].update(ru_toks)
     


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
model.resize_token_embeddings(len(new_tokenizer))

n_extra = len(extra_vocab)
print(n_extra)
old_vocab_size = len(tokenizer.sp_model) + 1

for old_token_id in range(old_vocab_size, len(tokenizer)):
    old_token = tokenizer.convert_ids_to_tokens(old_token_id)
    new_token_id = new_tokenizer.convert_tokens_to_ids(old_token)
    
    print(old_token_id, old_token, new_token_id)
    # model.model.shared.weight.data[i + n_extra] = model.model.shared.weight.data[i]
    model.model.shared.weight.data[new_token_id] = model.model.shared.weight.data[old_token_id]
     

for i in trange(n_extra):
    myv_tok = i + old_vocab_size
    ru_weights, ru_toks = get_ru_toks(myv_tok)
    if len(ru_toks) > 0:
        new_embedding = (model.model.shared.weight.data[ru_toks].T * ru_weights).sum(1)
        model.model.shared.weight.data[myv_tok] = new_embedding
        
        
model.save_pretrained(os.path.join(cur_path,'tokenizers/new_tokenizer'))
new_tokenizer.save_pretrained(os.path.join(cur_path,'tokenizers/new_tokenizer'))