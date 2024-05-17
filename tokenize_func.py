
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from collections import Counter, defaultdict
from tqdm.auto import tqdm, trange

from copy import deepcopy
from heapdict import heapdict
import numpy as np 

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
    
    return res

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

def tokenize_fn(corpus):
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    all_sentences = corpus['Tatar']
    char_count = Counter()
    for text in tqdm(all_sentences):
        char_count.update(text)
    

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
            old_toks = word2toks[word]
            new_toks = replace_pair(old_toks, pair, new_token)
            word2toks[word] = new_toks
            wc = word_count[word]
            for old_pair in zip(old_toks[:-1], old_toks[1:]):
                delta[old_pair] -= wc
                if word in pair2word[old_pair]:
                    pair2word[old_pair].remove(word)
            for new_pair in zip(new_toks[:-1], new_toks[1:]):
                delta[new_pair] += wc
                pair2word[new_pair].add(word)
        for a_pair, a_delta in delta.items():
            if a_delta == 0:
                continue
            if a_pair not in hd:
                hd[a_pair] = 0
            hd[a_pair] -= a_delta
            
    return extra_vocab, extra_counts, extra_pairs

PUNCT = '.,-—:)(»«!?–/;„"“…*́№Ёҥ[]”^%+І=і•_􏰀²|}{#‘■>⁠’á<°\§\''
SPACE = '▁'