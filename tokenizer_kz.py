import os 
import pandas as pd
import numpy as np 
import sentencepiece_model_pb2 as model

from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from collections import Counter, defaultdict
from tqdm.auto import tqdm, trange
from tokenize_fn import tokenize_fn
from transformers.models.mbart50.tokenization_mbart50 import FAIRSEQ_LANGUAGE_CODES


def get_ru_toks(tat_tok):
    ru_toks = []
    ru_weights = []
    for t, w in token_to_others[tat_tok].items():
        ru_toks.append(t)
        ru_weights.append(w**2 / token_priors[t])
    ru_weights = np.array(ru_weights)
    ru_weights = ru_weights / (sum(ru_weights) + 1e-4)
    return ru_weights, ru_toks

PUNCT = '.,-—:)(»«!?–/;„"“…*́№Ёҥ[]”^%+І=і•_􏰀²|}{#‘■>⁠’á<°\§\''
SPACE = '▁'


cur_path = os.path.dirname(os.path.realpath(__file__))
corpus = pd.read_csv(os.path.join(cur_path, 'data/all_data.csv'))
corpus = corpus.dropna(ignore_index=True)
extra_vocab, extra_counts, extra_pairs = tokenize_fn(corpus)
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
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