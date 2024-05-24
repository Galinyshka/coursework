import os 
import pandas as pd
import numpy as np 
import sentencepiece_model_pb2 as model

from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from collections import Counter, defaultdict
from tqdm.auto import tqdm, trange
from transformers.models.mbart50.tokenization_mbart50 import FAIRSEQ_LANGUAGE_CODES
from tokenize_fn import tokenize_fn

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
  
# сохранили токенайзер с новыми токенами     
with open(os.path.join(cur_path,"tokenizers/old_tokenizer/sentencepiece.bpe.model"), 'wb') as f:
    f.write(m.SerializeToString())


new_tokenizer = MBart50Tokenizer.from_pretrained(
    os.path.join(cur_path,"tokenizers/old_tokenizer/sentencepiece.bpe.model"), 
    additional_special_tokens = tokenizer.additional_special_tokens
)

print(f'New tokenizer has {new_tokenizer.vocab_size} tokens')


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
    #tokenizer.added_tokens_encoder = {}
    
#print(new_tokenizer.vocab_size)
fix_tokenizer(new_tokenizer)
#print(new_tokenizer.vocab_size)
#print(new_tokenizer.additional_special_tokens_ids)

token_priors = Counter()
token_to_others = defaultdict(Counter)

all_pairs = zip(corpus['Tatar'], corpus['Russian'])
for tat, ru in tqdm(all_pairs):
    tat_toks = new_tokenizer.convert_tokens_to_ids(new_tokenizer.tokenize(tat))
    ru_toks = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ru))
    token_priors.update(ru_toks)
    for tat_tok in tat_toks:
        token_to_others[tat_tok].update(ru_toks)
     


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
     
# инициализируем вес нового специального токена tt_XX
model.model.shared.weight.data[new_tokenizer.convert_tokens_to_ids('tt_XX')] = (
    model.model.shared.weight.data[tokenizer.convert_tokens_to_ids('tr_TR')] * 0.25
    + model.model.shared.weight.data[tokenizer.convert_tokens_to_ids('kk_KZ')] * 0.3
    + model.model.shared.weight.data[tokenizer.convert_tokens_to_ids('az_AZ')] * 0.25
    + model.model.shared.weight.data[tokenizer.convert_tokens_to_ids('ru_RU')] * 0.2
)

for i in trange(n_extra):
    tat_tok = i + old_vocab_size
    ru_weights, ru_toks = get_ru_toks(tat_tok)
    if len(ru_toks) > 0:
        new_embedding = (model.model.shared.weight.data[ru_toks].T * ru_weights).sum(1)
        model.model.shared.weight.data[tat_tok] = new_embedding
        
        
model.save_pretrained(os.path.join(cur_path,'tokenizers/new_tokenizer'))
new_tokenizer.save_pretrained(os.path.join(cur_path,'tokenizers/new_tokenizer'))