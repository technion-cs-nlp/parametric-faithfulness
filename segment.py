import nltk
import spacy
import torch
import random

import numpy as np

from dataclasses import dataclass

TARGET_TAGS = set([
    'VERB',
    'NUM',
    'ADJ',
    'NOUN',
    'PROPN',
])

@dataclass
class Word:
    word: str
    pos: str
    span_start: int
    span_end: int

    def is_content(self):
      return self.pos in TARGET_TAGS

WHITESPACE_CHARS = {
    'meta-llama/Meta-Llama-3-8B-Instruct': 'Ġ',
    'microsoft/Phi-3-mini-4k-instruct': '▁',
    'mistralai/Mistral-7B-Instruct-v0.2': '▁',
    'meta-llama/Llama-3.2-3B-Instruct': 'Ġ',
}

def sentencize(text):
    return nltk.sent_tokenize(text)

def pos_tag(text, nlp):
    doc = nlp(text)
    return [(w.text, w.pos_) for w in doc]

def words_to_token_spans(wpos, tokens, W):
    # Filter out space tokens
    toks_pos = [(t, p) for t, p in wpos if p != "SPACE"]

    # Iterate over words
    i = 0
    cur_word, cur_pos = toks_pos[i]
    
    word_start = 0
    words = []

    for j, subword in enumerate(tokens):
        # if W == subword: continue
    
        if W in subword: # new word
            word_start = j 
    
        # Convert span to string, filter out whitespace
        span = tokens[word_start:j+1]
        # print(span[0], dir(span[0]))
        span = [e.replace(W, "") for e in span]
        cur = ''.join(span)
    
        # equality check
        if cur == cur_word:
    
            # Store span
            w = Word(cur_word, cur_pos, word_start, j+1)
            words.append(w)
    
            # Goto next
            i += 1
            if i >= len(toks_pos): break
            
            cur_word, cur_pos = toks_pos[i]
            word_start = j
    
    
    if not len(words) == len(toks_pos):
      print("Length mismatch")  
      print(words)
      print("-"*30)
      print([t for t,p in toks_pos])

    return words

def align_cot_to_pos(cot_step_text, tokenizer, model_id, nlp):    
    W = WHITESPACE_CHARS[model_id]
    w_p = pos_tag(cot_step_text, nlp)
    pretokenized_text = [f" {w}" for w,_ in w_p] # Take words, prefix whitespace
    tokens = tokenizer.tokenize(pretokenized_text, is_split_into_words=True, add_special_tokens=False)
    indices = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

    return indices, words_to_token_spans(w_p, tokens, W)
