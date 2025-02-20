import os, json, copy, random
from pathlib import Path

import random
import spacy
import datasets
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from evaluate import generate_dataset_cots
from segment import align_cot_to_pos


IGNORE_IDX = -100

model_name_dict = {
    'Phi-3-mini-4k-instruct': 'Phi-3',
    'Meta-Llama-3-8B-Instruct': 'LLaMA-3',
    'Llama-3.2-3B-Instruct': 'LLaMA-3-3B',
    'Meta-Llama-3-70B-Instruct': 'LLaMA-3-70B',
    'Mistral-7B-Instruct-v0.2': 'Mistral-2',
    'phi-2': 'Phi-2',
    'llama-2-hf': 'LLaMA-2',
    'Llama-2-7b-chat-hf': 'LLaMA-2',
    'Mistral-7B-Instruct-v0.1': 'Mistral-1',
}

def cache_cots(dataset_cots, root, model_id, dataset_id, seed, temp):
  floc = f"{root}/{dataset_id}/{model_id}_s={seed}_t={temp}_cots.jsonl"
  dir = f"{root}/{dataset_id}"
  os.makedirs(dir, exist_ok=True)
  with open(floc, 'w') as outfile:
    for line in dataset_cots:
      outfile.write(json.dumps(line) + "\n")

def load_or_generate_dataset_cots(model_id, tokenizer, dataset_id, seed, temperature, force_generate=False, sentencize=True, atomic=False):
    root = 'final_cot' if not atomic else 'atomic_cot'
    temp = f"{temperature:.{2}}"
    short_model_id = model_id.split("/")[-1]
    floc = f"{root}/{dataset_id}/{short_model_id}_s={seed}_t={temp}_cots.jsonl"
    if not os.path.exists(floc) or force_generate:
        dataset_cots = generate_dataset_cots(model_id, tokenizer, dataset_id, temperature=temperature, sentencize=sentencize)
        cache_cots(dataset_cots, root,  short_model_id, dataset_id, seed, temp)
        # Store dependent on seed/temperature
        return dataset_cots
    else:
        return load_jsonl(floc)

def left_pad_sequence(vector_list, padding_value):
    # print(vector_list)
    N = len(vector_list)
    T = max([len(f) for f in vector_list])
    # print(N, T)

    ret = torch.full((N,T), fill_value=padding_value, dtype=vector_list[0].dtype)
    for i, v_i in enumerate(vector_list):
        L = len(v_i)
        ret[i, T-L:] = v_i # Leave padding on left
    return ret


def qcot_encoder(tokenizer, question, cot, pos_filter=False, nlp=None):
    question += "\n\n"
    question_tokens = tokenizer.encode(question, add_special_tokens=False, return_tensors='pt')[0]

    # input = question + cot # newlines are already prepended
  
    if pos_filter:
      cot_tokens, word_to_span = align_cot_to_pos(cot, tokenizer, tokenizer.name_or_path, nlp=nlp)
    else:
      cot_tokens = tokenizer.encode(cot, add_special_tokens=False, return_tensors='pt').squeeze()

    encoded_input = torch.cat((question_tokens, cot_tokens), dim=0)

    labels = encoded_input.clone() # IMPORTANT: Shift by one when computing loss
    attention_mask = torch.ones_like(encoded_input)

    # Do not unlearn the question, only the CoT
    QL = len(question_tokens)
    for i in range(QL): labels[i] = IGNORE_IDX
    if pos_filter:
      for w in word_to_span:
        if not w.is_content():
          # Mask out function words from loss
          labels[QL + w.span_start: QL + w.span_end] = IGNORE_IDX

    L = (labels != IGNORE_IDX).sum()
    return encoded_input, labels, attention_mask, L

#On-the-Fly
class OTFDataset(Dataset):
    def __init__(self, forget, retain):
        self.forget = forget # Either a single sentence or a set of atomic statements
        self.retain = retain

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        # 1. Take a forget sample at the given index
        forget_sample = self.forget[idx]

        # 2. Take a (random?) retain sample 
        retain_sample = self.retain[idx]

        # Tokenization, padding etc done in collator
        return [forget_sample, retain_sample]

class SegmentOTFDataset(Dataset):
    def __init__(self, forget, retain, tokenizer, stepwise=False, pos_filter=False, step_idx=0):
        self.forget = forget # Either a single sentence or a set of atomic statements
        self.retain = retain
        self.tokenizer = tokenizer
        self.stepwise = stepwise
        self.step = step_idx
        self.retain_idx = 0
        self.min_targets = 2
        self.pos_filter = pos_filter
        self.NLP = None
        if pos_filter:
            self.NLP = spacy.load("en_core_web_sm", disable=['ner'])

        self._forget_sample = None
        self._retain_sample = None

    def __len__(self):
        # If stepwise, we unlearn only one step for each dataset instantiation
        return len(self.forget) if not self.stepwise else 1

    def num_targets(self):
        print(f"L = {len(self)}")
        total_targets = 0
        for idx in range(len(self)):
            (_, L, _), (_, _, _) = self[idx]
            total_targets += self.targets(L)
            print(total_targets)
        return total_targets

    @staticmethod
    def targets(aten):
        return (aten != IGNORE_IDX).sum()

    def __getitem__(self, idx):
        # 1. Take a forget sample at the given index
        # If we go stepwise, we unlearn one step of a cot in time, so
        # index of forget is hardcoded
        idx = self.step if self.stepwise else idx
        # This is a dict
        forget_sample = self.forget[idx]
        if 'prefix' in forget_sample:
          prompt = '\n'.join(
              [forget_sample['prompt'], forget_sample['prefix']]
            )
        else:
            prompt = forget_sample['prompt']

        self._forget_sample = prompt + "\nTarget:" + forget_sample['completion'] # Bookkeeping
        (E_f, L_f, A_f, T_f) = qcot_encoder(self.tokenizer, prompt, forget_sample['completion'], pos_filter=self.pos_filter, nlp=self.NLP)

        # 2. Take a sufficiently long enough (random?) retain sample
        N_retain = len(self.retain) 
        found_retain = False
        print(f"Looking for a retain sample, idx={idx}")
        for an_idx in range(N_retain):
          # Rotate over samples
          cur_retain_idx = (self.retain_idx + an_idx) % N_retain
          retain_sample = self.retain[cur_retain_idx]
          if 'prefix' in retain_sample:
            retain_prompt = '\n'.join(
                [retain_sample['prompt'], retain_sample['prefix']]
              )
          else:
            retain_prompt = retain_sample['prompt']
          
          self._retain_sample = retain_prompt + "\nTarget:" + retain_sample['completion']
          (E_r, L_r, A_r, T_l) = qcot_encoder(self.tokenizer, retain_prompt, retain_sample['completion'], pos_filter=self.pos_filter, nlp=self.NLP) # Maybe don't use a CoT sample here (~demonstration)
          if T_l > self.min_targets:
              found_retain = True
              print(f"Using sample #{cur_retain_idx}, N={T_l}") # \n{retain_sample}
              break

        if not found_retain:
            raise ValueError("No long enough retain samples")

        # Padding etc done in collator
        return (E_f, L_f, A_f), (E_r, L_r, A_r)

class FRCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
        # print(self.pad_token_id)

    def __call__(self, samples):
        # bsz, [F, R], (3)

        F, R = zip(*samples)
        
        # Alt: turn this into a matrix and then slice rows
        # Alt: transpose somehow? but it's 2x2x3
        E_fb = [f[0] for f in F]
        L_fb = [f[1] for f in F]
        A_fb = [f[2] for f in F]
        
        E_rb = [r[0] for r in R]
        L_rb = [r[1] for r in R]
        A_rb = [r[2] for r in R]

        # print(E_rb)
        # print(L_rb)
        # print(A_rb)
        
        E_fib = left_pad_sequence(E_fb, padding_value=self.pad_token_id)
        L_fib = left_pad_sequence(L_fb, padding_value=IGNORE_IDX)
        A_fib = left_pad_sequence(A_fb, padding_value=0) # 0 > ignore for attention

        E_rib = left_pad_sequence(E_rb, padding_value=self.pad_token_id)
        L_rib = left_pad_sequence(L_rb, padding_value=IGNORE_IDX)
        A_rib = left_pad_sequence(A_rb, padding_value=0) # 0 > ignore for attention

        E_fib = E_fib.to(self.device)
        L_fib = L_fib.to(self.device)
        A_fib = A_fib.to(self.device)

        E_rib = E_rib.to(self.device)
        L_rib = L_rib.to(self.device)
        A_rib = A_rib.to(self.device)

        return (E_fib, L_fib, A_fib), (E_rib, L_rib, A_rib)

# Forget-Retain
class DualCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        forgets, retain = zip(*samples)
        # Tokenize, check max length, stack together
        forget_batch = self.tokenizer.batch_encode_plus(forgets, pad_to_max_length=True)
        retain_batch = self.tokenizer.batch_encode_plus(retain, pad_to_max_length=True)
        return forget_batch, retain_batch

COT_ROOT = Path('cots/full')
SEGMENT_COT_ROOT = Path('cots/sentencized')

def load_jsonl(floc):
    data = []
    with open(floc, 'r') as infile:
        for line in infile:
            data.append(json.loads(line))
    return data

# sports_Phi-3-mini-4k-instruct_cots.json
def load_cotfiles(model='Phi-3-mini-4k-instruct', dataset='sports', root=COT_ROOT):
    short_model = model_name_dict[model]
    cot_data = []
    with open(root / dataset / f"{short_model}_cots.jsonl") as infile:
        for line in infile:
            cot_data.append(json.loads(line))
    return cot_data

def make_targets(cot_dict, segment=lambda d: [(d['cot'], None)]):
    DELIM = '\n\n'
    # We want to have prompt & completion fields
    prompt = cot_dict['cot_prompt']
    ret = []

    # Segment into components to be unlearned
    completions = segment(cot_dict)
    for completion, prefix in completions:
        # So far, only full
        if prefix is not None:
            ret.append(
                {'prompt': prompt, 
                  'completion': completion,
                  'prefix': '\n'.join(prefix)
                })
        else:
            ret.append(
                {'prompt': prompt, 
                 'completion': completion
                })
    return ret

# strategies = full, newline, sentencize, atomic statements
def cot_to_otfd(target, all, tokenizer, n=4, strategy='full', stepwise=True, step_idx=0, pos=False):
    # Target cot, other cots, generate a dataset
    if strategy == 'full':
        all = copy.deepcopy(all)
        all.remove(target)
        
        target = make_targets(target)

        retain = random.sample(all, n)
        # Format into dict for segment dataset by transforming content into prompt & completion
        retain = [rr for r in retain for rr in make_targets(r)]

        return SegmentOTFDataset(target, retain, tokenizer, stepwise, pos_filter=pos)
    
    elif strategy == 'sentencize':
        all = copy.deepcopy(all)
        all.remove(target)

        def segment(d):
            cot_segments = d['segmented_cot']
            outs = []
            prefixes = []
            for s in cot_segments:
                outs.append((s, list(prefixes)))
                prefixes.append(s)
            return outs

        targets = make_targets(target, segment=segment)

        retain = random.sample(all, n)
        # Format into dict for segment dataset by transforming content into prompt & completion
        retain = [rr for r in retain for rr in make_targets(r, segment=segment)]

        return SegmentOTFDataset(targets, retain, tokenizer, stepwise, step_idx=step_idx, pos_filter=pos)

    return None
