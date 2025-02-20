import random
import torch
import transformers
import numpy as np

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name, half=True):
  trust_remote_code=True
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
  
  half_dtype = torch.bfloat16
  model = AutoModelForCausalLM.from_pretrained(model_name,
                                              torch_dtype=half_dtype,
                                              device_map="auto",
                                              trust_remote_code=True)
  return model, tokenizer
