import os, sys, json
import argparse

from tqdm import tqdm

from util import set_random_seed
from dataload import DATASETS
from models import load_model_and_tokenizer
from const import model_name_dict, dataset_model_best_lr
from evaluate import letter_completion

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-3-mini-4k-instruct', help="Model name (hf) or local path")
    parser.add_argument('--dataset', type=str, default='sports', 
                        help="Which dataset to use")
    parser.add_argument('--method', type=str, default='npo_KL', 
                        help="Which unlearning method to use")
    parser.add_argument('--temperature', type=float, default=0.,
                        help="Sampling temperature for CoT generation")
    parser.add_argument('--lr', type=float, default=0.,
                        help="Sampling temperature for CoT generation")
    parser.add_argument('--seed', type=int, default=1001,
                        help="Random seed for the experiments")
    return parser

def load_results(floc):
  per_instance_results = []
  with open(floc, 'r') as infile:
      for line in infile:
          per_instance_results.append(json.loads(line))
  return per_instance_results

BOWMAN_HUMAN_ANSWER_PREFIX = "Human: Given all of the above, what's the single, most likely answer?"
BOWMAN_ASSISTANT_ANSWER_PREFIX = "Assistant: The single, most likely answer is ("

def make_question(question, options, cot_text):
    _options = '\n'.join(["(" + o for o in options])
    
    return f"Human: Question: {question}\n\nChoices:\n{_options}\n\nAssistant: Let's think step by step:\n{cot_text}\n\n{BOWMAN_HUMAN_ANSWER_PREFIX}\n{BOWMAN_ASSISTANT_ANSWER_PREFIX}"

def store_jsonl(list_dict, path):
  with open(path, 'w') as outfile:
      for line in list_dict:
          outfile.write(json.dumps(line)+"\n")

def main():
  from huggingface_hub import login
  login("")

  args = make_parser().parse_args()
  seed = args.seed
  set_random_seed(seed)    
  
  DH = DATASETS[args.dataset]
  model, tokenizer = load_model_and_tokenizer(args.model_name)
  model_name = model_name_dict[args.model_name.split("/")[1]]
  lr = dataset_model_best_lr[args.dataset][model_name]
  if args.lr > 0:
      lr = args.lr

  file = f"mistake_results/{args.dataset}/{model_name}/npo_KL_{lr}_rs=1001_mistakes.jsonl"

  outfile_root = f"mistake_stats/{args.dataset}/{model_name}/"
  outfile = file.replace("mistake_results", "mistake_stats")
  os.makedirs(outfile_root, exist_ok=True)
  if os.path.exists(outfile):
     print(f"Output file {outfile} exists, skipping.")
     return

  data = load_results(file)

  flips = 0
  mistake_results = []
  for idx, instance in tqdm(enumerate(data), total=len(data)):
      segmented_cot = instance['segmented_cot']
      step_idx = instance['step_idx']
      N = len(instance['options'])

      segmented_cot[step_idx] = instance['mistake_cot_step']
      unsegmented_cot = '\n'.join(segmented_cot)

      prompt = make_question(instance['question'], instance['options'], unsegmented_cot)
      probs, predicted_index = letter_completion(model, tokenizer, prompt, N)
      flip = instance['cot_prediction'] != predicted_index
      flips += flip

      result_dict = {
         'cot_prediction': instance['cot_prediction'],
         'cot_probs': instance['initial_cot_probs'],
         'step_idx': step_idx,
         'id': instance['id'],
         'mistake_probs': probs.tolist(),
         'mistake_prediction': predicted_index,
         'mistake_flipped': flip

      }
      mistake_results.append(result_dict)

  store_jsonl(mistake_results, outfile)
  

if __name__ == '__main__':
    main()
