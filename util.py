import os, json, random
import torch

import numpy as np

from const import LETTERS, datasets, models, dataset_model_best_lr

def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sort_key(t):
    p,a,r,t = t.split(",")
    return (p,a)

def load_results(floc, sample=0, N_ablation=30):
    per_instance_results = []
    with open(floc, 'r') as infile:
        for idx, line in enumerate(infile):
            if sample and idx >= N_ablation: break
            per_instance_results.append(json.loads(line))
    return per_instance_results

def load_specific_results(model, dataset, lr, path_root="results", type="sentencize", method='npo_KL', rs=1001, pos=True, ff2=True):

    floc = f'{path_root}/{dataset}/{model}/{method}_{type}_s=True_lr={lr}_rs={rs}_pos={pos}_ff2={ff2}.out'
    if not os.path.exists(floc):
        print("File doesn't exist")

    per_instance_results = load_results(floc)
    return per_instance_results

def load_best_full_lrs(path_root, type="sentencize", method='npo_KL', rs=1001, ff2=True, pos=True):
    results = {}
    for dataset in datasets:
        for model in models:
            k = f"{dataset}_{model}"
            lr = dataset_model_best_lr[dataset][model] # best_model_dataset_lr[k]

            floc = f'{path_root}/{dataset}/{model}/{method}_{type}_s=True_lr={lr}_rs={rs}_pos={pos}_ff2={ff2}.out'
    
            per_instance_results = load_results(floc)

            if per_instance_results:
                key = f"{dataset},{model},{method},{lr}"
                results[key] = per_instance_results
    return results

def store_jsonl(list_dict, path):
  with open(path, 'w') as outfile:
    for line in list_dict:
      outfile.write(json.dumps(line)+"\n")

def list_learning_rates(path_root):
    dataset_model_lrs = {}
    for dataset in datasets:
        for model in models:
            key = f"{dataset}_{model}"
            lrs = set()
            fdir = f'{path_root}/{dataset}/{model}/'
            for file in os.listdir(fdir):
                parts = file.split('_')
                lr = float(parts[4].replace('lr=', ''))
                lrs.add(lr)
            dataset_model_lrs[key] = sorted(list(lrs))
    return dataset_model_lrs

def unique_instances(result_dict):
    unique_ids = set()
    for an_inst in result_dict:
        unique_ids.add(an_inst['question'])
    return len(unique_ids)

def filter_for_agreement_after(results):
    filtered_results = []
    for r in results:
        stepwise_results = r['unlearning_results']
        if all([
            np.argmax(rr['probs']) == np.argmax(rr['new_cot_probs']) for _, rr in stepwise_results.items()
        ]):
            filtered_results.append(r)
    return filtered_results

def filter_for_agreement(results):
    return [r for r in results if r['prediction'] == r['cot_prediction']]

def filter_for_correctness(results):
    return [r for r in results if r['prediction'] == r['cot_prediction'] and r['prediction'] == LETTERS.index(r['correct'])]

def group_results(some_results):
    grouped_results = {}
    for a_key, results in some_results.items(): # correct_results
        dataset, model, _, _ = a_key.split(",")
        single_group = {}
        for instance in results:
            q = instance['question']
            if q not in single_group:
                single_group[q] = []
            single_group[q].append(instance)
        grouped_results[a_key] = single_group
    return grouped_results

def renorm(ps):
    norm_p = [p/sum(ps) for p in ps]
    return norm_p

