import os, sys

models = {
    'meta-llama/Meta-Llama-3-8B-Instruct': True,
    'microsoft/Phi-3-mini-4k-instruct': False,
    'mistralai/Mistral-7B-Instruct-v0.2': True,
    'meta-llama/Llama-3.2-3B-Instruct': False,
}

datasets = ['arc-challenge', 'openbook', 'sports', 'sqa'] # 'aqua',

lrs = [5e-5, 3e-5, 5e-6]

model_to_short = {
  'microsoft/Phi-3-mini-4k-instruct': 'Phi-3',
  'meta-llama/Meta-Llama-3-8B-Instruct': 'LLaMA-3',
  'meta-llama/Llama-3.2-3B-Instruct': 'LLaMA-3-3B',
  'mistralai/Mistral-7B-Instruct-v0.2': 'Mistral-2',
}

ablate_ff2_small = 'ablate_ff2.job'
ablate_ff2_large = 'ablate_ff2_large.job'

small_script = 'ul_step_pos_ff2.job'
big_script = 'ul_step_pos_ff2_L2.job'

method = 'npo_KL'

# sbatch --job-name=phi3-sqa-stepwise-lr5e5 ul_seg_step_pos.job microsoft/Phi-3-mini-4k-instruct sqa 5e-5 npo_grad_diff
script_template = "sbatch --job-name={}-{}-stepwise-lr{} {} {} {} {} {}"

for dataset in datasets:
  for model, big in models.items():
    print("-"*30)
    print(f"{dataset} => {model}")
    print("-"*30)
    for lr in [3e-5]: # lrs
      mod = model_to_short[model]
      script = big_script if big else small_script
      print(script_template.format(
        mod, dataset, lr, script, model, dataset, lr, method
      ))
