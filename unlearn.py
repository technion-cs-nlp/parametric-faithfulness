import os, sys, gc, json, copy, random, argparse, subprocess, shutil
from pprint import pprint

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from transformers import AutoTokenizer as TOK
from transformers import AutoModelForCausalLM as CLM

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from evaluate import completion_probabilities, answer_probabilities, complete, generation_fixed_cot
from data import FRCollator, cot_to_otfd, model_name_dict, load_or_generate_dataset_cots
from dataload import DATASETS
from util import set_random_seed

def memory_stats():
    print(torch.cuda.memory_allocated()/1024**2)
    print(torch.cuda.memory_reserved()/1024**2)


def run_lm_eval(model_path, log_path): 
  run_cmd = ["lm_eval","--model","hf",
    "--model_args", "pretrained={}",
    "--tasks", "mmlu",
    "--device","cuda:0",
    "--batch_size", "auto:4",
    "--num_fewshot=0",
    ]

  run_cmd[4] = run_cmd[4].format(model_path)

  result = subprocess.run(
      run_cmd,
      text=True,  # Return output as a string (not bytes)
      capture_output=True,  # Capture stdout and stderr
      check=True  # Raise CalledProcessError if the command fails
  )

  return result.stdout

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss

def compute_loss(model, oracle_model, inputs, loss_type='npo_grad_diff', ref_policy='fine_tuned', beta=0.1, npo_coeff=1.0, grad_diff_coeff=1.0, KL_coeff=1.0, return_outputs=False):
        ### Implement the NPO
        if loss_type == 'npo':
            forget_inputs, _ = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)

            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            loss = - 2 / beta * F.logsigmoid(beta * neg_log_ratios).mean()

        elif loss_type == 'npo_grad_diff':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs

            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 

            if ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits

                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = npo_coeff * forget_loss + grad_diff_coeff * retain_loss
            
        elif loss_type == 'npo_KL':
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss_current = get_batch_loss(outputs.logits, labels) 
            
            if ref_policy == 'fine_tuned':
                with torch.no_grad():
                    forget_outputs_oracle = oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
                    forget_logits_oracle = forget_outputs_oracle.logits
                    forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
                neg_log_ratios = forget_loss_current - forget_loss_oracle
            else:
                raise NotImplementedError
            forget_loss = -F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            # minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss = npo_coeff * forget_loss + KL_coeff * retain_loss

        return (loss, outputs) if return_outputs else loss

def compute_specificity(model, tokenizer, DH, specificity_split):
  specificity = []
  specificity_probs = []
  for instance in specificity_split:
      sp_c_b, sp_pr_b, sp_p_b = answer_probabilities(model, tokenizer, DH, instance['raw_instance']) # question_prefix
      specificity.append(sp_p_b)
      specificity_probs.append(sp_pr_b.tolist())
  
  return specificity, specificity_probs

def evaluate(model, tokenizer, DH, target, specificity_split, step_idx):
  model.eval()
  # (0) efficacy: how does the probability of the initial CoT change after unlearning
  # model, tokenizer, prefix, target
  unlearned_cot = target['cot']
  cot_prefix = DH.make_cot_prompt(target['raw_instance'])
  cot_probability = completion_probabilities(model, tokenizer, cot_prefix, [unlearned_cot])

  # (0.1) how does the probability of the _unlearned step_ change after unlearning
  unlearned_step = target['segmented_cot'][step_idx]
  previous_steps = target['segmented_cot'][:step_idx]

  if previous_steps:
      unlearned_step_prefix = '\n'.join([cot_prefix]+previous_steps)
  else:
      unlearned_step_prefix = cot_prefix # First cot step

  step_probability = completion_probabilities(model, tokenizer, unlearned_step_prefix, [unlearned_cot])

  # (1) faithfulness: how does the model perform wrt. unlearning target
  completion_after, probs_after, prediction_after = answer_probabilities(
  model, tokenizer, DH, target['raw_instance']) # question_prefix

  # (2) "specificity": currently, checks how the model performs on a heldout set of instances from the same dataset: (a) pred, (b) prob
  specificity_predictions, specificity_probabilities = compute_specificity(model, tokenizer, DH, specificity_split)

  # (3) new CoT: check how the model generated CoT looks like after unlearning
  new_cot = complete(model, tokenizer, DH.make_cot_prompt(target['raw_instance']))

  # (4) probability under new CoT (agreement before/after unlearning)
  new_cot_probs, _  = generation_fixed_cot(model, tokenizer, DH, target['raw_instance'], new_cot)

  return_dict = {
      'completion': completion_after,
      'probs': probs_after.tolist(),
      'prediction': prediction_after,
      
      'target_cot_step': unlearned_step,
      'target_cot_step_prefix': unlearned_step_prefix,
      
      'specificity_preds': specificity_predictions,
      'specificity_probs': specificity_probabilities,
      
      'new_cot': new_cot,
      'new_cot_probs': new_cot_probs.tolist(),

      'cot_prob': cot_probability.detach().cpu().float().numpy().tolist(),
      'cot_step_prob': step_probability.detach().cpu().float().numpy().tolist(),
  }

  return return_dict

def unlearn_single(model_id, tokenizer, args, target, step_idx, cots_train, cots_verify, dh, instance_idx):

    # Load models and dataset, fresh every time
    model = CLM.from_pretrained(model_id, 
                                torch_dtype=torch.bfloat16,
                                trust_remote_code=True,
                                device_map="auto"
                                )
    # Oracle model is frozen
    oracle_model = CLM.from_pretrained(model_id,
                                        torch_dtype=torch.bfloat16,
                                        trust_remote_code=True, 
                                        device_map="auto")
    device = model.device
    collator = FRCollator(tokenizer, device=device)

    dataset = cot_to_otfd(target, cots_train, tokenizer, strategy=args.strategy, stepwise=args.stepwise, step_idx=step_idx, pos=args.pos)

    NT = dataset.num_targets()
    print(f"Num targets: {NT}")
    print(target['segmented_cot'][step_idx])
    if NT <= 2:
         print("-"*20)
         print(f"Too few targets")
         print("-"*20)
         return {'unlearning_results': None, 'mmlu_results':None}

    EPOCHS = args.epochs
    batch_size = 1 # cfg.batch_size

    # For loop for each of the steps in a Cot
    steps_per_epoch = len(dataset) # Unlearning only one statement
    max_steps = EPOCHS * steps_per_epoch

    print(f"Training, #E={EPOCHS}")
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)

    if args.ff2:
        print("Setting only FF2 parameters to be optimized")
        # model.layers.[num].mlp.down_proj
        # mlp.down_proj.weight is the key for all considered models
        param_key = 'mlp.down_proj.weight'
        for name, param in model.named_parameters():
          if param_key in name:
            param.requires_grad = True
          else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=max_steps) # warmup_steps

    results_per_epoch = {}
    # Results before training, for comparison
    results_per_epoch[0] = evaluate(model, tokenizer, dh, target, cots_verify, step_idx=step_idx)
    
    for epoch in range(EPOCHS):
      model.train()
      optimizer.zero_grad()

      for step, batch in enumerate(train_dataloader):
        loss = compute_loss(model, oracle_model, batch, loss_type=args.method) 

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

      # Eval step
      epoch_result = evaluate(model, tokenizer, dh, target, cots_verify,step_idx=step_idx)
      results_per_epoch[(epoch+1)] = epoch_result


    # So, this is very ugly but I had to do it quickly and it works ¯\_(ツ)_/¯
    # Calls lm_eval_harness on a saved model checkpoint, extracts the performance from stdout
    #  and then deletes the checkpoint.
    if args.mmlu or args.gsm:
      mod = model_id.split("/")[-1]
      short_model = model_name_dict[mod]
      name = f"{instance_idx}_{step_idx}"
      resdir = f"chkp/{args.dataset}/{short_model}/"
      print(f"Instance and step idx: {name}")
      os.makedirs(resdir+name, exist_ok=True)
      model.save_pretrained(resdir+name, from_pt=True)
      tokenizer.save_pretrained(resdir+name)        

    # Delete model and clean cuda to free up space for lm eval
    del collator, train_dataloader, dataset, scheduler, optimizer, model, oracle_model
    gc.collect()
    torch.cuda.empty_cache()

    return_dict = {
        'unlearning_results': results_per_epoch,

    }

    if args.mmlu or args.gsm:
      logdir = resdir.replace("chkp", "gen_cap") + f"{args.lr}/"
      os.makedirs(logdir, exist_ok=True)

      print("Running evaluation from python")
      result = run_lm_eval(resdir + name, logdir + name)
      result_lines = result.split("\n")
      score_line = result_lines[-7]
      result_line_parts = score_line.split("|")
      assert result_line_parts[1].strip() == 'mmlu', "Error when retrieving scores"
      mmlu_acc, err = result_line_parts[-4], result_line_parts[-2]
      # print(f"Accuracy and error: {acc} +- {err}")
      key = 'mmlu_results' if args.mmlu else 'gsm8k_results'
      return_dict[key] = mmlu_acc

      print("Deleting model directory")
      shutil.rmtree(resdir + name, ignore_errors=False, onerror=None)

    return return_dict

def load_ids(fin, stepwise=False):
    ids = set()
    if os.path.exists(fin):
      with open(fin, 'r') as infile:
          for line in infile:
              jsonline = json.loads(line)
              id = jsonline['question']
              if stepwise:
                  id = f"{id}_{jsonline['step_idx']}"
              ids.add(id)
    return ids

def store(instance_info, fout):
    with open(fout, 'a') as outfile:
      outfile.write(json.dumps(instance_info)+"\n")

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-3-mini-4k-instruct', help="Model name (hf) or local path")
    parser.add_argument('--dataset', type=str, default='sports', 
                        help="Which dataset to use")
    parser.add_argument('--method', type=str, default='npo_KL', 
                        help="Which unlearning method to use")
    parser.add_argument('--strategy', type=str, default='segmented', 
                        help="Which unlearning strategy to use: full, segmented")
    parser.add_argument('--stepwise', action='store_false', help="Unlearn all steps, or one at a time.")
    parser.add_argument('--temperature', type=float, default=0.,
                        help="Sampling temperature for CoT generation")
    parser.add_argument('--seed', type=int, default=1001,
                        help="Random seed for the experiments")
    parser.add_argument('--epochs', type=int, default=5,
                        help="Number of unlearning epochs")
    parser.add_argument('--lr', type=float, default=5e-5,
                        help="Learning rate for NPO")
    parser.add_argument('--new_cot', action='store_true', help="Force generation of a fresh batch of CoTs.")
    parser.add_argument('--pos', action='store_true', help="Filter out function tokens in unlearning.")
    parser.add_argument('--ff2', action='store_true', help="Optimize only the ff2 layers")
    parser.add_argument('--ablation', action='store_true', help="Run on subsample of instances, change logging dir.")
    parser.add_argument('--mmlu', type=int, default=0, help="Evaluate MMLU on a subsample of --mmlu model instances post-unlearning")
    parser.add_argument('--gsm', type=int, default=0, help="Evaluate GSM8K on a subsample of --gsm model instances post-unlearning [WIP]")
    
    return parser

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = make_parser().parse_args()

    # Reproducibility
    seed = args.seed
    set_random_seed(seed)

    from huggingface_hub import login
    login("") # Set your user id

    model_id = args.model_name 
    tokenizer = TOK.from_pretrained(model_id)

    # Fix missing pad token if necessary
    if 'Phi' in model_id:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.eos_token

    # Data loading (needs tokenizer)
    # Question, CoT, Answer
    # Always unlearn either (a) one step of a cot or (b) entire cot, 
    #  then evaluate performance / probabilities on the same instance
    #  as well as a small held-out set
    DH = DATASETS[args.dataset]

    # Sentencize option, pos tag option
    cot_data = load_or_generate_dataset_cots(model_id=model_id, tokenizer=tokenizer,
                                              dataset_id=args.dataset,force_generate=args.new_cot, 
                                              sentencize=args.strategy == 'sentencize',
                                              temperature=args.temperature, seed=args.seed, atomic=args.atomic)

    # Shuffle data
    random.shuffle(cot_data)

    # "Specificity" split = same task, different instances
    N_verify = 20
    cots_train, cots_verify = cot_data[:-N_verify], cot_data[-N_verify:] #

    # Results / dataset / model_id
    mod = model_id.split("/")[-1]
    short_model = model_name_dict[mod]

    # Logging
    if args.mmlu:
      root_name = "mmlu_results"
      N_unlearn = args.mmlu
    elif args.gsm:
      root_name = "gsm8k_results"
      N_unlearn = args.gsm
    elif args.ablation:
      root_name = "ablation"
      N_unlearn = 30
    else:
      root_name = "final_results"
      N_unlearn = 250
    
    resdir = f"{root_name}/{args.dataset}/{short_model}/"
    os.makedirs(resdir, exist_ok=True)
    # No POS, no ff2, unlearn full
    logfile_name = f"{args.method}_{args.strategy}_s={args.stepwise}_lr={str(args.lr)}_rs={args.seed}_pos={args.pos}_ff2={args.ff2}.out"
    
    # Restore previous results 
    ids = load_ids(resdir + logfile_name, stepwise=args.stepwise)
    print(f"Ids so far: {len(ids)}")

    for idx, target in enumerate(cots_train[:N_unlearn]):
        # Clunky for now
        n_steps = 1
        if args.stepwise:
          n_steps = len(target['segmented_cot'])

        for step_idx in range(n_steps):
          
          check_id = target['id']
          if args.stepwise: check_id = f"{check_id}_{step_idx}"

          if check_id in ids: continue

          instance_info = {
              'id': target['id'],
              'question': target['question'],
              'step_idx': step_idx,
              'options': target['options'],
              'correct': target['correct_letter'],
              'initial_cot': target['cot'],
              'initial_cot_probs': target['cot_probs'],
              'initial_probs': target['nocot_probs'],
              'prediction': int(np.argmax(target['nocot_probs'])),
              'cot_prediction': int(np.argmax(target['cot_probs']))
          }

          if args.stepwise:
              instance_info['cot_step'] = target['segmented_cot'][step_idx]
              instance_info['segmented_cot'] = target['segmented_cot']

          # Logging: model name, cot source, dataset, strategy, stepwise, lr, seed, correct answer
          return_dict = unlearn_single(model_id, tokenizer, args, target, step_idx, cots_train, cots_verify, DH, idx)

          results = return_dict['unlearning_results']

          if results is None:
              # Too few targets, skipping instance
              continue

          # Log the results
          instance_info['unlearning_results'] = results
          if args.mmlu:
            instance_info['mmlu_results'] = return_dict['mmlu_results']
          if args.gsm:
            instance_info['gsm8k_results'] = return_dict['gsm8k_results']
          store(instance_info, resdir+logfile_name)
          del instance_info

if __name__ == '__main__':
    main()
