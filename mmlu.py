from lm_eval import simple_evaluate

from const import model_name_to_path

def evaluate_mmlu_local(model_path):
    mmlu_eval_results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=["mmlu"],
        batch_size='auto:4', # Batch size selection can be automated by setting the flag to auto. This will perform automatic detection of the largest batch size that will fit on your device. On tasks where there is a large difference between the longest and shortest example, it can be helpful to periodically recompute the largest batch size, to gain a further speedup. To do this, append :N to above flag to automatically recompute the largest batch size N times.
        device="auto",
        num_fewshot=0, # 5
    )
    # print("Evaluation Results:")
    # print(mmlu_eval_results['results']['mmlu']['acc,none'])
    return mmlu_eval_results

if __name__ == "__main__":
    model_results = {}
    for model in model_name_to_path.values():
      result = evaluate_mmlu_local(model)
      model_results[model] = result['results']['mmlu']['acc,none']
    from pprint import pprint
    pprint(model_results)

#{'meta-llama/Llama-3.2-3B-Instruct': 0.6039025779803446,
# 'meta-llama/Meta-Llama-3-8B-Instruct': 0.6385130323315767,
# 'microsoft/Phi-3-mini-4k-instruct': 0.6990457199829084,
# 'mistralai/Mistral-7B-Instruct-v0.2': 0.5901580971371599}
# 00:37:55 min to evaluate all base models on MMLU

# GSM8k results
# Phi-3
# |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
# |-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
# |gsm8k|      3|flexible-extract|     0|exact_match|↑  |0.6914|±  |0.0127|
# |     |       |strict-match    |     0|exact_match|↑  |0.1312|±  |0.0093|

# Meta-llama/Meta-Llama-3-8B-Instruct
# Submitted batch job 71236427
# Meta-llama/Meta-Llama-3.2-3B-Instruct
# Submitted batch job 71236430
# mistralai/Mistral-7B-Instruct-v0.2
# Submitted batch job 71236435
