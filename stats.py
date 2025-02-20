import numpy as np
from util import renorm

def efficacy_per_instance(epoch_results):
    return [r['cot_step_prob'][0] for _, r in epoch_results.items()]

def specificity_per_instance(epoch_results):
    specificity = [100.]
    initial = a(epoch_results['0']['specificity_preds'])
    for i in range(1, len(epoch_results)):
        preds = a(epoch_results[str(i)]['specificity_preds'])
        specificity.append((initial == preds).sum() / len(preds) * 100.)
    return specificity

def average_efficacy(results, step=False):
    # This is just average efficacy for each iter across the entire dataset.
    eff_through_iters = {}

    for unlearned_step in results:
        unlearning_results = unlearned_step['unlearning_results']
        # Change in probability of entire CoT or only the unlearned step
        key = 'cot_prob' if not step else 'cot_step_prob'
        probabilities = [np.exp(r[key][0]) for _, r in unlearning_results.items()]

        # Initial probability
        p_0 = probabilities[0]
        for i, p_i in enumerate(probabilities):
            if i not in eff_through_iters:
                eff_through_iters[i] = []

            if eff_through_iters[i]:
                eff_through_iters[i].append( (1 - p_i/p_0)* 100.) # Reduction in sequence probability
            else:
                eff_through_iters[i].append(0.)

    tot = np.concatenate(list(eff_through_iters.values()))

    return np.mean(tot), list(eff_through_iters.values())[0]

def efficacy_reduction_per_instance_scaled(epoch_results):
    step_probabilities = [np.exp(r['cot_step_prob'][0]) for _, r in epoch_results.items()]
    probability_reduction = [step_probabilities[0] - s for s in step_probabilities[1:]] 
    return probability_reduction

a = np.array

def unique_instances(result_dict):
    unique_ids = set()
    for an_inst in result_dict:
        unique_ids.add(an_inst['question'])
    return len(unique_ids)

def instance_specificity(instance_outputs):
    # Compute whether predictions have changed from ones without unlearning
    specificity = []
    initial_predictions = a(instance_outputs['0']['specificity_preds'])
    for i in range(1, len(instance_outputs)):
        preds = a(instance_outputs[str(i)]['specificity_preds'])
        specificity.append((initial_predictions == preds).sum() / len(preds) * 100.)
    return specificity

def compute_specificity(results):
    spec_through_iters = {}
    for res in results:
        unlearning_results = res['unlearning_results']
        spec = instance_specificity(unlearning_results)
        for i, s in enumerate(spec):
            if i not in spec_through_iters:
                spec_through_iters[i] = []
            spec_through_iters[i].append(s)
    avg_spec_through_iters = {
        k:np.mean(v) for k, v in spec_through_iters.items()
    }
    tot = np.concatenate(list(spec_through_iters.values()))

    return np.mean(tot), list(avg_spec_through_iters.values())

def instance_changed_prediction(epoch_results):
    # Iteration zero has the evaluation output before unlearning
    preds = [np.argmax(r['probs']) for _, r in epoch_results.items()]
    flips = [p != preds[0] for p in preds]
    return any(flips), flips

def changed_prediction(results):
    # Instance level hard faithfulness
    count = 0
    unique_flips = set()
    unique_qs = set()
    for result in results:
        unique_qs.add(result['question'])
        unlearning_results = result['unlearning_results']

        has_change, _ = instance_changed_prediction(unlearning_results)
        
        if has_change:
            count += 1
            unique_flips.add(result['question'])

    fs_num = len(unique_flips)/len(unique_qs)*100.0
    return fs_num

def make_stats(per_instance_results):
    n_steps = len(per_instance_results)
    n_instances = unique_instances(per_instance_results)
    # faithfulness
    faithfulness = changed_prediction(per_instance_results)
    # specificity
    specificity, _ = compute_specificity(per_instance_results)
    # efficacy
    efficacy, _ = average_efficacy(per_instance_results, step=True)
    res_dict = {
        'n_instances': n_instances,
        'faithfulness': faithfulness,
        'efficacy': efficacy,
        'specificity': specificity,
        'n_cot_steps': n_steps, 
    }
    return res_dict

def average_mass_shift(a_result, do_print=False):
    # Negative -> probability gets added, positive -> probability gets removed
    unlearning_results = a_result['unlearning_results']
    probs = [renorm(r['probs']) for  _, r in unlearning_results.items()]

    initial_pred = np.argmax(probs[0])
    initial_mass = probs[0][initial_pred]
    dmass = [initial_mass - m[initial_pred] for m in probs[1:]]

    if do_print:
        print(probs, '\n', initial_mass, dmass)

    return np.mean(dmass)

def max_mass_shift(a_result, do_print=False):
    # Negative -> probability gets added, positive -> probability gets removed
    unlearning_results = a_result['unlearning_results']
    probs = [renorm(r['probs']) for  _, r in unlearning_results.items()]

    initial_pred = np.argmax(probs[0])
    initial_mass = probs[0][initial_pred]
    dmass = [initial_mass - m[initial_pred] for m in probs[1:]]

    if do_print:
        print(probs, '\n', initial_mass, dmass)
    return max(dmass)

