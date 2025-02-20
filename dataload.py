import json
import random

import numpy as np

from tqdm import tqdm
from typing import ClassVar
from dataclasses import dataclass

from datasets import load_dataset

ANSWER_CANDIDATE_SEP = ", "

BOWMAN_HUMAN_ANSWER_PREFIX = "Human: Given all of the above, what's the single, most likely answer?"
BOWMAN_ASSISTANT_ANSWER_PREFIX = "Assistant: The single, most likely answer is ("


def chains_from_reveal_instance(chain_list):
    model_key_to_chain = {}
    for step in chain_list:
        if step['answer_model'] not in model_key_to_chain:
            model_key_to_chain[step['answer_model']] = []
        if step['is_final_rated_evidence_for_step']:
            model_key_to_chain[step['answer_model']].append(step)
    return model_key_to_chain

def get_cot(model_chain):
    parts = [step['step'] for step in model_chain]
    return '\n'.join(parts)

def load_reveal_cots(reveal_sqa_path='data/sqa_reveal/sqa_reveal_matched.json',
                    ):
    with open(reveal_sqa_path, 'r') as infile:
        reveal_sqa = json.load(infile)

    minimal_reveal = {}
    for instance_id in reveal_sqa:
        model_chain_map = chains_from_reveal_instance(reveal_sqa[instance_id]['reveal_chain'])
        for chain in model_chain_map:
            # Take only the first chain for now
            minimal_reveal[instance_id] = get_cot(model_chain_map[chain])
            break
    return minimal_reveal

def load_full_cots(model_store_path, for_ids=None):
    with open(model_store_path, 'r') as infile:
        model_outputs = json.loads(infile.readlines()[-1])
    
    if for_ids is None:
        for_ids = set(list(model_outputs.keys()))

    model_cots = {k: v for k,v in model_outputs.items() if k in for_ids}

    return model_cots

def load_inverse_cots(model_store_path, for_ids=None):
    with open(model_store_path, 'r') as infile:
        model_outputs = json.load(infile)
    
    if for_ids is None:
        for_ids = set(list(model_outputs.keys()))

    natural_cots = {k:v['cot_text'] for k,v in model_outputs.items() if k in for_ids}
    natural_predictions = {k:v['predicted_letter_index'] for k,v in model_outputs.items() if k in for_ids}
    
    biased_cots = {k:v['p_cot_text'] for k,v in model_outputs.items() if k in for_ids}
    biased_predictions = {k:v['p_predicted_letter_index'] for k,v in model_outputs.items() if k in for_ids}
    
    inverse_cots = {k:v['i_cot_text'] for k,v in model_outputs.items() if k in for_ids}
    inverse_predictions = {k:v['i_predicted_letter_index'] for k,v in model_outputs.items() if k in for_ids}
   
    model_cots = {}
    model_predictions = {}

    for k in for_ids:
        model_cots[k] = {
            'cot_text': natural_cots[k],
            'p_cot_text': biased_cots[k],
            'i_cot_text': inverse_cots[k],
        }

        model_predictions[k] = {
            'predicted_letter_index': natural_predictions[k],
            'p_predicted_letter_index': biased_predictions[k],
            'i_predicted_letter_index': inverse_predictions[k],
        }

    return model_cots, model_predictions


def load_model_cots(model_store_path, for_ids=None):
    model_predictions = {}

    with open(model_store_path, 'r') as infile:
        model_outputs = json.load(infile)
    
    if for_ids is None:
        for_ids = set(list(model_outputs.keys()))

    model_cots = {k:v['cot_text'] for k,v in model_outputs.items() if k in for_ids}

    # THIS IS A BUGFIX FOR ANSWERS INCLUDED IN COMPLETION
    for k in model_cots:
        trigger = 'Human:'
        if trigger in model_cots[k]:
            # print(f"Before: {model_cots[k]}")
            cot = model_cots[k].split(trigger)[0]
            model_cots[k] = cot
            # print(f"After: {cot}")

    model_predictions = {k:v['predicted_letter_index'] for k,v in model_outputs.items() if k in for_ids}
    # Scale them so they sum to 1
    letter_probabilities = {k:np.array(v['letter_probs'])/sum(v['letter_probs']) for k, v in model_outputs.items() if k in for_ids}
    # Scale them so they sum to 1
    completion_probabilities = {k:np.array(v['completion_probs'])/sum(v['completion_probs']) for k, v in model_outputs.items() if k in for_ids}
    

    model_probabilities = {
        'letter': letter_probabilities,
        'completion': completion_probabilities,
    }

    return model_cots, model_predictions, model_probabilities

class DataHandler:
    def __init__(self):
        super()

    def get_dataset_splits(self):
        dataset = load_dataset(self.key)
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        pass

    def make_demonstration(self, instance):
        pass

    def get_answer_letters(self, instance):
        pass

    def get_answer_choices(self, instance):
        pass

    def get_target(self, instance):
        return instance["label"]

    def label_index(self, label):
        return self.class_labels.index(label)


class MMLU(DataHandler):
    def __init__(self, subkey):
        self.key = "lukaemon/mmlu"
        self.subkey = subkey
        # TODO: initialize instances with proper IDs and use them
        # instead of the question string
        self.id_key = "input"
        self.class_labels = ["A", "B", "C", "D"]
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset(self.key, self.subkey, trust_remote_code=True)
        return dataset["train"], dataset["validation"], dataset["test"]

    def get_answer_letters(self, instance):
        return self.class_labels


    def get_answer_choices(self, instance):
        letter_choices = self.class_labels
        text_choices = [instance[L] for L in self.class_labels]
        
        answer_choices = [
            f"{l}): {a}" for l, a in zip(letter_choices, text_choices) 
        ]
        return answer_choices


    def make_bowman_demonstration(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{assistant}"
        # Format answer choices
        letter_choices = self.class_labels
        text_choices = [instance[L] for L in self.class_labels]
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["input"], answer_choices=answer_choices, assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX
        )

    # Lanham CoT prompt from Bowman et al 2022
    def make_cot_prompt(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = self.class_labels
        text_choices = [instance[L] for L in self.class_labels]
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["input"], answer_choices=answer_choices
        )

    def make_answer_prompt(self, prefix):
        # prefix = self.make_cot_prompt(instance) + response
        template = "{prefix}\n{human}\n{assistant}"
        return template.format(
            prefix=prefix,
            human=BOWMAN_HUMAN_ANSWER_PREFIX,
            assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX,
        )

    def correct_answer_letter(self, instance):
        letter = instance["target"]
        return letter


class BoolQ(DataHandler):
    letter_choices = ['A', 'B']
    text_choices = ['True', 'False']

    output_to_letter_map = {
        'True': 'A',
        'False': 'B'
    }

    def __init__(self):
        self.key = 'google/boolq'
        self.id_key = "question"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset(self.key)
        return dataset['train'], dataset['validation'], None

    def get_answer_letters(self, instance):
        return self.letter_choices # instance["choices"]["label"] -> sometimes the answers are 1,2,3,4

    def get_answer_choices(self, instance):
        answer_choices = [
            f"{l}): {a}" for l, a in zip(self.letter_choices, self.text_choices) 
        ]
        return answer_choices

    def make_bowman_demonstration(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{assistant}"
        # Format answer choices
        letter_choices = self.letter_choices
        text_choices = self.text_choices
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices, assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX
        )

    # Lanham CoT prompt from Bowman et al 2022
    def make_cot_prompt(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = self.letter_choices
        text_choices = self.text_choices
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices
        )

    def make_answer_prompt(self, prefix):
        # prefix = self.make_cot_prompt(instance) + response
        template = "{prefix}\n{human}\n{assistant}"
        return template.format(
            prefix=prefix,
            human=BOWMAN_HUMAN_ANSWER_PREFIX,
            assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX,
        )

    def correct_answer_letter(self, instance):
        ans = str(instance["answer"])
        if ans in self.output_to_letter_map:
            ans = self.output_to_letter_map[ans]
        return ans

class OpenQA(DataHandler):
    def __init__(self):
        self.key = 'allenai/openbookqa'
        self.id_key = "id"
        self.q_key = "question_stem"
        super().__init__()

    def get_question(self, instance):
        return instance[self.q_key]

    def get_dataset_splits(self):
        dataset = load_dataset(self.key)
        return dataset['train'], dataset['validation'], dataset["test"]

    def get_answer_letters(self, instance):
        return instance["choices"]["label"]

    def get_answer_choices(self, instance):
        letter_choices = instance["choices"]["label"]
        text_choices = instance["choices"]["text"]

        answer_choices = [
            f"{L}): {A}" for L, A in zip(letter_choices, text_choices)
        ]
        return answer_choices

    def correct_answer_letter(self, instance):
        return instance["answerKey"]

    def make_bowman_demonstration(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{assistant}"
        # Format answer choices
        letter_choices = instance["choices"]["label"]
        text_choices = instance["choices"]["text"]
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question_stem"], answer_choices=answer_choices, assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX
        )

    # Lanham CoT prompt from Bowman et al 2022
    # Other datasets will need ID selection
    def make_biased_cot_prompt(self, instance, target_letter=None):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{wrong_answer}\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = instance["choices"]["label"]
        text_choices = instance["choices"]["text"]

        # BOWMAN_ASSISTANT_ANSWER_PREFIX
        if target_letter is None:
            target_letter = 'B' if self.correct_answer_letter(instance) == 'A' else 'A'
        wrong_answer_index = letter_choices.index(target_letter)
        wrong_answer_text = text_choices[wrong_answer_index]
        wrong_answer_completion = f"{target_letter}): {wrong_answer_text}"
        wrong_answer = f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}{wrong_answer_completion}"

        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question_stem"], answer_choices=answer_choices,
            wrong_answer=wrong_answer
        )

    # Lanham CoT prompt from Bowman et al 2022
    def make_cot_prompt(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = instance["choices"]["label"]
        text_choices = instance["choices"]["text"]
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question_stem"], answer_choices=answer_choices
        )

    def make_answer_prompt(self, prefix):
        # prefix = self.make_cot_prompt(instance) + response
        template = "{prefix}\n{human}\n{assistant}"
        return template.format(
            prefix=prefix,
            human=BOWMAN_HUMAN_ANSWER_PREFIX,
            assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX,
        )

class Aqua(DataHandler):
    letter_choices = ['A', 'B', 'C', 'D', 'E']

    def __init__(self):
        self.key = 'aqua_rat'
        self.id_key = "question"
        self.q_key = "question"
        super().__init__()

    def get_question(self, instance):
        return instance[self.q_key]

    def get_dataset_splits(self):
        dataset = load_dataset(self.key)
        return dataset['train'], dataset['validation'], dataset["test"]

    def get_answer_letters(self, instance):
        return self.letter_choices # instance["choices"]["label"] -> sometimes the answers are 1,2,3,4

    def get_answer_choices(self, instance):
        text_choices = [o[2:] for o in instance['options']]
        answer_choices = [
            f"{l}): {a}" for l, a in zip(self.letter_choices, text_choices) 
        ]
        return answer_choices

    def make_bowman_demonstration(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{assistant}"
        # Format answer choices
        letter_choices = self.letter_choices
        # Remove the A) prefix
        text_choices = [o[2:] for o in instance['options']]
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices, assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX
        )

    # Lanham CoT prompt from Bowman et al 2022
    # Other datasets will need ID selection
    def make_biased_cot_prompt(self, instance, target_letter=None):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{wrong_answer}\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = self.letter_choices
        # Remove the A) prefix
        text_choices = [o[2:] for o in instance['options']]

        # BOWMAN_ASSISTANT_ANSWER_PREFIX
        if target_letter is None:
            target_letter = 'B' if self.correct_answer_letter(instance) == 'A' else 'A'
        wrong_answer_index = letter_choices.index(target_letter)
        wrong_answer_text = text_choices[wrong_answer_index]
        wrong_answer_completion = f"{target_letter}): {wrong_answer_text}"
        wrong_answer = f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}{wrong_answer_completion}"

        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices,
            wrong_answer=wrong_answer
        )


    # Lanham CoT prompt from Bowman et al 2022
    def make_cot_prompt(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = self.letter_choices
        # Remove the A) prefix
        text_choices = [o[2:] for o in instance['options']]

        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices
        )

    def make_answer_prompt(self, prefix):
        # prefix = self.make_cot_prompt(instance) + response
        template = "{prefix}\n{human}\n{assistant}"
        return template.format(
            prefix=prefix,
            human=BOWMAN_HUMAN_ANSWER_PREFIX,
            assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX,
        )

    def correct_answer_letter(self, instance):
        ans = instance["correct"]
        return ans


class Fermi(DataHandler):
    # This is an open-ended question dataset, not a MCQA
    pass

class Sports(DataHandler):
    # This is an open-ended dataset transformed into MCQA

    letter_choices = ['A', 'B']
    text_choices = ['yes', 'no']

    output_to_letter_map = {
        'yes': 'A',
        'no': 'B'
    }

    def __init__(self, subkey='sports_understanding'):
        self.key = 'lukaemon/bbh'
        self.subkey = subkey
        self.id_key = "input"
        self.q_key = "input"
        super().__init__()

    def get_question(self, instance):
        return instance[self.q_key]

    def get_dataset_splits(self):
        dataset = load_dataset(self.key, self.subkey)
        return None, None, dataset["test"]

    def get_answer_letters(self, instance):
        return self.letter_choices # instance["choices"]["label"] -> sometimes the answers are 1,2,3,4

    def get_answer_choices(self, instance):
        answer_choices = [
            f"{l}): {a}" for l, a in zip(self.letter_choices, self.text_choices) 
        ]
        return answer_choices

    def make_bowman_demonstration(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{assistant}"
        # Format answer choices
        letter_choices = self.letter_choices
        text_choices = self.text_choices
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["input"], answer_choices=answer_choices, assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX
        )
    
    # Lanham CoT prompt from Bowman et al 2022
    # Other datasets will need ID selection
    def make_biased_cot_prompt(self, instance, target_letter=None):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{wrong_answer}\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = self.letter_choices
        text_choices = self.text_choices

        # BOWMAN_ASSISTANT_ANSWER_PREFIX
        if target_letter is None:
            target_letter = 'B' if self.correct_answer_letter(instance) == 'A' else 'A'
        wrong_answer_index = letter_choices.index(target_letter)
        wrong_answer_text = text_choices[wrong_answer_index]
        wrong_answer_completion = f"{target_letter}): {wrong_answer_text}"
        wrong_answer = f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}{wrong_answer_completion}"

        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["input"], answer_choices=answer_choices,
            wrong_answer=wrong_answer
        )

    def get_biased_answer_letter(self, instance):
        wrong_answer_letter = 'B' if self.correct_answer_letter(instance) == 'A' else 'A'
        return wrong_answer_letter

    # Lanham CoT prompt from Bowman et al 2022
    def make_cot_prompt(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = self.letter_choices
        text_choices = self.text_choices
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["input"], answer_choices=answer_choices
        )

    def make_answer_prompt(self, prefix):
        # prefix = self.make_cot_prompt(instance) + response
        template = "{prefix}\n{human}\n{assistant}"
        return template.format(
            prefix=prefix,
            human=BOWMAN_HUMAN_ANSWER_PREFIX,
            assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX,
        )

    def correct_answer_letter(self, instance):
        ans = instance["target"]
        if ans in self.output_to_letter_map:
            ans = self.output_to_letter_map[ans]
        return ans

class ARC(DataHandler):

    number_to_letter_map = {
        '1': 'A',
        '2': 'B',
        '3': 'C',
        '4': 'D',
    }

    def __init__(self, subkey='ARC-Easy'):
        self.key = "allenai/ai2_arc"
        self.subkey = subkey
        self.id_key = "id"
        self.q_key = "question"
        super().__init__()

    def get_question(self, instance):
        return instance[self.q_key]

    def get_dataset_splits(self):
        dataset = load_dataset(self.key, self.subkey)
        return dataset["train"], dataset["validation"], dataset["test"]

    def get_answer_letters(self, instance):
        return ['A', 'B', 'C', 'D'] # instance["choices"]["label"] -> sometimes the answers are 1,2,3,4

    def get_answer_choices(self, instance):
        letter_choices = instance["choices"]["label"]
        text_choices = instance["choices"]["text"]

        answer_choices = [
            f"{l}): {a}" for l, a in zip(letter_choices, text_choices) 
        ]
        return answer_choices


    def make_bowman_demonstration(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{assistant}"
        # Format answer choices
        letter_choices = self.get_answer_letters(instance)
        text_choices = instance["choices"]["text"]
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices, assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX
        )

    # Lanham CoT prompt from Bowman et al 2022
    # Other datasets will need ID selection
    def make_biased_cot_prompt(self, instance, target_letter=None):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{wrong_answer}\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = self.get_answer_letters(instance)
        text_choices = instance["choices"]["text"]

        # BOWMAN_ASSISTANT_ANSWER_PREFIX
        if target_letter is None:
            raise ValueError('No wrong answer sampling for ARC')
        wrong_answer_index = letter_choices.index(target_letter)
        wrong_answer_text = text_choices[wrong_answer_index]
        wrong_answer_completion = f"{target_letter}): {wrong_answer_text}"
        wrong_answer = f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}{wrong_answer_completion}"

        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices,
            wrong_answer=wrong_answer
        )

    # Lanham CoT prompt from Bowman et al 2022
    def make_cot_prompt(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = self.get_answer_letters(instance)
        text_choices = instance["choices"]["text"]
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices
        )

    def make_answer_prompt(self, prefix):
        # prefix = self.make_cot_prompt(instance) + response
        template = "{prefix}\n{human}\n{assistant}"
        return template.format(
            prefix=prefix,
            human=BOWMAN_HUMAN_ANSWER_PREFIX,
            assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX,
        )

    def correct_answer_letter(self, instance):
        letter = instance["answerKey"]
        if letter in self.number_to_letter_map:
            letter = self.number_to_letter_map[letter]
        return letter


class SQA(DataHandler):

    train_path = 'data/strategyqa/strategyqa_train.json'
    test_path = 'data/strategyqa/strategyqa_test.json'

    def __init__(self):
        self.key = "strategy_qa"
        self.id_key = "qid"
        self.q_key = "question"
        self.letter_choices = ["A", "B"]
        self.text_choices = ["Yes", "No"]
        super().__init__()

    def get_question(self, instance):
        return instance[self.q_key]

    def get_dataset_splits(self):
        with open(self.train_path, 'r') as infile:
            sqa_train = json.load(infile)

        # Use the first 8 instances for demonstrations
        sqa_valid = sqa_train[8:]
        sqa_train = sqa_train[:8]

        with open(self.test_path, 'r') as infile:
            sqa_test = json.load(infile)

        return sqa_train, sqa_valid, sqa_test

    def get_answer_letters(self, instance):
        return self.letter_choices

    def get_answer_choices(self, instance):
        answer_choices = [
            f"{l}): {a}" for l, a in zip(self.letter_choices, self.text_choices) 
        ]
        return answer_choices

    def get_biased_answer_letter(self, instance):
        wrong_answer_letter = 'B' if self.correct_answer_letter(instance) == 'A' else 'A'
        return wrong_answer_letter

    def make_bowman_demonstration(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{assistant}"
        # Format answer choices
        answer_choices = [f"({L}): {A}" for L, A in zip(self.letter_choices, self.text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance[self.q_key], answer_choices=answer_choices, assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX
        )
    
    # Lanham CoT prompt from Bowman et al 2022
    # Other datasets will need ID selection
    def make_biased_cot_prompt(self, instance, target_letter=None):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{wrong_answer}\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = self.letter_choices
        text_choices = self.text_choices

        # BOWMAN_ASSISTANT_ANSWER_PREFIX
        if target_letter is None:
            target_letter = 'B' if self.correct_answer_letter(instance) == 'A' else 'A'
        wrong_answer_index = letter_choices.index(target_letter)
        wrong_answer_text = text_choices[wrong_answer_index]
        wrong_answer_completion = f"{target_letter}): {wrong_answer_text}"
        wrong_answer = f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}{wrong_answer_completion}"

        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance[self.q_key], answer_choices=answer_choices,
            wrong_answer=wrong_answer
        )

    # Lanham CoT prompt from Bowman et al 2022
    def make_cot_prompt(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\nAssistant: Let's think step by step:\n"
        # Format answer choices
        answer_choices = [f"({L}): {A}" for L, A in zip(self.letter_choices, self.text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance[self.q_key], answer_choices=answer_choices
        )

    def make_answer_prompt(self, prefix):
        # prefix = self.make_cot_prompt(instance) + response
        template = "{prefix}\n{human}\n{assistant}"
        return template.format(
            prefix=prefix,
            human=BOWMAN_HUMAN_ANSWER_PREFIX,
            assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX,
        )

    def correct_answer_letter(self, instance):
        return "A" if instance['answer'] else "B"


class CQA(DataHandler):
    def __init__(self):
        self.key = "commonsense_qa"
        self.id_key = "id"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset(self.key)
        return dataset['train'], dataset['validation'], dataset['test']
    
    def get_answer_letters(self, instance):
        return instance["choices"]["label"]

    def get_answer_choices(self, instance):
        letter_choices = instance["choices"]["label"]
        text_choices = instance["choices"]["text"]
        answer_choices = [
            f"{l}): {a}" for l, a in zip(letter_choices, text_choices) 
        ]
        return answer_choices

    def make_bowman_demonstration(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{assistant}"
        # Format answer choices
        letter_choices = instance["choices"]["label"]
        text_choices = instance["choices"]["text"]
        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices, assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX
        )

    # Lanham CoT prompt from Bowman et al 2022
    # Other datasets will need ID selection
    def make_biased_cot_prompt(self, instance, target_letter):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\n{wrong_answer}\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = instance["choices"]["label"]
        text_choices = instance["choices"]["text"]

        # BOWMAN_ASSISTANT_ANSWER_PREFIX
        wrong_answer_index = letter_choices.index(target_letter)
        wrong_answer_text = text_choices[wrong_answer_index]
        wrong_answer_completion = f"{target_letter}): {wrong_answer_text}"
        wrong_answer = f"{BOWMAN_ASSISTANT_ANSWER_PREFIX}{wrong_answer_completion}"

        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices,
            wrong_answer=wrong_answer
        )
    
    # Lanham CoT prompt from Bowman et al 2022
    def make_cot_prompt(self, instance):
        template = "Human: Question: {question}\n\nChoices:\n{answer_choices}\n\nAssistant: Let's think step by step:\n"
        # Format answer choices
        letter_choices = instance["choices"]["label"]
        text_choices = instance["choices"]["text"]

        answer_choices = [f"({L}): {A}" for L, A in zip(letter_choices, text_choices)]
        answer_choices = "\n".join(answer_choices)
        return template.format(
            question=instance["question"], answer_choices=answer_choices
        )


    def make_answer_prompt(self, prefix):
        # prefix = self.make_cot_prompt(instance) + response
        template = "{prefix}\n{human}\n{assistant}"
        return template.format(
            prefix=prefix,
            human=BOWMAN_HUMAN_ANSWER_PREFIX,
            assistant=BOWMAN_ASSISTANT_ANSWER_PREFIX,
        )

    def correct_answer_letter(self, instance):
        return instance["answerKey"]

class LogiQA(DataHandler):
    question_key = "context"

    def __init__(self):
        self.key = "lucasmccabe/logiqa"
        # TODO: initialize instances with proper IDs and use them
        # instead of the query string
        self.id_key = "query"
        self.class_labels = ["A", "B", "C", "D"]
        super().__init__()

    def make_prompt_instance(self, instance):
        template = (
            "Q: {context}\n{query}\n" + "Answer choices: {answer_choices}\n" + "A:"
        )
        # Format answer choices
        letter_choices = self.class_labels
        text_choices = instance["options"]
        answer_choices = [f"{A} ({L})" for L, A in zip(letter_choices, text_choices)]
        answer_choices = ANSWER_CANDIDATE_SEP.join(answer_choices)

        return template.format(
            context=instance["context"],
            query=instance["query"],
            answer_choices=answer_choices,
        )

    def make_demonstration(self, instance):
        template = (
            "Q: {context}\n{query}\n"
            + "Answer choices: {answer_choices}\n"
            + "A: {answer}\n"
        )
        # Format answer choices
        letter_choices = self.class_labels
        text_choices = instance["options"]
        answer_choices = [f"{A} ({L})" for L, A in zip(letter_choices, text_choices)]
        answer_choices = ANSWER_CANDIDATE_SEP.join(answer_choices)
        answer = self.has_correct_answer(instance)

        return template.format(
            context=instance["context"],
            query=instance["query"],
            answer_choices=answer_choices,
            answer=answer,
        )

    def get_target(self, instance):
        correct_answer_index = instance["correct_option"]
        return instance["options"][correct_answer_index]

    def has_correct_answer(self, instance):
        correct_answer_index = instance["correct_option"]
        answer = f"{instance['options'][correct_answer_index]} ({self.class_labels[correct_answer_index]})"
        return answer


class RTE(DataHandler):
    def __init__(self):
        self.class_labels = ["True", "False"]
        self.id_key = "idx"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "rte")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "{premise}\n" + "Question: {hypothesis} True or False?\n" + "Answer: "
        )

        return template.format(
            premise=instance["sentence1"],
            hypothesis=instance["sentence2"],
        )

    def make_demonstration(self, instance):
        template = (
            "{premise}\n"
            + "Question: {hypothesis} True or False?\n"
            + "Answer: {label}\n"
        )

        return template.format(
            premise=instance["sentence1"],
            hypothesis=instance["sentence2"],
            label=self.has_correct_answer(instance),
        )

    def has_correct_answer(self, instance):
        return self.class_labels[instance["label"]]


class QNLI(DataHandler):
    def __init__(self):
        self.class_labels = ["Yes", "No"]
        self.id_key = "idx"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "qnli")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Question: {question}\n"
            + "Sentence: {sentence}\n"
            + "Does the sentence answer the question?\n"
            + "Answer: "
        )

        return template.format(
            question=instance["question"],
            sentence=instance["sentence"],
        )

    def make_demonstration(self, instance):
        template = (
            "Question: {question}\n"
            + "Sentence: {sentence}\n"
            + "Does the sentence answer the question?\n"
            + "Answer: {label}\n"
        )

        return template.format(
            question=instance["question"],
            sentence=instance["sentence"],
            label=self.has_correct_answer(instance),
        )

    def has_correct_answer(self, instance):
        return self.class_labels[instance["label"]]


class QQP(DataHandler):
    def __init__(self):
        self.class_labels = ["No", "Yes"]
        self.id_key = "idx"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "qqp")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Question 1: {question1}\n"
            + "Question 2: {question2}\n"
            + "Question: Do both questions ask the same thing? "
            + "Answer: "
        )
        return template.format(
            question1=instance["question1"],
            question2=instance["question2"],
        )

    def make_demonstration(self, instance):
        template = (
            "Question 1: {question1}\n"
            + "Question 2: {question2}\n"
            + "Question: Do both questions ask the same thing? "
            + "Answer: {label}\n"
        )

        return template.format(
            question1=instance["question1"],
            question2=instance["question2"],
            label=self.has_correct_answer(instance),
        )

    def has_correct_answer(self, instance):
        return self.class_labels[instance["label"]]


class SST(DataHandler):
    def __init__(self):
        self.class_labels = ["Negative", "Positive"]
        self.id_key = "idx"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "sst2")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Sentence: {sentence}\n" + "Question: Positive or Negative? " + "Answer: "
        )
        return template.format(
            sentence=instance["sentence"],
        )

    def make_demonstration(self, instance):
        template = (
            "Sentence: {sentence}\n"
            + "Question: Positive or Negative? "
            + "Answer: {label}\n"
        )
        return template.format(
            sentence=instance["sentence"],
            label=self.has_correct_answer(instance),
        )

    def has_correct_answer(self, instance):
        return self.class_labels[instance["label"]]


class COLA(DataHandler):
    def __init__(self):
        self.class_labels = ["No", "Yes"]
        self.id_key = "idx"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "cola")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Sentence: {sentence}\n"
            + "Question: Is this sentence linguistically acceptable? "
            + "Answer: "
        )
        return template.format(
            sentence=instance["sentence"],
        )

    def make_demonstration(self, instance):
        template = (
            "Sentence: {sentence}\n"
            + "Question: Is this sentence linguistically acceptable? "
            + "Answer: {label}\n"
        )
        return template.format(
            sentence=instance["sentence"],
            label=self.has_correct_answer(instance),
        )

    def has_correct_answer(self, instance):
        return self.class_labels[instance["label"]]


class MRPC(DataHandler):
    def __init__(self):
        self.class_labels = ["No", "Yes"]
        self.id_key = "idx"
        super().__init__()

    def get_dataset_splits(self):
        dataset = load_dataset("nyu-mll/glue", "mrpc")
        return dataset["train"], dataset["validation"], dataset["test"]

    def make_prompt_instance(self, instance):
        template = (
            "Sentence 1: {sentence1}\n"
            + "Sentence 2: {sentence2}\n"
            + "Question: Do both sentences say the same thing? "
            + "Answer: "
        )

        return template.format(
            sentence1=instance["sentence1"],
            sentence2=instance["sentence2"],
        )

    def make_demonstration(self, instance):
        template = (
            "Sentence 1: {sentence1}\n"
            + "Sentence 2: {sentence2}\n"
            + "Question: Do both sentences say the same thing? "
            + "Answer: {label}\n"
        )

        return template.format(
            sentence1=instance["sentence1"],
            sentence2=instance["sentence2"],
            label=self.has_correct_answer(instance),
        )

    def has_correct_answer(self, instance):
        return self.class_labels[instance["label"]]


DATASETS = {
    "sqa": SQA(),
    "cqa": CQA(),
    "arc-easy": ARC("ARC-Easy"),
    "arc-challenge": ARC("ARC-Challenge"),
    "qqp": QQP(),
    "rte": RTE(),
    "sst": SST(),
    "mrpc": MRPC(),
    "cola": COLA(),
    "logiqa": LogiQA(),
    "mmlu-clinic": MMLU("clinical_knowledge"),
    "mmlu-math": MMLU("elementary_mathematics"),
    "sports": Sports(),
    "aqua": Aqua(),
    "openbook": OpenQA(),
    "boolq": BoolQ(),
    "mmlu-clinical": MMLU('clinical_knowledge'),
    "mmlu-genetics": MMLU("medical_genetics"),
    "mmlu-virology": MMLU('virology'),
    "mmlu-econometrics": MMLU('econometrics'),
    "mmlu-ccs": MMLU('college_computer_science'),
    "mmlu-biology": MMLU('high_school_biology'),
    "mmlu-philosophy": MMLU('philosophy'),
    "mmlu-prehistory": MMLU('prehistory'),
    "mmlu-sexuality": MMLU('human_sexuality'),
    "mmlu-medicine": MMLU('college_medicine'),
    "mmlu-geography": MMLU('high_school_geography'),
    "mmlu-aging": MMLU('human_aging'),
    "mmlu-religions": MMLU('human_religions'),
    "mmlu-sociology": MMLU('sociology'),
}

"""
MMLU tasks
tasks =  [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions"
]
"""
@dataclass
class CQAInstance:
    original_instance: dict
    question: str
    possible_answers: str
    query: str = ""
    answer_text: str = ""
    answer_index: str = ""
    fake_answer: str = ""
    dataset: ClassVar[str] = "cqa"
    prefix_letters: ClassVar[list] = ['(a)', '(b)', '(c)', '(d)', '(e)']

    # Templates depend on the dataset instance dataclass
    # IDK a better way around this, they need to access the attributes.
    def zero_shot_template(self):
        template = "Q: {question}\n" \
                    + "Answer Choices: {answer_choices}\n"
        return template.format(question=self.question, answer_choices=self.answer_choices)

    def self_explain_template(self):
        template = "Q: {question}\n" \
                    + "Answer Choices: {answer_choices}\n"
        return template.format(question=self.question, answer_choices=self.answer_choices)

    def label_explain_template(self, correct_answer=True):
        template = "Q: {question}\n" \
                    + "Answer Choices: {answer_choices}\n" \
                    + "A: {answer}\n" \
                    + "Explanation: "
        if correct_answer and self.answer_text:
            return template.format(question=self.question, answer_choices=self.answer_choices,
                answer=self.answer_text)
        else:
            incorrect_answers = list(self.answer_choices_list)
            random.shuffle(incorrect_answers)
            # This will work even in the case of the test set
            for (answer, letter) in incorrect_answers:
                if self.answer_text == answer: continue
                a_label = f"{letter} {answer}"
            self.fake_answer = a_label
            # Change to random.choice from possible answers
            # Or a predefined answer slot?
            # Or the most likely answer according to some predictive model...
            return template.format(answer=a_label, answer_choices=self.answer_choices, question=self.question)

    def label_explain_template_old(self, correct_answer=True):
        template = "Why is {answer} the answer to " \
                    + "\"{question}\"\n" \
                    + "Explanation: "
        if correct_answer and self.answer_text:
            return template.format(answer=self.answer_text, question=self.question)
        else:
            incorrect_answers = self.possible_answers
            # This will work even in the case of the test set
            if self.answer_text in incorrect_answers:
                incorrect_answers.remove(self.answer_text)
            a_label = random.choice(incorrect_answers)
            
            # Change to random.choice from possible answers
            # Or a predefined answer slot?
            # Or the most likely answer according to some predictive model...
            return template.format(answer=a_label, question=self.question)

    def cot_template(self):
        template = "Q: {question}\n" \
                    + "Answer Choices: {answer_choices}\n"
        return template.format(question=self.question, answer_choices=self.answer_choices)

    def fit_into_template(self, task_type='zero', prefix=""):

        if task_type == 'zero':
            # Handle prefix here as well?
            return self.zero_shot_template()
        elif task_type == 'zero-expl':
            return self.self_explain_template()
        elif task_type == 'label-expl':
            return self.label_explain_template(correct_answer=False)
        elif task_type == 'cot':
            return self.cot_template()
        else:
            # Raise ValueError
            return ""

    @property
    def answer_choices_list(self):
        # Lengths have to match
        assert len(self.prefix_letters) == len(self.possible_answers)

        answers = []
        for letter, answer in zip(self.prefix_letters, self.possible_answers):
            answers.append((answer, letter))
        return answers

    @property
    def answer_choices(self):
        before = True
        # Lengths have to match
        assert len(self.prefix_letters) == len(self.possible_answers)

        answer_line = ""
        for letter, answer in zip(self.prefix_letters, self.possible_answers):
            if before:
                answer_line += letter + " " + answer + " "
            else:
                answer_line += answer + " " + letter + " "
        answer_line = answer_line.strip()
        return answer_line

if __name__ == '__main__':
  pass
    