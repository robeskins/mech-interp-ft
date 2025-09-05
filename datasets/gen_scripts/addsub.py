import random
from transformers import AutoTokenizer
from pathlib import Path
from utils import is_single_token, compare_token_count, generate_csvs

def get_valid_combos(tokenizer: AutoTokenizer):
    add_combos = [
        (a, b) for a in range(1, 100) for b in range(1, 100)
        if is_single_token(str(a + b), tokenizer)
    ]
    
    sub_combos = [
        (a, b) for a in range(1, 100) for b in range(1, 100)
        if a > b and is_single_token(str(a - b), tokenizer)
    ]

    random.shuffle(add_combos)
    random.shuffle(sub_combos)

    return add_combos, sub_combos

def corrupt_result(b: int):
    while True:
        c = random.randint(10, 99)
        if c != b and compare_token_count(b, c, tokenizer):
            return c

def create_prompts(prefix_prompt: str, total_examples: int, tokenizer: AutoTokenizer):
    add_combos, sub_combos = get_valid_combos(tokenizer)
    prompts = []
    add_counter = 0
    sub_counter = 0
    half = total_examples // 2

    if len(add_combos) < half or len(sub_combos) < half:
        raise ValueError("Not enough combos to generate the required number of examples.")

    for i in range(half):
        a, b = add_combos[i]
        prompt_add = prefix_prompt + f'{a} + {b} ='
        answer_add = a + b
        c = corrupt_result(b)
        corrupted_add = prefix_prompt + f'{a} + {c} ='
        
        if not compare_token_count(prompt_add, corrupted_add, tokenizer):
            raise ValueError("Token count different for add prompt and corrupted!")

        prompts.append([prompt_add, corrupted_add, answer_add])
        add_counter += 1

        a, b = sub_combos[i]
        prompt_sub = prefix_prompt + f'{a} - {b} ='
        answer_sub = a - b
        c = corrupt_result(b)
        corrupted_sub = prefix_prompt + f'{a} - {c} ='
        if not compare_token_count(prompt_sub, corrupted_sub, tokenizer):
            raise ValueError("Token count different for sub prompt and corrupted!")

        prompts.append([prompt_sub, corrupted_sub, answer_sub])
        sub_counter += 1

    if sub_counter != total_examples / 2:
        raise ValueError("Expected half of the examples to be from subtraction, but got a different count. May not be enough combos for the amount of exmaples.")
    if add_counter != total_examples / 2:
        raise ValueError("Expected half of the examples to be from addition, but got a different count. May not be enough combos for the amount of exmaples.")

    return prompts

total_examples = 4000
prefix_prompt = 'Solve the following and respond with only the final answer: '
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b-deduped")
prompts = create_prompts(prefix_prompt, total_examples, tokenizer)

generate_csvs(prompts = prompts,
              split_percent = 0.9,
              results_dir = '/homes/rje41/mech-interp-ft/experiment2/task_set_0_spacing/AddSub/datasets_csv')